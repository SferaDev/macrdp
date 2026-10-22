//! Changed regions of EGFX AVC420 frames.
//!
//! Every H.264 frame still encodes the whole surface; the AVC420 metablock's
//! `regionRects` tell the client which parts of the decoded picture to copy
//! into its surface, so a client only converts and blits what changed
//! (MS-RDPEGFX 2.2.4.4.1). A pixel that changed but is outside every rect
//! stays stale on the client, so everything here errs towards covering more.
//!
//! The capture side collects the changed rects of each frame into a
//! [`Damage`]; [`DamageTracker`] pairs it with the encoded frame by
//! presentation timestamp and decides the rects the frame ships with.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// H.264 macroblock size. Rects are grown to whole macroblocks so a changed
/// pixel's block is copied whole.
const MACROBLOCK: u32 = 16;

/// Margin added around each changed rect before macroblock alignment: the
/// deblocking filter changes up to 3 pixels into the neighbouring block, and
/// the encoder may spend bits on blocks next to a change.
const MARGIN: u32 = MACROBLOCK;

/// Most rects a frame ships with. More are merged into bounding boxes.
pub const MAX_REGION_RECTS: usize = 8;

/// Above this many input rects the greedy merge would cost too much per
/// frame; their bounding box ships instead.
const MAX_MERGE_INPUT: usize = 64;

/// Rects covering at least this percentage of the surface ship as the whole
/// surface: the client saves little and one rect is simpler.
const FULL_SURFACE_PCT: u64 = 50;

/// Raw rects a [`Damage`] keeps before collapsing to their bounding box, so a
/// long run of dropped frames can't grow it without bound.
const MAX_ACCUMULATED_RECTS: usize = 256;

/// Frames after a keyframe that ship whole: the encoder refines the picture
/// over the frames that follow an IDR, all over the surface.
const FULL_FRAMES_AFTER_KEYFRAME: u32 = 8;

/// Submitted frames a tracker remembers; past this the oldest collapse into
/// a full-surface entry.
const MAX_QUEUED: usize = 64;

/// A rectangle in surface pixels, right and bottom exclusive (the
/// `RDPGFX_RECT16` convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl Rect {
    pub fn from_xywh(x: usize, y: usize, w: usize, h: usize) -> Self {
        let c = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        Self {
            left: c(x),
            top: c(y),
            right: c(x.saturating_add(w)),
            bottom: c(y.saturating_add(h)),
        }
    }

    fn is_empty(&self) -> bool {
        self.left >= self.right || self.top >= self.bottom
    }

    fn area(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            u64::from(self.right - self.left) * u64::from(self.bottom - self.top)
        }
    }

    fn union(&self, o: &Rect) -> Rect {
        Rect {
            left: self.left.min(o.left),
            top: self.top.min(o.top),
            right: self.right.max(o.right),
            bottom: self.bottom.max(o.bottom),
        }
    }

    fn intersection_area(&self, o: &Rect) -> u64 {
        Rect {
            left: self.left.max(o.left),
            top: self.top.max(o.top),
            right: self.right.min(o.right),
            bottom: self.bottom.min(o.bottom),
        }
        .area()
    }

    fn clip(&self, width: u32, height: u32) -> Rect {
        Rect {
            left: self.left.min(width),
            top: self.top.min(height),
            right: self.right.min(width),
            bottom: self.bottom.min(height),
        }
    }

    /// Grown by `margin` and out to the macroblock grid, within the surface.
    fn aligned(&self, margin: u32, width: u32, height: u32) -> Rect {
        let down = |v: u32| v.saturating_sub(margin) / MACROBLOCK * MACROBLOCK;
        let up = |v: u32| v.saturating_add(margin).div_ceil(MACROBLOCK) * MACROBLOCK;
        Rect {
            left: down(self.left),
            top: down(self.top),
            right: up(self.right),
            bottom: up(self.bottom),
        }
        .clip(width, height)
    }
}

/// What changed in a frame, or since the last frame the encoder took.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Damage {
    /// Unknown or everything: the frame ships as the whole surface.
    Full,
    /// These rects, in surface pixels. Empty means nothing changed.
    Rects(Vec<Rect>),
}

impl Default for Damage {
    fn default() -> Self {
        Damage::Rects(Vec::new())
    }
}

impl Damage {
    pub fn is_empty(&self) -> bool {
        matches!(self, Damage::Rects(r) if r.is_empty())
    }

    pub fn add_rects(&mut self, rects: impl IntoIterator<Item = Rect>) {
        if let Damage::Rects(own) = self {
            own.extend(rects.into_iter().filter(|r| !r.is_empty()));
            if own.len() > MAX_ACCUMULATED_RECTS {
                let bbox = own.iter().skip(1).fold(own[0], |a, r| a.union(r));
                *own = vec![bbox];
            }
        }
    }

    pub fn add(&mut self, other: &Damage) {
        match other {
            Damage::Full => *self = Damage::Full,
            Damage::Rects(r) => self.add_rects(r.iter().copied()),
        }
    }
}

/// The rects a frame of a `width`×`height` surface ships with for `rects`
/// of changed pixels: clipped, grown to whole macroblocks, merged to at most
/// [`MAX_REGION_RECTS`]. `None` means the whole surface.
pub fn region_rects(rects: &[Rect], width: u16, height: u16) -> Option<Vec<Rect>> {
    let (w, h) = (u32::from(width), u32::from(height));
    let mut out: Vec<Rect> = rects
        .iter()
        .map(|r| r.clip(w, h))
        .filter(|r| !r.is_empty())
        .map(|r| r.aligned(MARGIN, w, h))
        .collect();
    if out.is_empty() {
        return None;
    }
    if out.len() > MAX_MERGE_INPUT {
        let bbox = out.iter().skip(1).fold(out[0], |a, r| a.union(r));
        out = vec![bbox];
    }
    merge(&mut out, MAX_REGION_RECTS);
    let covered: u64 = out.iter().map(Rect::area).sum();
    if covered * 100 >= u64::from(w) * u64::from(h) * FULL_SURFACE_PCT {
        return None;
    }
    Some(out)
}

/// Merges the pair whose bounding box adds the least uncovered area, as long
/// as there are more than `max` rects or a merge adds nothing.
fn merge(rects: &mut Vec<Rect>, max: usize) {
    while rects.len() > 1 {
        let mut best = (u64::MAX, 0, 0);
        for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let (a, b) = (rects[i], rects[j]);
                let covered = a.area() + b.area() - a.intersection_area(&b);
                let waste = a.union(&b).area().saturating_sub(covered);
                if waste < best.0 {
                    best = (waste, i, j);
                }
            }
        }
        let (waste, i, j) = best;
        if waste > 0 && rects.len() <= max {
            break;
        }
        let b = rects.swap_remove(j);
        rects[i] = rects[i].union(&b);
    }
}

/// Pairs each encoded frame of one encoder with the damage of the frames
/// submitted up to it, and picks the rects it ships with. VideoToolbox can
/// drop a frame without output, so a shipped frame carries the damage of
/// every earlier frame that never shipped.
#[derive(Debug, Clone)]
pub struct DamageTracker {
    enabled: bool,
    full_refresh: Option<Duration>,
    queue: VecDeque<(i64, Damage)>,
    frames_since_keyframe: u32,
    last_full: Option<Instant>,
}

impl DamageTracker {
    /// `full_refresh` ships a frame whole when the last whole one is older,
    /// so encoder refinements outside the changed rects reach the client.
    pub fn new(enabled: bool, full_refresh: Option<Duration>) -> Self {
        Self {
            enabled,
            full_refresh,
            queue: VecDeque::new(),
            frames_since_keyframe: 0,
            last_full: None,
        }
    }

    /// From `MACRDP_EGFX_DIRTY_RECTS` (default on) and
    /// `MACRDP_EGFX_FULL_REFRESH_MS` (default 1000, 0 = never).
    pub fn from_env() -> Self {
        let enabled = std::env::var("MACRDP_EGFX_DIRTY_RECTS")
            .map(|v| {
                !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
            })
            .unwrap_or(true);
        let refresh_ms = std::env::var("MACRDP_EGFX_FULL_REFRESH_MS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(1000);
        Self::new(
            enabled,
            (refresh_ms > 0).then(|| Duration::from_millis(refresh_ms)),
        )
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Records the damage of the frame submitted with presentation time `pts`.
    pub fn submitted(&mut self, pts: i64, damage: Damage) {
        if !self.enabled {
            return;
        }
        self.queue.push_back((pts, damage));
        if self.queue.len() > MAX_QUEUED {
            self.queue.pop_front();
            if let Some(front) = self.queue.front_mut() {
                front.1 = Damage::Full;
            }
        }
    }

    /// The rects the encoded frame `pts` ships with; `None` = whole surface.
    pub fn regions(
        &mut self,
        pts: i64,
        is_keyframe: bool,
        width: u16,
        height: u16,
        now: Instant,
    ) -> Option<Vec<Rect>> {
        if !self.enabled {
            return None;
        }
        let mut damage = Damage::default();
        let mut found = false;
        while self.queue.front().is_some_and(|(p, _)| *p <= pts) {
            let (p, d) = self.queue.pop_front().expect("front checked");
            damage.add(&d);
            found |= p == pts;
        }
        if is_keyframe {
            self.frames_since_keyframe = 0;
        } else {
            self.frames_since_keyframe = self.frames_since_keyframe.saturating_add(1);
        }
        let stale = match (self.full_refresh, self.last_full) {
            (Some(every), Some(at)) => now.saturating_duration_since(at) >= every,
            (Some(_), None) => true,
            (None, _) => false,
        };
        let regions = if !found
            || is_keyframe
            || stale
            || self.frames_since_keyframe <= FULL_FRAMES_AFTER_KEYFRAME
        {
            None
        } else {
            match &damage {
                Damage::Full => None,
                Damage::Rects(r) => region_rects(r, width, height),
            }
        };
        if regions.is_none() {
            self.last_full = Some(now);
        }
        regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(left: u32, top: u32, right: u32, bottom: u32) -> Rect {
        Rect {
            left,
            top,
            right,
            bottom,
        }
    }

    fn covers(rects: &[Rect], x: u32, y: u32) -> bool {
        rects
            .iter()
            .any(|r| x >= r.left && x < r.right && y >= r.top && y < r.bottom)
    }

    #[test]
    fn aligns_to_macroblocks_with_a_margin() {
        let out = region_rects(&[r(100, 50, 102, 66)], 1920, 1080).unwrap();
        assert_eq!(out, vec![r(80, 32, 128, 96)]);
        for rect in &out {
            assert_eq!(rect.left % 16, 0);
            assert_eq!(rect.top % 16, 0);
            assert_eq!(rect.right % 16, 0);
            assert_eq!(rect.bottom % 16, 0);
        }
    }

    #[test]
    fn clips_to_the_surface_edges() {
        // 1080 isn't a multiple of 16: the bottom edge is clipped, not padded.
        let out = region_rects(&[r(1910, 1075, 1950, 1200)], 1920, 1080).unwrap();
        assert_eq!(out, vec![r(1888, 1056, 1920, 1080)]);
        let out = region_rects(&[r(0, 0, 3, 3)], 1920, 1080).unwrap();
        assert_eq!(out, vec![r(0, 0, 32, 32)]);
    }

    #[test]
    fn no_usable_rects_ship_the_whole_surface() {
        assert_eq!(region_rects(&[r(5, 5, 5, 9)], 64, 64), None);
        assert_eq!(region_rects(&[r(100, 100, 120, 120)], 64, 64), None);
        assert_eq!(region_rects(&[], 64, 64), None);
    }

    #[test]
    fn merges_overlapping_rects_without_waste() {
        let out =
            region_rects(&[r(100, 100, 110, 110), r(105, 105, 120, 118)], 1920, 1080).unwrap();
        assert_eq!(out, vec![r(80, 80, 144, 144)]);
    }

    #[test]
    fn keeps_distant_rects_apart() {
        let out = region_rects(&[r(0, 0, 10, 10), r(1800, 1000, 1810, 1010)], 1920, 1080).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn merges_down_to_the_cap_and_still_covers_every_input() {
        let input: Vec<Rect> = (0..30u32)
            .map(|i| {
                let (x, y) = ((i * 397) % 3300, (i * 211) % 1400);
                r(x, y, x + 5, y + 3)
            })
            .collect();
        let out = region_rects(&input, 3440, 1440).unwrap();
        assert!(out.len() <= MAX_REGION_RECTS);
        for i in &input {
            for (x, y) in [(i.left, i.top), (i.right - 1, i.bottom - 1)] {
                assert!(covers(&out, x, y), "({x},{y}) not covered by {out:?}");
            }
        }
    }

    #[test]
    fn too_many_rects_collapse_to_their_bounding_box() {
        let input: Vec<Rect> = (0..100u32).map(|i| r(i * 32, 64, i * 32 + 2, 66)).collect();
        let out = region_rects(&input, 3440, 1440).unwrap();
        assert_eq!(out, vec![r(0, 48, 3200, 96)]);
    }

    #[test]
    fn a_large_change_ships_the_whole_surface() {
        assert_eq!(region_rects(&[r(0, 0, 1000, 1000)], 1920, 1080), None);
        assert!(region_rects(&[r(0, 0, 900, 500)], 1920, 1080).is_some());
    }

    #[test]
    fn damage_accumulates_and_full_wins() {
        let mut d = Damage::default();
        assert!(d.is_empty());
        d.add_rects([r(0, 0, 4, 4), r(3, 3, 3, 3)]);
        assert_eq!(d, Damage::Rects(vec![r(0, 0, 4, 4)]));
        d.add(&Damage::Full);
        assert_eq!(d, Damage::Full);
        d.add_rects([r(0, 0, 4, 4)]);
        assert_eq!(d, Damage::Full);
    }

    #[test]
    fn damage_collapses_past_the_accumulation_cap() {
        let mut d = Damage::default();
        d.add_rects((0..=MAX_ACCUMULATED_RECTS as u32).map(|i| r(i, i, i + 1, i + 1)));
        let n = MAX_ACCUMULATED_RECTS as u32 + 1;
        assert_eq!(d, Damage::Rects(vec![r(0, 0, n, n)]));
    }

    fn settled(tracker: &mut DamageTracker, now: Instant) -> i64 {
        tracker.submitted(0, Damage::Full);
        assert_eq!(tracker.regions(0, true, 1920, 1080, now), None);
        for pts in 1..=i64::from(FULL_FRAMES_AFTER_KEYFRAME) {
            tracker.submitted(pts, Damage::Rects(vec![r(0, 0, 4, 4)]));
            assert_eq!(tracker.regions(pts, false, 1920, 1080, now), None);
        }
        i64::from(FULL_FRAMES_AFTER_KEYFRAME) + 1
    }

    #[test]
    fn tracker_ships_keyframes_and_their_followers_whole() {
        let now = Instant::now();
        let mut t = DamageTracker::new(true, None);
        let pts = settled(&mut t, now);
        t.submitted(pts, Damage::Rects(vec![r(0, 0, 4, 4)]));
        assert_eq!(
            t.regions(pts, false, 1920, 1080, now),
            Some(vec![r(0, 0, 32, 32)])
        );
        t.submitted(pts + 1, Damage::Rects(vec![r(0, 0, 4, 4)]));
        assert_eq!(
            t.regions(pts + 1, true, 1920, 1080, now),
            None,
            "VT's own IDR"
        );
    }

    #[test]
    fn tracker_carries_damage_of_frames_the_encoder_dropped() {
        let now = Instant::now();
        let mut t = DamageTracker::new(true, None);
        let pts = settled(&mut t, now);
        t.submitted(pts, Damage::Rects(vec![r(0, 0, 4, 4)]));
        t.submitted(pts + 1, Damage::Rects(vec![r(1000, 600, 1004, 604)]));
        let out = t.regions(pts + 1, false, 1920, 1080, now).unwrap();
        assert!(covers(&out, 0, 0) && covers(&out, 1000, 600), "{out:?}");
    }

    #[test]
    fn tracker_ships_whole_without_a_matching_submit() {
        let now = Instant::now();
        let mut t = DamageTracker::new(true, None);
        let pts = settled(&mut t, now);
        assert_eq!(t.regions(pts, false, 1920, 1080, now), None);
        t.submitted(pts + 1, Damage::Full);
        assert_eq!(t.regions(pts + 1, false, 1920, 1080, now), None);
    }

    #[test]
    fn tracker_refreshes_the_whole_surface_periodically() {
        let start = Instant::now();
        let mut t = DamageTracker::new(true, Some(Duration::from_secs(1)));
        let pts = settled(&mut t, start);
        let small = || Damage::Rects(vec![r(0, 0, 4, 4)]);
        t.submitted(pts, small());
        assert!(t
            .regions(pts, false, 1920, 1080, start + Duration::from_millis(500))
            .is_some());
        t.submitted(pts + 1, small());
        assert_eq!(
            t.regions(pts + 1, false, 1920, 1080, start + Duration::from_secs(1)),
            None
        );
        t.submitted(pts + 2, small());
        assert!(t
            .regions(
                pts + 2,
                false,
                1920,
                1080,
                start + Duration::from_millis(1500)
            )
            .is_some());
    }

    #[test]
    fn disabled_tracker_always_ships_whole() {
        let now = Instant::now();
        let mut t = DamageTracker::new(false, None);
        for pts in 0..20 {
            t.submitted(pts, Damage::Rects(vec![r(0, 0, 4, 4)]));
            assert_eq!(t.regions(pts, pts == 0, 1920, 1080, now), None);
        }
        assert!(t.queue.is_empty());
    }

    #[test]
    fn tracker_queue_is_bounded() {
        let now = Instant::now();
        let mut t = DamageTracker::new(true, None);
        let pts = settled(&mut t, now);
        for p in pts..pts + 2 * MAX_QUEUED as i64 {
            t.submitted(p, Damage::Rects(vec![r(0, 0, 4, 4)]));
        }
        assert_eq!(t.queue.len(), MAX_QUEUED);
        let last = pts + 2 * MAX_QUEUED as i64 - 1;
        assert_eq!(
            t.regions(last, false, 1920, 1080, now),
            None,
            "collapsed entry is full"
        );
    }
}
