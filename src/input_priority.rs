//! Input-priority tiles (`MACRDP_INPUT_PRIORITY`): the user's own change ships ahead of a busy screen's H.264.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::damage::{Rect, MAX_REGION_RECTS};
use crate::egfx_fast_path::FastPath;

/// How long after an input its captures take the priority path.
const DEFAULT_WINDOW: Duration = Duration::from_millis(150);

/// Busy-map cell size in surface pixels.
const CELL: u32 = 32;

/// A cell is busy when it changed in this many captures within [`BUSY_WINDOW_MS`] before the input.
const BUSY_HITS: usize = 8;
const BUSY_WINDOW_MS: u32 = 400;

/// Smallest byte share the in-flight budget keeps free for tiles.
const MIN_RESERVE_BYTES: u64 = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub tiles: FastPath,
    pub window: Duration,
}

impl Config {
    /// `MACRDP_INPUT_PRIORITY`, `MACRDP_INPUT_PRIORITY_MAX_PX` and `MACRDP_INPUT_PRIORITY_WINDOW_MS`.
    pub fn from_env() -> Option<Self> {
        let var = |k: &str| std::env::var(k).ok();
        Self::parse(
            var("MACRDP_INPUT_PRIORITY").as_deref(),
            var("MACRDP_INPUT_PRIORITY_MAX_PX").as_deref(),
            var("MACRDP_INPUT_PRIORITY_WINDOW_MS").as_deref(),
        )
    }

    fn parse(mode: Option<&str>, max_px: Option<&str>, window_ms: Option<&str>) -> Option<Self> {
        let tiles = FastPath::parse(mode, max_px)?;
        let window = window_ms
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .map_or(DEFAULT_WINDOW, Duration::from_millis);
        Some(Self { tiles, window })
    }
}

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Milliseconds since [`epoch`], plus one so 0 means never.
fn now_ms() -> u64 {
    epoch().elapsed().as_millis() as u64 + 1
}

static LAST_INPUT_MS: AtomicU64 = AtomicU64::new(0);

/// Records a key or pointer-button press.
pub fn note_input() {
    LAST_INPUT_MS.store(now_ms(), Ordering::Relaxed);
}

fn last_input_ms() -> Option<u64> {
    Some(LAST_INPUT_MS.load(Ordering::Relaxed)).filter(|&ms| ms != 0)
}

/// Bytes of the in-flight `budget` H.264 frames leave to tiles.
pub fn tile_reserve(budget: u64) -> u64 {
    (budget / 4).max(MIN_RESERVE_BYTES).min(budget / 2)
}

/// Per-cell times of the last [`BUSY_HITS`] captures that changed it.
#[derive(Debug, Default)]
struct BusyMap {
    cols: u32,
    rows: u32,
    hits: Vec<[u32; BUSY_HITS]>,
    next: Vec<u8>,
}

impl BusyMap {
    fn cells(&self, r: &Rect) -> impl Iterator<Item = usize> + '_ {
        let (c0, c1) = (r.left / CELL, r.right.div_ceil(CELL).min(self.cols));
        let (r0, r1) = (r.top / CELL, r.bottom.div_ceil(CELL).min(self.rows));
        (r0..r1).flat_map(move |y| (c0..c1).map(move |x| (y * self.cols + x) as usize))
    }

    fn fit(&mut self, r: &Rect) {
        let (cols, rows) = (r.right.div_ceil(CELL), r.bottom.div_ceil(CELL));
        if cols <= self.cols && rows <= self.rows {
            return;
        }
        let (cols, rows) = (cols.max(self.cols), rows.max(self.rows));
        let mut grown = BusyMap {
            cols,
            rows,
            hits: vec![[0; BUSY_HITS]; (cols * rows) as usize],
            next: vec![0; (cols * rows) as usize],
        };
        for y in 0..self.rows {
            for x in 0..self.cols {
                let (from, to) = ((y * self.cols + x) as usize, (y * cols + x) as usize);
                grown.hits[to] = self.hits[from];
                grown.next[to] = self.next[from];
            }
        }
        *self = grown;
    }

    fn hit(&mut self, r: &Rect, at_ms: u32) {
        self.fit(r);
        let cells: Vec<usize> = self.cells(r).collect();
        for i in cells {
            let n = usize::from(self.next[i]);
            self.hits[i][n] = at_ms;
            self.next[i] = ((n + 1) % BUSY_HITS) as u8;
        }
    }

    /// Cells that changed in every one of [`BUSY_HITS`] captures within [`BUSY_WINDOW_MS`] before `at_ms`.
    fn busy_at(&self, at_ms: u32) -> Vec<bool> {
        let since = at_ms.saturating_sub(BUSY_WINDOW_MS);
        self.hits
            .iter()
            .map(|h| h.iter().all(|&t| t != 0 && t >= since && t <= at_ms))
            .collect()
    }
}

/// What a capture in the input window ships ahead of H.264.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub rects: Vec<Rect>,
    /// Some post-input change was too large for tiles and waits for H.264.
    pub untiled: bool,
    /// The input this capture answers, in [`now_ms`] units.
    pub input_ms: u64,
}

/// Per-connection state, fed by the capture loop and drained by the submit.
#[derive(Debug, Default)]
pub struct Priority {
    busy: BusyMap,
    /// The input the busy snapshot was taken for, and the snapshot.
    snapshot: Option<(u64, Vec<bool>)>,
    pending: Vec<Rect>,
    pending_input: u64,
    /// The last input whose capture skipped the frame-rate gates.
    bypassed_input: u64,
    /// The last input whose first tile was logged.
    logged_input: u64,
}

impl Priority {
    /// Records one capture's own changed rects, not the accumulated damage.
    pub fn observe(&mut self, rects: &[Rect], window: Duration) {
        // Input first: one stored after `now` would read as in the future.
        let input = last_input_ms();
        self.observe_at(rects, now_ms(), input, window);
    }

    fn observe_at(&mut self, rects: &[Rect], now: u64, input: Option<u64>, window: Duration) {
        let live = input.filter(|&i| i <= now && now - i <= window.as_millis() as u64);
        match live {
            Some(input) => {
                let cells = self.busy.hits.len();
                if self
                    .snapshot
                    .as_ref()
                    .is_none_or(|(i, b)| *i != input || b.len() != cells)
                {
                    self.snapshot = Some((input, self.busy.busy_at(input as u32)));
                }
                if self.pending_input != input {
                    self.pending.clear();
                    self.pending_input = input;
                }
                let busy = &self.snapshot.as_ref().expect("set above").1;
                let quiet = |r: &Rect| !self.busy.cells(r).any(|i| busy.get(i) == Some(&true));
                let fresh: Vec<Rect> = rects.iter().copied().filter(quiet).collect();
                self.pending.extend(fresh);
            }
            None => self.pending.clear(),
        }
        for r in rects {
            self.busy.hit(r, now as u32);
        }
    }

    /// The pending rects, clipped, as at most `max_px` tile pixels; `None` when nothing changed since the input.
    pub fn take(&mut self, max_px: u64, width: u16, height: u16) -> Option<Selection> {
        let pending = std::mem::take(&mut self.pending);
        let (w, h) = (u32::from(width), u32::from(height));
        let mut rects: Vec<Rect> = Vec::new();
        let mut untiled = false;
        let mut px = 0u64;
        for r in pending {
            let r = Rect {
                left: r.left.min(w),
                top: r.top.min(h),
                right: r.right.min(w),
                bottom: r.bottom.min(h),
            };
            if r.area() == 0 || rects.iter().any(|t| t.contains(&r)) {
                continue;
            }
            if px + r.area() > max_px || rects.len() >= MAX_REGION_RECTS {
                untiled = true;
                continue;
            }
            px += r.area();
            rects.push(r);
        }
        (!rects.is_empty() || untiled).then_some(Selection {
            rects,
            untiled,
            input_ms: self.pending_input,
        })
    }

    /// Whether this capture skips the frame-rate gates: once per input, when tiles can't carry its change.
    pub fn bypass_fps_gate(&mut self, sel: &Selection) -> bool {
        if !sel.untiled || self.bypassed_input == sel.input_ms {
            return false;
        }
        self.bypassed_input = sel.input_ms;
        true
    }

    /// Time since `sel`'s input, and whether it is the first tile for it.
    pub fn shipped(&mut self, sel_input_ms: u64) -> (Duration, bool) {
        let first = self.logged_input != sel_input_ms;
        self.logged_input = sel_input_ms;
        (
            Duration::from_millis(now_ms().saturating_sub(sel_input_ms)),
            first,
        )
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

    const WINDOW: Duration = DEFAULT_WINDOW;

    /// A busy right side redrawn every 16 ms up to `until`.
    fn busy_until(p: &mut Priority, until: u64) {
        let mut t = 1;
        while t < until {
            p.observe_at(&[r(1800, 0, 1920, 1080)], t, None, WINDOW);
            t += 16;
        }
    }

    #[test]
    fn parses_like_the_fast_path_with_its_own_window() {
        assert_eq!(Config::parse(None, None, None), None);
        assert_eq!(Config::parse(Some("0"), None, None), None);
        let c = Config::parse(Some("1"), Some("4096"), Some("90")).unwrap();
        assert_eq!(c.tiles.max_px, 4096);
        assert_eq!(c.window, Duration::from_millis(90));
        assert_eq!(
            Config::parse(Some("on"), None, Some("x")).unwrap().window,
            WINDOW
        );
    }

    #[test]
    fn splits_the_input_change_from_the_busy_area() {
        let mut p = Priority::default();
        busy_until(&mut p, 1000);
        let busy = r(1800, 0, 1920, 1080);
        let square = r(200, 200, 248, 248);
        p.observe_at(&[busy, square], 1010, Some(1000), WINDOW);
        let sel = p.take(32 * 1024, 1920, 1080).unwrap();
        assert_eq!(sel.rects, vec![square]);
        assert!(!sel.untiled);
        assert_eq!(sel.input_ms, 1000);
        assert_eq!(p.take(32 * 1024, 1920, 1080), None, "taken once");
    }

    #[test]
    fn nothing_ships_outside_the_window_or_before_any_input() {
        let mut p = Priority::default();
        p.observe_at(&[r(0, 0, 10, 10)], 500, None, WINDOW);
        assert_eq!(p.take(32 * 1024, 1920, 1080), None);
        p.observe_at(&[r(0, 0, 10, 10)], 1200, Some(1000), WINDOW);
        assert_eq!(p.take(32 * 1024, 1920, 1080), None);
        p.observe_at(&[r(0, 0, 10, 10)], 990, Some(1000), WINDOW);
        assert_eq!(
            p.take(32 * 1024, 1920, 1080),
            None,
            "change before the input"
        );
    }

    #[test]
    fn typing_on_a_quiet_line_is_never_busy() {
        let mut p = Priority::default();
        // A key every 70 ms, each redrawing the caret cell once.
        for k in 0..20u64 {
            let at = 1000 + k * 70;
            p.observe_at(&[r(300, 300, 316, 330)], at + 15, Some(at), WINDOW);
            let sel = p.take(32 * 1024, 1920, 1080).unwrap();
            assert_eq!(sel.rects.len(), 1, "key {k}");
        }
    }

    #[test]
    fn a_large_or_overflowing_change_waits_for_h264_and_bypasses_once() {
        let mut p = Priority::default();
        let small = r(0, 0, 16, 16);
        let large = r(100, 100, 600, 600);
        p.observe_at(&[small, large], 1010, Some(1000), WINDOW);
        let sel = p.take(32 * 1024, 1920, 1080).unwrap();
        assert_eq!(sel.rects, vec![small]);
        assert!(sel.untiled);
        assert!(p.bypass_fps_gate(&sel));
        p.observe_at(&[large], 1030, Some(1000), WINDOW);
        let sel = p.take(32 * 1024, 1920, 1080).unwrap();
        assert!(sel.rects.is_empty() && sel.untiled);
        assert!(!p.bypass_fps_gate(&sel), "once per input");
        let many: Vec<Rect> = (0..12).map(|i| r(i * 40, 0, i * 40 + 8, 8)).collect();
        p.observe_at(&many, 1060, Some(1050), WINDOW);
        let sel = p.take(32 * 1024, 1920, 1080).unwrap();
        assert_eq!(sel.rects.len(), MAX_REGION_RECTS);
        assert!(sel.untiled && p.bypass_fps_gate(&sel));
    }

    #[test]
    fn rects_are_clipped_and_deduplicated() {
        let mut p = Priority::default();
        p.observe_at(&[r(1900, 1070, 1940, 1100)], 1010, Some(1000), WINDOW);
        p.observe_at(&[r(1900, 1070, 1940, 1100)], 1026, Some(1000), WINDOW);
        let sel = p.take(32 * 1024, 1920, 1080).unwrap();
        assert_eq!(sel.rects, vec![r(1900, 1070, 1920, 1080)]);
    }

    #[test]
    fn a_new_input_drops_the_previous_inputs_leftovers() {
        let mut p = Priority::default();
        p.observe_at(&[r(0, 0, 8, 8)], 1010, Some(1000), WINDOW);
        p.observe_at(&[r(64, 64, 72, 72)], 1110, Some(1100), WINDOW);
        let sel = p.take(32 * 1024, 1920, 1080).unwrap();
        assert_eq!(sel.rects, vec![r(64, 64, 72, 72)]);
        assert_eq!(sel.input_ms, 1100);
    }

    #[test]
    fn reserve_is_a_quarter_within_bounds() {
        assert_eq!(tile_reserve(400_000), 100_000);
        assert_eq!(tile_reserve(20_000), 10_000, "never more than half");
        assert_eq!(tile_reserve(64 * 1024), 16 * 1024);
    }
}
