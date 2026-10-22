//! Maps client wheel and trackpad scrolls to the scroll events a local device
//! would produce on macOS. Pure, so the mapping is unit-tested on every target;
//! `input.rs` posts the result.
//!
//! Deltas come in as 16.16 fixed-point RDP wheel units (120 per notch), the
//! encoding of the FreeRDP ainput channel; plain wheel PDUs are shifted up to
//! it. Positive vertical scrolls up and positive horizontal scrolls right, as
//! in RDP. A notch is worth `PIXELS_PER_NOTCH` pixels, and our FreeRDP client
//! scales its pixel deltas by the inverse, so a client pixel is a server pixel.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

const FIXED_ONE: i64 = 1 << 16;
const UNITS_PER_NOTCH: i64 = 120;
const PIXELS_PER_NOTCH: i64 = 100;
/// Cap for streams that cannot tell a flick from a backlog; see `Source::Legacy`.
const LEGACY_MAX_PIXELS: i64 = 600;

/// Bits our FreeRDP client adds to the ainput mouse flags, above the 32 that
/// FreeRDP defines. Stock clients leave them clear.
pub mod ainput {
    /// The scroll bits below are set.
    pub const SCROLL_EXT: u64 = 1 << 32;
    /// Continuous deltas from a trackpad or high-resolution wheel, as opposed
    /// to the notches of a plain wheel.
    pub const SCROLL_PRECISE: u64 = 1 << 33;
    pub const SCROLL_PHASE_SHIFT: u32 = 36;
    pub const SCROLL_MOMENTUM_SHIFT: u32 = 40;
    pub const SCROLL_PHASE_MASK: u64 = 0x7;
    pub const SCROLL_MOMENTUM_MASK: u64 = 0x3;
}

/// The finger-on-trackpad part of a gesture (`NSEvent.phase`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    None,
    Began,
    Changed,
    Ended,
    Cancelled,
    MayBegin,
}

impl Phase {
    /// `kCGScrollWheelEventScrollPhase` value.
    pub fn cg_value(self) -> i64 {
        match self {
            Phase::None => 0,
            Phase::Began => 1,
            Phase::Changed => 2,
            Phase::Ended => 4,
            Phase::Cancelled => 8,
            Phase::MayBegin => 128,
        }
    }
}

/// The inertia after the fingers lift (`NSEvent.momentumPhase`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Momentum {
    #[default]
    None,
    Began,
    Changed,
    Ended,
}

impl Momentum {
    /// `kCGScrollWheelEventMomentumPhase` value.
    pub fn cg_value(self) -> i64 {
        match self {
            Momentum::None => 0,
            Momentum::Began => 1,
            Momentum::Changed => 2,
            Momentum::Ended => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Plain wheel PDUs and stock ainput clients. A trackpad arrives sliced
    /// into small deltas, a notch as 120, and nothing says which.
    Legacy,
    /// Continuous deltas with the client's gesture phases.
    Precise { phase: Phase, momentum: Momentum },
    /// A notched wheel, already accelerated by the client.
    Lines,
}

impl Source {
    pub fn from_ainput_flags(flags: u64) -> Source {
        use ainput::*;
        if flags & SCROLL_EXT == 0 {
            return Source::Legacy;
        }
        if flags & SCROLL_PRECISE == 0 {
            return Source::Lines;
        }
        let phase = match (flags >> SCROLL_PHASE_SHIFT) & SCROLL_PHASE_MASK {
            1 => Phase::Began,
            2 => Phase::Changed,
            3 => Phase::Ended,
            4 => Phase::Cancelled,
            5 => Phase::MayBegin,
            _ => Phase::None,
        };
        let momentum = match (flags >> SCROLL_MOMENTUM_SHIFT) & SCROLL_MOMENTUM_MASK {
            1 => Momentum::Began,
            2 => Momentum::Changed,
            3 => Momentum::Ended,
            _ => Momentum::None,
        };
        Source::Precise { phase, momentum }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Pixel,
    Line,
}

/// One scroll event to post, with deltas in `CGEventCreateScrollWheelEvent`'s
/// sense: positive `v` scrolls up, positive `h` scrolls left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Post {
    pub unit: Unit,
    pub v: i32,
    pub h: i32,
    pub phase: Phase,
    pub momentum: Momentum,
}

/// Remainders of whole pixels or lines not yet posted, per axis, in input
/// units times the output units per notch.
#[derive(Debug, Default)]
pub struct Scroller {
    v: i64,
    h: i64,
    unit: Option<Unit>,
    /// A gesture or its momentum is open on the server and needs an end.
    open: Option<(Phase, Momentum)>,
}

impl Scroller {
    pub fn feed(&mut self, v: i64, h: i64, source: Source) -> Option<Post> {
        let (unit, per_notch) = match source {
            Source::Lines => (Unit::Line, 1),
            _ => (Unit::Pixel, PIXELS_PER_NOTCH),
        };
        if self.unit != Some(unit) {
            self.v = 0;
            self.h = 0;
            self.unit = Some(unit);
        }
        let v = carry(&mut self.v, v.saturating_mul(per_notch));
        let h = carry(&mut self.h, h.saturating_mul(per_notch));
        let (v, h) = match source {
            Source::Legacy => {
                // Drop what the cap cuts off: a backlog from a fast scroll
                // otherwise swallowed the next scroll the other way.
                let cap = LEGACY_MAX_PIXELS;
                (v.clamp(-cap, cap), h.clamp(-cap, cap))
            }
            _ => (v, h),
        };
        let (phase, momentum) = match source {
            Source::Precise { phase, momentum } => (phase, momentum),
            _ => (Phase::None, Momentum::None),
        };
        self.open = match (phase, momentum) {
            (Phase::Began | Phase::Changed | Phase::MayBegin, _) => {
                Some((Phase::Changed, Momentum::None))
            }
            (_, Momentum::Began | Momentum::Changed) => Some((Phase::None, Momentum::Changed)),
            _ => None,
        };
        // Phase changes are posted even without movement, as a trackpad does.
        if v == 0 && h == 0 && phase == Phase::None && momentum == Momentum::None {
            return None;
        }
        Some(Post {
            unit,
            v: clamp_i32(v),
            h: clamp_i32(-h),
            phase,
            momentum,
        })
    }

    /// Closes a gesture the client never finished (it went away mid-scroll),
    /// so the app under the cursor does not stay rubber-banded or scrolling.
    pub fn close(&mut self) -> Option<Post> {
        let (phase, momentum) = self.open.take()?;
        self.v = 0;
        self.h = 0;
        Some(Post {
            unit: Unit::Pixel,
            v: 0,
            h: 0,
            phase: if phase == Phase::None {
                Phase::None
            } else {
                Phase::Cancelled
            },
            momentum: if momentum == Momentum::None {
                Momentum::None
            } else {
                Momentum::Ended
            },
        })
    }
}

/// Adds `delta` to the remainder and takes out the whole part. A reversal
/// starts from zero so a leftover fraction never moves the wrong way. Within
/// 1/1024 of a whole step counts as whole: the client's pixels are not exact
/// in wheel units.
fn carry(rest: &mut i64, delta: i64) -> i64 {
    const STEP: i64 = UNITS_PER_NOTCH * FIXED_ONE;
    if delta != 0 && *rest != 0 && (*rest < 0) != (delta < 0) {
        *rest = 0;
    }
    *rest = rest.saturating_add(delta);
    let whole = (*rest + rest.signum() * (STEP / 1024)) / STEP;
    *rest -= whole * STEP;
    whole
}

fn clamp_i32(v: i64) -> i32 {
    i32::try_from(v).unwrap_or(if v < 0 { i32::MIN } else { i32::MAX })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: i64 = FIXED_ONE;

    /// What the client sends for `px` pixels.
    fn units(px: f64) -> i64 {
        (px * 1.2 * ONE as f64).round() as i64
    }

    fn px(v: i32, h: i32) -> Option<Post> {
        Some(Post {
            unit: Unit::Pixel,
            v,
            h,
            phase: Phase::None,
            momentum: Momentum::None,
        })
    }

    fn precise(phase: Phase, momentum: Momentum) -> Source {
        Source::Precise { phase, momentum }
    }

    #[test]
    fn legacy_notch_is_a_hundred_pixels() {
        let mut s = Scroller::default();
        assert_eq!(s.feed(120 * ONE, 0, Source::Legacy), px(100, 0));
        assert_eq!(s.feed(-120 * ONE, 0, Source::Legacy), px(-100, 0));
    }

    #[test]
    fn legacy_slices_add_up_without_losing_the_fraction() {
        let mut s = Scroller::default();
        assert_eq!(s.feed(ONE, 0, Source::Legacy), None);
        assert_eq!(s.feed(ONE, 0, Source::Legacy), px(1, 0));
        let total: i32 = (0..118)
            .filter_map(|_| s.feed(ONE, 0, Source::Legacy))
            .map(|p| p.v)
            .sum();
        assert_eq!(total + 1, 100);
    }

    #[test]
    fn legacy_drops_what_the_cap_cuts_off() {
        let mut s = Scroller::default();
        assert_eq!(s.feed(-12_000 * ONE, 0, Source::Legacy), px(-600, 0));
        assert_eq!(s.feed(units(-1.0), 0, Source::Legacy), px(-1, 0));
    }

    #[test]
    fn reversal_ignores_the_leftover_fraction() {
        let mut s = Scroller::default();
        assert_eq!(s.feed(-ONE, 0, Source::Legacy), None);
        assert_eq!(s.feed(120 * ONE, 0, Source::Legacy), px(100, 0));
    }

    #[test]
    fn horizontal_right_is_negative_for_core_graphics() {
        let mut s = Scroller::default();
        assert_eq!(s.feed(0, 120 * ONE, Source::Legacy), px(0, -100));
    }

    #[test]
    fn a_client_pixel_is_a_server_pixel() {
        // The client sends 1.2 wheel units per pixel.
        let mut s = Scroller::default();
        let src = precise(Phase::None, Momentum::None);
        assert_eq!(s.feed(units(103.0), 0, src).map(|p| p.v), Some(103));
        assert_eq!(s.feed(units(1.0), 0, src).map(|p| p.v), Some(1));
        assert_eq!(s.feed(units(0.5), 0, src), None);
        assert_eq!(s.feed(units(0.5), 0, src).map(|p| p.v), Some(1));
        // Rounding on the client costs at most a pixel over a long stream.
        let total: i32 = (0..1000)
            .filter_map(|_| s.feed(units(0.1), 0, src))
            .map(|p| p.v)
            .sum();
        assert!((99..=100).contains(&total), "{total}");
    }

    #[test]
    fn precise_streams_are_not_capped() {
        let mut s = Scroller::default();
        let src = precise(Phase::None, Momentum::Changed);
        assert_eq!(s.feed(units(2000.0), 0, src).map(|p| p.v), Some(2000));
    }

    #[test]
    fn phases_are_posted_without_movement() {
        let mut s = Scroller::default();
        let began = s.feed(0, 0, precise(Phase::Began, Momentum::None)).unwrap();
        assert_eq!((began.v, began.phase), (0, Phase::Began));
        let ended = s.feed(0, 0, precise(Phase::Ended, Momentum::None)).unwrap();
        assert_eq!(ended.phase, Phase::Ended);
        let mom = s.feed(0, 0, precise(Phase::None, Momentum::Ended)).unwrap();
        assert_eq!(mom.momentum, Momentum::Ended);
    }

    #[test]
    fn idle_precise_movement_below_a_pixel_posts_nothing() {
        let mut s = Scroller::default();
        assert_eq!(
            s.feed(ONE / 2, 0, precise(Phase::None, Momentum::None)),
            None
        );
    }

    #[test]
    fn lines_post_whole_lines() {
        let mut s = Scroller::default();
        let p = s.feed(3 * 120 * ONE, 0, Source::Lines).unwrap();
        assert_eq!((p.unit, p.v), (Unit::Line, 3));
        assert_eq!(s.feed(60 * ONE, 0, Source::Lines), None);
        assert_eq!(s.feed(60 * ONE, 0, Source::Lines).map(|p| p.v), Some(1));
    }

    #[test]
    fn switching_units_drops_the_remainder() {
        let mut s = Scroller::default();
        assert_eq!(s.feed(60 * ONE, 0, Source::Lines), None);
        assert_eq!(s.feed(ONE / 2, 0, Source::Legacy), None);
        assert_eq!(s.feed(60 * ONE, 0, Source::Lines), None);
    }

    #[test]
    fn close_ends_an_open_gesture_or_momentum() {
        let mut s = Scroller::default();
        assert_eq!(s.close(), None);
        s.feed(units(10.0), 0, precise(Phase::Changed, Momentum::None));
        let p = s.close().unwrap();
        assert_eq!((p.phase, p.momentum), (Phase::Cancelled, Momentum::None));
        assert_eq!(s.close(), None);
        s.feed(units(10.0), 0, precise(Phase::None, Momentum::Changed));
        let p = s.close().unwrap();
        assert_eq!((p.phase, p.momentum), (Phase::None, Momentum::Ended));
        s.feed(0, 0, precise(Phase::None, Momentum::Ended));
        assert_eq!(s.close(), None);
    }

    #[test]
    fn decodes_the_ainput_extension() {
        use ainput::*;
        assert_eq!(Source::from_ainput_flags(0x1), Source::Legacy);
        assert_eq!(Source::from_ainput_flags(0x1 | SCROLL_EXT), Source::Lines);
        let f = 0x1 | SCROLL_EXT | SCROLL_PRECISE | (1 << SCROLL_PHASE_SHIFT);
        assert_eq!(
            Source::from_ainput_flags(f),
            precise(Phase::Began, Momentum::None)
        );
        let f = SCROLL_EXT | SCROLL_PRECISE | (3 << SCROLL_MOMENTUM_SHIFT);
        assert_eq!(
            Source::from_ainput_flags(f),
            precise(Phase::None, Momentum::Ended)
        );
        let f = SCROLL_EXT | SCROLL_PRECISE | (5 << SCROLL_PHASE_SHIFT);
        assert_eq!(
            Source::from_ainput_flags(f),
            precise(Phase::MayBegin, Momentum::None)
        );
    }
}
