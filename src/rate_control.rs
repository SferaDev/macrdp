//! Delay-targeted rate control for the adaptive H.264 bitrate.
//!
//! The controller keeps the standing queue (frame ack RTT above its windowed
//! minimum) inside a band of `low_ms..high_ms`. It grows only while the queue
//! is below `low_ms` and not rising, holds inside the band, and above
//! `high_ms` cuts to the measured delivery rate less a margin proportional to
//! the excess, so the queue drains instead of being chased by repeated
//! multiplicative cuts. The delivery rate while a queue stands is the
//! bottleneck's capacity: the controller remembers it and converges just under
//! it instead of climbing back into it at full speed.
//!
//! A relayed path jitters by tens of ms in bursts, which a per-interval sample
//! reads as a queue. The signal is therefore the minimum over a [`DelayWindow`]
//! spanning several frames, the marks rise with the measured jitter, a cut needs
//! the queue to outlast `PERSIST_S`, and a sender far under its target (a static
//! screen) never cuts on delay at all: its in-flight gate is the safety net.
//!
//! Everything here is pure (no clock, no I/O) so the tests can drive it against
//! a simulated bottleneck link.

use std::collections::VecDeque;
use std::time::Duration;

/// Excess queue the controller plans to drain within about this long.
const DRAIN_MS: f64 = 1000.0;
/// The deepest single cut, as a fraction of the delivery rate.
const MAX_CUT: f64 = 0.5;
/// Above `high_ms` the new rate is at least this far below what the link delivered.
const OVER_MARGIN: f64 = 0.9;
/// The cut below the delivered rate when the unacked budget, not the queue, absorbed the excess.
const BOUND_MARGIN: f64 = 0.92;
/// A per-interval rise larger than this counts as a rising queue.
const RISE_MS: f64 = 5.0;
/// Half-life of the delivered-rate peak used for the app-limited growth cap.
const PEAK_HALF_LIFE_S: f64 = 3.0;
/// Growth per second with no capacity estimate, after slow start.
const FREE_GROWTH_PER_S: f64 = 0.25;
/// Probe growth per second near a known capacity, as a fraction of the rate.
const PROBE_PER_S: f64 = 0.01;
/// Downward drift per second while the queue sits in the upper half of the band.
const DRIFT_PER_S: f64 = 0.03;
/// Where the controller parks relative to the remembered capacity.
const PARK: f64 = 0.92;
/// Seconds of clean probing per doubling of the probe rate.
const PROBE_DOUBLING_S: f64 = 2.0;
/// Sending under this share of the target is the screen's doing, not the link's.
const APP_LIMITED: f64 = 0.6;
/// A queue of D ms at rate R holds R x D bytes; in-flight bytes under this share of that cannot be one.
const QUEUE_ROOM: f64 = 0.5;
/// Cuts stay above this share of the remembered capacity until loss, or a busy sender delivering less.
const CAPACITY_FLOOR: f64 = 0.5;
/// Time constant of the sent-to-target average that tells a static screen's lone repaint from a busy one.
const BUSY_TAU_S: f64 = 1.0;
/// Congestion must hold this long before the frame rate follows the cut target.
const FPS_CONGESTED_S: f64 = 2.0;
/// A queue must stand above the high mark this long before it cuts.
const PERSIST_S: f64 = 0.5;
/// A queue this many times the high mark cuts at once.
const DEEP: f64 = 3.0;
/// Each ms of measured jitter raises the high mark by this many ms, up to the mark itself.
const JITTER_K: f64 = 2.0;
/// The windowed minimum spans at least this long and this many samples, within `WINDOW_MAX_S`.
const WINDOW_S: f64 = 0.5;
const WINDOW_SAMPLES: usize = 8;
const WINDOW_MAX_S: f64 = 2.0;
/// Gain of the RFC 3550 interarrival jitter estimate.
const JITTER_GAIN: f64 = 1.0 / 16.0;
/// A standing delay held this long with no sign of being a queue is path latency.
const SHIFT_HOLD_S: f64 = 2.1;
/// How far the delay may wander and still count as standing, plus a slow fall a draining queue would far outrun.
const SHIFT_FLAT_MS: f64 = 10.0;
const SHIFT_SAG_MS_PER_S: f64 = 15.0;
/// Delivering under this share of the capacity drains any queue of ours within the hold.
const SHIFT_DRAIN: f64 = 0.8;

#[derive(Clone, Copy, Debug)]
pub struct RateConfig {
    pub floor_bps: u32,
    pub ceiling_bps: u32,
    /// Grow only below this standing queue.
    pub low_ms: f64,
    /// Cut above this standing queue.
    pub high_ms: f64,
    /// Fixed additive step per interval; `None` grows proportionally.
    pub increase_bps: Option<u32>,
    /// Multiplicative cut on loss, or on a queue with no delivery measurement.
    pub loss_decrease: f64,
    /// Per-interval growth factor until the first queue rise; `None` disables slow start.
    pub slow_start: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RateInput {
    pub dt: Duration,
    /// Filtered standing queue for the interval.
    pub delay_ms: f64,
    /// Bytes acknowledged over the interval as a rate; `None` when nothing was measurable.
    pub delivered_bps: Option<f64>,
    pub loss: bool,
    /// Whether the delay signal can be trusted this interval.
    pub usable: bool,
    /// Queue a just-shipped keyframe is expected to cause, added to both thresholds.
    pub allowance_ms: f64,
    /// The unacked-byte budget held back a capture this interval.
    pub gated: bool,
    /// Bytes shipped over the interval as a rate.
    pub sent_bps: Option<f64>,
    /// Delay noise from [`DelayWindow::jitter_ms`], which raises the marks.
    pub jitter_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Increase,
    Hold,
    Decrease,
}

#[derive(Clone, Debug)]
pub struct RateController {
    cfg: RateConfig,
    target: f64,
    slow_start: bool,
    capacity: Option<f64>,
    peak_delivered: f64,
    prev_delay_ms: f64,
    over_high: u32,
    cut_s: f64,
    /// The capacity before the current cut episode, which the episode's own deliveries cannot erode.
    episode_capacity: Option<f64>,
    probe_s: f64,
    congested: bool,
    /// How long the queue has stood above the high mark.
    over_s: f64,
    /// Average share of the target the sender ships.
    busy: f64,
    /// The last step cut on a standing queue, above the capacity floor.
    cut_on_queue: bool,
    /// How long `congested` has held.
    congested_s: f64,
    shift: PathShift,
}

impl RateController {
    /// A controller starting at `initial_bps`; a remembered `capacity_bps` skips slow start.
    pub fn new(cfg: RateConfig, initial_bps: u32, capacity_bps: Option<u32>) -> Self {
        let capacity = capacity_bps.filter(|&c| c > 0).map(f64::from);
        let mut c = Self {
            cfg,
            target: f64::from(initial_bps),
            slow_start: cfg.slow_start.is_some() && capacity.is_none(),
            capacity,
            peak_delivered: 0.0,
            prev_delay_ms: 0.0,
            over_high: 0,
            cut_s: 0.0,
            episode_capacity: None,
            probe_s: 0.0,
            congested: false,
            over_s: 0.0,
            busy: 1.0,
            cut_on_queue: false,
            congested_s: 0.0,
            shift: PathShift::default(),
        };
        c.target = c.clamp(c.target);
        c
    }

    pub fn target_bps(&self) -> u32 {
        self.target as u32
    }

    /// The bottleneck rate learned from the last standing queue, if any.
    pub fn capacity_bps(&self) -> Option<u32> {
        self.capacity.map(|c| c as u32)
    }

    /// The link rate a frame's own transmission time is estimated at.
    pub fn serialization_bps(&self) -> f64 {
        self.peak_delivered.max(self.target)
    }

    pub fn in_slow_start(&self) -> bool {
        self.slow_start
    }

    /// Whether a confirmed queue or loss holds, until the queue falls back under the low mark.
    pub fn congested(&self) -> bool {
        self.congested
    }

    /// The rate the frame rate is chosen for: an idle screen's low target is not a thin link, nor is a fresh cut.
    pub fn fps_basis_bps(&self) -> f64 {
        match self.capacity {
            Some(c) if self.congested_s < FPS_CONGESTED_S => self.target.max(self.clamp(c * PARK)),
            _ => self.target,
        }
    }

    fn app_limited(&self, gated: bool, sent: Option<f64>, delivered: Option<f64>) -> bool {
        let t = self.target;
        !gated
            && sent.is_some_and(|s| s < APP_LIMITED * t)
            && delivered.is_none_or(|d| d < APP_LIMITED * t)
    }

    /// Whether the unacked bytes are too few to hold a queue of `delay_ms` at the link's rate.
    fn no_room_for(&self, inflight: Option<f64>, delay_ms: f64) -> bool {
        let rate = self.capacity.unwrap_or(self.target);
        inflight.is_some_and(|b| b * 8.0 < QUEUE_ROOM * rate * delay_ms / 1000.0)
    }

    /// A rate safe to start the next connection on this path at.
    pub fn safe_bps(&self) -> u32 {
        let t = self
            .capacity
            .map_or(self.target, |c| self.target.min(c * PARK));
        self.clamp(t) as u32
    }

    /// Jumps to `bps` (a transport change), keeping what was learned about the path.
    pub fn reset_to(&mut self, bps: u32) {
        self.target = self.clamp(f64::from(bps));
        self.end_episode();
        self.shift = PathShift::default();
        self.prev_delay_ms = 0.0;
        self.congested = false;
        self.congested_s = 0.0;
        self.over_s = 0.0;
    }

    fn clamp(&self, bps: f64) -> f64 {
        let ceiling = f64::from(self.cfg.ceiling_bps.max(1));
        let floor = f64::from(self.cfg.floor_bps).min(ceiling);
        bps.clamp(floor, ceiling)
    }

    fn note_capacity(&mut self, delivered: f64) {
        self.capacity = Some(match self.capacity {
            Some(c) if (0.8 * c..=1.2 * c).contains(&delivered) => 0.5 * c + 0.5 * delivered,
            _ => delivered,
        });
    }

    pub fn step(&mut self, i: RateInput) -> Verdict {
        let dt = i.dt.as_secs_f64().clamp(0.01, 2.0);
        let delivered = i.delivered_bps.filter(|d| *d > 0.0);
        self.peak_delivered *= 0.5f64.powf(dt / PEAK_HALF_LIFE_S);
        if let Some(d) = delivered {
            self.peak_delivered = self.peak_delivered.max(d);
        }
        let noise = (JITTER_K * i.jitter_ms.max(0.0)).min(self.cfg.high_ms);
        let low = self.cfg.low_ms + i.allowance_ms;
        // Jitter only raises the cut mark: the band below it still steers a standing queue down.
        let mid = 0.5 * (low + self.cfg.high_ms + i.allowance_ms);
        let high = self.cfg.high_ms + i.allowance_ms + noise;
        let rising = i.delay_ms > self.prev_delay_ms + RISE_MS;
        let falling = i.delay_ms < self.prev_delay_ms - RISE_MS;
        self.prev_delay_ms = i.delay_ms;
        let t = self.target;
        // Held captures while the link delivers less than the target: the budget, not the queue, is absorbing the excess.
        // Delivery far under the target is an idle screen, not a full link: only near-target delivery counts.
        let link_bound = i.gated
            && i.allowance_ms <= 0.0
            && delivered.is_some_and(|d| (0.6 * t..0.97 * t).contains(&d));
        // A held capture is a sender wanting more than it shipped.
        if let Some(share) = i
            .sent_bps
            .map(|s| if i.gated { 1.0 } else { s / t.max(1.0) })
        {
            self.busy += (share - self.busy) * (1.0 - (-dt / BUSY_TAU_S).exp());
        }
        // A full-screen repaint on a static screen fills one interval and the gate, but not the link.
        let app_limited =
            self.app_limited(i.gated, i.sent_bps, delivered) || self.busy < APP_LIMITED;
        let over = i.usable && !app_limited && (i.delay_ms > high || link_bound);
        self.over_s = if over { self.over_s + dt } else { 0.0 };
        let persisted = self.over_s >= PERSIST_S - 1e-6;
        // A queue this deep is no jitter burst.
        let confirmed = persisted || (over && i.delay_ms > DEEP * high);
        if i.usable && i.delay_ms <= low {
            self.congested = false;
        }
        self.cut_on_queue = false;

        let (verdict, next) = if i.loss {
            self.slow_start = false;
            self.congested = true;
            self.over_high = 0;
            // A queue that follows a loss cut persists after cutting: no floor for it.
            self.cut_s = f64::INFINITY;
            if let Some(d) = delivered.filter(|d| *d >= 0.5 * t) {
                self.note_capacity(d);
            }
            (Verdict::Decrease, t * self.cfg.loss_decrease)
        } else if !i.usable {
            (Verdict::Hold, t)
        } else if app_limited && i.delay_ms > low {
            // A static screen's delay is its own burst or the path's jitter; its in-flight gate catches a real queue.
            self.over_high = 0;
            (Verdict::Hold, t)
        } else if over && !confirmed {
            // One window over the mark can be a jitter burst: wait for the queue to persist.
            self.slow_start = false;
            (Verdict::Hold, t)
        } else if over && falling && self.over_high > 0 && !link_bound {
            // The last cut is draining the queue: let it, rather than cut on samples from before it.
            (Verdict::Hold, t)
        } else if over {
            self.slow_start = false;
            self.congested = true;
            self.cut_on_queue = true;
            self.over_high += 1;
            self.begin_episode(dt);
            // A busy sender's queue delivering under the floor and its own target is a link that lost its capacity: no floor.
            if delivered.is_some_and(|d| {
                d < OVER_MARGIN * t
                    && self
                        .episode_capacity
                        .or(self.capacity)
                        .is_some_and(|c| d < CAPACITY_FLOOR * c)
            }) {
                self.cut_s = f64::INFINITY;
            }
            let excess = i.delay_ms - mid;
            // Without a deep queue to drain, settling just under the delivered rate is enough.
            let margin = if i.delay_ms > high {
                OVER_MARGIN
            } else {
                BOUND_MARGIN
            };
            let factor = (1.0 - excess / DRAIN_MS).clamp(MAX_CUT, margin);
            let next = match delivered {
                // A queue that persists means the link is carrying all it can.
                Some(d) if persisted || self.over_high >= 2 || link_bound => {
                    // A draining queue delivers less than the link carries: keep the episode's best.
                    if self.over_high > 2 {
                        self.capacity = Some(self.capacity.map_or(d, |c| c.max(d)));
                    } else {
                        self.note_capacity(d);
                    }
                    let base = self.capacity.unwrap_or(d);
                    (base * factor).min(t * margin)
                }
                Some(d) => (d * factor).max(t * MAX_CUT).min(t * margin),
                None => t * self.cfg.loss_decrease,
            };
            (Verdict::Decrease, next)
        } else if i.delay_ms > low || (rising && i.delay_ms > 0.5 * low) {
            if self.slow_start {
                self.slow_start = false;
                if let Some(d) = delivered.filter(|d| *d >= 0.5 * t) {
                    self.note_capacity(d);
                }
            }
            self.over_high = 0;
            if (i.delay_ms > mid || i.gated) && i.allowance_ms <= 0.0 {
                self.begin_episode(dt);
                // A rate that only holds the queue steady is the capacity: ease under it.
                (Verdict::Decrease, t * (1.0 - DRIFT_PER_S * dt))
            } else {
                (Verdict::Hold, t)
            }
        } else if i.allowance_ms > 0.0 || i.gated {
            // A keyframe's burst says nothing about the link's steady capacity, and a held capture says it is full.
            self.end_episode();
            (Verdict::Hold, t)
        } else {
            self.end_episode();
            (Verdict::Increase, self.grow(t, dt, app_limited))
        };
        let floor = self
            .episode_capacity
            .or(self.capacity)
            .map(|c| c * CAPACITY_FLOOR);
        let next = match floor {
            Some(f) if verdict == Verdict::Decrease && !i.loss && self.cut_s.is_finite() => {
                next.max(f.min(t))
            }
            _ => next,
        };
        if verdict == Verdict::Decrease || i.delay_ms > low {
            self.probe_s = 0.0;
        }
        self.target = self.clamp(next);
        self.congested_s = if self.congested {
            self.congested_s + dt
        } else {
            0.0
        };
        if verdict == Verdict::Increase && self.target <= t {
            return Verdict::Hold;
        }
        verdict
    }

    /// The delay to fold into the base RTT, once a standing delay has proven not to be a queue.
    pub fn observe_path(&mut self, i: ShiftInput) -> Option<f64> {
        let dt = i.dt.as_secs_f64().clamp(0.01, 2.0);
        let d = i.delay_ms;
        if d.is_nan() || d <= self.cfg.low_ms {
            self.shift = PathShift::default();
            return None;
        }
        let s = &self.shift;
        let band = SHIFT_FLAT_MS + SHIFT_SAG_MS_PER_S * s.held_s;
        if s.held_s == 0.0 || d > s.lo + SHIFT_FLAT_MS || d.max(s.hi) - d.min(s.lo) > band {
            self.shift = PathShift {
                lo: d,
                hi: d,
                ..PathShift::default()
            };
        }
        let delivered = i.delivered_bps.filter(|d| *d > 0.0);
        let idle = self.app_limited(i.gated, i.sent_bps, delivered)
            && self.no_room_for(i.inflight_bytes, d);
        // A queue of ours drains while the link delivers well under what it carried, so one that holds is not ours.
        let carried = self.episode_capacity.or(self.capacity);
        let undrained = carried.is_some_and(|c| {
            self.target <= SHIFT_DRAIN * c && delivered.is_some_and(|x| x <= SHIFT_DRAIN * c)
        });
        // A cut that left the delay where it was did not touch it.
        let unmoved = self.cut_on_queue && self.cut_s.is_finite() && d > self.shift.lo - RISE_MS;
        let s = &mut self.shift;
        s.lo = s.lo.min(d);
        s.hi = s.hi.max(d);
        s.held_s += dt;
        if idle || undrained || unmoved {
            s.evidence_s += dt;
        }
        if s.held_s < SHIFT_HOLD_S || s.evidence_s < 0.75 * s.held_s {
            return None;
        }
        let shift = s.lo;
        self.shift = PathShift::default();
        // The episode was a false alarm: what it taught about the link does not stand.
        if let Some(c) = self.episode_capacity {
            self.capacity = Some(self.capacity.map_or(c, |k| k.max(c)));
        }
        self.end_episode();
        self.prev_delay_ms = (self.prev_delay_ms - shift).max(0.0);
        Some(shift)
    }

    /// Counts the time a cut episode lasts, remembering the capacity it started from.
    fn begin_episode(&mut self, dt: f64) {
        if self.cut_s == 0.0 {
            self.episode_capacity = self.capacity;
        }
        self.cut_s += dt;
    }

    fn end_episode(&mut self) {
        self.over_high = 0;
        self.cut_s = 0.0;
        self.episode_capacity = None;
    }

    fn grow(&mut self, t: f64, dt: f64, app_limited: bool) -> f64 {
        let fixed = self.cfg.increase_bps.map(f64::from);
        let grown = if let Some(c) = self.capacity.filter(|_| !self.slow_start) {
            let park = c * PARK;
            if t < park * 0.97 {
                self.probe_s = 0.0;
                t + (park - t) * (dt * 1.5).min(1.0)
            } else if app_limited {
                // An idle screen cannot probe past the capacity: its bursts are not the link's.
                t
            } else {
                // Probe gently, then faster the longer the link stays clean past the old capacity.
                self.probe_s += dt;
                let per_s = (PROBE_PER_S * 2f64.powf(self.probe_s / PROBE_DOUBLING_S))
                    .min(FREE_GROWTH_PER_S);
                t + fixed.map_or(t * per_s * dt, |f| f.min(t * per_s * dt))
            }
        } else if app_limited {
            // Without a known capacity an idle screen's bursts say nothing about the link.
            t
        } else if self.slow_start {
            let factor = self.cfg.slow_start.unwrap_or(1.0);
            (t * factor).max(t + fixed.unwrap_or(0.0))
        } else {
            t + fixed.unwrap_or(t * FREE_GROWTH_PER_S * dt)
        };
        // Growth on an idle screen proves nothing about the link.
        let app_cap = (2.0 * self.peak_delivered).max(f64::from(self.cfg.floor_bps));
        grown.min(app_cap.max(t))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ShiftInput {
    pub dt: Duration,
    /// The interval's unsmoothed standing delay.
    pub delay_ms: f64,
    pub delivered_bps: Option<f64>,
    pub sent_bps: Option<f64>,
    pub inflight_bytes: Option<f64>,
    pub gated: bool,
}

/// Tells path latency from queue: a standing delay that neither the unacked bytes nor the cuts can explain.
#[derive(Clone, Debug, Default)]
struct PathShift {
    held_s: f64,
    evidence_s: f64,
    lo: f64,
    hi: f64,
}

/// Per-ack queue-delay samples: their recent minimum is the standing queue, their spread the noise.
#[derive(Clone, Debug, Default)]
pub struct DelayWindow {
    /// (seconds since an arbitrary epoch, queue delay ms), oldest first.
    samples: VecDeque<(f64, f64)>,
    jitter_ms: f64,
    last_ms: Option<f64>,
}

impl DelayWindow {
    pub fn push(&mut self, at: Duration, delay_ms: f64) {
        let t = at.as_secs_f64();
        if let Some(prev) = self.last_ms {
            self.jitter_ms += ((delay_ms - prev).abs() - self.jitter_ms) * JITTER_GAIN;
        }
        self.last_ms = Some(delay_ms);
        self.samples.push_back((t, delay_ms));
        while self.samples.front().is_some_and(|s| t - s.0 > WINDOW_MAX_S) {
            self.samples.pop_front();
        }
    }

    /// The smallest sample over the last `WINDOW_S` and at least the last `WINDOW_SAMPLES`; `None` with none recent.
    pub fn min_ms(&self, now: Duration) -> Option<f64> {
        let now = now.as_secs_f64();
        let n = self.samples.len();
        self.samples
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                let age = now - s.0;
                age <= WINDOW_MAX_S && (age <= WINDOW_S || n - i <= WINDOW_SAMPLES)
            })
            .map(|(_, s)| s.1)
            .reduce(f64::min)
    }

    /// The smallest sample newer than `since`, for the per-interval reading.
    pub fn min_since(&self, since: Duration) -> Option<f64> {
        let since = since.as_secs_f64();
        self.samples
            .iter()
            .filter(|s| s.0 > since)
            .map(|s| s.1)
            .reduce(f64::min)
    }

    pub fn jitter_ms(&self) -> f64 {
        self.jitter_ms
    }

    /// Takes `shift_ms` of re-baselined path latency out of the samples.
    pub fn rebase(&mut self, shift_ms: f64) {
        for s in &mut self.samples {
            s.1 = (s.1 - shift_ms).max(0.0);
        }
        self.last_ms = self.last_ms.map(|m| (m - shift_ms).max(0.0));
    }
}

/// Frame rates the governor steps through, highest first.
fn fps_ladder(max_fps: u32, min_fps: u32) -> Vec<u32> {
    let max_fps = max_fps.max(1);
    let min_fps = min_fps.clamp(1, max_fps);
    let mut ladder: Vec<u32> = [1, 2, 3, 4, 6]
        .iter()
        .map(|d| max_fps / d)
        .filter(|&f| f >= min_fps)
        .collect();
    ladder.push(min_fps);
    ladder.dedup();
    ladder
}

/// Raising the frame rate needs this much more than the minimum bits per pixel.
const FPS_UP_MARGIN: f64 = 1.3;

/// The frame rate that keeps at least `min_bpp` bits per pixel per frame at `rate_bps`.
pub fn next_fps_level(
    level: u32,
    max_fps: u32,
    min_fps: u32,
    rate_bps: f64,
    pixels: u64,
    min_bpp: f64,
) -> u32 {
    if min_bpp <= 0.0 || pixels == 0 {
        return max_fps.max(1);
    }
    let ladder = fps_ladder(max_fps, min_fps);
    let bpp = |fps: u32| rate_bps / (pixels as f64 * f64::from(fps));
    let pos = ladder
        .iter()
        .position(|&f| f <= level)
        .unwrap_or(ladder.len() - 1);
    if bpp(ladder[pos]) < min_bpp {
        return ladder[pos..]
            .iter()
            .copied()
            .find(|&f| bpp(f) >= min_bpp)
            .unwrap_or(*ladder.last().unwrap());
    }
    match pos.checked_sub(1).map(|p| ladder[p]) {
        Some(up) if bpp(up) >= min_bpp * FPS_UP_MARGIN => up,
        _ => ladder[pos],
    }
}

/// Whether a capture arriving `since_pass` after the last encoded one exceeds `fps`.
pub fn fps_gate_drops(since_pass: Duration, fps: u32, max_fps: u32) -> bool {
    if fps >= max_fps || fps == 0 {
        return false;
    }
    // A little slack so captures on the display's own cadence are not aliased away.
    since_pass.as_secs_f64() < 0.9 / f64::from(fps)
}

/// Longest periodic keyframe interval the rate scaling stretches to.
const MAX_KEYFRAME_SECS: f64 = 30.0;

/// Periodic keyframe interval that keeps keyframes under `share` of the link time at `rate_bps`.
pub fn keyframe_interval_secs(normal_secs: f64, idr_bytes: u64, rate_bps: f64, share: f64) -> f64 {
    if share <= 0.0 || idr_bytes == 0 || rate_bps <= 0.0 {
        return normal_secs;
    }
    let link_s = idr_bytes as f64 * 8.0 / rate_bps;
    (link_s / share).min(MAX_KEYFRAME_SECS).max(normal_secs)
}

/// Queue a keyframe of `idr_bytes` adds at `rate_bps`, while it is still draining.
pub fn idr_allowance_ms(
    idr_bytes: u64,
    rate_bps: f64,
    since_idr: Duration,
    interval: Duration,
) -> f64 {
    if idr_bytes == 0 || rate_bps <= 0.0 {
        return 0.0;
    }
    let link_ms = idr_bytes as f64 * 8.0 / rate_bps * 1000.0;
    let window_ms = link_ms + 2.0 * interval.as_secs_f64() * 1000.0;
    if since_idr.as_secs_f64() * 1000.0 <= window_ms {
        link_ms
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ceiling_bps: u32) -> RateConfig {
        RateConfig {
            floor_bps: 500_000,
            ceiling_bps,
            low_ms: 30.0,
            high_ms: 50.0,
            increase_bps: None,
            loss_decrease: 0.7,
            slow_start: Some(1.25),
        }
    }

    fn input(delay_ms: f64, delivered: f64) -> RateInput {
        RateInput {
            dt: Duration::from_millis(300),
            delay_ms,
            delivered_bps: Some(delivered),
            loss: false,
            usable: true,
            allowance_ms: 0.0,
            gated: false,
            sent_bps: None,
            jitter_ms: 0.0,
        }
    }

    #[test]
    fn slow_start_ends_at_the_first_queue_rise() {
        let mut c = RateController::new(cfg(100_000_000), 2_000_000, None);
        assert_eq!(c.step(input(0.0, 2_000_000.0)), Verdict::Increase);
        assert_eq!(c.target_bps(), 2_500_000);
        assert_eq!(c.step(input(35.0, 2_500_000.0)), Verdict::Hold);
        assert!(!c.in_slow_start());
        assert_eq!(c.target_bps(), 2_500_000);
        assert!(c.capacity_bps().is_some());
    }

    #[test]
    fn a_standing_queue_cuts_to_the_delivered_rate() {
        let mut c = RateController::new(cfg(100_000_000), 25_000_000, None);
        assert_eq!(c.step(input(200.0, 6_000_000.0)), Verdict::Decrease);
        assert_eq!(c.target_bps(), 12_500_000);
        assert_eq!(c.step(input(250.0, 6_000_000.0)), Verdict::Decrease);
        assert!(c.target_bps() < 6_000_000 && c.target_bps() > 4_000_000);
        assert_eq!(c.capacity_bps(), Some(6_000_000));
    }

    #[test]
    fn a_small_excess_cuts_a_little_once_it_persists() {
        let mut c = RateController::new(cfg(100_000_000), 10_000_000, Some(10_000_000));
        assert_eq!(c.step(input(55.0, 10_000_000.0)), Verdict::Hold);
        assert!(!c.congested());
        assert_eq!(c.step(input(55.0, 10_000_000.0)), Verdict::Decrease);
        assert_eq!(c.target_bps(), 9_000_000);
        assert!(c.congested());
    }

    #[test]
    fn a_jitter_burst_over_the_mark_does_not_cut() {
        let mut c = RateController::new(cfg(100_000_000), 10_000_000, Some(10_000_000));
        for _ in 0..10 {
            assert_eq!(c.step(input(80.0, 10_000_000.0)), Verdict::Hold);
            c.step(input(10.0, 10_000_000.0));
        }
        assert!(c.target_bps() >= 9_000_000, "{}", c.target_bps());
        assert!(!c.congested());
    }

    #[test]
    fn measured_jitter_raises_the_cut_mark() {
        let mut c = RateController::new(cfg(100_000_000), 10_000_000, Some(10_000_000));
        for _ in 0..5 {
            c.step(RateInput {
                jitter_ms: 15.0,
                ..input(70.0, 10_000_000.0)
            });
        }
        // Only the band's slow drift, no cut.
        assert!(c.target_bps() >= 9_500_000, "{}", c.target_bps());
        c.step(input(70.0, 10_000_000.0));
        c.step(input(70.0, 10_000_000.0));
        assert!(c.target_bps() <= 9_000_000, "{}", c.target_bps());
    }

    #[test]
    fn a_busy_queue_that_cuts_cannot_drain_stops_at_half_the_capacity() {
        let mut c = RateController::new(cfg(100_000_000), 10_000_000, Some(10_000_000));
        for _ in 0..30 {
            let t = f64::from(c.target_bps());
            c.step(input(80.0, t));
        }
        assert_eq!(c.target_bps(), 5_000_000);
        // Delivering under the floor and the target is a link that lost its capacity.
        for _ in 0..10 {
            c.step(input(80.0, 2_000_000.0));
        }
        assert!(c.target_bps() < 2_000_000, "{}", c.target_bps());
    }

    #[test]
    fn growth_is_capped_on_an_idle_screen() {
        let mut c = RateController::new(cfg(100_000_000), 4_000_000, None);
        for _ in 0..20 {
            c.step(input(0.0, 300_000.0));
        }
        assert_eq!(c.target_bps(), 4_000_000);
    }

    #[test]
    fn a_remembered_capacity_parks_just_under_it() {
        let mut c = RateController::new(cfg(100_000_000), 2_000_000, Some(10_000_000));
        assert!(!c.in_slow_start());
        for _ in 0..10 {
            c.step(input(0.0, f64::from(c.target_bps())));
        }
        let t = c.target_bps();
        assert!((9_000_000..=9_800_000).contains(&t), "{t}");
    }

    #[test]
    fn loss_and_blind_queues_cut_multiplicatively() {
        let mut c = RateController::new(cfg(100_000_000), 10_000_000, None);
        let mut i = input(0.0, 10_000_000.0);
        i.loss = true;
        assert_eq!(c.step(i), Verdict::Decrease);
        assert_eq!(c.target_bps(), 7_000_000);
        let mut i = input(500.0, 0.0);
        i.delivered_bps = None;
        c.step(i);
        assert_eq!(c.target_bps(), 4_900_000);
    }

    #[test]
    fn an_unusable_signal_holds() {
        let mut c = RateController::new(cfg(100_000_000), 10_000_000, None);
        let mut i = input(0.0, 10_000_000.0);
        i.usable = false;
        assert_eq!(c.step(i), Verdict::Hold);
        assert_eq!(c.target_bps(), 10_000_000);
    }

    #[test]
    fn a_keyframe_allowance_keeps_its_queue_from_cutting() {
        let mut c = RateController::new(cfg(100_000_000), 6_000_000, Some(6_500_000));
        let mut i = input(140.0, 6_000_000.0);
        i.allowance_ms = 150.0;
        assert_eq!(c.step(i), Verdict::Hold);
        assert_eq!(c.target_bps(), 6_000_000);
    }

    #[test]
    fn fixed_increase_step_still_applies() {
        let mut config = cfg(100_000_000);
        config.increase_bps = Some(250_000);
        config.slow_start = None;
        let mut c = RateController::new(config, 4_000_000, None);
        c.step(input(0.0, 4_000_000.0));
        assert_eq!(c.target_bps(), 4_250_000);
    }

    fn idle(delay_ms: f64) -> RateInput {
        RateInput {
            sent_bps: Some(200_000.0),
            ..input(delay_ms, 200_000.0)
        }
    }

    #[test]
    fn an_idle_sender_never_cuts_on_delay_alone() {
        let mut c = RateController::new(cfg(25_000_000), 4_000_000, Some(4_400_000));
        for _ in 0..20 {
            assert_eq!(c.step(idle(80.0)), Verdict::Hold);
        }
        assert_eq!(c.target_bps(), 4_000_000);
        assert!(!c.congested());
        assert!(c.fps_basis_bps() >= 4_000_000.0);
        // Loss still cuts.
        let i = RateInput {
            loss: true,
            ..idle(80.0)
        };
        assert_eq!(c.step(i), Verdict::Decrease);
    }

    #[test]
    fn the_delay_window_keeps_several_frames_and_measures_their_spread() {
        let ms = Duration::from_millis;
        let mut w = DelayWindow::default();
        assert_eq!(w.min_ms(ms(0)), None);
        // Ten frames 100 ms apart: the last eight count although they span 700 ms.
        for k in 0..10u64 {
            w.push(ms(100 * k), if k == 1 { 0.0 } else { 40.0 + k as f64 });
        }
        assert_eq!(w.min_ms(ms(900)), Some(42.0));
        assert_eq!(w.min_since(ms(850)), Some(49.0));
        // Samples past the longest window drop out.
        assert_eq!(w.min_ms(ms(10_000)), None);
        let mut noisy = DelayWindow::default();
        for k in 0..200u64 {
            noisy.push(ms(16 * k), if k % 2 == 0 { 0.0 } else { 20.0 });
        }
        assert!(
            (15.0..=20.0).contains(&noisy.jitter_ms()),
            "{}",
            noisy.jitter_ms()
        );
        noisy.rebase(5.0);
        assert_eq!(noisy.min_ms(ms(16 * 199)), Some(0.0));
    }

    #[test]
    fn a_standing_queue_at_capacity_is_never_rebaselined() {
        let mut c = RateController::new(cfg(25_000_000), 5_000_000, Some(5_000_000));
        for _ in 0..100 {
            let i = RateInput {
                sent_bps: Some(5e6),
                ..input(40.0, 5e6)
            };
            let shift = c.observe_path(ShiftInput {
                dt: i.dt,
                delay_ms: 40.0,
                delivered_bps: i.delivered_bps,
                sent_bps: i.sent_bps,
                inflight_bytes: Some(5e6 / 8.0 * 0.070),
                gated: false,
            });
            assert_eq!(shift, None);
            c.step(i);
        }
    }

    #[test]
    fn a_flat_delay_an_idle_sender_cannot_cause_is_rebaselined() {
        let mut c = RateController::new(cfg(25_000_000), 4_000_000, Some(4_400_000));
        let shift = (0..10).find_map(|_| {
            c.observe_path(ShiftInput {
                dt: Duration::from_millis(300),
                delay_ms: 52.0,
                delivered_bps: Some(200_000.0),
                sent_bps: Some(200_000.0),
                inflight_bytes: Some(3_000.0),
                gated: false,
            })
        });
        assert_eq!(shift, Some(52.0));
    }

    #[test]
    fn fps_steps_down_when_bits_per_pixel_run_short() {
        let px = 2560 * 1440;
        assert_eq!(next_fps_level(60, 60, 10, 30_000_000.0, px, 0.05), 60);
        assert_eq!(next_fps_level(60, 60, 10, 6_000_000.0, px, 0.05), 30);
        assert_eq!(next_fps_level(60, 60, 10, 4_500_000.0, px, 0.05), 20);
        assert_eq!(next_fps_level(60, 60, 10, 500_000.0, px, 0.05), 10);
        // Up one rung at a time, and only with margin.
        assert_eq!(next_fps_level(20, 60, 10, 30_000_000.0, px, 0.05), 30);
        assert_eq!(next_fps_level(30, 60, 10, 12_000_000.0, px, 0.05), 30);
        assert_eq!(next_fps_level(30, 60, 10, 14_500_000.0, px, 0.05), 60);
        // Disabled.
        assert_eq!(next_fps_level(20, 60, 10, 1.0, px, 0.0), 60);
    }

    #[test]
    fn fps_gate_spaces_captures() {
        let ms = Duration::from_millis;
        assert!(!fps_gate_drops(ms(1), 60, 60));
        assert!(fps_gate_drops(ms(17), 30, 60));
        assert!(!fps_gate_drops(ms(33), 30, 60));
        assert!(fps_gate_drops(ms(33), 20, 60));
        assert!(!fps_gate_drops(ms(50), 20, 60));
    }

    #[test]
    fn keyframes_stretch_on_a_thin_link_only() {
        // 200 KB keyframe on a LAN: the configured 2 s stands.
        assert_eq!(keyframe_interval_secs(2.0, 200_000, 200e6, 0.05), 2.0);
        // On 6 Mbit it is 267 ms of link time: stretched to keep it at 5%.
        let s = keyframe_interval_secs(2.0, 200_000, 6e6, 0.05);
        assert!((5.0..6.0).contains(&s), "{s}");
        assert_eq!(keyframe_interval_secs(2.0, 200_000, 6e6, 0.0), 2.0);
    }

    #[test]
    fn keyframe_allowance_lasts_while_it_drains() {
        let ms = Duration::from_millis;
        let a = idr_allowance_ms(75_000, 6e6, ms(100), ms(300));
        assert!((a - 100.0).abs() < 1.0);
        assert_eq!(idr_allowance_ms(75_000, 6e6, ms(800), ms(300)), 0.0);
    }

    /// A bottleneck link (fixed capacity, FIFO queue, base RTT) fed by an encoder
    /// that hits the target rate, with the controller in the loop the way
    /// `h264.rs` wires it: per-interval minimum queue delay, acked throughput,
    /// the unacked-byte budget, dynamic frame rate and keyframes whose size does
    /// not shrink with the rate (VideoToolbox ignores DataRateLimits for them).
    struct Frame {
        sent: f64,
        bytes: f64,
        acked: f64,
    }

    struct Run {
        /// (time s, target bps, interval delay ms, fps)
        trace: Vec<(f64, f64, f64, u32)>,
        /// (time s, ms folded into the base RTT)
        rebaselined: Vec<(f64, f64)>,
    }

    impl Run {
        fn window(&self, from: f64, to: f64) -> Vec<(f64, f64, f64, u32)> {
            self.trace
                .iter()
                .copied()
                .filter(|s| s.0 >= from && s.0 < to)
                .collect()
        }

        fn mean_rate(&self, from: f64, to: f64) -> f64 {
            let w = self.window(from, to);
            w.iter().map(|s| s.1).sum::<f64>() / w.len() as f64
        }

        fn p95_delay(&self, from: f64, to: f64) -> f64 {
            let mut d: Vec<f64> = self.window(from, to).iter().map(|s| s.2).collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            d[(d.len() as f64 * 0.95) as usize - 1]
        }

        fn swing(&self, from: f64, to: f64) -> f64 {
            let w = self.window(from, to);
            let max = w.iter().map(|s| s.1).fold(0.0, f64::max);
            let min = w.iter().map(|s| s.1).fold(f64::MAX, f64::min);
            (max - min) / self.mean_rate(from, to)
        }
    }

    fn simulate(
        capacity: impl Fn(f64) -> f64,
        base_rtt_ms: f64,
        ceiling_bps: u32,
        initial_bps: u32,
        secs: f64,
    ) -> Run {
        simulate_jittery(capacity, base_rtt_ms, 0.0, ceiling_bps, initial_bps, secs)
    }

    fn simulate_jittery(
        capacity: impl Fn(f64) -> f64,
        base_rtt_ms: f64,
        jitter_ms: f64,
        ceiling_bps: u32,
        initial_bps: u32,
        secs: f64,
    ) -> Run {
        simulate_path(Path {
            capacity: &capacity,
            rtt_ms: &|_| base_rtt_ms,
            jitter: Jitter::Uniform(jitter_ms),
            idle: &|_| false,
            rtt_window_s: f64::INFINITY,
            ceiling_bps,
            initial_bps,
            secs,
            ..Path::DEFAULT
        })
    }

    #[derive(Clone, Copy)]
    enum Jitter {
        /// Independent per-frame noise up to this many ms; acks may reorder.
        Uniform(f64),
        /// The measured tailnet relay: per-packet noise plus bursts of ~50 ms, acks in order.
        Relay,
    }

    /// xorshift64, so every run sees the same noise.
    struct Rng(u64);

    impl Rng {
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Extra one-way delay per frame, in seconds.
    struct Noise {
        model: Jitter,
        rng: Rng,
        burst: bool,
        toggle_at: f64,
    }

    impl Noise {
        fn new(model: Jitter) -> Self {
            Self {
                model,
                rng: Rng(0x9E37_79B9_7F4A_7C15),
                burst: false,
                toggle_at: 1.5,
            }
        }

        fn sample(&mut self, now: f64) -> f64 {
            match self.model {
                Jitter::Uniform(ms) => self.rng.unit() * ms / 1000.0,
                Jitter::Relay => {
                    // A fifth of the time in a 100-400 ms burst near +50 ms: RTT 29-82 ms, mean ~48, sd ~14.
                    if now >= self.toggle_at {
                        self.burst = !self.burst;
                        let u = self.rng.unit();
                        self.toggle_at = now
                            + if self.burst {
                                0.1 + 0.3 * u
                            } else {
                                0.4 + 1.2 * u
                            };
                    }
                    let ms = if self.burst {
                        35.0 + 18.0 * self.rng.unit()
                    } else {
                        25.0 * self.rng.unit()
                    };
                    ms / 1000.0
                }
            }
        }

        fn in_order(&self) -> bool {
            matches!(self.model, Jitter::Relay)
        }
    }

    struct Path<'a> {
        capacity: &'a dyn Fn(f64) -> f64,
        /// Propagation round trip, which a relay can change mid-session.
        rtt_ms: &'a dyn Fn(f64) -> f64,
        jitter: Jitter,
        /// A static screen: a trickle of small frames and an occasional large repaint.
        idle: &'a dyn Fn(f64) -> bool,
        /// Size of the idle screen's repaint every 4 s.
        repaint_bytes: f64,
        /// Base-RTT bucket width; `RTT_MIN_WINDOW` in `h264.rs` is 30 s.
        rtt_window_s: f64,
        pixels: u64,
        min_bpp: f64,
        ceiling_bps: u32,
        initial_bps: u32,
        secs: f64,
    }

    impl Path<'static> {
        const DEFAULT: Self = Path {
            capacity: &|_| 5e6,
            rtt_ms: &|_| 30.0,
            jitter: Jitter::Uniform(0.0),
            idle: &|_| false,
            repaint_bytes: 60_000.0,
            rtt_window_s: f64::INFINITY,
            pixels: 1920 * 1080,
            min_bpp: 0.05,
            ceiling_bps: 25_000_000,
            initial_bps: 8_000_000,
            secs: 60.0,
        };
    }

    fn simulate_path(p: Path) -> Run {
        let mut noise = Noise::new(p.jitter);
        let mut last_acked = 0.0f64;
        let max_fps = 60u32;
        let pixels = p.pixels;
        let interval = Duration::from_millis(300);
        let tick = 1.0 / f64::from(max_fps);
        let keyframe_natural = 150_000.0f64;
        let mut c = RateController::new(cfg(p.ceiling_bps), p.initial_bps, None);
        let mut frames: Vec<Frame> = Vec::new();
        let mut link_free = 0.0f64;
        let mut fps = max_fps;
        let mut fps_changed = 0.0;
        let mut last_pass = -1.0;
        let mut last_keyframe = -100.0;
        let mut last_repaint = 0.0;
        let mut keyframe_bytes = 0.0;
        let mut last_control = 0.0;
        let (mut min_cur, mut min_prev, mut bucket_at) = (f64::INFINITY, f64::INFINITY, 0.0);
        let mut window = DelayWindow::default();
        let mut gated = false;
        let mut sent = 0.0;
        let mut trace = Vec::new();
        let mut rebaselined = Vec::new();
        let mut ticks = 0u64;
        let mut now = 0.0;
        while now < p.secs {
            let link_bps = (p.capacity)(now);
            let rtt_s = (p.rtt_ms)(now) / 1000.0;
            let idle = (p.idle)(now);
            let target = f64::from(c.target_bps());
            let unacked: f64 = frames
                .iter()
                .filter(|f| f.acked > now)
                .map(|f| f.bytes)
                .sum();
            // h264.rs budgets on the measured base; the fixed-path scenarios keep the true one.
            let base_ms = if p.rtt_window_s.is_finite() && min_cur.min(min_prev).is_finite() {
                min_cur.min(min_prev) * 1000.0
            } else {
                (p.rtt_ms)(now)
            };
            let budget = (target / 8.0 * (base_ms + 80.0) / 1000.0).max(64.0 * 1024.0);
            let since_pass = Duration::from_secs_f64((now - last_pass).max(0.0));
            let repaint = idle && now - last_repaint >= 4.0;
            let changed = !idle || ticks.is_multiple_of(4) || repaint;
            ticks += 1;
            let due = changed && !fps_gate_drops(since_pass, fps, max_fps);
            gated |= due && unacked > budget;
            if due && unacked <= budget {
                last_pass = now;
                let period = keyframe_interval_secs(2.0, keyframe_bytes as u64, target, 0.05);
                let bytes = if now - last_keyframe >= period {
                    last_keyframe = now;
                    keyframe_bytes = keyframe_natural;
                    keyframe_bytes
                } else if repaint {
                    last_repaint = now;
                    p.repaint_bytes
                } else if idle {
                    1_500.0
                } else {
                    target / 8.0 / f64::from(fps)
                };
                sent += bytes;
                link_free = now.max(link_free) + bytes * 8.0 / link_bps;
                let mut acked = link_free + rtt_s + noise.sample(now);
                if noise.in_order() {
                    acked = acked.max(last_acked);
                    last_acked = acked;
                }
                frames.push(Frame {
                    sent: now,
                    bytes,
                    acked,
                });
            }
            now += tick;
            if now - last_control < interval.as_secs_f64() {
                continue;
            }
            let dt = now - last_control;
            let own = |f: &Frame| f.bytes * 8.0 / c.serialization_bps();
            let mut delivered_bytes = 0.0;
            let mut acked: Vec<&Frame> = frames
                .iter()
                .filter(|f| f.acked > last_control && f.acked <= now)
                .collect();
            acked.sort_by(|a, b| a.acked.total_cmp(&b.acked));
            for f in acked {
                if f.acked - bucket_at >= p.rtt_window_s {
                    min_prev = min_cur;
                    min_cur = f64::INFINITY;
                    bucket_at = f.acked;
                }
                let rtt = f.acked - f.sent;
                min_cur = min_cur.min(rtt);
                let delay = ((rtt - min_cur.min(min_prev) - own(f)) * 1000.0).max(0.0);
                window.push(Duration::from_secs_f64(f.acked), delay);
                delivered_bytes += f.bytes;
            }
            // As in h264.rs, the fallback is for acks that have gone quiet.
            let quiet = delivered_bytes == 0.0;
            let shipping = now - last_pass < 1.0 && quiet;
            let base_s = min_cur.min(min_prev);
            let base_s = if base_s.is_finite() { base_s } else { 0.0 };
            // As in h264.rs: the oldest unacked frame's wait past the base RTT bounds the queue from below.
            let waiting = frames
                .iter()
                .filter(|f| f.acked > now)
                .min_by(|a, b| a.sent.total_cmp(&b.sent))
                .map_or(0.0, |f| {
                    ((now - f.sent - base_s - own(f)) * 1000.0).max(0.0)
                });
            let waiting = if shipping { waiting } else { 0.0 };
            let mut sample = window
                .min_ms(Duration::from_secs_f64(now))
                .unwrap_or(0.0)
                .max(waiting);
            let mut interval_ms = window
                .min_since(Duration::from_secs_f64(last_control))
                .unwrap_or(sample)
                .max(waiting);
            let delivered = delivered_bytes * 8.0 / dt;
            let sent_bps = std::mem::take(&mut sent) * 8.0 / dt;
            let gated_now = std::mem::take(&mut gated);
            let inflight: f64 = frames
                .iter()
                .filter(|f| f.acked > now)
                .map(|f| f.bytes)
                .sum();
            if let Some(ms) = c.observe_path(ShiftInput {
                dt: Duration::from_secs_f64(dt),
                delay_ms: interval_ms,
                delivered_bps: Some(delivered),
                sent_bps: Some(sent_bps),
                inflight_bytes: Some(inflight),
                gated: gated_now,
            }) {
                let base = min_cur.min(min_prev) + ms / 1000.0;
                (min_cur, min_prev, bucket_at) = (base, base, now);
                sample = (sample - ms).max(0.0);
                interval_ms = (interval_ms - ms).max(0.0);
                window.rebase(ms);
                rebaselined.push((now, ms));
            }
            let allowance = idr_allowance_ms(
                keyframe_bytes as u64,
                c.serialization_bps(),
                Duration::from_secs_f64(now - last_keyframe),
                interval,
            );
            c.step(RateInput {
                dt: Duration::from_secs_f64(dt),
                delay_ms: sample,
                delivered_bps: Some(delivered),
                loss: false,
                usable: true,
                allowance_ms: allowance,
                gated: gated_now,
                sent_bps: Some(sent_bps),
                jitter_ms: window.jitter_ms(),
            });
            if now - fps_changed >= 2.0 {
                let next = next_fps_level(fps, max_fps, 10, c.fps_basis_bps(), pixels, p.min_bpp);
                if next != fps {
                    fps = next;
                    fps_changed = now;
                }
            }
            trace.push((now, f64::from(c.target_bps()), interval_ms, fps));
            last_control = now;
            frames.retain(|f| f.acked > now - 1.0);
        }
        Run { trace, rebaselined }
    }

    fn report(name: &str, run: &Run, capacity: f64, from: f64, to: f64) {
        let w = run.window(from, to);
        if std::env::var("RC_TRACE").is_ok() {
            for s in &w {
                eprintln!(
                    "  {:6.1}s {:7.2} Mbit {:6.0} ms {} fps",
                    s.0,
                    s.1 / 1e6,
                    s.2,
                    s.3
                );
            }
        }
        eprintln!(
            "{name}: mean {:.2} Mbit ({:.0}% of capacity), p95 queue {:.0} ms, swing {:.0}%, fps {}",
            run.mean_rate(from, to) / 1e6,
            run.mean_rate(from, to) / capacity * 100.0,
            run.p95_delay(from, to),
            run.swing(from, to) * 100.0,
            w.last().unwrap().3,
        );
    }

    fn assert_converged(name: &str, run: &Run, capacity: f64, from: f64, to: f64) {
        let start = run.window(0.0, from);
        let worst = start.iter().map(|s| s.2).fold(0.0, f64::max);
        let settled = start.iter().rev().find(|s| s.2 > 60.0).map_or(0.0, |s| s.0);
        eprintln!("{name}: startup worst queue {worst:.0} ms, last above 60 ms at {settled:.1} s");
        report(name, run, capacity, from, to);
        let mean = run.mean_rate(from, to);
        assert!(mean < capacity, "{name}: mean {mean} not below capacity");
        assert!(
            mean > 0.75 * capacity,
            "{name}: mean {mean} leaves the link idle"
        );
        assert!(run.p95_delay(from, to) < 60.0, "{name}: queue too deep");
        assert!(run.swing(from, to) < 0.3, "{name}: sawtooth");
        assert!(
            run.rebaselined.is_empty(),
            "{name}: a queue was taken for path latency: {:?}",
            run.rebaselined
        );
    }

    /// A relayed 5 Mbit path whose round trip and screen activity change over time.
    fn relayed(rtt_ms: &dyn Fn(f64) -> f64, idle: &dyn Fn(f64) -> bool, secs: f64) -> Run {
        simulate_path(Path {
            rtt_ms,
            idle,
            rtt_window_s: 30.0,
            secs,
            ..Path::DEFAULT
        })
    }

    /// From `from` on the rate stays near capacity and the frame rate where the busy link had it.
    fn assert_unharmed(name: &str, run: &Run, from: f64, to: f64) {
        let fps = run.window(from - 5.0, from - 4.0)[0].3;
        let w = run.window(from, to);
        let low = w.iter().map(|s| s.1).fold(f64::MAX, f64::min);
        let min_fps = w.iter().map(|s| s.3).min().unwrap();
        eprintln!(
            "{name}: lowest rate {:.2} Mbit ({:.0}% of capacity), lowest fps {min_fps} (was {fps}), rebaselined {:?}",
            low / 1e6,
            low / 5e6 * 100.0,
            run.rebaselined
        );
        report(name, run, 5e6, from, to);
        assert!(low >= 0.7 * 5e6, "{name}: rate fell to {low}");
        assert!(min_fps >= fps, "{name}: fps fell to {min_fps} from {fps}");
    }

    #[test]
    fn converges_under_a_relayed_5_mbit_link() {
        let run = simulate(|_| 5e6, 32.0, 25_000_000, 8_000_000, 60.0);
        assert_converged("5 Mbit, 25 Mbit ceiling", &run, 5e6, 20.0, 60.0);
        let run = simulate(|_| 5e6, 32.0, 8_000_000, 1_000_000, 60.0);
        assert_converged("5 Mbit, 8 Mbit ceiling", &run, 5e6, 20.0, 60.0);
        let run = simulate_jittery(|_| 5e6, 32.0, 20.0, 25_000_000, 8_000_000, 60.0);
        assert_converged("5 Mbit, 20 ms jitter", &run, 5e6, 20.0, 60.0);
    }

    #[test]
    fn converges_under_a_30_mbit_tailnet_link() {
        let run = simulate(|_| 30e6, 30.0, 100_000_000, 8_000_000, 60.0);
        assert_converged("30 Mbit", &run, 30e6, 20.0, 60.0);
        let run = simulate(|_| 30e6, 30.0, 8_000_000, 8_000_000, 30.0);
        assert!(run
            .window(5.0, 30.0)
            .iter()
            .all(|s| s.1 == 8e6 && s.2 < 10.0));
        let run = simulate_jittery(|_| 30e6, 30.0, 10.0, 100_000_000, 8_000_000, 60.0);
        assert_converged("30 Mbit, 10 ms jitter", &run, 30e6, 20.0, 60.0);
    }

    #[test]
    fn converges_under_a_200_mbit_lan() {
        let run = simulate(|_| 200e6, 2.0, 400_000_000, 100_000_000, 60.0);
        assert_converged("200 Mbit", &run, 200e6, 20.0, 60.0);
    }

    #[test]
    fn a_ceiling_below_capacity_is_held() {
        let run = simulate(|_| 200e6, 2.0, 50_000_000, 50_000_000, 30.0);
        report("200 Mbit, 50 Mbit ceiling", &run, 200e6, 10.0, 30.0);
        assert!(run.window(10.0, 30.0).iter().all(|s| s.1 == 50e6));
    }

    #[test]
    fn recovers_after_a_capacity_drop() {
        let run = simulate(
            |t| if t < 30.0 { 30e6 } else { 6e6 },
            30.0,
            100_000_000,
            8_000_000,
            70.0,
        );
        let settle = run
            .window(30.0, 70.0)
            .iter()
            .find(|s| s.2 < 60.0 && s.1 < 6e6 && s.0 > 30.5)
            .map(|s| s.0 - 30.0)
            .expect("never recovered");
        eprintln!("30 -> 6 Mbit drop: queue back under 60 ms in {settle:.1} s");
        assert!(settle < 4.0, "took {settle} s");
        assert_converged("after the drop", &run, 6e6, 40.0, 70.0);
    }

    #[test]
    fn climbs_back_after_capacity_returns() {
        let run = simulate(
            |t| if t < 30.0 { 6e6 } else { 30e6 },
            30.0,
            100_000_000,
            8_000_000,
            90.0,
        );
        let reach = run
            .window(30.0, 90.0)
            .iter()
            .find(|s| s.1 >= 24e6)
            .map(|s| s.0 - 30.0)
            .expect("never climbed");
        eprintln!("6 -> 30 Mbit: 80% of the new capacity after {reach:.1} s");
        report("6 -> 30 Mbit, climbing", &run, 30e6, 29.0, 52.0);
        assert!(reach < 20.0, "took {reach} s");
        assert_converged("6 -> 30 Mbit", &run, 30e6, 60.0, 90.0);
    }

    #[test]
    fn an_idle_screen_keeps_the_rate_on_a_relayed_link() {
        let run = relayed(&|_| 30.0, &|t| t >= 20.0, 60.0);
        assert_unharmed("idle screen, 5 Mbit relayed", &run, 20.0, 60.0);
        assert!(run.rebaselined.is_empty(), "{:?}", run.rebaselined);
    }

    #[test]
    fn a_relay_rtt_step_is_rebaselined_not_cut() {
        let run = relayed(&|t| if t < 30.0 { 30.0 } else { 117.0 }, &|_| false, 70.0);
        let (at, ms) = *run.rebaselined.first().expect("never rebaselined");
        assert!(
            at - 30.0 <= 3.0,
            "rebaselined {ms} ms after {} s",
            at - 30.0
        );
        // The base already held some serialization time, so less than the 87 ms step reads as delay.
        assert!((50.0..=87.0).contains(&ms), "folded {ms} ms");
        assert_unharmed("busy screen, RTT 30 -> 117 ms", &run, 33.0, 70.0);
        assert!(run.p95_delay(33.0, 70.0) < 60.0, "queue left standing");
    }

    #[test]
    fn an_idle_screen_survives_a_relay_rtt_step() {
        let run = relayed(
            &|t| if t < 30.0 { 30.0 } else { 117.0 },
            &|t| t >= 20.0,
            70.0,
        );
        let (at, _) = *run.rebaselined.first().expect("never rebaselined");
        assert!(at - 30.0 <= 3.0, "rebaselined after {} s", at - 30.0);
        assert_unharmed("idle screen, RTT 30 -> 117 ms", &run, 20.0, 70.0);
    }

    /// The measured tailnet relay to endor: 29 ms base RTT, bursty jitter, a 3024x1898 desktop.
    fn relay_jitter(capacity: &dyn Fn(f64) -> f64, idle: &dyn Fn(f64) -> bool, secs: f64) -> Run {
        simulate_path(Path {
            capacity,
            rtt_ms: &|_| 29.0,
            jitter: Jitter::Relay,
            idle,
            repaint_bytes: 250_000.0,
            rtt_window_s: 30.0,
            pixels: 3024 * 1898,
            min_bpp: 0.02,
            initial_bps: 8_300_000,
            secs,
            ..Path::DEFAULT
        })
    }

    /// Lowest rate and frame rate over the window, against the capacity.
    fn assert_jitter_proof(name: &str, run: &Run, capacity: f64, from: f64, to: f64, fps: u32) {
        let w = run.window(from, to);
        let low = w.iter().map(|s| s.1).fold(f64::MAX, f64::min);
        let min_fps = w.iter().map(|s| s.3).min().unwrap();
        eprintln!(
            "{name}: lowest rate {:.2} Mbit ({:.0}% of capacity), lowest fps {min_fps}, rebaselined {:?}",
            low / 1e6,
            low / capacity * 100.0,
            run.rebaselined
        );
        report(name, run, capacity, from, to);
        assert!(low >= 0.6 * capacity, "{name}: rate fell to {low}");
        assert!(min_fps >= fps, "{name}: fps fell to {min_fps}");
    }

    #[test]
    fn the_relay_noise_matches_the_live_pings() {
        let mut n = Noise::new(Jitter::Relay);
        let rtt: Vec<f64> = (0..20_000)
            .map(|i| 29.0 + n.sample(f64::from(i) / 60.0) * 1000.0)
            .collect();
        let mean = rtt.iter().sum::<f64>() / rtt.len() as f64;
        let sd = (rtt.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rtt.len() as f64).sqrt();
        let max = rtt.iter().copied().fold(0.0, f64::max);
        eprintln!("relay noise: mean {mean:.1} ms, sd {sd:.1} ms, max {max:.0} ms");
        assert!((44.0..52.0).contains(&mean), "mean {mean}");
        assert!((10.0..17.0).contains(&sd), "sd {sd}");
        assert!((75.0..=84.0).contains(&max), "max {max}");
    }

    #[test]
    fn relay_jitter_on_a_static_screen_keeps_the_rate() {
        for cap in [13e6, 5e6] {
            let run = relay_jitter(&|_| cap, &|_| true, 60.0);
            let fps = if cap > 10e6 { 30 } else { 1 };
            assert_jitter_proof(
                &format!("relay jitter, static, {} Mbit", cap / 1e6),
                &run,
                cap,
                0.0,
                60.0,
                fps,
            );
        }
    }

    #[test]
    fn relay_jitter_on_a_busy_screen_keeps_the_rate() {
        let run = relay_jitter(&|_| 13e6, &|_| false, 60.0);
        assert_jitter_proof("relay jitter, busy, 13 Mbit", &run, 13e6, 0.0, 60.0, 30);
        let run = relay_jitter(&|_| 5e6, &|_| false, 60.0);
        assert_jitter_proof("relay jitter, busy, 5 Mbit", &run, 5e6, 10.0, 60.0, 1);
        assert!(
            run.mean_rate(10.0, 60.0) < 5e6,
            "busy 5 Mbit sits over capacity"
        );
    }

    #[test]
    fn relay_jitter_with_a_mixed_screen_keeps_the_rate() {
        let run = relay_jitter(&|_| 13e6, &|t| (t % 20.0) >= 8.0, 60.0);
        assert_jitter_proof("relay jitter, mixed, 13 Mbit", &run, 13e6, 0.0, 60.0, 30);
        let run = relay_jitter(&|_| 5e6, &|t| (t % 20.0) >= 8.0, 60.0);
        assert_jitter_proof("relay jitter, mixed, 5 Mbit", &run, 5e6, 10.0, 60.0, 1);
    }

    #[test]
    fn relay_jitter_still_cuts_on_a_capacity_drop() {
        let run = relay_jitter(&|t| if t < 30.0 { 13e6 } else { 3e6 }, &|_| false, 60.0);
        let before = run.window(29.0, 30.0)[0].1;
        let first_cut = run
            .window(30.0, 60.0)
            .iter()
            .find(|s| s.1 < 0.95 * before)
            .map(|s| s.0 - 30.0)
            .expect("never cut");
        let under = run
            .window(30.0, 60.0)
            .iter()
            .find(|s| s.1 < 3e6)
            .map(|s| s.0 - 30.0)
            .expect("never got under the new capacity");
        eprintln!("relay jitter, 13 -> 3 Mbit: first cut after {first_cut:.1} s, under capacity after {under:.1} s");
        report("relay jitter, after 13 -> 3 Mbit", &run, 3e6, 40.0, 60.0);
        assert!(first_cut <= 2.0, "first cut after {first_cut} s");
        assert!(under <= 4.0, "under capacity after {under} s");
        let mean = run.mean_rate(40.0, 60.0);
        assert!((0.6 * 3e6..3e6).contains(&mean), "mean {mean}");
    }
}
