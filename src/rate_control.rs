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
//! Everything here is pure (no clock, no I/O) so the tests can drive it against
//! a simulated bottleneck link.

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
    probe_s: f64,
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
            probe_s: 0.0,
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
        self.over_high = 0;
        self.prev_delay_ms = 0.0;
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
        let low = self.cfg.low_ms + i.allowance_ms;
        let high = self.cfg.high_ms + i.allowance_ms;
        let rising = i.delay_ms > self.prev_delay_ms + RISE_MS;
        self.prev_delay_ms = i.delay_ms;
        let t = self.target;
        // Held captures while the link delivers less than the target: the budget, not the queue, is absorbing the excess.
        // Delivery far under the target is an idle screen, not a full link: only near-target delivery counts.
        let link_bound = i.gated
            && i.allowance_ms <= 0.0
            && delivered.is_some_and(|d| (0.6 * t..0.97 * t).contains(&d));

        let (verdict, next) = if i.loss {
            self.slow_start = false;
            self.over_high = 0;
            if let Some(d) = delivered.filter(|d| *d >= 0.5 * t) {
                self.note_capacity(d);
            }
            (Verdict::Decrease, t * self.cfg.loss_decrease)
        } else if !i.usable {
            (Verdict::Hold, t)
        } else if i.delay_ms > high || link_bound {
            self.slow_start = false;
            self.over_high += 1;
            let excess = i.delay_ms - 0.5 * (low + high);
            // Without a deep queue to drain, settling just under the delivered rate is enough.
            let margin = if i.delay_ms > high {
                OVER_MARGIN
            } else {
                BOUND_MARGIN
            };
            let factor = (1.0 - excess / DRAIN_MS).clamp(MAX_CUT, margin);
            let next = match delivered {
                // A queue that outlasts one interval means the link is carrying all it can.
                Some(d) if self.over_high >= 2 || link_bound => {
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
            if (i.delay_ms > 0.5 * (low + high) || i.gated) && i.allowance_ms <= 0.0 {
                // A rate that only holds the queue steady is the capacity: ease under it.
                (Verdict::Decrease, t * (1.0 - DRIFT_PER_S * dt))
            } else {
                (Verdict::Hold, t)
            }
        } else if i.allowance_ms > 0.0 || i.gated {
            // A keyframe's burst says nothing about the link's steady capacity, and a held capture says it is full.
            self.over_high = 0;
            (Verdict::Hold, t)
        } else {
            self.over_high = 0;
            (Verdict::Increase, self.grow(t, dt))
        };
        if verdict == Verdict::Decrease || i.delay_ms > low {
            self.probe_s = 0.0;
        }
        self.target = self.clamp(next);
        if verdict == Verdict::Increase && self.target <= t {
            return Verdict::Hold;
        }
        verdict
    }

    fn grow(&mut self, t: f64, dt: f64) -> f64 {
        let fixed = self.cfg.increase_bps.map(f64::from);
        let grown = if self.slow_start {
            let factor = self.cfg.slow_start.unwrap_or(1.0);
            (t * factor).max(t + fixed.unwrap_or(0.0))
        } else if let Some(c) = self.capacity {
            let park = c * PARK;
            if t < park * 0.97 {
                self.probe_s = 0.0;
                t + (park - t) * (dt * 1.5).min(1.0)
            } else {
                // Probe gently, then faster the longer the link stays clean past the old capacity.
                self.probe_s += dt;
                let per_s = (PROBE_PER_S * 2f64.powf(self.probe_s / PROBE_DOUBLING_S))
                    .min(FREE_GROWTH_PER_S);
                t + fixed.map_or(t * per_s * dt, |f| f.min(t * per_s * dt))
            }
        } else {
            t + fixed.unwrap_or(t * FREE_GROWTH_PER_S * dt)
        };
        // Growth on an idle screen proves nothing about the link.
        let app_cap = (2.0 * self.peak_delivered).max(f64::from(self.cfg.floor_bps));
        grown.min(app_cap.max(t))
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
    fn a_small_excess_cuts_a_little() {
        let mut c = RateController::new(cfg(100_000_000), 10_000_000, Some(10_000_000));
        c.step(input(55.0, 10_000_000.0));
        assert_eq!(c.target_bps(), 9_000_000);
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
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut jitter = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64 * jitter_ms / 1000.0
        };
        let max_fps = 60u32;
        let pixels = 1920u64 * 1080;
        let interval = Duration::from_millis(300);
        let tick = 1.0 / f64::from(max_fps);
        let keyframe_natural = 150_000.0f64;
        let base_rtt_s = base_rtt_ms / 1000.0;
        let mut c = RateController::new(cfg(ceiling_bps), initial_bps, None);
        let mut frames: Vec<Frame> = Vec::new();
        let mut link_free = 0.0f64;
        let mut fps = max_fps;
        let mut fps_changed = 0.0;
        let mut last_pass = -1.0;
        let mut last_keyframe = -100.0;
        let mut keyframe_bytes = 0.0;
        let mut last_control = 0.0;
        let mut last_ack = 0.0f64;
        let mut min_rtt = f64::INFINITY;
        let mut ewma = 0.0;
        let mut gated = false;
        let mut trace = Vec::new();
        let mut now = 0.0;
        while now < secs {
            let link_bps = capacity(now);
            let target = f64::from(c.target_bps());
            let unacked: f64 = frames
                .iter()
                .filter(|f| f.acked > now)
                .map(|f| f.bytes)
                .sum();
            let budget = (target / 8.0 * (base_rtt_ms + 80.0) / 1000.0).max(64.0 * 1024.0);
            let since_pass = Duration::from_secs_f64((now - last_pass).max(0.0));
            let due = !fps_gate_drops(since_pass, fps, max_fps);
            gated |= due && unacked > budget;
            if due && unacked <= budget {
                last_pass = now;
                let period = keyframe_interval_secs(2.0, keyframe_bytes as u64, target, 0.05);
                let bytes = if now - last_keyframe >= period {
                    last_keyframe = now;
                    keyframe_bytes = keyframe_natural;
                    keyframe_bytes
                } else {
                    target / 8.0 / f64::from(fps)
                };
                link_free = now.max(link_free) + bytes * 8.0 / link_bps;
                frames.push(Frame {
                    sent: now,
                    bytes,
                    acked: link_free + base_rtt_s + jitter(),
                });
            }
            now += tick;
            if now - last_control < interval.as_secs_f64() {
                continue;
            }
            let dt = now - last_control;
            let window: Vec<&Frame> = frames
                .iter()
                .filter(|f| f.acked > last_control && f.acked <= now)
                .collect();
            let own = |f: &Frame| f.bytes * 8.0 / c.serialization_bps();
            for f in &window {
                min_rtt = min_rtt.min(f.acked - f.sent);
                last_ack = last_ack.max(f.acked);
            }
            let interval_min = window
                .iter()
                .map(|f| ((f.acked - f.sent - own(f) - min_rtt) * 1000.0).max(0.0))
                .fold(f64::INFINITY, f64::min);
            let outstanding = frames.iter().any(|f| f.acked > now);
            let sample = if interval_min.is_finite() {
                interval_min
            } else if outstanding {
                (now - last_ack) * 1000.0
            } else {
                0.0
            };
            ewma = 0.5 * sample + 0.5 * ewma;
            let delivered = window.iter().map(|f| f.bytes).sum::<f64>() * 8.0 / dt;
            let allowance = idr_allowance_ms(
                keyframe_bytes as u64,
                c.serialization_bps(),
                Duration::from_secs_f64(now - last_keyframe),
                interval,
            );
            c.step(RateInput {
                dt: Duration::from_secs_f64(dt),
                delay_ms: ewma,
                delivered_bps: Some(delivered),
                loss: false,
                usable: true,
                allowance_ms: allowance,
                gated: std::mem::take(&mut gated),
            });
            if now - fps_changed >= 2.0 {
                let rate = f64::from(c.target_bps());
                let next = next_fps_level(fps, max_fps, 10, rate, pixels, 0.05);
                if next != fps {
                    fps = next;
                    fps_changed = now;
                }
            }
            trace.push((now, f64::from(c.target_bps()), sample, fps));
            last_control = now;
            frames.retain(|f| f.acked > now - 1.0);
        }
        Run { trace }
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
}
