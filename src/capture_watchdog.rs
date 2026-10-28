//! Capture watchdog: rebuilds a session's capture whose ScreenCaptureKit stream
//! stopped and did not come back.
//!
//! `sck_stream` restarts a stopped stream on its own once the display returns,
//! but only for a display it can still find. When the display was recreated or
//! never re-registered, the session sits connected with no frames. The capture
//! loop feeds this watchdog the stream state; once a stream has been down for
//! [`STALL_LIMIT`] while the displays are awake and the client is not
//! minimized, the loop asks for a same-size reactivation, whose `updates()`
//! re-resolves the virtual display, wakes the displays and recreates a missing
//! one. The frame flow cannot drive this: SCK sends nothing for a static screen.

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a stream may stay down, displays awake, before the rebuild.
pub const STALL_LIMIT: Duration = Duration::from_secs(2);
/// How often a capture loop blocked on SCK wakes to check.
pub const POLL: Duration = Duration::from_millis(500);
/// Rebuilds closer together than this double the next stall limit.
const BACKOFF_WINDOW: Duration = Duration::from_secs(60);
const MAX_STALL_LIMIT: Duration = Duration::from_secs(30);

static REBUILDS: AtomicU64 = AtomicU64::new(0);

/// Capture rebuilds since the process started.
pub fn rebuilds() -> u64 {
    REBUILDS.load(Ordering::Relaxed)
}

/// What the capture loop sees on one check.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    /// Some stream of the session is stopped and not yet restarted.
    pub stream_down: bool,
    pub displays_awake: bool,
    /// The client is not minimized (no SuppressOutput in force).
    pub active: bool,
}

/// Recent rebuilds, shared across the per-connection capture objects so a
/// rebuild that does not help backs off instead of looping.
#[derive(Clone, Default)]
pub struct RebuildHistory(Arc<Mutex<Vec<Instant>>>);

impl RebuildHistory {
    /// The process-wide history the capture loops share.
    pub fn global() -> Self {
        static GLOBAL: std::sync::OnceLock<RebuildHistory> = std::sync::OnceLock::new();
        GLOBAL.get_or_init(RebuildHistory::default).clone()
    }

    fn stall_limit(&self, now: Instant) -> Duration {
        let mut recent = self.0.lock().unwrap();
        recent.retain(|t| now.duration_since(*t) < BACKOFF_WINDOW);
        let shift = u32::try_from(recent.len()).unwrap_or(u32::MAX).min(4);
        (STALL_LIMIT * 2u32.pow(shift)).min(MAX_STALL_LIMIT)
    }

    fn record(&self, now: Instant) {
        self.0.lock().unwrap().push(now);
    }
}

/// Per-capture watchdog state.
pub struct CaptureWatchdog {
    stalled_since: Option<Instant>,
    history: RebuildHistory,
}

impl CaptureWatchdog {
    pub fn new(history: RebuildHistory) -> Self {
        Self {
            stalled_since: None,
            history,
        }
    }

    /// Feeds one observation; true when the capture should be rebuilt now.
    pub fn observe(&mut self, now: Instant, obs: Observation) -> bool {
        if !(obs.stream_down && obs.displays_awake && obs.active) {
            self.stalled_since = None;
            return false;
        }
        let since = *self.stalled_since.get_or_insert(now);
        let stalled = now.duration_since(since);
        let limit = self.history.stall_limit(now);
        if stalled < limit {
            return false;
        }
        self.stalled_since = None;
        self.history.record(now);
        let total = REBUILDS.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!(
            stalled_ms = stalled.as_millis() as u64,
            limit_ms = limit.as_millis() as u64,
            total,
            "capture stream stayed down; rebuilding the session's capture"
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOWN: Observation = Observation {
        stream_down: true,
        displays_awake: true,
        active: true,
    };

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn fires_once_the_stream_stays_down_past_the_limit() {
        let mut w = CaptureWatchdog::new(RebuildHistory::default());
        let t0 = Instant::now();
        assert!(!w.observe(t0, DOWN));
        assert!(!w.observe(t0 + ms(1500), DOWN));
        let before = rebuilds();
        assert!(w.observe(t0 + STALL_LIMIT, DOWN));
        assert!(rebuilds() > before);
        // Re-arms from scratch after firing.
        assert!(!w.observe(t0 + STALL_LIMIT + ms(100), DOWN));
    }

    #[test]
    fn a_running_stream_resets_the_clock() {
        let mut w = CaptureWatchdog::new(RebuildHistory::default());
        let t0 = Instant::now();
        assert!(!w.observe(t0, DOWN));
        let up = Observation {
            stream_down: false,
            ..DOWN
        };
        assert!(!w.observe(t0 + ms(1900), up));
        assert!(!w.observe(t0 + ms(2000), DOWN));
        assert!(!w.observe(t0 + ms(3900), DOWN));
        assert!(w.observe(t0 + ms(4000), DOWN));
    }

    #[test]
    fn asleep_displays_or_a_minimized_client_hold_it_off() {
        let mut w = CaptureWatchdog::new(RebuildHistory::default());
        let t0 = Instant::now();
        let asleep = Observation {
            displays_awake: false,
            ..DOWN
        };
        let minimized = Observation {
            active: false,
            ..DOWN
        };
        assert!(!w.observe(t0, asleep));
        assert!(!w.observe(t0 + ms(10_000), asleep));
        assert!(!w.observe(t0 + ms(20_000), minimized));
        // Counts only from when the displays woke.
        assert!(!w.observe(t0 + ms(30_000), DOWN));
        assert!(!w.observe(t0 + ms(31_000), DOWN));
        assert!(w.observe(t0 + ms(32_000), DOWN));
    }

    #[test]
    fn repeated_rebuilds_back_off() {
        let history = RebuildHistory::default();
        let t0 = Instant::now();
        let mut w = CaptureWatchdog::new(history.clone());
        assert!(!w.observe(t0, DOWN));
        assert!(w.observe(t0 + ms(2000), DOWN));
        // A fresh capture after the rebuild shares the history: 4 s now.
        let mut w = CaptureWatchdog::new(history.clone());
        let t1 = t0 + ms(3000);
        assert!(!w.observe(t1, DOWN));
        assert!(!w.observe(t1 + ms(3900), DOWN));
        assert!(w.observe(t1 + ms(4000), DOWN));
        // Capped, and forgotten after the backoff window.
        assert!(history.stall_limit(t1 + ms(4000)) <= MAX_STALL_LIMIT);
        assert_eq!(
            history.stall_limit(t1 + ms(4000) + BACKOFF_WINDOW),
            STALL_LIMIT
        );
    }
}
