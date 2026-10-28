//! Connect timeline: when the latest connection reached each connect step,
//! logged at info as one line once its first frame ships.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Auth,
    /// The displays were seen awake.
    Awake,
    Remode,
    CaptureStarted,
    FirstFrame,
}

const STEPS: usize = 5;

#[derive(Default)]
struct Timeline {
    accepted: Option<Instant>,
    at: [Option<Duration>; STEPS],
    /// How long the primary capture took to find its SCK display, and how it found it.
    sck: Option<(Duration, &'static str)>,
}

impl Timeline {
    fn accept(&mut self, now: Instant) {
        *self = Self {
            accepted: Some(now),
            ..Self::default()
        };
    }

    /// Records the first time `step` is reached; returns the summary on the first frame.
    fn mark(&mut self, step: Step, now: Instant) -> Option<String> {
        let accepted = self.accepted?;
        let slot = &mut self.at[step as usize];
        if slot.is_none() {
            *slot = Some(now.saturating_duration_since(accepted));
        }
        if step != Step::FirstFrame {
            return None;
        }
        self.accepted = None;
        Some(self.summary())
    }

    fn sck_lookup(&mut self, waited: Duration, path: &'static str) {
        if self.accepted.is_some() && self.sck.is_none() {
            self.sck = Some((waited, path));
        }
    }

    fn summary(&self) -> String {
        let ms = |s: Step| match self.at[s as usize] {
            Some(d) => d.as_millis().to_string(),
            None => "-".to_owned(),
        };
        let after_auth = match (
            self.at[Step::Auth as usize],
            self.at[Step::FirstFrame as usize],
        ) {
            (Some(auth), Some(frame)) => frame.saturating_sub(auth).as_millis().to_string(),
            _ => "-".to_owned(),
        };
        let (sck_wait, sck) = self.sck.map_or_else(
            || ("-".to_owned(), "-"),
            |(d, path)| (d.as_millis().to_string(), path),
        );
        format!(
            "auth={} wake={} remode={} sck_wait={} sck={} capture={} first_frame={} first_frame_after_auth={}",
            ms(Step::Auth),
            ms(Step::Awake),
            ms(Step::Remode),
            sck_wait,
            sck,
            ms(Step::CaptureStarted),
            ms(Step::FirstFrame),
            after_auth
        )
    }
}

static TIMELINE: Mutex<Timeline> = Mutex::new(Timeline {
    accepted: None,
    at: [None; STEPS],
    sck: None,
});
/// Lets the per-frame call skip the lock once the first frame is logged.
static ARMED: AtomicBool = AtomicBool::new(false);

/// Starts a new timeline for a just-accepted connection.
pub fn accepted() {
    TIMELINE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .accept(Instant::now());
    ARMED.store(true, Ordering::Relaxed);
}

/// Marks `step` for the latest connection; later repeats (reactivations) are ignored.
pub fn mark(step: Step) {
    if !ARMED.load(Ordering::Relaxed) {
        return;
    }
    let summary = TIMELINE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .mark(step, Instant::now());
    if let Some(timeline) = summary {
        ARMED.store(false, Ordering::Relaxed);
        tracing::info!(%timeline, "connect timeline (ms since accept)");
    }
}

/// Records how the latest connection's primary capture found its SCK display
/// (`cached`, `listed` or `relisted`) and how long that took.
pub fn sck_lookup(waited: Duration, path: &'static str) {
    if ARMED.load(Ordering::Relaxed) {
        TIMELINE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sck_lookup(waited, path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn summarizes_each_step_once_the_first_frame_ships() {
        let t0 = Instant::now();
        let mut t = Timeline::default();
        t.accept(t0);
        assert_eq!(t.mark(Step::Awake, t0 + ms(150)), None);
        assert_eq!(t.mark(Step::Auth, t0 + ms(320)), None);
        t.sck_lookup(ms(90), "listed");
        t.sck_lookup(ms(400), "cached");
        assert_eq!(t.mark(Step::CaptureStarted, t0 + ms(600)), None);
        assert_eq!(t.mark(Step::CaptureStarted, t0 + ms(900)), None);
        assert_eq!(
            t.mark(Step::FirstFrame, t0 + ms(800)).as_deref(),
            Some(
                "auth=320 wake=150 remode=- sck_wait=90 sck=listed capture=600 first_frame=800 \
                 first_frame_after_auth=480"
            )
        );
        // Done until the next accept.
        assert_eq!(t.mark(Step::FirstFrame, t0 + ms(2000)), None);
    }

    #[test]
    fn nothing_is_recorded_before_an_accept() {
        let t0 = Instant::now();
        let mut t = Timeline::default();
        assert_eq!(t.mark(Step::FirstFrame, t0), None);
        t.accept(t0);
        t.mark(Step::Remode, t0 + ms(50));
        t.sck_lookup(ms(50), "cached");
        t.accept(t0 + ms(100));
        assert_eq!(
            t.mark(Step::FirstFrame, t0 + ms(400)).as_deref(),
            Some(
                "auth=- wake=- remode=- sck_wait=- sck=- capture=- first_frame=300 \
                 first_frame_after_auth=-"
            )
        );
    }
}
