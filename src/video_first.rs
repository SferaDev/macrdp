//! Holds a connection's audio capture back until its video capture has started,
//! so SCK's audio setup never delays the first frame.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_millis(10);

/// Counts accepted connections and the latest one whose video capture started.
struct Gate {
    accepted: AtomicU64,
    video_started: AtomicU64,
}

impl Gate {
    const fn new() -> Self {
        Self {
            accepted: AtomicU64::new(0),
            video_started: AtomicU64::new(0),
        }
    }

    fn accept(&self) {
        self.accepted.fetch_add(1, Ordering::SeqCst);
    }

    fn start_video(&self) {
        self.video_started
            .store(self.accepted.load(Ordering::SeqCst), Ordering::SeqCst);
    }

    fn open(&self) -> bool {
        self.video_started.load(Ordering::SeqCst) == self.accepted.load(Ordering::SeqCst)
    }
}

static GATE: Gate = Gate::new();

pub fn connection_accepted() {
    GATE.accept();
}

pub fn video_started() {
    GATE.start_video();
}

/// Waits until the latest connection's video capture started, or `timeout` passes.
pub async fn wait_for_video(timeout: Duration) {
    let started = Instant::now();
    while !GATE.open() && started.elapsed() < timeout {
        tokio::time::sleep(POLL).await;
    }
    tracing::info!(
        waited_ms = started.elapsed().as_millis() as u64,
        video_started = GATE.open(),
        "starting the audio capture"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_once_the_latest_connection_starts_video() {
        let g = Gate::new();
        assert!(g.open());
        g.accept();
        assert!(!g.open());
        g.start_video();
        assert!(g.open());
        // A new connection closes it again until its own video starts.
        g.accept();
        assert!(!g.open());
        g.start_video();
        assert!(g.open());
    }
}
