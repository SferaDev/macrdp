//! A ScreenCaptureKit display stream that restarts itself when SCK stops it.
//!
//! SCK stops a stream when its display goes away, which a display sleep does
//! to every virtual display: on wake the displays come back but the stream
//! stays dead, and `AsyncSCStream` (no delegate) never even reports it, so the
//! capture loop waits forever and the session freezes on its last frame. This
//! stream watches for the stop, waits for the display to return and opens a
//! new capture of it with the same configuration, feeding the same queue. The
//! capture loops see a pause in frames, not an end.

#![cfg(target_os = "macos")]

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use screencapturekit::prelude::{
    CMSampleBuffer, CMTime, PixelFormat, SCContentFilter, SCDisplay, SCError, SCShareableContent,
    SCStream, SCStreamConfiguration, SCStreamDelegateTrait, SCStreamOutputTrait,
    SCStreamOutputType,
};

/// Samples kept for a consumer that falls behind; the oldest is dropped.
const CAPACITY: usize = 4;

/// SCK's surface-pool depth: its default (3) plus one for the frame the flush
/// burst keeps retained (`ScreenCaptureUpdates::last_frame`), so holding it
/// doesn't shrink the headroom SCK had before.
pub(crate) const SCK_QUEUE_DEPTH: u32 = 4;
/// How long a stopped stream waits for its display to come back.
const RESTART_WINDOW: Duration = Duration::from_secs(120);
const RESTART_POLL: Duration = Duration::from_millis(500);

/// The capture's minimum frame interval: `Some(n)` caps it at 1/n s, `None`
/// leaves it uncapped. A display refreshing no faster than `fps` already keeps
/// SCK within `fps`, and a cap there makes SCK skip refreshes that come a hair
/// early and hold each frame back toward its presentation time.
fn frame_cap(fps: u32, display_hz: Option<f64>, paced: bool) -> Option<u32> {
    let fps = fps.max(1);
    match display_hz {
        Some(hz) if !paced && hz > 0.0 && hz <= f64::from(fps) + 0.5 => None,
        _ => Some(fps),
    }
}

fn display_hz(display_id: u32) -> Option<f64> {
    core_graphics::display::CGDisplay::new(display_id)
        .display_mode()
        .map(|m| m.refresh_rate())
        .filter(|hz| *hz > 0.0)
}

/// `MACRDP_SCK_PACED=1` restores the 1/fps cap on every capture.
fn paced() -> bool {
    matches!(
        std::env::var("MACRDP_SCK_PACED").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

#[derive(Clone, Copy)]
pub struct StreamParams {
    pub width: u16,
    pub height: u16,
    pub fps: u32,
    pub preserve_aspect: bool,
}

#[derive(Default)]
struct Queue {
    /// Each sample with its arrival time (`latency_trace::now_ns`, 0 untraced).
    samples: VecDeque<(CMSampleBuffer, u64)>,
    waker: Option<Waker>,
    closed: bool,
}

struct Shared {
    /// Follows the virtual display when WindowServer re-registers it mid-session.
    display_id: AtomicU32,
    params: StreamParams,
    queue: Mutex<Queue>,
    stream: Mutex<Option<SCStream>>,
    /// Set by `stop_capture`, so a stop we asked for is not restarted.
    stopping: Mutex<bool>,
    /// Arrival time of the sample last taken from the queue.
    taken_arrival_ns: AtomicU64,
}

impl Shared {
    fn push(&self, sample: CMSampleBuffer) {
        let arrived = if crate::latency_trace::enabled() {
            crate::latency_trace::now_ns()
        } else {
            0
        };
        let mut q = self.queue.lock().unwrap();
        if q.samples.len() >= CAPACITY {
            q.samples.pop_front();
        }
        q.samples.push_back((sample, arrived));
        if let Some(w) = q.waker.take() {
            w.wake();
        }
    }

    fn take(&self, q: &mut Queue) -> Option<CMSampleBuffer> {
        let (sample, arrived) = q.samples.pop_front()?;
        self.taken_arrival_ns.store(arrived, Ordering::Relaxed);
        Some(sample)
    }

    fn close(&self) {
        let mut q = self.queue.lock().unwrap();
        q.closed = true;
        if let Some(w) = q.waker.take() {
            w.wake();
        }
    }
}

struct Output(Weak<Shared>);

impl SCStreamOutputTrait for Output {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if of_type == SCStreamOutputType::Screen {
            if let Some(shared) = self.0.upgrade() {
                shared.push(sample);
            }
        }
    }
}

struct Delegate(Weak<Shared>);

impl SCStreamDelegateTrait for Delegate {
    fn did_stop_with_error(&self, error: SCError) {
        let Some(shared) = self.0.upgrade() else {
            return;
        };
        if *shared.stopping.lock().unwrap() {
            return;
        }
        tracing::warn!(display_id = shared.display_id.load(Ordering::Relaxed), %error, "SCK stopped the capture; restarting it once the display is back");
        let weak = self.0.clone();
        let spawned = std::thread::Builder::new()
            .name("sck-restart".into())
            .spawn(move || restart(weak));
        if spawned.is_err() {
            shared.close();
        }
    }
}

fn open(shared: &Arc<Shared>, display: &SCDisplay) -> Result<SCStream, SCError> {
    let p = shared.params;
    let filter = SCContentFilter::create()
        .with_display(display)
        .with_excluding_windows(&[])
        .build();
    let hz = display_hz(shared.display_id.load(Ordering::Relaxed));
    let cap = frame_cap(p.fps, hz, paced());
    tracing::info!(
        display_id = shared.display_id.load(Ordering::Relaxed),
        ?hz,
        fps = p.fps,
        ?cap,
        "SCK frame cap"
    );
    let interval = match cap {
        Some(fps) => CMTime::new(1, fps as i32),
        None => CMTime::new(0, 1),
    };
    let config = SCStreamConfiguration::new()
        .with_width(u32::from(p.width))
        .with_height(u32::from(p.height))
        .with_pixel_format(PixelFormat::BGRA)
        .with_minimum_frame_interval(&interval)
        // The client draws the cursor; see capture.rs.
        .with_shows_cursor(false)
        .with_scales_to_fit(true)
        .with_preserves_aspect_ratio(p.preserve_aspect)
        .with_queue_depth(SCK_QUEUE_DEPTH);
    let mut stream =
        SCStream::new_with_delegate(&filter, &config, Delegate(Arc::downgrade(shared)));
    stream.add_output_handler(Output(Arc::downgrade(shared)), SCStreamOutputType::Screen);
    stream.start_capture()?;
    Ok(stream)
}

fn restart(weak: Weak<Shared>) {
    let started = Instant::now();
    while started.elapsed() < RESTART_WINDOW {
        std::thread::sleep(RESTART_POLL);
        let Some(shared) = weak.upgrade() else {
            return;
        };
        if *shared.stopping.lock().unwrap() {
            return;
        }
        let Ok(content) = SCShareableContent::get() else {
            continue;
        };
        let displays = content.displays();
        let old = shared.display_id.load(Ordering::Relaxed);
        let ids: Vec<u32> = displays.iter().map(|d| d.display_id()).collect();
        let size = (
            u32::from(shared.params.width),
            u32::from(shared.params.height),
        );
        let Some(id) = crate::virtual_display::follow_reregistered(old, size, &ids) else {
            continue;
        };
        let Some(display) = displays.into_iter().find(|d| d.display_id() == id) else {
            continue;
        };
        shared.display_id.store(id, Ordering::Relaxed);
        match open(&shared, &display) {
            Ok(stream) => {
                let old = shared.stream.lock().unwrap().replace(stream);
                drop(old);
                tracing::info!(
                    display_id = shared.display_id.load(Ordering::Relaxed),
                    after_ms = started.elapsed().as_millis() as u64,
                    "SCK capture restarted"
                );
                return;
            }
            Err(e) => tracing::debug!(error = %e, "SCK capture restart failed; retrying"),
        }
    }
    if let Some(shared) = weak.upgrade() {
        tracing::warn!(
            display_id = shared.display_id.load(Ordering::Relaxed),
            "display did not come back; ending its capture"
        );
        shared.close();
    }
}

pub struct CaptureStream {
    shared: Arc<Shared>,
}

impl CaptureStream {
    /// Opens and starts a capture of `display`.
    pub fn start(display: &SCDisplay, params: StreamParams) -> Result<Self, SCError> {
        let shared = Arc::new(Shared {
            display_id: AtomicU32::new(display.display_id()),
            params,
            queue: Mutex::new(Queue::default()),
            stream: Mutex::new(None),
            stopping: Mutex::new(false),
            taken_arrival_ns: AtomicU64::new(0),
        });
        let stream = open(&shared, display)?;
        *shared.stream.lock().unwrap() = Some(stream);
        Ok(Self { shared })
    }

    /// The next sample; `None` once the capture has ended for good.
    pub fn next(&self) -> NextSample<'_> {
        NextSample(&self.shared)
    }

    pub fn try_next(&self) -> Option<CMSampleBuffer> {
        let mut q = self.shared.queue.lock().unwrap();
        self.shared.take(&mut q)
    }

    /// When the sample last returned arrived from SCK, in
    /// `latency_trace::now_ns`; 0 unless the latency trace is on.
    pub fn taken_arrival_ns(&self) -> u64 {
        self.shared.taken_arrival_ns.load(Ordering::Relaxed)
    }

    pub fn stop_capture(&self) -> Result<(), SCError> {
        *self.shared.stopping.lock().unwrap() = true;
        match self.shared.stream.lock().unwrap().as_ref() {
            Some(s) => s.stop_capture(),
            None => Ok(()),
        }
    }
}

pub struct NextSample<'a>(&'a Shared);

impl Future for NextSample<'_> {
    type Output = Option<CMSampleBuffer>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut q = self.0.queue.lock().unwrap();
        if let Some(s) = self.0.take(&mut q) {
            return Poll::Ready(Some(s));
        }
        if q.closed {
            return Poll::Ready(None);
        }
        q.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::frame_cap;

    #[test]
    fn uncaps_a_display_no_faster_than_fps() {
        assert_eq!(frame_cap(120, Some(120.0), false), None);
        assert_eq!(frame_cap(60, Some(60.0), false), None);
        assert_eq!(frame_cap(60, Some(59.94), false), None);
        assert_eq!(frame_cap(120, Some(60.0), false), None);
    }

    #[test]
    fn caps_a_faster_display_an_unknown_rate_or_when_paced() {
        assert_eq!(frame_cap(60, Some(120.0), false), Some(60));
        assert_eq!(frame_cap(120, Some(165.0), false), Some(120));
        assert_eq!(frame_cap(60, None, false), Some(60));
        assert_eq!(frame_cap(120, Some(120.0), true), Some(120));
        assert_eq!(frame_cap(0, None, false), Some(1));
    }
}
