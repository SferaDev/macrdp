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
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use screencapturekit::prelude::{
    CMSampleBuffer, PixelFormat, SCContentFilter, SCDisplay, SCError, SCShareableContent, SCStream,
    SCStreamConfiguration, SCStreamDelegateTrait, SCStreamOutputTrait, SCStreamOutputType,
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

#[derive(Clone, Copy)]
pub struct StreamParams {
    pub width: u16,
    pub height: u16,
    pub fps: u32,
    pub preserve_aspect: bool,
}

#[derive(Default)]
struct Queue {
    samples: VecDeque<CMSampleBuffer>,
    waker: Option<Waker>,
    closed: bool,
}

struct Shared {
    display_id: u32,
    params: StreamParams,
    queue: Mutex<Queue>,
    stream: Mutex<Option<SCStream>>,
    /// Set by `stop_capture`, so a stop we asked for is not restarted.
    stopping: Mutex<bool>,
}

impl Shared {
    fn push(&self, sample: CMSampleBuffer) {
        let mut q = self.queue.lock().unwrap();
        if q.samples.len() >= CAPACITY {
            q.samples.pop_front();
        }
        q.samples.push_back(sample);
        if let Some(w) = q.waker.take() {
            w.wake();
        }
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
        tracing::warn!(display_id = shared.display_id, %error, "SCK stopped the capture; restarting it once the display is back");
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
    let config = SCStreamConfiguration::new()
        .with_width(u32::from(p.width))
        .with_height(u32::from(p.height))
        .with_pixel_format(PixelFormat::BGRA)
        .with_fps(p.fps)
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
        let Some(display) = content
            .displays()
            .into_iter()
            .find(|d| d.display_id() == shared.display_id)
        else {
            continue;
        };
        match open(&shared, &display) {
            Ok(stream) => {
                let old = shared.stream.lock().unwrap().replace(stream);
                drop(old);
                tracing::info!(
                    display_id = shared.display_id,
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
            display_id = shared.display_id,
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
            display_id: display.display_id(),
            params,
            queue: Mutex::new(Queue::default()),
            stream: Mutex::new(None),
            stopping: Mutex::new(false),
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
        self.shared.queue.lock().unwrap().samples.pop_front()
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
        if let Some(s) = q.samples.pop_front() {
            return Poll::Ready(Some(s));
        }
        if q.closed {
            return Poll::Ready(None);
        }
        q.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}
