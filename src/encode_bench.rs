//! Encoder throughput with full-frame motion; `cargo test --release encode_bench -- --ignored --nocapture`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use crate::videotoolbox::{EncodedFrame, Encoder};

const SCROLL_ROWS: usize = 12;

/// A tall BGRA texture; frame `k` is the window starting `k * SCROLL_ROWS` rows down.
struct Source {
    pixels: Vec<u8>,
    stride: usize,
    height: usize,
    frames: usize,
}

impl Source {
    fn new(width: usize, height: usize, frames: usize) -> Self {
        let stride = width * 4;
        let rows = height + frames * SCROLL_ROWS;
        let mut pixels = vec![0u8; stride * rows];
        let mut seed = 0x9E37_79B9u32;
        for y in 0..rows {
            let line = y % 18;
            for x in 0..width {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                // Glyph-like strokes on text lines, flat gaps between them.
                let ink = line < 12 && (x / 7 + y / 18) % 9 != 0 && seed.is_multiple_of(3);
                let base = if ink { 30 } else { 235 };
                let tint = ((x * 255) / width) as u8;
                let i = y * stride + x * 4;
                pixels[i] = base;
                pixels[i + 1] = base.wrapping_add(tint / 16);
                pixels[i + 2] = base;
                pixels[i + 3] = 255;
            }
        }
        Self {
            pixels,
            stride,
            height,
            frames,
        }
    }

    fn frame(&self, k: usize) -> &[u8] {
        let start = (k % self.frames) * SCROLL_ROWS * self.stride;
        &self.pixels[start..start + self.height * self.stride]
    }
}

#[derive(Default)]
struct Stats {
    encoded: u64,
    bytes: u64,
    latency_us: Vec<u64>,
    submit_us: Vec<u64>,
}

fn pct(v: &mut [u64], p: f64) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn sizes() -> Vec<(u16, u16)> {
    let spec = std::env::var("MACRDP_BENCH_SIZES").unwrap_or_else(|_| "3440x1440".into());
    spec.split(',')
        .filter_map(|s| {
            let (w, h) = s.trim().split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .collect()
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Default)]
struct Counters {
    submitted: AtomicU64,
    shipped: AtomicU64,
}

impl Counters {
    fn outstanding(&self) -> u64 {
        self.submitted
            .load(Ordering::Acquire)
            .saturating_sub(self.shipped.load(Ordering::Acquire))
    }
}

/// `paced` drops a sample the busy encoder can't take, like SCK; `primary` makes this output yield to it.
#[allow(clippy::too_many_arguments)]
fn run_output(
    width: u16,
    height: u16,
    fps: u32,
    in_flight: u64,
    paced: bool,
    duration: Duration,
    own: Arc<Counters>,
    primary: Option<Arc<Counters>>,
) -> (Stats, f64) {
    let limit = || match &primary {
        Some(p) => crate::h264::output_in_flight_limit(in_flight, p.outstanding()),
        None => in_flight,
    };
    let source = Source::new(usize::from(width), usize::from(height), 240);
    let bitrate = env_u64("MACRDP_BENCH_MBPS", 6) as u32 * 1_000_000;
    let mut encoder = Encoder::new(width, height, fps, bitrate, 1.0).expect("encoder");
    let rx = encoder.take_receiver().expect("receiver");
    let submitted_at: Arc<Mutex<Vec<(i64, Instant)>>> = Arc::default();
    let (done_tx, done_rx) = mpsc::channel::<Stats>();
    let slot = Arc::new((Mutex::new(()), std::sync::Condvar::new()));
    {
        let submitted_at = submitted_at.clone();
        let own = own.clone();
        let slot = slot.clone();
        std::thread::spawn(move || {
            let mut stats = Stats::default();
            while let Ok(f) = rx.recv() {
                let f: EncodedFrame = f;
                let now = Instant::now();
                if let Some(t) = submitted_at
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|(pts, _)| *pts == f.pts)
                    .map(|(_, t)| *t)
                {
                    stats
                        .latency_us
                        .push(now.duration_since(t).as_micros() as u64);
                }
                stats.encoded += 1;
                stats.bytes += f.data.len() as u64;
                own.shipped.fetch_add(1, Ordering::Release);
                slot.1.notify_all();
            }
            let _ = done_tx.send(stats);
        });
    }
    let interval = Duration::from_secs_f64(1.0 / f64::from(fps));
    let start = Instant::now();
    let mut next_sample = start;
    let mut submit_us = Vec::new();
    let mut k = 0usize;
    while start.elapsed() < duration {
        if paced {
            let now = Instant::now();
            if now < next_sample {
                std::thread::sleep(next_sample - now);
            }
            next_sample += interval;
            k += 1;
            if own.outstanding() >= limit() {
                continue;
            }
        } else {
            let mut g = slot.0.lock().unwrap();
            while own.outstanding() >= limit() {
                g = slot.1.wait_timeout(g, interval).unwrap().0;
            }
            drop(g);
            k += 1;
        }
        let t = Instant::now();
        let pts = encoder
            .encode_bgra(source.frame(k), source.stride, k <= 1)
            .expect("encode");
        submit_us.push(t.elapsed().as_micros() as u64);
        submitted_at.lock().unwrap().push((pts, t));
        own.submitted.fetch_add(1, Ordering::Release);
    }
    let wall = start.elapsed().as_secs_f64();
    let _ = encoder.flush();
    drop(encoder);
    let mut stats = done_rx.recv().expect("ship stats");
    stats.submit_us = submit_us;
    (stats, wall)
}

fn report(label: &str, sizes: &[(u16, u16)], runs: Vec<(Stats, f64)>) {
    for ((w, h), (mut s, wall)) in sizes.iter().zip(runs) {
        let fps = s.encoded as f64 / wall;
        let mpx = f64::from(*w) * f64::from(*h) / 1e6;
        println!(
            "{label} {w}x{h}: {fps:6.1} frames/s ({:5.2} ms/frame, {:6.0} Mpx/s), \
             vt latency p50 {:5.1} p95 {:5.1} ms, submit p50 {:4.2} ms, {:5.1} KB/frame",
            1000.0 / fps,
            fps * mpx,
            pct(&mut s.latency_us, 0.5) as f64 / 1000.0,
            pct(&mut s.latency_us, 0.95) as f64 / 1000.0,
            pct(&mut s.submit_us, 0.5) as f64 / 1000.0,
            s.bytes as f64 / s.encoded.max(1) as f64 / 1024.0,
        );
    }
}

fn run_all(fps: u32, in_flight: u64, paced: bool, yield_to_primary: bool) {
    let sizes = sizes();
    let secs = env_u64("MACRDP_BENCH_SECS", 4);
    let counters: Vec<Arc<Counters>> = sizes.iter().map(|_| Arc::default()).collect();
    let handles: Vec<_> = sizes
        .iter()
        .enumerate()
        .map(|(i, &(w, h))| {
            let own = counters[i].clone();
            let primary = (yield_to_primary && i > 0).then(|| counters[0].clone());
            std::thread::spawn(move || {
                let d = Duration::from_secs(secs);
                run_output(w, h, fps, in_flight, paced, d, own, primary)
            })
        })
        .collect();
    let runs = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let mode = if paced { "paced" } else { "unpaced" };
    let policy = if yield_to_primary { " yield" } else { "" };
    report(
        &format!("[{mode} {fps}Hz in_flight={in_flight}{policy}]"),
        &sizes,
        runs,
    );
}

#[test]
#[ignore = "benchmark; needs the hardware encoder"]
fn encode_bench() {
    let fps = env_u64("MACRDP_BENCH_FPS", 120) as u32;
    for in_flight in [1, 2, 3] {
        run_all(fps, in_flight, false, false);
    }
    for in_flight in [1, 2, 3] {
        run_all(fps, in_flight, true, false);
    }
    if sizes().len() > 1 {
        for in_flight in [2, 3] {
            run_all(fps, in_flight, true, true);
        }
    }
}
