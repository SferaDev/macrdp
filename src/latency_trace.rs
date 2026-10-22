//! Opt-in per-stage latency histograms (`MACRDP_LATENCY_TRACE=1`). Each stage
//! collects samples in microseconds and logs p50/p95/max at INFO every
//! [`REPORT_EVERY`]; with the env var unset every call is one relaxed load.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const REPORT_EVERY: Duration = Duration::from_secs(5);

static STATE: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 off, 2 on

pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = matches!(
                std::env::var("MACRDP_LATENCY_TRACE").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE")
            );
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

struct Hist {
    since: Instant,
    stages: BTreeMap<String, Vec<u32>>,
}

fn hist() -> &'static Mutex<Hist> {
    static H: OnceLock<Mutex<Hist>> = OnceLock::new();
    H.get_or_init(|| {
        Mutex::new(Hist {
            since: Instant::now(),
            stages: BTreeMap::new(),
        })
    })
}

/// Monotonic nanoseconds since the first call; small enough to ride a
/// pointer-sized refcon through VideoToolbox.
pub fn now_ns() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

static LAST_KEY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static KEY_SEEN: [std::sync::atomic::AtomicU64; 12] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 12];

/// Marks a key event, so the first frame each monitor ships after it can be
/// timed against it.
pub fn mark_key() {
    if enabled() {
        LAST_KEY_NS.store(now_ns().max(1), Ordering::Relaxed);
    }
}

/// Records `{stage}` as the time since the last key event, once per key and
/// monitor slot, and only within 500 ms of the key.
pub fn record_since_key(stage: &str, slot: usize, at_ns: u64) {
    if !enabled() || slot >= KEY_SEEN.len() {
        return;
    }
    let key = LAST_KEY_NS.load(Ordering::Relaxed);
    if key == 0 || at_ns < key || at_ns - key > 500_000_000 {
        return;
    }
    if KEY_SEEN[slot].swap(key, Ordering::Relaxed) != key {
        record_us(stage, (at_ns - key) / 1000);
    }
}

pub fn record_us(stage: &str, us: u64) {
    if !enabled() {
        return;
    }
    let mut h = hist().lock().unwrap();
    match h.stages.get_mut(stage) {
        Some(v) => v.push(us.min(u64::from(u32::MAX)) as u32),
        None => {
            h.stages
                .insert(stage.to_owned(), vec![us.min(u64::from(u32::MAX)) as u32]);
        }
    }
    if h.since.elapsed() >= REPORT_EVERY {
        let window_ms = h.since.elapsed().as_millis();
        h.since = Instant::now();
        for (stage, v) in &mut h.stages {
            if v.is_empty() {
                continue;
            }
            let (n, p50, p95, max) = summarize(v);
            tracing::info!(
                target: "macrdp::latency",
                stage = stage.as_str(),
                n,
                p50_us = p50,
                p95_us = p95,
                max_us = max,
                window_ms,
                "latency trace"
            );
            v.clear();
        }
    }
}

pub fn record_since(stage: &str, start: Instant) {
    if enabled() {
        record_us(stage, start.elapsed().as_micros() as u64);
    }
}

pub fn record_ns_delta(stage: &str, from_ns: u64, to_ns: u64) {
    if enabled() && from_ns != 0 {
        record_us(stage, to_ns.saturating_sub(from_ns) / 1000);
    }
}

fn summarize(v: &mut [u32]) -> (usize, u32, u32, u32) {
    v.sort_unstable();
    let n = v.len();
    let at = |q: usize| v[((n - 1) * q) / 100];
    (n, at(50), at(95), v[n - 1])
}

/// Age of a ScreenCaptureKit sample: host-clock now minus its presentation
/// timestamp, which SCK stamps on the host time clock.
#[cfg(target_os = "macos")]
pub fn host_clock_age_us(pts_value: i64, pts_timescale: i32) -> Option<u64> {
    #[repr(C)]
    struct CMTime {
        value: i64,
        timescale: i32,
        flags: u32,
        epoch: i64,
    }
    #[link(name = "CoreMedia", kind = "framework")]
    extern "C" {
        fn CMClockGetHostTimeClock() -> *const std::ffi::c_void;
        fn CMClockGetTime(clock: *const std::ffi::c_void) -> CMTime;
    }
    if pts_timescale <= 0 {
        return None;
    }
    let now = unsafe { CMClockGetTime(CMClockGetHostTimeClock()) };
    if now.timescale <= 0 {
        return None;
    }
    let now_s = now.value as f64 / f64::from(now.timescale);
    let pts_s = pts_value as f64 / f64::from(pts_timescale);
    let age = now_s - pts_s;
    (age >= 0.0).then_some((age * 1e6) as u64)
}

#[cfg(test)]
mod tests {
    use super::summarize;

    #[test]
    fn summarize_picks_nearest_rank() {
        let mut v: Vec<u32> = (1..=100).rev().collect();
        assert_eq!(summarize(&mut v), (100, 50, 95, 100));
        let mut one = vec![7];
        assert_eq!(summarize(&mut one), (1, 7, 7, 7));
    }
}
