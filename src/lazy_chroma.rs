//! Lazy chroma (`MACRDP_AVC444=lazy`): AVC444 v1 whose auxiliary view ships in small bands once the screen and
//! input are idle, each band a non-reference picture coded by `aux_coder`; see docs/cli.md.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ironrdp_egfx::pdu::{
    CapabilitiesV103Flags, CapabilitiesV104Flags, CapabilitiesV107Flags, CapabilitiesV10Flags,
    CapabilitySet,
};

use crate::aux_coder::{RefPicture, StreamParams};
use crate::damage::Rect;

/// Default idle time before the auxiliary view ships.
const DEFAULT_IDLE_MS: u64 = 250;

/// Default shortest periodic keyframe interval.
const DEFAULT_KEYFRAME_SECS: f32 = 30.0;

/// Default bytes a band stops at: 4.8 ms of a 20 Mbit/s link.
const DEFAULT_BAND_BYTES: usize = 12_000;

/// Default link rate the bands are paced for.
const DEFAULT_PACE_KBPS: u64 = 20_000;

/// Default QP of the band macroblocks: about 39 dB chroma on coloured text, against 29 dB for 4:2:0.
const DEFAULT_QP: i32 = 24;

/// Wait after the server refused a band.
const RETRY_AFTER: Duration = Duration::from_millis(100);

const MB: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LazyChromaConfig {
    /// How long the screen and input must stay idle before a band ships.
    pub idle: Duration,
    /// Shortest periodic keyframe interval, since each IDR makes the whole surface 4:2:0 again.
    pub keyframe_secs: f32,
    /// Encoded bytes a band stops at.
    pub band_bytes: usize,
    /// Link rate in kbit/s: a band waits until the previous one has had time to leave.
    pub pace_kbps: u64,
    pub qp: i32,
}

impl LazyChromaConfig {
    /// `MACRDP_AVC444=lazy`, tuned by `MACRDP_AVC444_IDLE_MS`, `_KEYFRAME_SECS`, `_BAND_BYTES`, `_PACE_KBPS` and `_QP`.
    pub fn from_env() -> Option<Self> {
        let var = |k: &str| std::env::var(k).ok();
        Self::parse(
            var("MACRDP_AVC444").as_deref(),
            var("MACRDP_AVC444_IDLE_MS").as_deref(),
            var("MACRDP_AVC444_KEYFRAME_SECS").as_deref(),
            var("MACRDP_AVC444_BAND_BYTES").as_deref(),
            var("MACRDP_AVC444_PACE_KBPS").as_deref(),
            var("MACRDP_AVC444_QP").as_deref(),
        )
    }

    fn parse(
        mode: Option<&str>,
        idle_ms: Option<&str>,
        keyframe_secs: Option<&str>,
        band_bytes: Option<&str>,
        pace_kbps: Option<&str>,
        qp: Option<&str>,
    ) -> Option<Self> {
        if !mode.is_some_and(|m| m.trim().eq_ignore_ascii_case("lazy")) {
            return None;
        }
        let num = |s: Option<&str>| s.and_then(|s| s.trim().parse::<u64>().ok());
        let keyframe_secs = keyframe_secs
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(DEFAULT_KEYFRAME_SECS);
        Some(Self {
            idle: Duration::from_millis(num(idle_ms).unwrap_or(DEFAULT_IDLE_MS)),
            keyframe_secs,
            band_bytes: num(band_bytes)
                .filter(|b| *b >= 1000)
                .map_or(DEFAULT_BAND_BYTES, |b| b as usize),
            pace_kbps: num(pace_kbps)
                .filter(|k| *k > 0)
                .unwrap_or(DEFAULT_PACE_KBPS),
            qp: num(qp)
                .filter(|q| (10..=51).contains(q))
                .map_or(DEFAULT_QP, |q| q as i32),
        })
    }
}

/// Whether the vendored server may send AVC444 under `cap` (V10 and later without `AVC_DISABLED`).
pub fn negotiated_avc444(cap: &CapabilitySet) -> bool {
    match cap {
        CapabilitySet::V8 { .. } | CapabilitySet::V8_1 { .. } => false,
        CapabilitySet::V10 { flags } | CapabilitySet::V10_2 { flags } => {
            !flags.contains(CapabilitiesV10Flags::AVC_DISABLED)
        }
        CapabilitySet::V10_1 => true,
        CapabilitySet::V10_3 { flags } => !flags.contains(CapabilitiesV103Flags::AVC_DISABLED),
        CapabilitySet::V10_4 { flags }
        | CapabilitySet::V10_5 { flags }
        | CapabilitySet::V10_6 { flags }
        | CapabilitySet::V10_6Err { flags } => !flags.contains(CapabilitiesV104Flags::AVC_DISABLED),
        CapabilitySet::V10_7 { flags } => !flags.contains(CapabilitiesV107Flags::AVC_DISABLED),
    }
}

static LAST_INPUT_MS: AtomicU64 = AtomicU64::new(0);

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Records a keyboard or mouse event: bands wait until input has been idle too.
pub fn note_input() {
    LAST_INPUT_MS.store(epoch().elapsed().as_millis() as u64 + 1, Ordering::Relaxed);
}

/// Whether input has been idle for `idle`.
pub fn input_idle(idle: Duration) -> bool {
    let at = LAST_INPUT_MS.load(Ordering::Relaxed);
    at == 0
        || epoch()
            .elapsed()
            .saturating_sub(Duration::from_millis(at - 1))
            >= idle
}

/// What the next band is coded against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandTarget {
    pub params: StreamParams,
    pub last: RefPicture,
}

/// One encoder's lazy chroma state, shared by its capture and ship threads.
#[derive(Debug)]
pub struct LazyChroma {
    width: usize,
    height: usize,
    cols: usize,
    /// Per macroblock: the client's chroma there is 4:2:0 only.
    stale: Vec<bool>,
    stale_count: usize,
    /// VideoToolbox's SPS and PPS as the band coder reads them; `None` until a keyframe, or when it cannot follow them.
    params: Option<StreamParams>,
    /// The last reference picture the client was sent.
    last: Option<RefPicture>,
    /// No band before this: the previous one is still leaving at the paced rate.
    next_band_at: Option<Instant>,
}

impl LazyChroma {
    pub fn new(width: u16, height: u16) -> Self {
        let (width, height) = (usize::from(width), usize::from(height));
        let (cols, rows) = (width.div_ceil(MB), height.div_ceil(MB));
        Self {
            width,
            height,
            cols,
            stale: vec![true; cols * rows],
            stale_count: cols * rows,
            params: None,
            last: None,
            next_band_at: None,
        }
    }

    pub fn stale_mbs(&self) -> usize {
        self.stale_count
    }

    /// The stream to code a band against, when a band may go now.
    pub fn band_due(&self, now: Instant) -> Option<BandTarget> {
        if self.stale_count == 0 || self.next_band_at.is_some_and(|t| now < t) {
            return None;
        }
        Some(BandTarget {
            params: self.params?,
            last: self.last?,
        })
    }

    /// Stale macroblocks in raster order: the band coder takes them from the front.
    pub fn candidates(&self) -> Vec<usize> {
        (0..self.stale.len()).filter(|&i| self.stale[i]).collect()
    }

    /// The macroblocks as region rects: runs per row, merged down where the columns match.
    pub fn rects_of(&self, mbs: &[usize]) -> Vec<Rect> {
        let mut rects: Vec<Rect> = Vec::new();
        let mut i = 0;
        while i < mbs.len() {
            let (row, start) = (mbs[i] / self.cols, mbs[i] % self.cols);
            let mut end = start + 1;
            while i + 1 < mbs.len() && mbs[i + 1] == mbs[i] + 1 && mbs[i + 1] / self.cols == row {
                i += 1;
                end += 1;
            }
            i += 1;
            let r = Rect {
                left: (start * MB) as u32,
                top: (row * MB) as u32,
                right: (end * MB).min(self.width) as u32,
                bottom: ((row + 1) * MB).min(self.height) as u32,
            };
            match rects
                .iter_mut()
                .rev()
                .find(|q| q.bottom == r.top && q.left == r.left && q.right == r.right)
            {
                Some(q) => q.bottom = r.bottom,
                None => rects.push(r),
            }
        }
        rects
    }

    /// A band of `mbs` went out in `bytes`: those macroblocks are refined.
    pub fn band_sent(&mut self, mbs: &[usize], bytes: usize, now: Instant, pace_kbps: u64) {
        for &i in mbs {
            if std::mem::replace(&mut self.stale[i], false) {
                self.stale_count -= 1;
            }
        }
        let leave = Duration::from_micros(bytes as u64 * 8 * 1000 / pace_kbps.max(1));
        self.next_band_at = Some(now + leave);
    }

    /// The server refused the band (backpressure): try again later.
    pub fn band_refused(&mut self, now: Instant) {
        self.next_band_at = Some(now + RETRY_AFTER);
    }

    fn mark_stale(&mut self, r: &Rect) {
        let (w, h) = (self.width as u32, self.height as u32);
        let (left, top) = (r.left.min(w) as usize, r.top.min(h) as usize);
        let (right, bottom) = (r.right.min(w) as usize, r.bottom.min(h) as usize);
        if left >= right || top >= bottom {
            return;
        }
        for row in top / MB..bottom.div_ceil(MB) {
            for col in left / MB..right.div_ceil(MB) {
                let i = row * self.cols + col;
                if !std::mem::replace(&mut self.stale[i], true) {
                    self.stale_count += 1;
                }
            }
        }
    }

    /// A motion frame shipped with `regions` (`None` = whole surface): the client shows 4:2:0 chroma there again.
    pub fn main_shipped(&mut self, regions: Option<&[Rect]>) {
        match regions {
            None => {
                self.stale.fill(true);
                self.stale_count = self.stale.len();
            }
            Some(r) => r.iter().for_each(|r| self.mark_stale(r)),
        }
    }

    /// Follows the frames the client was sent: parameter sets on keyframes, and each reference picture's numbering.
    pub fn stream_sent<'a>(
        &mut self,
        parameter_sets: &[Vec<u8>],
        nals: impl Iterator<Item = &'a [u8]>,
    ) {
        let kind = |k: u8| move |p: &&Vec<u8>| p.first().is_some_and(|b| b & 0x1f == k);
        let sps = parameter_sets.iter().find(kind(7));
        let pps = parameter_sets.iter().find(kind(8));
        if let (Some(sps), Some(pps)) = (sps, pps) {
            let params = StreamParams::parse(sps, pps);
            if params.is_none() && self.params.is_some() {
                tracing::warn!(
                    "EGFX lazy chroma: the H.264 stream is not Baseline CAVLC, no aux bands"
                );
            }
            self.params = params;
            self.last = None;
        }
        let Some(p) = self.params else {
            return;
        };
        if let Some(info) = nals.filter_map(|n| p.slice_info(n)).next() {
            if info.is_reference {
                self.last = Some(info.picture);
            }
        }
    }

    /// A frame the client never got: bands wait for the next keyframe, since the numbering no longer follows.
    pub fn stream_lost(&mut self) {
        self.last = None;
    }
}

#[cfg(all(test, target_os = "macos"))]
mod bench;

#[cfg(test)]
mod tests {
    use super::*;

    fn r(left: u32, top: u32, right: u32, bottom: u32) -> Rect {
        Rect {
            left,
            top,
            right,
            bottom,
        }
    }

    fn refined(w: u16, h: u16) -> LazyChroma {
        let mut l = LazyChroma::new(w, h);
        l.stale.fill(false);
        l.stale_count = 0;
        l
    }

    /// VideoToolbox's 1920x1080 Baseline SPS and PPS.
    const SPS: [u8; 16] = [
        0x27, 0x42, 0x00, 0x2a, 0xab, 0x40, 0x3c, 0x01, 0x13, 0xf2, 0xcd, 0xc0, 0x40, 0x40, 0x40,
        0x80,
    ];
    const PPS: [u8; 4] = [0x28, 0xce, 0x3c, 0x80];

    #[test]
    fn config_is_opt_in() {
        assert_eq!(
            LazyChromaConfig::parse(None, None, None, None, None, None),
            None
        );
        assert_eq!(
            LazyChromaConfig::parse(Some("off"), Some("10"), None, None, None, None),
            None
        );
        assert_eq!(
            LazyChromaConfig::parse(
                Some("lazy"),
                None,
                Some("-1"),
                Some("5"),
                Some("0"),
                Some("60")
            ),
            Some(LazyChromaConfig {
                idle: Duration::from_millis(250),
                keyframe_secs: 30.0,
                band_bytes: 12_000,
                pace_kbps: 20_000,
                qp: 24,
            })
        );
        assert_eq!(
            LazyChromaConfig::parse(
                Some(" LAZY "),
                Some("400"),
                Some("5"),
                Some("8000"),
                Some("50000"),
                Some("28")
            ),
            Some(LazyChromaConfig {
                idle: Duration::from_millis(400),
                keyframe_secs: 5.0,
                band_bytes: 8000,
                pace_kbps: 50_000,
                qp: 28,
            })
        );
    }

    #[test]
    fn stale_macroblocks_follow_the_region_rects() {
        let mut l = refined(1920, 1080);
        // One line of text, 624x18 at (96, 198), as the region rects grow it.
        l.main_shipped(Some(&[r(80, 176, 736, 240)]));
        assert_eq!(l.stale_mbs(), 41 * 4);
        let c = l.candidates();
        assert_eq!(c[0], 11 * 120 + 5);
        assert_eq!(l.rects_of(&c), vec![r(80, 176, 736, 240)]);
        l.main_shipped(None);
        assert_eq!(l.stale_mbs(), 120 * 68);
        assert_eq!(l.rects_of(&l.candidates()), vec![r(0, 0, 1920, 1080)]);
    }

    #[test]
    fn rects_of_merges_runs_down_and_keeps_ragged_ones_apart() {
        let l = LazyChroma::new(64, 48);
        // Row 0 columns 1-2, row 1 columns 1-2, row 2 column 0 only.
        assert_eq!(
            l.rects_of(&[1, 2, 5, 6, 8]),
            vec![r(16, 0, 48, 32), r(0, 32, 16, 48)]
        );
    }

    #[test]
    fn bands_follow_the_stream_and_the_pace() {
        let mut l = LazyChroma::new(64, 48);
        let now = Instant::now();
        assert!(l.band_due(now).is_none(), "no parameter sets yet");
        // A P slice header: first_mb 0, slice_type 0, pps 0, frame_num 3, poc lsb 5.
        let slice = [0x21, 0b1110_0011, 0b0001_0100, 0x80];
        l.stream_sent(&[SPS.to_vec(), PPS.to_vec()], [&slice[..]].into_iter());
        let target = l.band_due(now).expect("Baseline CAVLC");
        assert_eq!(
            target.last,
            RefPicture {
                frame_num: 3,
                poc_lsb: 5
            }
        );
        assert_eq!(target.params.width_mbs, 120);
        assert_eq!(target.params.height_mbs, 68);

        let c = l.candidates();
        l.band_sent(&c[..8], 25_000, now, 20_000);
        assert_eq!(l.stale_mbs(), 4);
        assert!(l.band_due(now).is_none(), "25 kB takes 10 ms at 20 Mbit/s");
        assert!(l.band_due(now + Duration::from_millis(10)).is_some());
        l.band_refused(now);
        assert!(l.band_due(now + Duration::from_millis(50)).is_none());
        l.stream_lost();
        assert!(
            l.band_due(now + Duration::from_secs(1)).is_none(),
            "numbering lost until a keyframe"
        );
    }

    #[test]
    fn a_cabac_stream_gets_no_bands() {
        let mut l = LazyChroma::new(64, 48);
        // The same PPS with entropy_coding_mode_flag set.
        l.stream_sent(
            &[SPS.to_vec(), vec![0x28, 0xee, 0x3c, 0x80]],
            std::iter::empty(),
        );
        assert!(l.band_due(Instant::now()).is_none());
    }

    #[test]
    fn negotiated_caps() {
        assert!(!negotiated_avc444(&CapabilitySet::V8_1 {
            flags: ironrdp_egfx::pdu::CapabilitiesV81Flags::AVC420_ENABLED
        }));
        assert!(negotiated_avc444(&CapabilitySet::V10_7 {
            flags: CapabilitiesV107Flags::empty()
        }));
        assert!(!negotiated_avc444(&CapabilitySet::V10_7 {
            flags: CapabilitiesV107Flags::AVC_DISABLED
        }));
    }
}
