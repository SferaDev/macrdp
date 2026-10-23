//! Lazy chroma (`MACRDP_AVC444=lazy`): AVC444 v1 whose auxiliary view ships once the screen is idle; see docs/cli.md.

use std::time::Duration;

use ironrdp_egfx::pdu::{
    CapabilitiesV103Flags, CapabilitiesV104Flags, CapabilitiesV107Flags, CapabilitiesV10Flags,
    CapabilitySet,
};

use crate::damage::{region_rects_with_margin, Damage, Rect};

/// Default idle time before the auxiliary view ships.
const DEFAULT_IDLE_MS: u64 = 250;

/// Default shortest periodic keyframe interval.
const DEFAULT_KEYFRAME_SECS: f32 = 30.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LazyChromaConfig {
    /// How long the screen must stay unchanged before the auxiliary view ships.
    pub idle: Duration,
    /// Shortest periodic keyframe interval, since each IDR ships the whole surface.
    pub keyframe_secs: f32,
}

impl LazyChromaConfig {
    /// `MACRDP_AVC444=lazy`, tuned by `MACRDP_AVC444_IDLE_MS` and `MACRDP_AVC444_KEYFRAME_SECS`.
    pub fn from_env() -> Option<Self> {
        Self::parse(
            std::env::var("MACRDP_AVC444").ok().as_deref(),
            std::env::var("MACRDP_AVC444_IDLE_MS").ok().as_deref(),
            std::env::var("MACRDP_AVC444_KEYFRAME_SECS").ok().as_deref(),
        )
    }

    fn parse(
        mode: Option<&str>,
        idle_ms: Option<&str>,
        keyframe_secs: Option<&str>,
    ) -> Option<Self> {
        if !mode.is_some_and(|m| m.trim().eq_ignore_ascii_case("lazy")) {
            return None;
        }
        let ms = idle_ms
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_IDLE_MS);
        let keyframe_secs = keyframe_secs
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(DEFAULT_KEYFRAME_SECS);
        Some(Self {
            idle: Duration::from_millis(ms),
            keyframe_secs,
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

/// One encoder's lazy chroma state, shared by its capture and ship threads.
#[derive(Debug)]
pub struct LazyChroma {
    /// Every luma rect shipped since the last aux view: where the client's chroma is 4:2:0 only.
    stale: Damage,
    /// The auxiliary picture submitted and not shipped yet.
    aux_in_flight: Option<i64>,
    /// The next motion frame must not predict from the auxiliary picture.
    refresh_pending: bool,
    /// The auxiliary picture came out as an IDR, so the next frame must be one.
    keyframe_pending: bool,
    /// A long-term reference was acknowledged since the last IDR.
    ltr_acked: bool,
}

impl Default for LazyChroma {
    fn default() -> Self {
        Self {
            stale: Damage::Full,
            aux_in_flight: None,
            refresh_pending: false,
            keyframe_pending: false,
            ltr_acked: false,
        }
    }
}

impl LazyChroma {
    /// Whether an auxiliary view would improve what the client shows.
    pub fn aux_due(&self) -> bool {
        self.aux_in_flight.is_none() && !self.stale.is_empty()
    }

    /// Records the auxiliary picture submitted with `pts`.
    pub fn aux_submitted(&mut self, pts: i64) {
        self.aux_in_flight = Some(pts);
        self.refresh_pending = true;
    }

    /// `(force_keyframe, ltr_refresh)` for the next motion frame; without an acknowledged LTR, an IDR.
    pub fn main_options(&mut self, force_keyframe: bool) -> (bool, bool) {
        let refresh = std::mem::take(&mut self.refresh_pending);
        let keyframe = std::mem::take(&mut self.keyframe_pending)
            || force_keyframe
            || (refresh && !self.ltr_acked);
        (keyframe, refresh && !keyframe)
    }

    pub fn is_aux(&self, pts: i64) -> bool {
        self.aux_in_flight == Some(pts)
    }

    /// A motion frame shipped with `regions` (`None` = the whole surface).
    pub fn main_shipped(&mut self, pts: i64, regions: Option<&[Rect]>, is_keyframe: bool) {
        // VT dropped the auxiliary picture: its rects are still stale.
        if self.aux_in_flight.is_some_and(|aux| aux < pts) {
            self.aux_in_flight = None;
        }
        match regions {
            None => self.stale = Damage::Full,
            Some(r) => self.stale.add_rects(r.iter().copied()),
        }
        if is_keyframe {
            self.ltr_acked = false;
        }
    }

    /// The motion frame's LTR token was handed to the encoder to acknowledge.
    pub fn ltr_acknowledged(&mut self) {
        self.ltr_acked = true;
    }

    /// The aux picture is shipping: the rects it refines (`None` = whole surface).
    pub fn aux_shipping(
        &mut self,
        is_keyframe: bool,
        width: u16,
        height: u16,
    ) -> Option<Vec<Rect>> {
        self.aux_in_flight = None;
        if is_keyframe {
            self.keyframe_pending = true;
            self.ltr_acked = false;
        }
        match std::mem::take(&mut self.stale) {
            Damage::Full => None,
            Damage::Rects(r) => region_rects_with_margin(&r, width, height, 0),
        }
    }
}

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

    #[test]
    fn config_is_opt_in() {
        assert_eq!(LazyChromaConfig::parse(None, None, None), None);
        assert_eq!(LazyChromaConfig::parse(Some("off"), Some("10"), None), None);
        assert_eq!(
            LazyChromaConfig::parse(Some("lazy"), None, Some("-1")),
            Some(LazyChromaConfig {
                idle: Duration::from_millis(250),
                keyframe_secs: 30.0
            })
        );
        assert_eq!(
            LazyChromaConfig::parse(Some(" LAZY "), Some("400"), Some("5")),
            Some(LazyChromaConfig {
                idle: Duration::from_millis(400),
                keyframe_secs: 5.0
            })
        );
    }

    #[test]
    fn aux_covers_every_luma_rect_since_the_last_one() {
        let mut l = LazyChroma::default();
        assert!(l.aux_due(), "nothing has been refined yet");
        l.aux_submitted(5);
        assert!(!l.aux_due());
        assert!(l.is_aux(5));
        assert_eq!(l.aux_shipping(false, 1920, 1080), None, "whole surface");
        assert!(!l.aux_due());

        l.main_shipped(6, Some(&[r(0, 0, 32, 32)]), false);
        l.main_shipped(7, Some(&[r(64, 64, 96, 80)]), false);
        assert!(l.aux_due());
        l.aux_submitted(8);
        assert_eq!(
            l.aux_shipping(false, 1920, 1080),
            Some(vec![r(0, 0, 32, 32), r(64, 64, 96, 80)])
        );
        l.main_shipped(9, None, false);
        l.aux_submitted(10);
        assert_eq!(l.aux_shipping(false, 1920, 1080), None);
    }

    #[test]
    fn a_dropped_aux_picture_leaves_its_rects_stale() {
        let mut l = LazyChroma::default();
        l.aux_submitted(3);
        l.main_shipped(4, Some(&[r(0, 0, 16, 16)]), false);
        assert!(l.aux_due());
        assert!(!l.is_aux(3));
    }

    #[test]
    fn the_frame_after_aux_refreshes_from_an_acknowledged_ltr() {
        let mut l = LazyChroma::default();
        l.main_shipped(0, None, true);
        l.aux_submitted(1);
        assert_eq!(
            l.main_options(false),
            (true, false),
            "no LTR acknowledged: IDR"
        );
        assert_eq!(l.main_options(false), (false, false));

        l.ltr_acknowledged();
        l.aux_submitted(3);
        assert_eq!(l.main_options(false), (false, true));
        assert_eq!(l.main_options(false), (false, false), "only the first one");
        assert_eq!(l.main_options(true), (true, false));
    }

    #[test]
    fn an_idr_aux_picture_makes_the_next_frame_an_idr() {
        let mut l = LazyChroma::default();
        l.ltr_acknowledged();
        l.aux_submitted(1);
        l.aux_shipping(true, 64, 64);
        assert_eq!(l.main_options(false), (true, false));
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
