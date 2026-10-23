//! EGFX small-change fast path (`MACRDP_EGFX_FAST_PATH`, off by default).
//!
//! A small change (a typed glyph, the caret) otherwise waits for the hardware
//! H.264 encoder, about 12 ms at 3440x1440. With the fast path on,
//! such a change first ships at once as lossless Planar (or Uncompressed)
//! WireToSurface1 tiles in an EGFX frame of its own; the H.264 frame of the
//! same capture follows as always, its region rects covering the same pixels,
//! so the surface ends exactly where the H.264-only path leaves it. Both
//! codecs are ones every EGFX client must decode (MS-RDPEGFX RDPGFX_CODECID).
//!
//! The caller only takes the fast path while its encoder has no frame in
//! flight: every H.264 frame the client receives after the tiles was then
//! captured at or after them, and never paints older content over them.

use ironrdp_egfx::pdu::Codec1Type;
use ironrdp_graphics::rdp6::{BgrAChannels, BitmapStreamEncoder};
use ironrdp_pdu::geometry::ExclusiveRectangle;

use crate::damage::{Damage, Rect, MAX_REGION_RECTS};

/// Changed pixels above which a frame waits for H.264 alone.
const DEFAULT_MAX_PX: u64 = 32 * 1024;

/// One WireToSurface1 tile: codec, destination on the surface, bitmap data.
pub type Tile = (Codec1Type, ExclusiveRectangle, Vec<u8>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// RLE planar (MS-RDPEGDI 2.2.2.5.1), raw planar when RLE doesn't pay.
    Planar,
    /// The surface's XRGB pixels as they are.
    Uncompressed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastPath {
    pub codec: Codec,
    pub max_px: u64,
}

impl FastPath {
    /// `MACRDP_EGFX_FAST_PATH` (`1`/`planar`, or `uncompressed`) and
    /// `MACRDP_EGFX_FAST_PATH_MAX_PX`.
    pub fn from_env() -> Option<Self> {
        Self::parse(
            std::env::var("MACRDP_EGFX_FAST_PATH").ok().as_deref(),
            std::env::var("MACRDP_EGFX_FAST_PATH_MAX_PX")
                .ok()
                .as_deref(),
        )
    }

    fn parse(mode: Option<&str>, max_px: Option<&str>) -> Option<Self> {
        let codec = match mode?.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "planar" => Codec::Planar,
            "uncompressed" | "raw" => Codec::Uncompressed,
            _ => return None,
        };
        let max_px = max_px
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MAX_PX);
        Some(Self { codec, max_px })
    }

    /// The rects to send ahead of H.264, clipped to the surface, or `None`
    /// when the change is unknown, empty, too large or in too many pieces.
    pub fn rects(&self, damage: &Damage, width: u16, height: u16) -> Option<Vec<Rect>> {
        let Damage::Rects(rects) = damage else {
            return None;
        };
        let (w, h) = (u32::from(width), u32::from(height));
        let rects: Vec<Rect> = rects
            .iter()
            .map(|r| Rect {
                left: r.left.min(w),
                top: r.top.min(h),
                right: r.right.min(w),
                bottom: r.bottom.min(h),
            })
            .filter(|r| r.left < r.right && r.top < r.bottom)
            .collect();
        if rects.is_empty() || rects.len() > MAX_REGION_RECTS {
            return None;
        }
        let px: u64 = rects
            .iter()
            .map(|r| u64::from(r.right - r.left) * u64::from(r.bottom - r.top))
            .sum();
        (px <= self.max_px).then_some(rects)
    }

    /// Encodes `rects` of a BGRA frame (`stride` bytes per row) as tiles, or
    /// `None` when a rect reaches past the buffer.
    pub fn tiles(&self, bgra: &[u8], stride: usize, rects: &[Rect]) -> Option<Vec<Tile>> {
        rects
            .iter()
            .map(|r| {
                let (left, right) = (r.left as usize, r.right as usize);
                let (top, bottom) = (r.top as usize, r.bottom as usize);
                let end = (bottom - 1)
                    .checked_mul(stride)?
                    .checked_add(right.checked_mul(4)?)?;
                if right * 4 > stride || end > bgra.len() {
                    return None;
                }
                let rows =
                    (top..bottom).map(|y| &bgra[y * stride + left * 4..y * stride + right * 4]);
                let destination = ExclusiveRectangle {
                    left: u16::try_from(r.left).ok()?,
                    top: u16::try_from(r.top).ok()?,
                    right: u16::try_from(r.right).ok()?,
                    bottom: u16::try_from(r.bottom).ok()?,
                };
                let (codec, data) = match self.codec {
                    Codec::Planar => (
                        Codec1Type::Planar,
                        planar(rows, right - left, bottom - top)?,
                    ),
                    Codec::Uncompressed => {
                        (Codec1Type::Uncompressed, rows.flatten().copied().collect())
                    }
                };
                Some((codec, destination, data))
            })
            .collect()
    }
}

/// A top-down planar bitmap stream of `width`x`height` BGRA rows, without
/// alpha: RLE when it is smaller, raw planes otherwise.
fn planar<'a>(
    rows: impl Iterator<Item = &'a [u8]> + Clone,
    width: usize,
    height: usize,
) -> Option<Vec<u8>> {
    let pixels = rows.flat_map(|row| row.as_chunks::<4>().0.iter().map(|p| p.as_slice()));
    // Header, three planes and the pad byte raw planes end with.
    let raw_len = 3 * width * height + 2;
    let mut out = vec![0u8; raw_len];
    let mut encoder = BitmapStreamEncoder::new(width, height);
    let len = match encoder.encode_pixels_stream::<_, BgrAChannels>(pixels.clone(), &mut out, true)
    {
        Ok(len) if len < raw_len => len,
        _ => encoder
            .encode_pixels_stream::<_, BgrAChannels>(pixels, &mut out, false)
            .ok()?,
    };
    out.truncate(len);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironrdp_graphics::rdp6::BitmapStreamDecoder;

    fn rect(left: u32, top: u32, right: u32, bottom: u32) -> Rect {
        Rect {
            left,
            top,
            right,
            bottom,
        }
    }

    fn planar_fp() -> FastPath {
        FastPath::parse(Some("1"), None).unwrap()
    }

    /// A `w`x`h` BGRA frame with `pad` spare bytes per row and a text-like
    /// pattern: mostly one colour with a few distinct pixels.
    fn frame(w: usize, h: usize, pad: usize) -> (Vec<u8>, usize) {
        let stride = w * 4 + pad;
        let mut bgra = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                let p = &mut bgra[y * stride + x * 4..y * stride + x * 4 + 4];
                let ink = (x * 7 + y * 3) % 11 == 0;
                p.copy_from_slice(&if ink {
                    [(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8, 255]
                } else {
                    [0xF0, 0xF0, 0xF0, 255]
                });
            }
        }
        (bgra, stride)
    }

    fn rgb24_of(bgra: &[u8], stride: usize, r: Rect) -> Vec<u8> {
        let mut out = Vec::new();
        for y in r.top as usize..r.bottom as usize {
            for x in r.left as usize..r.right as usize {
                let p = &bgra[y * stride + x * 4..];
                out.extend_from_slice(&[p[2], p[1], p[0]]);
            }
        }
        out
    }

    #[test]
    fn env_values_pick_the_codec() {
        assert_eq!(FastPath::parse(None, None), None);
        assert_eq!(FastPath::parse(Some("0"), None), None);
        assert_eq!(planar_fp().codec, Codec::Planar);
        assert_eq!(planar_fp().max_px, DEFAULT_MAX_PX);
        let raw = FastPath::parse(Some("uncompressed"), Some("4096")).unwrap();
        assert_eq!((raw.codec, raw.max_px), (Codec::Uncompressed, 4096));
        assert_eq!(
            FastPath::parse(Some("on"), Some("0")).unwrap().max_px,
            DEFAULT_MAX_PX
        );
    }

    #[test]
    fn only_small_known_changes_qualify() {
        let fp = planar_fp();
        assert_eq!(fp.rects(&Damage::Full, 1920, 1080), None);
        assert_eq!(fp.rects(&Damage::default(), 1920, 1080), None);
        let glyph = Damage::Rects(vec![rect(100, 200, 112, 220), rect(112, 200, 114, 220)]);
        assert_eq!(fp.rects(&glyph, 1920, 1080).unwrap().len(), 2);
        // 200x200 = 40000 px is over the 32768 default.
        assert_eq!(
            fp.rects(&Damage::Rects(vec![rect(0, 0, 200, 200)]), 1920, 1080),
            None
        );
        let nine = Damage::Rects((0..9).map(|i| rect(i * 20, 0, i * 20 + 4, 4)).collect());
        assert_eq!(fp.rects(&nine, 1920, 1080), None);
    }

    #[test]
    fn rects_are_clipped_to_the_surface() {
        let fp = planar_fp();
        let damage = Damage::Rects(vec![rect(1910, 1070, 1930, 1090), rect(2000, 0, 2010, 10)]);
        assert_eq!(
            fp.rects(&damage, 1920, 1080).unwrap(),
            vec![rect(1910, 1070, 1920, 1080)]
        );
    }

    #[test]
    fn planar_tiles_decode_to_the_captured_pixels() {
        let (bgra, stride) = frame(64, 48, 32);
        let rects = [rect(3, 5, 40, 30), rect(60, 40, 64, 48), rect(0, 0, 1, 1)];
        let tiles = planar_fp().tiles(&bgra, stride, &rects).unwrap();
        for ((codec, dest, data), r) in tiles.iter().zip(rects) {
            assert_eq!(*codec, Codec1Type::Planar);
            assert_eq!(
                (dest.left, dest.top, dest.right, dest.bottom),
                (r.left as u16, r.top as u16, r.right as u16, r.bottom as u16)
            );
            let (w, h) = ((r.right - r.left) as usize, (r.bottom - r.top) as usize);
            let mut rgb = Vec::new();
            BitmapStreamDecoder::default()
                .decode_bitmap_stream_to_rgb24(data, &mut rgb, w, h)
                .unwrap();
            assert_eq!(rgb, rgb24_of(&bgra, stride, r), "rect {r:?}");
        }
    }

    #[test]
    fn planar_falls_back_to_raw_planes_on_noise() {
        let (w, h) = (16usize, 16usize);
        let bgra: Vec<u8> = (0..w * h * 4)
            .map(|i| ((i * 2654435761usize) >> 7) as u8)
            .collect();
        let r = rect(0, 0, w as u32, h as u32);
        let tiles = planar_fp().tiles(&bgra, w * 4, &[r]).unwrap();
        let data = &tiles[0].2;
        assert!(data.len() <= 3 * w * h + 2);
        let mut rgb = Vec::new();
        BitmapStreamDecoder::default()
            .decode_bitmap_stream_to_rgb24(data, &mut rgb, w, h)
            .unwrap();
        assert_eq!(rgb, rgb24_of(&bgra, w * 4, r));
    }

    #[test]
    fn uncompressed_tiles_are_the_bgra_rows() {
        let (bgra, stride) = frame(32, 8, 8);
        let fp = FastPath::parse(Some("uncompressed"), None).unwrap();
        let tiles = fp.tiles(&bgra, stride, &[rect(2, 1, 5, 3)]).unwrap();
        let (codec, _, data) = &tiles[0];
        assert_eq!(*codec, Codec1Type::Uncompressed);
        let mut want = bgra[stride + 8..stride + 20].to_vec();
        want.extend_from_slice(&bgra[2 * stride + 8..2 * stride + 20]);
        assert_eq!(*data, want);
    }

    #[test]
    fn a_rect_past_the_buffer_is_refused() {
        let (bgra, stride) = frame(16, 16, 0);
        assert!(planar_fp()
            .tiles(&bgra, stride, &[rect(0, 10, 16, 17)])
            .is_none());
        assert!(planar_fp()
            .tiles(&bgra, stride, &[rect(10, 0, 17, 1)])
            .is_none());
    }
}
