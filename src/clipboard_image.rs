//! Clipboard image codecs: the Windows bitmap formats (`CF_DIB`,
//! `CF_DIBV5`) and the registered `PNG` format, mapped to and from the PNG
//! bytes the macOS pasteboard carries.
//!
//! Pure and platform-independent, like `clipboard_rich.rs`, so the header
//! arithmetic is unit-testable on Linux CI. `clipboard.rs` owns the protocol
//! flow; this module only converts bytes.
//!
//! What each format is for:
//!
//! - `PNG` (registered by name, as Office, browsers and FreeRDP do): lossless
//!   with alpha. Preferred both ways, and passed through untouched when the
//!   source already has PNG.
//! - `CF_DIBV5`: a `BITMAPV5HEADER` with explicit `BI_BITFIELDS` masks,
//!   including alpha. The bitmap format Windows apps read transparency from.
//! - `CF_DIB`: a plain `BITMAPINFOHEADER`, kept for clients and apps that know
//!   nothing newer. Its fourth byte is officially reserved, so alpha there is
//!   at best a convention.

use std::io::Cursor;

use anyhow::{anyhow, bail};
use image::{ImageEncoder, ImageFormat, ImageReader, RgbaImage};

/// Registered clipboard format name Windows apps use for PNG.
pub const PNG_FORMAT_NAME: &str = "PNG";

/// Largest image accepted from the remote, in pixels (8192 x 8192). A PNG
/// is compressed, so the incoming payload cap alone doesn't bound what a
/// Mac app would allocate when it decodes one.
const MAX_PIXELS: u64 = 1 << 26;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

const BI_RGB: u32 = 0;
const BI_BITFIELDS: u32 = 3;
const BI_ALPHABITFIELDS: u32 = 6;
const BITMAPINFOHEADER_SIZE: usize = 40;
const BITMAPV5HEADER_SIZE: usize = 124;
/// `LCS_sRGB`: the `'sRGB'` FourCC, stored little-endian.
const LCS_SRGB: u32 = 0x7352_4742;
const LCS_GM_IMAGES: u32 = 4;

/// Channel masks for a 32bpp BGRA pixel read as a little-endian `u32`.
const MASK_R: u32 = 0x00FF_0000;
const MASK_G: u32 = 0x0000_FF00;
const MASK_B: u32 = 0x0000_00FF;
const MASK_A: u32 = 0xFF00_0000;

fn check_dimensions(width: u32, height: u32, max_pixels: u64) -> anyhow::Result<()> {
    if width == 0 || height == 0 {
        bail!("empty image {width}x{height}");
    }
    if u64::from(width) * u64::from(height) > max_pixels {
        bail!("image {width}x{height} exceeds the {max_pixels}-pixel cap");
    }
    Ok(())
}

/// Decode PNG bytes (from the Mac pasteboard) into RGBA pixels.
pub fn decode_png(png: &[u8]) -> anyhow::Result<RgbaImage> {
    Ok(ImageReader::with_format(Cursor::new(png), ImageFormat::Png)
        .decode()?
        .to_rgba8())
}

pub fn encode_png(img: &RgbaImage) -> anyhow::Result<Vec<u8>> {
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png).write_image(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(png)
}

/// Check a remote `PNG` payload before it is put on the Mac pasteboard as
/// is: the signature, and dimensions (read from the header, without decoding
/// the pixels) under the cap.
pub fn validate_png(png: &[u8]) -> anyhow::Result<()> {
    validate_png_capped(png, MAX_PIXELS)
}

fn validate_png_capped(png: &[u8], max_pixels: u64) -> anyhow::Result<()> {
    if !png.starts_with(PNG_SIGNATURE) {
        bail!("not a PNG (bad signature)");
    }
    let (w, h) = ImageReader::with_format(Cursor::new(png), ImageFormat::Png).into_dimensions()?;
    check_dimensions(w, h, max_pixels)
}

/// Encode a `CF_DIB` payload: a `BITMAPINFOHEADER` followed by 32bpp BGRA
/// pixels, top-down (negative `biHeight`), `BI_RGB`. Unchanged from the
/// layout macrdp has always sent; transparency travels in `PNG` and
/// `CF_DIBV5` instead.
pub fn encode_dib(img: &RgbaImage) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let pixel_bytes = img.as_raw().len();
    let mut out = Vec::with_capacity(BITMAPINFOHEADER_SIZE + pixel_bytes);
    out.extend_from_slice(&(BITMAPINFOHEADER_SIZE as u32).to_le_bytes()); // biSize
    out.extend_from_slice(&(w as i32).to_le_bytes()); // biWidth
    out.extend_from_slice(&(-(h as i32)).to_le_bytes()); // biHeight (negative = top-down)
    out.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
    out.extend_from_slice(&32u16.to_le_bytes()); // biBitCount
    out.extend_from_slice(&BI_RGB.to_le_bytes()); // biCompression
    out.extend_from_slice(&(pixel_bytes as u32).to_le_bytes()); // biSizeImage
    out.extend_from_slice(&[0; 16]); // biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant
    for px in img.pixels() {
        let [r, g, b, a] = px.0;
        out.extend_from_slice(&[b, g, r, a]);
    }
    out
}

/// Encode a `CF_DIBV5` payload: a `BITMAPV5HEADER` declaring 32bpp
/// `BI_BITFIELDS` with an explicit alpha mask and sRGB, followed by straight
/// (not premultiplied) BGRA pixels. Rows are bottom-up, the orientation
/// Windows itself produces when it synthesizes `CF_DIBV5`, since some
/// readers mishandle top-down V5 bitmaps.
pub fn encode_dibv5(img: &RgbaImage) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let row_bytes = w as usize * 4;
    let pixel_bytes = img.as_raw().len();
    let mut out = Vec::with_capacity(BITMAPV5HEADER_SIZE + pixel_bytes);
    out.extend_from_slice(&(BITMAPV5HEADER_SIZE as u32).to_le_bytes()); // bV5Size
    out.extend_from_slice(&(w as i32).to_le_bytes()); // bV5Width
    out.extend_from_slice(&(h as i32).to_le_bytes()); // bV5Height (positive = bottom-up)
    out.extend_from_slice(&1u16.to_le_bytes()); // bV5Planes
    out.extend_from_slice(&32u16.to_le_bytes()); // bV5BitCount
    out.extend_from_slice(&BI_BITFIELDS.to_le_bytes()); // bV5Compression
    out.extend_from_slice(&(pixel_bytes as u32).to_le_bytes()); // bV5SizeImage
    out.extend_from_slice(&[0; 16]); // bV5XPelsPerMeter, bV5YPelsPerMeter, bV5ClrUsed, bV5ClrImportant
    for mask in [MASK_R, MASK_G, MASK_B, MASK_A] {
        out.extend_from_slice(&mask.to_le_bytes());
    }
    out.extend_from_slice(&LCS_SRGB.to_le_bytes()); // bV5CSType
    out.extend_from_slice(&[0; 36 + 12]); // bV5Endpoints, bV5GammaRed/Green/Blue
    out.extend_from_slice(&LCS_GM_IMAGES.to_le_bytes()); // bV5Intent
    out.extend_from_slice(&[0; 12]); // bV5ProfileData, bV5ProfileSize, bV5Reserved
    debug_assert_eq!(out.len(), BITMAPV5HEADER_SIZE);
    for row in img.as_raw().chunks_exact(row_bytes).rev() {
        for px in row.chunks_exact(4) {
            out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
        }
    }
    out
}

fn u32_at(buf: &[u8], at: usize) -> anyhow::Result<u32> {
    buf.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or_else(|| anyhow!("DIB truncated at byte {at}"))
}

/// Scale the bits `mask` selects from `px` to 0..=255. Masks are assumed
/// contiguous, as every real producer writes them.
fn channel(px: u32, mask: u32) -> u8 {
    if mask == 0 {
        return 0;
    }
    let max = u64::from(mask >> mask.trailing_zeros());
    let v = u64::from((px & mask) >> mask.trailing_zeros());
    ((v * 255 + max / 2) / max) as u8
}

/// Decode a `CF_DIB` or `CF_DIBV5` payload into RGBA pixels.
///
/// Accepts any header from `BITMAPINFOHEADER` (40 bytes) up to
/// `BITMAPV5HEADER` (124), 16, 24 or 32bpp, `BI_RGB`, `BI_BITFIELDS` or
/// `BI_ALPHABITFIELDS`, top-down or bottom-up. Bitfield masks are honoured
/// wherever the header version keeps them: inside the header from V2 (52
/// bytes) on, otherwise in the 12 (16 with alpha) bytes after it. A 32bpp
/// image whose alpha is zero everywhere is treated as opaque: Windows
/// producers commonly leave the reserved byte unset, which would otherwise
/// paste as a fully transparent image.
pub fn decode_dib(dib: &[u8]) -> anyhow::Result<RgbaImage> {
    decode_dib_capped(dib, MAX_PIXELS)
}

fn decode_dib_capped(dib: &[u8], max_pixels: u64) -> anyhow::Result<RgbaImage> {
    if dib.len() < BITMAPINFOHEADER_SIZE {
        bail!("DIB shorter than BITMAPINFOHEADER");
    }
    let bi_size = u32_at(dib, 0)? as usize;
    if bi_size < BITMAPINFOHEADER_SIZE || bi_size > dib.len() {
        bail!("bogus biSize {bi_size}");
    }
    let width = u32_at(dib, 4)? as i32;
    let height_signed = u32_at(dib, 8)? as i32;
    let bit_count = u16::from_le_bytes(dib[14..16].try_into().unwrap());
    let compression = u32_at(dib, 16)?;
    let clr_used = u32_at(dib, 32)? as usize;

    if width <= 0 {
        bail!("invalid width {width}");
    }
    if height_signed == 0 {
        bail!("invalid height 0");
    }
    let bitfields = matches!(compression, BI_BITFIELDS | BI_ALPHABITFIELDS);
    if compression != BI_RGB && !bitfields {
        bail!("unsupported BI_COMPRESSION {compression}");
    }
    if !matches!(bit_count, 16 | 24 | 32) {
        bail!("unsupported biBitCount {bit_count}");
    }
    if bitfields && bit_count == 24 {
        bail!("BI_BITFIELDS with biBitCount=24 not supported");
    }

    let (masks, mask_bytes) = if bitfields {
        if bi_size >= 52 {
            // BITMAPV2INFOHEADER and later: RGB masks inside the header,
            // and from V3 (56 bytes) the alpha mask too.
            let alpha = if bi_size >= 56 { u32_at(dib, 52)? } else { 0 };
            (
                [u32_at(dib, 40)?, u32_at(dib, 44)?, u32_at(dib, 48)?, alpha],
                0,
            )
        } else {
            let n = if compression == BI_ALPHABITFIELDS {
                4
            } else {
                3
            };
            let mut m = [0u32; 4];
            for (i, slot) in m.iter_mut().take(n).enumerate() {
                *slot = u32_at(dib, bi_size + i * 4)?;
            }
            (m, n * 4)
        }
    } else if bit_count == 16 {
        ([0x7C00, 0x03E0, 0x001F, 0], 0) // 5-5-5, top bit unused
    } else {
        // 24bpp ignores the alpha mask; 32bpp BI_RGB keeps the reserved
        // byte as alpha, subject to the all-zero check below.
        ([MASK_R, MASK_G, MASK_B, MASK_A], 0)
    };
    if masks[..3].iter().all(|&m| m == 0) {
        bail!("BI_BITFIELDS with empty colour masks");
    }

    let w = width as u32;
    let h = height_signed.unsigned_abs();
    check_dimensions(w, h, max_pixels)?;
    let top_down = height_signed < 0;
    let bpp = usize::from(bit_count / 8);
    // BMP rows are padded to a 4-byte multiple.
    let stride = (w as usize * bpp + 3) & !3;
    // An optional colour table may sit between the masks and the pixels
    // even at these depths (biClrUsed entries of 4 bytes).
    let pixel_start = clr_used
        .checked_mul(4)
        .and_then(|t| t.checked_add(bi_size + mask_bytes))
        .ok_or_else(|| anyhow!("overflow"))?;
    let need = stride
        .checked_mul(h as usize)
        .and_then(|n| n.checked_add(pixel_start))
        .ok_or_else(|| anyhow!("overflow"))?;
    if dib.len() < need {
        bail!("DIB payload truncated: have {}, need {need}", dib.len());
    }

    let mut rgba = Vec::with_capacity(w as usize * h as usize * 4);
    for row in 0..h {
        let src_row = if top_down { row } else { h - 1 - row };
        let row_off = pixel_start + src_row as usize * stride;
        for chunk in dib[row_off..row_off + w as usize * bpp].chunks_exact(bpp) {
            let px = match bpp {
                2 => u32::from(u16::from_le_bytes([chunk[0], chunk[1]])),
                3 => u32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0]),
                _ => u32::from_le_bytes(chunk.try_into().unwrap()),
            };
            let a = if masks[3] == 0 || bpp == 3 {
                0xFF
            } else {
                channel(px, masks[3])
            };
            rgba.extend_from_slice(&[
                channel(px, masks[0]),
                channel(px, masks[1]),
                channel(px, masks[2]),
                a,
            ]);
        }
    }
    if rgba.chunks_exact(4).all(|px| px[3] == 0) {
        for px in rgba.chunks_exact_mut(4) {
            px[3] = 0xFF;
        }
    }
    RgbaImage::from_raw(w, h, rgba).ok_or_else(|| anyhow!("RGBA buffer size mismatch"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3x2, every pixel distinct, alpha spanning opaque to fully transparent.
    /// Width 3 makes 24bpp rows need padding.
    fn sample() -> RgbaImage {
        RgbaImage::from_raw(
            3,
            2,
            vec![
                255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, // row 0
                10, 20, 30, 40, 50, 60, 70, 255, 200, 100, 50, 1, // row 1
            ],
        )
        .unwrap()
    }

    fn info_header(bi_size: u32, w: i32, h: i32, bits: u16, compression: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&bi_size.to_le_bytes());
        out.extend_from_slice(&w.to_le_bytes());
        out.extend_from_slice(&h.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(&compression.to_le_bytes());
        out.extend_from_slice(&[0; 20]);
        out.resize(bi_size as usize, 0);
        out
    }

    #[test]
    fn png_round_trip_preserves_alpha() {
        let img = sample();
        let png = encode_png(&img).unwrap();
        validate_png(&png).unwrap();
        assert_eq!(decode_png(&png).unwrap(), img);
    }

    #[test]
    fn png_validation_rejects_garbage_and_oversized_images() {
        assert!(validate_png(b"").is_err());
        assert!(validate_png(&encode_dib(&sample())).is_err());
        let png = encode_png(&sample()).unwrap();
        assert!(validate_png(&png[..PNG_SIGNATURE.len()]).is_err());
        assert!(validate_png_capped(&png, 6).is_ok());
        assert!(validate_png_capped(&png, 5).is_err());
    }

    #[test]
    fn dibv5_round_trip_preserves_alpha() {
        let img = sample();
        assert_eq!(decode_dib(&encode_dibv5(&img)).unwrap(), img);
    }

    #[test]
    fn dibv5_header_declares_bitfields_alpha_and_srgb_bottom_up() {
        let v5 = encode_dibv5(&sample());
        assert_eq!(u32_at(&v5, 0).unwrap(), 124);
        assert_eq!(
            u32_at(&v5, 8).unwrap() as i32,
            2,
            "positive height = bottom-up"
        );
        assert_eq!(u16::from_le_bytes([v5[14], v5[15]]), 32);
        assert_eq!(u32_at(&v5, 16).unwrap(), BI_BITFIELDS);
        assert_eq!(u32_at(&v5, 20).unwrap(), 3 * 2 * 4);
        assert_eq!(
            [40, 44, 48, 52].map(|at| u32_at(&v5, at).unwrap()),
            [0x00FF_0000, 0x0000_FF00, 0x0000_00FF, 0xFF00_0000]
        );
        assert_eq!(&v5[56..60], b"BGRs", "LCS_sRGB, little-endian");
        assert_eq!(u32_at(&v5, 108).unwrap(), LCS_GM_IMAGES);
        assert_eq!(v5.len(), 124 + 3 * 2 * 4);
        // First stored row is the image's bottom row, as straight BGRA.
        assert_eq!(&v5[124..128], &[30, 20, 10, 40]);
    }

    #[test]
    fn dib_round_trip_keeps_the_legacy_top_down_layout() {
        let img = sample();
        let dib = encode_dib(&img);
        assert_eq!(u32_at(&dib, 0).unwrap(), 40);
        assert_eq!(
            u32_at(&dib, 8).unwrap() as i32,
            -2,
            "negative height = top-down"
        );
        assert_eq!(u32_at(&dib, 16).unwrap(), BI_RGB);
        assert_eq!(
            &dib[40..44],
            &[0, 0, 255, 255],
            "top-left pixel first, BGRA"
        );
        assert_eq!(decode_dib(&dib).unwrap(), img);
    }

    #[test]
    fn bottom_up_and_top_down_24bpp_decode_alike_with_row_padding() {
        let img = sample();
        let row = |y: u32| -> Vec<u8> {
            let mut r: Vec<u8> = (0..3)
                .flat_map(|x| {
                    let [red, g, b, _] = img.get_pixel(x, y).0;
                    [b, g, red]
                })
                .collect();
            r.extend_from_slice(&[0xEE; 3]); // 9 bytes padded to a 12-byte stride
            r
        };
        let mut top_down = info_header(40, 3, -2, 24, BI_RGB);
        top_down.extend(row(0));
        top_down.extend(row(1));
        let mut bottom_up = info_header(40, 3, 2, 24, BI_RGB);
        bottom_up.extend(row(1));
        bottom_up.extend(row(0));

        let a = decode_dib(&top_down).unwrap();
        assert_eq!(a, decode_dib(&bottom_up).unwrap());
        assert_eq!(a.get_pixel(1, 0).0, [0, 255, 0, 255], "24bpp is opaque");
        assert_eq!(a.get_pixel(0, 1).0, [10, 20, 30, 255]);
    }

    #[test]
    fn bitfield_masks_after_a_plain_info_header_are_honoured() {
        // RGBX byte order: masks the reverse of the usual BGRA.
        let mut dib = info_header(40, 1, 1, 32, BI_BITFIELDS);
        for m in [0x0000_00FFu32, 0x0000_FF00, 0x00FF_0000] {
            dib.extend_from_slice(&m.to_le_bytes());
        }
        dib.extend_from_slice(&[11, 22, 33, 0]);
        let img = decode_dib(&dib).unwrap();
        assert_eq!(
            img.get_pixel(0, 0).0,
            [11, 22, 33, 255],
            "no alpha mask = opaque"
        );

        let mut dib = info_header(40, 1, 1, 32, BI_ALPHABITFIELDS);
        for m in [0x0000_00FFu32, 0x0000_FF00, 0x00FF_0000, 0xFF00_0000] {
            dib.extend_from_slice(&m.to_le_bytes());
        }
        dib.extend_from_slice(&[11, 22, 33, 44]);
        assert_eq!(
            decode_dib(&dib).unwrap().get_pixel(0, 0).0,
            [11, 22, 33, 44]
        );
    }

    #[test]
    fn bitfield_masks_inside_a_v5_header_are_honoured() {
        // Alpha in the low byte, then B, G, R: ABGR read as a u32.
        let mut dib = info_header(124, 1, 1, 32, BI_BITFIELDS);
        for (i, m) in [0xFF00_0000u32, 0x00FF_0000, 0x0000_FF00, 0x0000_00FF]
            .iter()
            .enumerate()
        {
            dib[40 + i * 4..44 + i * 4].copy_from_slice(&m.to_le_bytes());
        }
        dib.extend_from_slice(&[77, 33, 22, 11]);
        assert_eq!(
            decode_dib(&dib).unwrap().get_pixel(0, 0).0,
            [11, 22, 33, 77]
        );
    }

    #[test]
    fn sixteen_bit_565_bitfields_scale_to_full_range() {
        let mut dib = info_header(40, 2, 1, 16, BI_BITFIELDS);
        for m in [0xF800u32, 0x07E0, 0x001F] {
            dib.extend_from_slice(&m.to_le_bytes());
        }
        dib.extend_from_slice(&0xFFFFu16.to_le_bytes());
        dib.extend_from_slice(&0xF800u16.to_le_bytes());
        let img = decode_dib(&dib).unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255, 255]);
        assert_eq!(img.get_pixel(1, 0).0, [255, 0, 0, 255]);
    }

    #[test]
    fn all_zero_alpha_is_opaque_but_partial_alpha_is_kept() {
        let mut dib = info_header(40, 2, 1, 32, BI_RGB);
        dib.extend_from_slice(&[1, 2, 3, 0, 4, 5, 6, 0]);
        let img = decode_dib(&dib).unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [3, 2, 1, 255]);
        assert_eq!(img.get_pixel(1, 0).0, [6, 5, 4, 255]);

        let mut dib = info_header(40, 2, 1, 32, BI_RGB);
        dib.extend_from_slice(&[1, 2, 3, 0, 4, 5, 6, 9]);
        let img = decode_dib(&dib).unwrap();
        assert_eq!(img.get_pixel(0, 0).0[3], 0);
        assert_eq!(img.get_pixel(1, 0).0[3], 9);
    }

    #[test]
    fn colour_table_entries_are_skipped() {
        let mut dib = info_header(40, 1, 1, 24, BI_RGB);
        dib[32..36].copy_from_slice(&2u32.to_le_bytes()); // biClrUsed
        dib.extend_from_slice(&[0xAA; 8]);
        dib.extend_from_slice(&[3, 2, 1, 0]);
        assert_eq!(decode_dib(&dib).unwrap().get_pixel(0, 0).0, [1, 2, 3, 255]);
    }

    #[test]
    fn malformed_dibs_are_rejected() {
        assert!(decode_dib(&[0; 39]).is_err());
        let dib = encode_dibv5(&sample());
        assert!(
            decode_dib(&dib[..dib.len() - 1]).is_err(),
            "truncated pixels"
        );
        assert!(decode_dib(&dib[..60]).is_err(), "biSize past the end");
        assert!(decode_dib(&info_header(40, 0, 1, 32, BI_RGB)).is_err());
        assert!(decode_dib(&info_header(40, 1, 0, 32, BI_RGB)).is_err());
        assert!(decode_dib(&info_header(40, 1, 1, 8, BI_RGB)).is_err());
        assert!(
            decode_dib(&info_header(40, 1, 1, 32, 1)).is_err(),
            "BI_RLE8"
        );
        let mut empty_masks = info_header(40, 1, 1, 32, BI_BITFIELDS);
        empty_masks.extend_from_slice(&[0; 16]);
        assert!(decode_dib(&empty_masks).is_err());
        let mut huge = info_header(40, i32::MAX, -i32::MAX, 32, BI_RGB);
        huge.extend_from_slice(&[0; 4]);
        assert!(decode_dib(&huge).is_err(), "over the pixel cap");
        assert!(decode_dib_capped(&dib, 5).is_err());
        assert!(decode_dib_capped(&dib, 6).is_ok());
    }
}
