//! Read the actual system-rendered cursor bitmap via undocumented SkyLight
//! symbols. ALL touches to private CoreGraphics/SkyLight symbols live in this
//! file — when Apple changes the API in a future macOS release (renames a
//! function, drops a parameter, returns a different type), update only this
//! file.
//!
//! Why this exists: `NSCursor.currentSystemCursor` returns the calling
//! *process's* NSCursor stack, not the cursor the WindowServer is actually
//! compositing on screen. macrdp's process never installs an I-beam or
//! crosshair, so NSCursor just keeps reporting the default arrow — even when
//! the user is typing in a Safari text field and the screen shows an I-beam,
//! or when `screencapture -i` has put a crosshair on screen. The
//! WindowServer's connection exposes the real composited cursor via the
//! private functions below.
//!
//! Symbols live in `/System/Library/PrivateFrameworks/SkyLight.framework`,
//! aren't exported in the framework's `.tbd` stub, and weren't kept under
//! the legacy `CGS*` names — they're `SLS*` on modern macOS (12+). We
//! resolve them at runtime via `dlsym(RTLD_DEFAULT, ...)` instead of
//! link-time: if Apple removes or renames any of them in a future macOS,
//! the lookup returns `None`, this module returns `None`, and `cursor/mod.rs`
//! falls back to the public NSCursor path without the binary failing to
//! launch.
//!
//! Reverse-engineered signatures cross-checked against:
//!   - setomits/PrivateHeadersForMacOS — `SLSCursor.h`
//!   - macmade/CursorView — `SkyLight.h`
//!   - sapuapu/MouseCape (CGS-era naming, same shape)
#![cfg(target_os = "macos")]

use core_graphics::geometry::{CGPoint, CGRect};
use libc::{dlsym, RTLD_DEFAULT};
use std::ffi::{c_int, c_void, CString};
use std::sync::OnceLock;

#[allow(non_camel_case_types)]
type CGSConnectionID = c_int;

type SLSMainConnectionIDFn = unsafe extern "C" fn() -> CGSConnectionID;
type SLSGetGlobalCursorDataSizeFn = unsafe extern "C" fn(CGSConnectionID, *mut c_int) -> c_int;
// Signature (9 out params after the connection):
//   data:             buffer to fill (caller-allocated, size from -DataSize)
//   data_size:        IN size of buffer / OUT bytes actually written
//   row_bytes:        OUT row stride
//   rect:             OUT cursor bounding rect (size = bitmap dimensions)
//   hot_spot:         OUT hotspot in pixels relative to the cursor rect
//   depth:            OUT bits per pixel (typically 32)
//   components:       OUT components per pixel (typically 4)
//   bits_per_component: OUT bits per component (typically 8)
// Pixel format in memory is little-endian BGRA premultiplied — i.e., the
// CG-native bitmap layout (kCGBitmapByteOrder32Little |
// kCGImageAlphaPremultipliedFirst). Byte 0 of each pixel is blue. Rows are
// top-down (row 0 is the arrow's tip), checked on macOS 27.
type SLSGetGlobalCursorDataFn = unsafe extern "C" fn(
    CGSConnectionID,
    *mut c_void,
    *mut c_int,
    *mut c_int,
    *mut CGRect,
    *mut CGPoint,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;

// The same shape for one of WindowServer's built-in cursors, selected by id
// (0 = arrow) after the connection: (cid, id, data, data_size, row_bytes,
// rect, hot_spot, depth, components, bits_per_component). Read off the
// disassembly: the id indexes SkyLight's `cursor_names` table and the rest is
// forwarded exactly as `SLSGetGlobalCursorData` forwards its arguments.
type SLSGetSystemDefinedCursorDataSizeFn =
    unsafe extern "C" fn(CGSConnectionID, c_int, *mut c_int) -> c_int;
type SLSGetSystemDefinedCursorDataFn = unsafe extern "C" fn(
    CGSConnectionID,
    c_int,
    *mut c_void,
    *mut c_int,
    *mut c_int,
    *mut CGRect,
    *mut CGPoint,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;

const SYSTEM_CURSOR_ARROW: c_int = 0;

/// Resolve a private SkyLight/CoreGraphics symbol by name. Returns `None`
/// if the symbol isn't present (the OS version removed it, sandbox is
/// hiding it, etc.).
fn resolve<T: Copy>(name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let sym = unsafe { dlsym(RTLD_DEFAULT, cname.as_ptr()) };
    if sym.is_null() {
        return None;
    }
    // SAFETY: caller picks the right function-pointer type for the
    // symbol they asked for. We're internal-only here.
    Some(unsafe { *(&sym as *const *mut c_void as *const T) })
}

fn main_connection_fn() -> Option<SLSMainConnectionIDFn> {
    static F: OnceLock<Option<SLSMainConnectionIDFn>> = OnceLock::new();
    *F.get_or_init(|| resolve("SLSMainConnectionID"))
}

fn cursor_data_size_fn() -> Option<SLSGetGlobalCursorDataSizeFn> {
    static F: OnceLock<Option<SLSGetGlobalCursorDataSizeFn>> = OnceLock::new();
    *F.get_or_init(|| resolve("SLSGetGlobalCursorDataSize"))
}

fn cursor_data_fn() -> Option<SLSGetGlobalCursorDataFn> {
    static F: OnceLock<Option<SLSGetGlobalCursorDataFn>> = OnceLock::new();
    *F.get_or_init(|| resolve("SLSGetGlobalCursorData"))
}

fn system_cursor_data_size_fn() -> Option<SLSGetSystemDefinedCursorDataSizeFn> {
    static F: OnceLock<Option<SLSGetSystemDefinedCursorDataSizeFn>> = OnceLock::new();
    *F.get_or_init(|| resolve("SLSGetSystemDefinedCursorDataSize"))
}

fn system_cursor_data_fn() -> Option<SLSGetSystemDefinedCursorDataFn> {
    static F: OnceLock<Option<SLSGetSystemDefinedCursorDataFn>> = OnceLock::new();
    *F.get_or_init(|| resolve("SLSGetSystemDefinedCursorData"))
}

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    // Deprecated but current: it reads WindowServer's shared scoreboard, so it
    // reports the on-screen cursor hidden or obscured by any app, not just us.
    fn CGCursorIsVisible() -> u32;
}

/// Whether WindowServer is drawing the cursor at all. False while an app has
/// hidden it (full-screen video) or obscured it until the mouse moves (typing).
pub fn cursor_visible() -> bool {
    unsafe { CGCursorIsVisible() != 0 }
}

/// A cursor exactly as SkyLight returned it: premultiplied BGRA, top-down,
/// `row_bytes` apart, with the rect and hotspot it reported.
pub struct RawCursor {
    data: Vec<u8>,
    row_bytes: usize,
    rect_w: f64,
    rect_h: f64,
    hot_x: f64,
    hot_y: f64,
}

impl RawCursor {
    /// Same pixels, size and hotspot. How the arrow is recognised: the global
    /// cursor is byte-identical to the system arrow while the arrow is up, and
    /// differs once the user resizes or recolours the pointer.
    pub fn same_as(&self, other: &RawCursor) -> bool {
        self.row_bytes == other.row_bytes
            && self.rect_w == other.rect_w
            && self.rect_h == other.rect_h
            && self.hot_x == other.hot_x
            && self.hot_y == other.hot_y
            && self.data == other.data
    }

    pub fn content_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.data.hash(&mut h);
        self.row_bytes.hash(&mut h);
        self.rect_w.to_bits().hash(&mut h);
        self.rect_h.to_bits().hash(&mut h);
        self.hot_x.to_bits().hash(&mut h);
        self.hot_y.to_bits().hash(&mut h);
        h.finish()
    }

    /// Straight-alpha RGBA, top row first, with the hotspot in the same pixels.
    /// Returns `(rgba, width, height, hot_x, hot_y)`.
    pub fn to_rgba(&self) -> Option<(Vec<u8>, u16, u16, u16, u16)> {
        let (w, h, scale) =
            pixel_geometry(self.rect_w, self.rect_h, self.row_bytes, self.data.len())?;
        let rgba = bgra_premultiplied_to_rgba(&self.data, self.row_bytes, w, h)?;
        let hot =
            |v: f64, max: usize| ((v * scale as f64).round().max(0.0) as usize).min(max - 1) as u16;
        Some((
            rgba,
            w as u16,
            h as u16,
            hot(self.hot_x, w),
            hot(self.hot_y, h),
        ))
    }
}

/// The cursor WindowServer is compositing right now.
pub fn global_cursor() -> Option<RawCursor> {
    let main_conn = main_connection_fn()?;
    let size_fn = cursor_data_size_fn()?;
    let data_fn = cursor_data_fn()?;
    unsafe {
        let cid = main_conn();
        let mut data_size: c_int = 0;
        if size_fn(cid, &mut data_size) != 0 {
            return None;
        }
        read_cursor(
            data_size,
            |buf, size, row_bytes, rect, hot, depth, comps, bpc| {
                data_fn(cid, buf, size, row_bytes, rect, hot, depth, comps, bpc)
            },
        )
    }
}

/// WindowServer's stock arrow, at the scale and size the user has set.
pub fn system_arrow() -> Option<RawCursor> {
    let main_conn = main_connection_fn()?;
    let size_fn = system_cursor_data_size_fn()?;
    let data_fn = system_cursor_data_fn()?;
    unsafe {
        let cid = main_conn();
        let mut data_size: c_int = 0;
        if size_fn(cid, SYSTEM_CURSOR_ARROW, &mut data_size) != 0 {
            return None;
        }
        read_cursor(
            data_size,
            |buf, size, row_bytes, rect, hot, depth, comps, bpc| {
                data_fn(
                    cid,
                    SYSTEM_CURSOR_ARROW,
                    buf,
                    size,
                    row_bytes,
                    rect,
                    hot,
                    depth,
                    comps,
                    bpc,
                )
            },
        )
    }
}

/// Allocate `data_size` bytes, run one of the cursor-data calls into it and
/// validate what came back.
unsafe fn read_cursor(
    data_size: c_int,
    call: impl FnOnce(
        *mut c_void,
        *mut c_int,
        *mut c_int,
        *mut CGRect,
        *mut CGPoint,
        *mut c_int,
        *mut c_int,
        *mut c_int,
    ) -> c_int,
) -> Option<RawCursor> {
    // Refuse absurd sizes: a 256 pt cursor at 2x is 1 MiB. Anything bigger is
    // a wire-format mismatch we'd rather skip than honour.
    if data_size <= 0 || data_size as usize > 1024 * 1024 {
        return None;
    }
    let mut buf = vec![0u8; data_size as usize];
    let mut size_inout: c_int = data_size;
    let mut row_bytes: c_int = 0;
    let mut rect = CGRect {
        origin: CGPoint { x: 0.0, y: 0.0 },
        size: core_graphics::geometry::CGSize {
            width: 0.0,
            height: 0.0,
        },
    };
    let mut hot = CGPoint { x: 0.0, y: 0.0 };
    let mut depth: c_int = 0;
    let mut components: c_int = 0;
    let mut bits_per_component: c_int = 0;
    let err = call(
        buf.as_mut_ptr() as *mut c_void,
        &mut size_inout,
        &mut row_bytes,
        &mut rect,
        &mut hot,
        &mut depth,
        &mut components,
        &mut bits_per_component,
    );
    if err != 0 {
        return None;
    }
    // Sanity: BGRA premultiplied is what we expect (depth=32, comps=4,
    // bpc=8). If macOS ever returns something different — say a 16bpc
    // HDR cursor — skip rather than emit garbled colors.
    if depth != 32 || components != 4 || bits_per_component != 8 {
        return None;
    }
    if size_inout <= 0 || size_inout as usize > buf.len() || row_bytes <= 0 {
        return None;
    }
    if !(1.0..=256.0).contains(&rect.size.width.round())
        || !(1.0..=256.0).contains(&rect.size.height.round())
    {
        return None;
    }
    buf.truncate(size_inout as usize);
    Some(RawCursor {
        data: buf,
        row_bytes: row_bytes as usize,
        rect_w: rect.size.width.round(),
        rect_h: rect.size.height.round(),
        hot_x: hot.x,
        hot_y: hot.y,
    })
}

/// Pixel size of a SkyLight cursor buffer and how many pixels it holds per
/// unit of the rect it reported. Normally the buffer matches the rect. A buffer
/// with a whole multiple of the rect's rows, and room for as many columns, is a
/// Retina rendition described by a point-sized rect: reading it with the rect's
/// size would take its top-left corner, and its hotspot has to scale with it.
/// Returns `(width, height, scale)`.
fn pixel_geometry(
    rect_w: f64,
    rect_h: f64,
    row_bytes: usize,
    data_len: usize,
) -> Option<(usize, usize, usize)> {
    let (w, h) = (rect_w as usize, rect_h as usize);
    if w == 0 || h == 0 || row_bytes == 0 {
        return None;
    }
    let rows = data_len / row_bytes;
    (1..=4)
        .find(|&s| rows == h * s && row_bytes >= w * s * 4)
        .map(|s| (w * s, h * s, s))
        .or_else(|| (rows >= h && row_bytes >= w * 4).then_some((w, h, 1)))
}

/// Premultiplied BGRA rows `row_bytes` apart → tightly packed straight RGBA,
/// keeping the row order.
fn bgra_premultiplied_to_rgba(buf: &[u8], row_bytes: usize, w: usize, h: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(w * h * 4);
    for row in 0..h {
        let start = row * row_bytes;
        let src = buf.get(start..start + w * 4)?;
        for px in src.chunks_exact(4) {
            let (r, g, b) = un_premultiply(px[2], px[1], px[0], px[3]);
            out.extend_from_slice(&[r, g, b, px[3]]);
        }
    }
    Some(out)
}

fn un_premultiply(r: u8, g: u8, b: u8, a: u8) -> (u8, u8, u8) {
    match a {
        0 => (0, 0, 0),
        255 => (r, g, b),
        _ => {
            let a_u = a as u32;
            let r = ((r as u32 * 255 + a_u / 2) / a_u).min(255) as u8;
            let g = ((g as u32 * 255 + a_u / 2) / a_u).min(255) as u8;
            let b = ((b as u32 * 255 + a_u / 2) / a_u).min(255) as u8;
            (r, g, b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_x_buffer_matches_its_rect() {
        // What tycho returns for the arrow: 28x40, 112-byte rows.
        assert_eq!(pixel_geometry(28.0, 40.0, 112, 4480), Some((28, 40, 1)));
        // Padded rows are still 1x.
        assert_eq!(pixel_geometry(28.0, 40.0, 128, 5120), Some((28, 40, 1)));
    }

    #[test]
    fn retina_buffer_doubles_the_point_rect() {
        assert_eq!(pixel_geometry(28.0, 40.0, 224, 224 * 80), Some((56, 80, 2)));
    }

    #[test]
    fn short_buffers_are_rejected() {
        assert_eq!(
            pixel_geometry(28.0, 40.0, 100, 4000),
            None,
            "rows narrower than the rect"
        );
        assert_eq!(
            pixel_geometry(28.0, 40.0, 112, 112 * 39),
            None,
            "missing a row"
        );
        assert_eq!(pixel_geometry(0.0, 40.0, 112, 4480), None);
    }

    #[test]
    fn retina_hotspot_scales_with_the_pixels() {
        let raw = RawCursor {
            data: vec![0; 224 * 80],
            row_bytes: 224,
            rect_w: 28.0,
            rect_h: 40.0,
            hot_x: 5.0,
            hot_y: 5.0,
        };
        let (rgba, w, h, hx, hy) = raw.to_rgba().unwrap();
        assert_eq!((w, h, hx, hy), (56, 80, 10, 10));
        assert_eq!(rgba.len(), 56 * 80 * 4);
    }

    #[test]
    fn hotspot_is_clamped_inside_the_bitmap() {
        let raw = RawCursor {
            data: vec![0; 8 * 2],
            row_bytes: 8,
            rect_w: 2.0,
            rect_h: 2.0,
            hot_x: 9.0,
            hot_y: -3.0,
        };
        let (_, _, _, hx, hy) = raw.to_rgba().unwrap();
        assert_eq!((hx, hy), (1, 0));
    }

    #[test]
    fn conversion_swaps_to_rgb_unpremultiplies_and_keeps_row_order() {
        // Two rows, one pixel each, 8-byte stride with padding.
        // Row 0: opaque red (B=0,G=0,R=255). Row 1: half-transparent blue,
        // premultiplied (B=128 at A=128).
        let buf = [0, 0, 255, 255, 9, 9, 9, 9, 128, 0, 0, 128, 9, 9, 9, 9];
        let out = bgra_premultiplied_to_rgba(&buf, 8, 1, 2).unwrap();
        assert_eq!(out, vec![255, 0, 0, 255, 0, 0, 255, 128]);
    }

    #[test]
    fn identical_cursors_compare_equal_and_hash_alike() {
        let a = RawCursor {
            data: vec![1, 2, 3, 4],
            row_bytes: 4,
            rect_w: 1.0,
            rect_h: 1.0,
            hot_x: 0.0,
            hot_y: 0.0,
        };
        let mut b = RawCursor {
            data: a.data.clone(),
            ..a
        };
        assert!(a.same_as(&b) && a.content_hash() == b.content_hash());
        b.hot_x = 1.0;
        assert!(!a.same_as(&b) && a.content_hash() != b.content_hash());
    }
}
