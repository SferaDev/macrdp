//! Offline numbers for lazy chroma: band bytes, CPU and chroma quality of the software coder, a key-delay model,
//! and a decode check of VideoToolbox's stream with the bands spliced in.
//!
//!   cargo test --release bench_lazy_chroma -- --ignored --nocapture
//!   cargo test --release aux_bands_decode -- --ignored --nocapture   (needs ffmpeg on PATH)

#![allow(clippy::needless_range_loop)]

use std::time::Instant;

use super::{LazyChroma, DEFAULT_BAND_BYTES, DEFAULT_PACE_KBPS, DEFAULT_QP};
use crate::aux_coder::{encode_band, RefPicture, StreamParams};
use crate::damage::{region_rects, Rect};
use crate::videotoolbox::{bgra_to_aux_view, bgra_to_cbcr444, Encoder};

const LINE: usize = 18;
const PALETTE: [[u8; 3]; 7] = [
    [0x20, 0x20, 0x20],
    [0xd7, 0x3a, 0x49],
    [0x00, 0x5c, 0xc5],
    [0x22, 0x86, 0x3a],
    [0x6f, 0x42, 0xc1],
    [0xe3, 0x62, 0x09],
    [0x03, 0x2f, 0x62],
];

fn hash(a: usize, b: usize, c: usize) -> usize {
    let mut x = (a as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (b as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ (c as u64).wrapping_mul(0x1656_67b1_9e37_79f9);
    x ^= x >> 29;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (x ^ (x >> 32)) as usize
}

/// Syntax-coloured pseudo text in `r`, one line every 18 px, with a half-weight fringe like antialiased glyphs.
fn draw_text(frame: &mut [u8], stride: usize, r: Rect, seed: usize) {
    let (x0, y0, x1, y1) = (
        r.left as usize,
        r.top as usize,
        r.right as usize,
        r.bottom as usize,
    );
    for y in y0..y1 {
        for x in x0..x1 {
            frame[y * stride + x * 4..][..4].copy_from_slice(&[0xff; 4]);
        }
    }
    for line in y0 / LINE.. {
        let top = line * LINE;
        if top >= y1 {
            break;
        }
        let (mut col, mut word_left, mut colour) = (0, 0, PALETTE[0]);
        while x0 + col * 8 + 8 <= x1 {
            if word_left == 0 {
                word_left = 2 + hash(seed, line, col) % 8;
                colour = PALETTE[hash(seed, line, col + 7) % PALETTE.len()];
                col += 1;
                continue;
            }
            word_left -= 1;
            // A glyph: three strokes in an 8x13 cell.
            for k in 0..3 {
                let h = hash(seed, line * 131 + col, k);
                let vertical = h.is_multiple_of(2);
                let (gx, gy) = (1 + h / 2 % 5, 2 + h / 16 % 9);
                for i in 0..3 + h / 256 % 6 {
                    let (px, py) = if vertical { (gx, gy + i) } else { (gx + i, gy) };
                    let (x, y) = (x0 + col * 8 + px.min(7), top + 2 + py.min(12));
                    if y < y0 || y >= y1 {
                        continue;
                    }
                    let bgr = [colour[2], colour[1], colour[0]];
                    frame[y * stride + x * 4..][..3].copy_from_slice(&bgr);
                    if x + 1 < x1 {
                        let q = &mut frame[y * stride + (x + 1) * 4..][..3];
                        for (c, v) in q.iter_mut().zip(bgr) {
                            *c = ((u16::from(*c) + u16::from(v)) / 2) as u8;
                        }
                    }
                }
            }
            col += 1;
        }
    }
}

/// The NAL units of a length-prefixed access unit.
fn avcc(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut rest = data;
    std::iter::from_fn(move || {
        let len = u32::from_be_bytes(rest.get(..4)?.try_into().ok()?) as usize;
        let nal = rest.get(4..4 + len)?;
        rest = &rest[4 + len..];
        Some(nal)
    })
}

/// VideoToolbox's stream parameters for a `w`x`h` session, as the server runs it.
fn vt_params(w: u16, h: u16, frame: &[u8]) -> StreamParams {
    let mut enc = Encoder::new(w, h, 60, 6_000_000, 30.0).unwrap();
    enc.encode_bgra(frame, usize::from(w) * 4, true).unwrap();
    let f = enc.flush().unwrap().remove(0);
    StreamParams::parse(&f.parameter_sets[0], &f.parameter_sets[1]).expect("Baseline CAVLC")
}

/// PSNR of full-resolution chroma rebuilt from 4:2:0 plus `odd` (the aux view's samples as decoded), against the truth.
fn chroma_psnr(
    truth: &[u8],
    w: usize,
    h: usize,
    odd: Option<&dyn Fn(usize, usize) -> i32>,
    region: Rect,
) -> f64 {
    let (mut se, mut n) = (0f64, 0f64);
    for y in (region.top as usize..region.bottom as usize).step_by(2) {
        for x in (region.left as usize..region.right as usize).step_by(2) {
            if y + 1 >= h || x + 1 >= w {
                continue;
            }
            let t = [
                truth[y * w + x],
                truth[y * w + x + 1],
                truth[(y + 1) * w + x],
                truth[(y + 1) * w + x + 1],
            ]
            .map(i32::from);
            let avg = (t.iter().sum::<i32>() + 2) / 4;
            let got = match odd {
                None => [avg; 4],
                Some(a) => {
                    let (b, c, d) = (a(x + 1, y), a(x, y + 1), a(x + 1, y + 1));
                    [(4 * avg - b - c - d).clamp(0, 255), b, c, d]
                }
            };
            for k in 0..4 {
                se += f64::from((t[k] - got[k]).pow(2));
                n += 1.0;
            }
        }
    }
    10.0 * (255.0f64 * 255.0 / (se / n).max(1e-9)).log10()
}

struct Sent {
    bytes: usize,
    cpu_ms: f64,
}

const TICK_MS: f64 = 1000.0 / 60.0;

fn wire_ms(bytes: usize, bps: f64) -> f64 {
    bytes as f64 * 8.0 / bps * 1000.0
}

/// When each band starts, as the capture loop offers them: first tick after the idle wait, then paced.
fn schedule(bands: &[Sent], idle_ms: f64) -> Vec<f64> {
    let tick = |at: f64| (at / TICK_MS).ceil() * TICK_MS;
    let mut start = tick(idle_ms);
    bands
        .iter()
        .map(|b| {
            let at = start;
            let paced = at + b.cpu_ms + wire_ms(b.bytes, DEFAULT_PACE_KBPS as f64 * 1000.0);
            start = tick(paced.max(at + TICK_MS));
            at
        })
        .collect()
}

/// Extra delay of a key pressed `t` ms after the last change and input. A key while a band is being coded stops
/// the coder at the next macroblock (a few microseconds) and the band is dropped; a key after that waits for the
/// band's bytes to leave a `link_bps` link. Motion frames never wait for the encoder: bands do not use it.
fn key_delay(bands: &[Sent], starts: &[f64], t: f64, link_bps: f64) -> f64 {
    const MB_MS: f64 = 0.01;
    for (b, &start) in bands.iter().zip(starts) {
        if t < start {
            return 0.0;
        }
        let coded = start + b.cpu_ms;
        if t < coded {
            return MB_MS;
        }
        let gone = coded + wire_ms(b.bytes, link_bps);
        if t < gone {
            return gone - t;
        }
    }
    0.0
}

fn percentiles(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
    (at(0.5), at(0.99), *v.last().unwrap())
}

#[test]
#[ignore = "measurement; run with --release --ignored --nocapture"]
fn bench_lazy_chroma() {
    for (w, h) in [(1920u16, 1080u16), (2560, 1440)] {
        let (wu, hu) = (usize::from(w), usize::from(h));
        let stride = wu * 4;
        let all = Rect {
            left: 0,
            top: 0,
            right: u32::from(w),
            bottom: u32::from(h),
        };
        let mut frame = vec![0u8; stride * hu];
        draw_text(&mut frame, stride, all, 1);
        let p = vt_params(w, h, &frame);
        let started = Instant::now();
        let aux = bgra_to_aux_view(&frame, stride, wu, hu).unwrap();
        let view_ms = started.elapsed().as_secs_f64() * 1000.0;
        let (mut cb, mut cr) = (vec![0u8; wu * hu], vec![0u8; wu * hu]);
        bgra_to_cbcr444(&frame, stride, wu, hu, &mut cb, &mut cr).unwrap();
        let edits = [
            ("one line", Rect::from_xywh(96, 198, 624, LINE)),
            ("paragraph", Rect::from_xywh(96, 306, 1200, 10 * LINE)),
            ("full screen", all),
        ];
        println!("\n{w}x{h}, coloured text everywhere; aux view conversion {view_ms:.1} ms");
        println!(
            "  {:<12} {:>3} {:>6} {:>5} {:>8} {:>5} {:>8} {:>7} {:>11} {:>7} | key delay at 20 Mbit/s p50/p99/max, keys delayed",
            "edit", "qp", "MBs", "bands", "bytes", "B/MB", "band max", "cpu max", "Cb/Cr dB", "done ms"
        );
        for (name, r) in edits {
            let regions = region_rects(&[r], w, h);
            let region = regions.as_ref().map_or(all, |v| v[0]);
            let base = (
                chroma_psnr(&cb, wu, hu, None, region),
                chroma_psnr(&cr, wu, hu, None, region),
            );
            println!(
                "  {:<12} {:>3} {:>6} {:>5} {:>8} {:>5} {:>8} {:>7} {:>5.1}/{:<5.1} 4:2:0",
                name, "-", "", "", "", "", "", "", base.0, base.1
            );
            for qp in [DEFAULT_QP, 28] {
                let mut l = LazyChroma::new(w, h);
                l.stale.fill(false);
                l.stale_count = 0;
                l.main_shipped(regions.as_deref());
                let mbs = l.stale_mbs();
                let (mut dy, mut duv) = (aux.y.clone(), aux.uv.clone());
                let mut sent = Vec::new();
                while l.stale_mbs() > 0 {
                    let started = Instant::now();
                    let band = encode_band(
                        &p,
                        RefPicture::default(),
                        &aux,
                        &l.candidates(),
                        qp,
                        DEFAULT_BAND_BYTES,
                        &|| false,
                    )
                    .unwrap();
                    let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
                    for (mb, ry, rcb, rcr) in &band.recon {
                        let (mx, my) = (mb % p.width_mbs * 16, mb / p.width_mbs * 16);
                        for i in 0..256 {
                            let (yy, xx) = (my + i / 16, mx + i % 16);
                            if yy < hu && xx < wu {
                                dy[yy * wu + xx] = ry[i] as u8;
                            }
                        }
                        for i in 0..64 {
                            let (yy, xx) = (my / 2 + i / 8, mx / 2 + i % 8);
                            if yy < hu / 2 && xx < wu / 2 {
                                duv[yy * wu + 2 * xx] = rcb[i] as u8;
                                duv[yy * wu + 2 * xx + 1] = rcr[i] as u8;
                            }
                        }
                    }
                    l.band_sent(&band.mbs, band.bytes, Instant::now(), DEFAULT_PACE_KBPS);
                    sent.push(Sent {
                        bytes: band.bytes,
                        cpu_ms,
                    });
                }
                // Where `pack_aux_v1_nv12` keeps each odd sample, read back from the decoded aux view.
                let odd = |v: bool| {
                    let (dy, duv) = (&dy, &duv);
                    move |x: usize, y: usize| -> i32 {
                        if y % 2 == 1 {
                            let i = (y - 1) / 2;
                            let row = i / 8 * 16 + i % 8 + if v { 8 } else { 0 };
                            if row < hu {
                                i32::from(dy[row * wu + x])
                            } else {
                                128
                            }
                        } else {
                            i32::from(duv[(y / 2) * wu + (x - 1) + usize::from(v)])
                        }
                    }
                };
                let (fcb, fcr) = (odd(false), odd(true));
                let q = (
                    chroma_psnr(&cb, wu, hu, Some(&fcb), region),
                    chroma_psnr(&cr, wu, hu, Some(&fcr), region),
                );
                let bytes: usize = sent.iter().map(|b| b.bytes).sum();
                let band_max = sent.iter().map(|b| b.bytes).max().unwrap_or(0);
                let cpu_max = sent.iter().map(|b| b.cpu_ms).fold(0.0, f64::max);
                let starts = schedule(&sent, 250.0);
                let done = starts
                    .last()
                    .zip(sent.last())
                    .map_or(0.0, |(s, b)| s + b.cpu_ms + wire_ms(b.bytes, 20e6));
                // Keys at random times in the 3 s after a pause, every 0.1 ms.
                let delays: Vec<f64> = (0..30_000)
                    .map(|i| key_delay(&sent, &starts, i as f64 * 0.1, 20e6))
                    .collect();
                let hit = delays.iter().filter(|&&d| d > 0.0).count() as f64 / delays.len() as f64;
                let (p50, p99, max) = percentiles(delays);
                println!(
                    "  {:<12} {:>3} {:>6} {:>5} {:>8} {:>5} {:>8} {:>7.2} {:>5.1}/{:<5.1} {:>7.0} | {p50:.1}/{p99:.1}/{max:.1} ms, {:.2}%",
                    name,
                    qp,
                    mbs,
                    sent.len(),
                    bytes,
                    bytes / mbs.max(1),
                    band_max,
                    cpu_max,
                    q.0,
                    q.1,
                    done,
                    hit * 100.0
                );
            }
        }
    }
}

/// VideoToolbox main frames with software aux bands spliced in, decoded by ffmpeg: the bands decode to exactly the
/// samples the coder reconstructed, and the main frames decode as if the bands were never there.
/// With `KEEP=1` the stream and an access-unit index stay in the temp dir, for a VideoToolbox decode as well.
#[test]
#[ignore = "needs VideoToolbox and ffmpeg on PATH"]
fn aux_bands_decode_bit_exact_in_ffmpeg() {
    let (w, h) = (640u16, 360u16);
    let (wu, hu) = (usize::from(w), usize::from(h));
    let stride = wu * 4;
    let all = Rect {
        left: 0,
        top: 0,
        right: u32::from(w),
        bottom: u32::from(h),
    };
    let mut base = vec![0u8; stride * hu];
    draw_text(&mut base, stride, all, 1);
    let aux = bgra_to_aux_view(&base, stride, wu, hu).unwrap();
    let mut enc = Encoder::new(w, h, 60, 6_000_000, 30.0).unwrap();
    let (mut stream, mut plain, mut aus) = (Vec::new(), Vec::new(), Vec::new());
    let put = |out: &mut Vec<u8>, nal: &[u8]| {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    };
    let mut lazy = LazyChroma::new(w, h);
    let mut pictures: Vec<Option<usize>> = Vec::new();
    let mut expect = Vec::new();
    for i in 0..6 {
        let mut f = base.clone();
        draw_text(
            &mut f,
            stride,
            Rect::from_xywh(16 + i * 8, 300, 8, LINE),
            5 + i,
        );
        enc.encode_bgra(&f, stride, i == 0).unwrap();
        for fr in enc.flush().unwrap() {
            aus.push(stream.len());
            for ps in &fr.parameter_sets {
                put(&mut stream, ps);
                put(&mut plain, ps);
            }
            for nal in avcc(&fr.data) {
                put(&mut stream, nal);
                put(&mut plain, nal);
            }
            lazy.stream_sent(&fr.parameter_sets, avcc(&fr.data));
            pictures.push(None);
        }
        // Bands after frames 2 to 4, two in a row after frame 3: consecutive non-reference pictures too.
        if (2..=4).contains(&i) {
            let target = lazy
                .band_due(Instant::now())
                .expect("Baseline CAVLC stream followed");
            let cols = target.params.width_mbs;
            let candidates: Vec<usize> = match i {
                2 => (cols * 2 + 3..cols * 2 + 20)
                    .chain(cols * 3..cols * 3 + 5)
                    .collect(),
                3 => (cols * 6..cols * 8).collect(),
                _ => (0..cols * target.params.height_mbs).collect(),
            };
            for cap in if i == 3 {
                vec![900, 100_000]
            } else {
                vec![100_000]
            } {
                aus.push(stream.len());
                let band = encode_band(
                    &target.params,
                    target.last,
                    &aux,
                    &candidates,
                    DEFAULT_QP,
                    cap,
                    &|| false,
                )
                .unwrap();
                assert!(!band.mbs.is_empty());
                for nal in &band.nals {
                    put(&mut stream, nal);
                }
                pictures.push(Some(expect.len()));
                expect.push(band.recon);
            }
        }
    }
    let dir = std::env::temp_dir().join(format!("aux-bands-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let decode = |name: &str, data: &[u8]| -> Vec<u8> {
        let input = dir.join(format!("{name}.h264"));
        let output = dir.join(format!("{name}.yuv"));
        std::fs::write(&input, data).unwrap();
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "warning",
                "-xerror",
                "-y",
                "-i",
            ])
            .arg(&input)
            .args([
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "passthrough",
            ])
            .arg(&output)
            .status()
            .expect("ffmpeg");
        assert!(status.success(), "ffmpeg failed on {name}");
        std::fs::read(&output).unwrap()
    };
    let yuv = decode("in", &stream);
    let plain_yuv = decode("plain", &plain);
    let starts: Vec<String> = aus.iter().map(ToString::to_string).collect();
    std::fs::write(dir.join("aus.txt"), starts.join("\n")).unwrap();

    let frame_len = wu * hu * 3 / 2;
    assert_eq!(
        yuv.len() / frame_len,
        pictures.len(),
        "one picture out per picture in"
    );
    let cols = wu.div_ceil(16);
    for (k, which) in pictures.iter().enumerate() {
        let Some(e) = which else { continue };
        let pic = &yuv[k * frame_len..(k + 1) * frame_len];
        let (y, u, v) = (
            &pic[..wu * hu],
            &pic[wu * hu..wu * hu * 5 / 4],
            &pic[wu * hu * 5 / 4..],
        );
        for (mb, ry, rcb, rcr) in &expect[*e] {
            let (mx, my) = (mb % cols * 16, mb / cols * 16);
            for i in 0..256 {
                let (yy, xx) = (my + i / 16, mx + i % 16);
                if yy < hu && xx < wu {
                    assert_eq!(i32::from(y[yy * wu + xx]), ry[i], "Y mb {mb} picture {k}");
                }
            }
            for i in 0..64 {
                let (yy, xx) = (my / 2 + i / 8, mx / 2 + i % 8);
                if yy < hu / 2 && xx < wu / 2 {
                    assert_eq!(
                        i32::from(u[yy * wu / 2 + xx]),
                        rcb[i],
                        "Cb mb {mb} picture {k}"
                    );
                    assert_eq!(
                        i32::from(v[yy * wu / 2 + xx]),
                        rcr[i],
                        "Cr mb {mb} picture {k}"
                    );
                }
            }
        }
    }
    let mains: Vec<usize> = (0..pictures.len())
        .filter(|&k| pictures[k].is_none())
        .collect();
    assert_eq!(plain_yuv.len() / frame_len, mains.len());
    for (n, &k) in mains.iter().enumerate() {
        assert!(
            yuv[k * frame_len..(k + 1) * frame_len]
                == plain_yuv[n * frame_len..(n + 1) * frame_len],
            "main frame {n} changed by the bands"
        );
    }
    if std::env::var("KEEP").is_ok() {
        println!("kept {}", dir.display());
    } else {
        let _ = std::fs::remove_dir_all(&dir);
    }
}
