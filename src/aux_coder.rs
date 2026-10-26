//! Software H.264 for lazy chroma's aux bands: a non-reference P picture of VideoToolbox's stream whose band macroblocks
//! are Intra16x16 (CAVLC) and whose other macroblocks are skipped, so the main stream's references and rate control
//! never see it. Each run of band macroblocks is its own slice: intra prediction then only reads samples coded here.

// Block and transform code indexes rows and columns the way the H.264 spec writes them.
#![allow(clippy::needless_range_loop)]

use crate::avc444::Nv12;

/// Stream parameters from VideoToolbox's SPS and PPS that the band slices must follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamParams {
    log2_max_frame_num: u32,
    /// `None`: `pic_order_cnt_type` 2.
    log2_max_poc_lsb: Option<u32>,
    pub width_mbs: usize,
    pub height_mbs: usize,
    pps_id: u32,
    pic_init_qp: i32,
    chroma_qp_offset: i32,
    bottom_field_poc: bool,
}

/// The last reference picture VideoToolbox shipped: the band's `frame_num` and POC follow it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RefPicture {
    pub frame_num: u32,
    pub poc_lsb: u32,
}

/// What a VideoToolbox slice header says about its picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceInfo {
    pub is_reference: bool,
    pub picture: RefPicture,
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn bit(&mut self) -> Option<u32> {
        let byte = *self.data.get(self.pos / 8)?;
        let b = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Some(u32::from(b))
    }

    fn u(&mut self, n: u32) -> Option<u32> {
        (0..n).try_fold(0u32, |acc, _| Some((acc << 1) | self.bit()?))
    }

    fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        Some((1u32 << zeros) - 1 + self.u(zeros)?)
    }

    fn se(&mut self) -> Option<i32> {
        let k = self.ue()?;
        Some(if k % 2 == 1 {
            k.div_ceil(2) as i32
        } else {
            -((k / 2) as i32)
        })
    }

    /// `more_rbsp_data()`: a set bit remains before the stop bit.
    fn more_data(&self) -> bool {
        let total = self.data.len() * 8;
        let last_one = (0..total)
            .rev()
            .find(|&i| (self.data[i / 8] >> (7 - i % 8)) & 1 == 1);
        last_one.is_some_and(|stop| self.pos < stop)
    }
}

/// The RBSP of a NAL unit (header byte dropped, emulation prevention removed).
fn rbsp(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0;
    for &b in nal.iter().skip(1) {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

impl StreamParams {
    /// `None` for anything the band coder cannot follow: CABAC, interlace, scaling lists, slice groups, 4:4:4, forced deblocking.
    pub fn parse(sps: &[u8], pps: &[u8]) -> Option<Self> {
        if sps.first()? & 0x1f != 7 || pps.first()? & 0x1f != 8 {
            return None;
        }
        let s = rbsp(sps);
        let mut r = BitReader { data: &s, pos: 0 };
        let profile = r.u(8)?;
        r.u(16)?;
        r.ue()?;
        if [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135].contains(&profile) {
            let chroma_format = r.ue()?;
            if chroma_format != 1 || r.ue()? != 0 || r.ue()? != 0 {
                return None;
            }
            r.u(1)?;
            if r.u(1)? != 0 {
                return None;
            }
        }
        let log2_max_frame_num = r.ue()? + 4;
        let log2_max_poc_lsb = match r.ue()? {
            0 => Some(r.ue()? + 4),
            2 => None,
            _ => return None,
        };
        r.ue()?;
        r.u(1)?;
        let width_mbs = r.ue()? as usize + 1;
        let height_mbs = r.ue()? as usize + 1;
        if r.u(1)? != 1 {
            return None;
        }

        let p = rbsp(pps);
        let mut r = BitReader { data: &p, pos: 0 };
        let pps_id = r.ue()?;
        r.ue()?;
        let cabac = r.u(1)? == 1;
        let bottom_field_poc = r.u(1)? == 1;
        if cabac || r.ue()? != 0 {
            return None;
        }
        r.ue()?;
        r.ue()?;
        if r.u(1)? != 0 {
            return None;
        }
        r.u(2)?;
        let pic_init_qp = 26 + r.se()?;
        r.se()?;
        let chroma_qp_offset = r.se()?;
        // Without deblocking_filter_control_present_flag a band could not turn the filter off.
        if r.u(1)? != 1 {
            return None;
        }
        r.u(1)?;
        if r.u(1)? != 0 {
            return None;
        }
        if r.more_data() {
            r.u(1)?;
            if r.u(1)? != 0 {
                return None;
            }
        }
        Some(Self {
            log2_max_frame_num,
            log2_max_poc_lsb,
            width_mbs,
            height_mbs,
            pps_id,
            pic_init_qp,
            chroma_qp_offset,
            bottom_field_poc,
        })
    }

    /// The picture a VideoToolbox slice NAL belongs to; `None` for other NAL types.
    pub fn slice_info(&self, nal: &[u8]) -> Option<SliceInfo> {
        let header = *nal.first()?;
        let nal_type = header & 0x1f;
        if nal_type != 1 && nal_type != 5 {
            return None;
        }
        let d = rbsp(&nal[..nal.len().min(32)]);
        let mut r = BitReader { data: &d, pos: 0 };
        r.ue()?;
        r.ue()?;
        r.ue()?;
        let frame_num = r.u(self.log2_max_frame_num)?;
        if nal_type == 5 {
            r.ue()?;
        }
        let poc_lsb = match self.log2_max_poc_lsb {
            Some(bits) => r.u(bits)?,
            None => 0,
        };
        Some(SliceInfo {
            is_reference: header & 0x60 != 0,
            picture: RefPicture { frame_num, poc_lsb },
        })
    }
}

struct BitWriter {
    bytes: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            acc: 0,
            n: 0,
        }
    }

    fn u(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.acc = (self.acc << 1) | ((v >> i) & 1);
            self.n += 1;
            if self.n == 8 {
                self.bytes.push(self.acc as u8);
                self.acc = 0;
                self.n = 0;
            }
        }
    }

    fn ue(&mut self, v: u32) {
        let x = v + 1;
        let len = 32 - x.leading_zeros();
        self.u(len - 1, 0);
        self.u(len, x);
    }

    fn se(&mut self, v: i32) {
        self.ue(if v > 0 {
            2 * v as u32 - 1
        } else {
            (-2 * v) as u32
        });
    }

    fn bits(&self) -> usize {
        self.bytes.len() * 8 + self.n as usize
    }

    /// The RBSP with its stop bit, wrapped as a NAL unit with emulation prevention.
    fn into_nal(mut self, header: u8) -> Vec<u8> {
        self.u(1, 1);
        while self.n != 0 {
            self.u(1, 0);
        }
        let mut out = Vec::with_capacity(self.bytes.len() + 8);
        out.push(header);
        let mut zeros = 0;
        for b in self.bytes {
            if zeros >= 2 && b <= 3 {
                out.push(3);
                zeros = 0;
            }
            zeros = if b == 0 { zeros + 1 } else { 0 };
            out.push(b);
        }
        out
    }
}

// CAVLC tables (H.264 9.2), as lengths and codes indexed [nC class][total_coeff * 4 + trailing_ones].
const COEFF_TOKEN_LEN: [[u8; 68]; 4] = [
    [
        1, 0, 0, 0, 6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6, 11, 10, 9, 7, 13, 11, 10, 8,
        13, 13, 11, 9, 13, 13, 13, 10, 14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15,
        14, 16, 15, 15, 15, 16, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    [
        2, 0, 0, 0, 6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4, 8, 7, 7, 5, 9, 8, 8, 6, 11, 9,
        9, 6, 11, 11, 11, 7, 12, 11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12, 13, 13,
        13, 13, 13, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 14,
    ],
    [
        4, 0, 0, 0, 6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 6, 6, 4, 7, 6,
        6, 4, 8, 7, 7, 5, 8, 8, 7, 6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8, 10, 9, 9, 9, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10,
    ],
    [6; 68],
];

const COEFF_TOKEN_BITS: [[u8; 68]; 4] = [
    [
        1, 0, 0, 0, 5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3, 7, 6, 5, 4, 15, 6, 5, 4, 11,
        14, 5, 4, 8, 10, 13, 4, 15, 14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8, 15, 1,
        9, 12, 11, 14, 13, 8, 7, 10, 9, 12, 4, 6, 5, 8,
    ],
    [
        3, 0, 0, 0, 11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4, 4, 6, 5, 6, 7, 6, 5, 8, 15,
        6, 5, 4, 11, 14, 13, 4, 15, 10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12, 11, 10,
        9, 12, 7, 11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4,
    ],
    [
        15, 0, 0, 0, 15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11, 11, 8, 9, 10, 9,
        14, 13, 9, 8, 10, 9, 8, 15, 14, 13, 13, 11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8,
        10, 13, 8, 13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2,
    ],
    [0; 68],
];

const CHROMA_DC_TOKEN_LEN: [u8; 20] = [2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7];
const CHROMA_DC_TOKEN_BITS: [u8; 20] = [1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0];

const TOTAL_ZEROS_LEN: [&[u8]; 15] = [
    &[1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9],
    &[3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6],
    &[4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6],
    &[5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5],
    &[4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5],
    &[6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6],
    &[6, 5, 3, 3, 3, 2, 3, 4, 3, 6],
    &[6, 4, 5, 3, 2, 2, 3, 3, 6],
    &[6, 6, 4, 2, 2, 3, 2, 5],
    &[5, 5, 3, 2, 2, 2, 4],
    &[4, 4, 3, 3, 1, 3],
    &[4, 4, 2, 1, 3],
    &[3, 3, 1, 2],
    &[2, 2, 1],
    &[1, 1],
];

const TOTAL_ZEROS_BITS: [&[u8]; 15] = [
    &[1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1],
    &[7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0],
    &[5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0],
    &[3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0],
    &[5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0],
    &[1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0],
    &[1, 1, 5, 4, 3, 3, 2, 1, 1, 0],
    &[1, 1, 1, 3, 3, 2, 2, 1, 0],
    &[1, 0, 1, 3, 2, 1, 1, 1],
    &[1, 0, 1, 3, 2, 1, 1],
    &[0, 1, 1, 2, 1, 3],
    &[0, 1, 1, 1, 1],
    &[0, 1, 1, 1],
    &[0, 1, 1],
    &[0, 1],
];

const CHROMA_DC_TOTAL_ZEROS_LEN: [&[u8]; 3] = [&[1, 2, 3, 3], &[1, 2, 2], &[1, 1]];
const CHROMA_DC_TOTAL_ZEROS_BITS: [&[u8]; 3] = [&[1, 1, 1, 0], &[1, 1, 0], &[1, 0]];

const RUN_LEN: [&[u8]; 7] = [
    &[1, 1],
    &[1, 2, 2],
    &[2, 2, 2, 2],
    &[2, 2, 2, 3, 3],
    &[2, 2, 3, 3, 3, 3],
    &[2, 3, 3, 3, 3, 3, 3],
    &[3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11],
];

const RUN_BITS: [&[u8]; 7] = [
    &[1, 0],
    &[1, 1, 0],
    &[3, 2, 1, 0],
    &[3, 2, 1, 1, 0],
    &[3, 2, 3, 2, 1, 0],
    &[3, 0, 1, 3, 2, 5, 4],
    &[7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

/// 4x4 zigzag: scan position to raster index.
const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Forward quantisation multipliers and dequantisation scales per `qp % 6`, for position classes a, b, c.
const MF: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];
const V: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

const CHROMA_QP: [i32; 22] = [
    29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
];

fn pos_class(i: usize) -> usize {
    let (r, c) = (i / 4, i % 4);
    match (r % 2, c % 2) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

fn chroma_qp(qp: i32, offset: i32) -> i32 {
    let q = (qp + offset).clamp(0, 51);
    if q < 30 {
        q
    } else {
        CHROMA_QP[(q - 30) as usize]
    }
}

fn forward4x4(x: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    for r in 0..4 {
        let (a, b, c, d) = (x[r * 4], x[r * 4 + 1], x[r * 4 + 2], x[r * 4 + 3]);
        let (s0, s1, s2, s3) = (a + d, b + c, b - c, a - d);
        t[r * 4] = s0 + s1;
        t[r * 4 + 1] = 2 * s3 + s2;
        t[r * 4 + 2] = s0 - s1;
        t[r * 4 + 3] = s3 - 2 * s2;
    }
    let mut y = [0i32; 16];
    for c in 0..4 {
        let (a, b, cc, d) = (t[c], t[4 + c], t[8 + c], t[12 + c]);
        let (s0, s1, s2, s3) = (a + d, b + cc, b - cc, a - d);
        y[c] = s0 + s1;
        y[4 + c] = 2 * s3 + s2;
        y[8 + c] = s0 - s1;
        y[12 + c] = s3 - 2 * s2;
    }
    y
}

/// The decoder's inverse transform (8.5.12.2), result already `(h + 32) >> 6`.
fn inverse4x4(d: &[i32; 16]) -> [i32; 16] {
    let mut f = [0i32; 16];
    for r in 0..4 {
        let (d0, d1, d2, d3) = (d[r * 4], d[r * 4 + 1], d[r * 4 + 2], d[r * 4 + 3]);
        let (e0, e1, e2, e3) = (d0 + d2, d0 - d2, (d1 >> 1) - d3, d1 + (d3 >> 1));
        f[r * 4] = e0 + e3;
        f[r * 4 + 1] = e1 + e2;
        f[r * 4 + 2] = e1 - e2;
        f[r * 4 + 3] = e0 - e3;
    }
    let mut out = [0i32; 16];
    for c in 0..4 {
        let (d0, d1, d2, d3) = (f[c], f[4 + c], f[8 + c], f[12 + c]);
        let (e0, e1, e2, e3) = (d0 + d2, d0 - d2, (d1 >> 1) - d3, d1 + (d3 >> 1));
        out[c] = (e0 + e3 + 32) >> 6;
        out[4 + c] = (e1 + e2 + 32) >> 6;
        out[8 + c] = (e1 - e2 + 32) >> 6;
        out[12 + c] = (e0 - e3 + 32) >> 6;
    }
    out
}

fn hadamard4x4(x: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    for r in 0..4 {
        let (a, b, c, d) = (x[r * 4], x[r * 4 + 1], x[r * 4 + 2], x[r * 4 + 3]);
        t[r * 4] = a + b + c + d;
        t[r * 4 + 1] = a + b - c - d;
        t[r * 4 + 2] = a - b - c + d;
        t[r * 4 + 3] = a - b + c - d;
    }
    let mut y = [0i32; 16];
    for c in 0..4 {
        let (a, b, cc, d) = (t[c], t[4 + c], t[8 + c], t[12 + c]);
        y[c] = a + b + cc + d;
        y[4 + c] = a + b - cc - d;
        y[8 + c] = a - b - cc + d;
        y[12 + c] = a - b + cc - d;
    }
    y
}

fn quant(w: i32, mf: i32, qbits: i32, f: i32) -> i32 {
    let l = ((w.abs() as i64 * mf as i64 + f as i64) >> qbits) as i32;
    (if w < 0 { -l } else { l }).clamp(-2047, 2047)
}

/// Codes one block's levels (scan order) with CAVLC; returns its TotalCoeff.
fn cavlc_block(bw: &mut BitWriter, levels: &[i32], nc: i32) -> u8 {
    let max = levels.len();
    let nz: Vec<usize> = (0..max).filter(|&i| levels[i] != 0).collect();
    let total = nz.len();
    let mut t1 = 0;
    for &i in nz.iter().rev() {
        if t1 < 3 && levels[i].abs() == 1 {
            t1 += 1;
        } else {
            break;
        }
    }
    let idx = total * 4 + t1;
    if nc == -1 {
        bw.u(
            u32::from(CHROMA_DC_TOKEN_LEN[idx]),
            u32::from(CHROMA_DC_TOKEN_BITS[idx]),
        );
    } else {
        let class = match nc {
            0..=1 => 0,
            2..=3 => 1,
            4..=7 => 2,
            _ => 3,
        };
        if class == 3 {
            // 6-bit FLC: TotalCoeff - 1 and TrailingOnes; 000011 for none.
            if total == 0 {
                bw.u(6, 3);
            } else {
                bw.u(6, (((total - 1) << 2) | t1) as u32);
            }
        } else {
            bw.u(
                u32::from(COEFF_TOKEN_LEN[class][idx]),
                u32::from(COEFF_TOKEN_BITS[class][idx]),
            );
        }
    }
    if total == 0 {
        return 0;
    }
    // Levels from the highest frequency down.
    let rev: Vec<i32> = nz.iter().rev().map(|&i| levels[i]).collect();
    for l in &rev[..t1] {
        bw.u(1, u32::from(*l < 0));
    }
    let mut suffix_len: u32 = if total > 10 && t1 < 3 { 1 } else { 0 };
    for (k, &l) in rev.iter().enumerate().skip(t1) {
        let mut code = if l > 0 { 2 * l - 2 } else { -2 * l - 1 };
        if k == t1 && t1 < 3 {
            code -= 2;
        }
        let code = code as u32;
        if suffix_len == 0 {
            if code < 14 {
                bw.u(code + 1, 1);
            } else if code < 30 {
                bw.u(15, 1);
                bw.u(4, code - 14);
            } else {
                bw.u(16, 1);
                bw.u(12, code - 30);
            }
        } else if code < (15 << suffix_len) {
            bw.u((code >> suffix_len) + 1, 1);
            bw.u(suffix_len, code & ((1 << suffix_len) - 1));
        } else {
            bw.u(16, 1);
            bw.u(12, code - (15 << suffix_len));
        }
        if suffix_len == 0 {
            suffix_len = 1;
        }
        if l.unsigned_abs() > (3 << (suffix_len - 1)) && suffix_len < 6 {
            suffix_len += 1;
        }
    }
    let last = *nz.last().unwrap_or(&0);
    let total_zeros = last + 1 - total;
    if total < max {
        if max == 4 {
            bw.u(
                u32::from(CHROMA_DC_TOTAL_ZEROS_LEN[total - 1][total_zeros]),
                u32::from(CHROMA_DC_TOTAL_ZEROS_BITS[total - 1][total_zeros]),
            );
        } else {
            bw.u(
                u32::from(TOTAL_ZEROS_LEN[total - 1][total_zeros]),
                u32::from(TOTAL_ZEROS_BITS[total - 1][total_zeros]),
            );
        }
    }
    let mut zeros_left = total_zeros;
    for w in (1..nz.len()).rev() {
        if zeros_left == 0 {
            break;
        }
        let run = nz[w] - nz[w - 1] - 1;
        let t = zeros_left.min(7) - 1;
        bw.u(u32::from(RUN_LEN[t][run]), u32::from(RUN_BITS[t][run]));
        zeros_left -= run;
    }
    total as u8
}

/// One plane's samples for a macroblock-sized block, edge-replicated past the picture.
fn block(plane: &[u8], stride: usize, rows: usize, x0: usize, y0: usize, size: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(size * size);
    for y in 0..size {
        let row = (y0 + y).min(rows - 1);
        for x in 0..size {
            let col = (x0 + x).min(stride - 1);
            out.push(i32::from(plane[row * stride + col]));
        }
    }
    out
}

/// Left neighbour column and the TotalCoeff of its right-edge blocks, from the macroblock coded before in this slice.
#[derive(Clone)]
struct Left {
    y: [i32; 16],
    cb: [i32; 8],
    cr: [i32; 8],
    luma_nc: [u8; 4],
    cb_nc: [u8; 2],
    cr_nc: [u8; 2],
}

fn sad(src: &[i32], pred: &[i32]) -> i64 {
    src.iter()
        .zip(pred)
        .map(|(a, b)| i64::from((a - b).abs()))
        .sum()
}

struct Coded {
    left: Left,
    /// Reconstructed samples: 256 luma, 64 Cb, 64 Cr.
    #[cfg_attr(not(test), allow(dead_code))]
    recon: (Vec<i32>, Vec<i32>, Vec<i32>),
}

/// Codes one Intra16x16 macroblock, `left` the macroblock before it in the slice.
fn code_mb(
    bw: &mut BitWriter,
    aux: &Nv12,
    mbx: usize,
    mby: usize,
    qp: i32,
    qpc: i32,
    left: Option<&Left>,
) -> Coded {
    let (w, h) = (aux.width, aux.height);
    let src_y = block(&aux.y, w, h, mbx * 16, mby * 16, 16);
    // NV12 chroma: interleaved, so read the 16 bytes and split.
    let mut src_cb = Vec::with_capacity(64);
    let mut src_cr = Vec::with_capacity(64);
    for y in 0..8 {
        let row = (mby * 8 + y).min(h / 2 - 1);
        for x in 0..8 {
            let col = (mbx * 8 + x).min(w / 2 - 1);
            src_cb.push(i32::from(aux.uv[row * w + 2 * col]));
            src_cr.push(i32::from(aux.uv[row * w + 2 * col + 1]));
        }
    }

    // Luma prediction: DC (2) or horizontal (1); the top macroblock is in another slice.
    let dc = left.map_or(128, |l| (l.y.iter().sum::<i32>() + 8) >> 4);
    let pred_dc = vec![dc; 256];
    let (mode, pred_y) = match left {
        Some(l) => {
            let pred_h: Vec<i32> = (0..256).map(|i| l.y[i / 16]).collect();
            if sad(&src_y, &pred_h) < sad(&src_y, &pred_dc) {
                (1u32, pred_h)
            } else {
                (2, pred_dc)
            }
        }
        None => (2, pred_dc),
    };
    // Chroma: DC (0) or horizontal (1), one mode for both components.
    let chroma_dc = |l: Option<&[i32; 8]>| -> Vec<i32> {
        (0..64)
            .map(|i| match l {
                Some(l) => {
                    let y0 = (i / 8) / 4 * 4;
                    (l[y0..y0 + 4].iter().sum::<i32>() + 2) >> 2
                }
                None => 128,
            })
            .collect()
    };
    let chroma_h = |l: &[i32; 8]| -> Vec<i32> { (0..64).map(|i| l[i / 8]).collect() };
    let (chroma_mode, pred_cb, pred_cr) = match left {
        Some(l) => {
            let (dc_cb, dc_cr) = (chroma_dc(Some(&l.cb)), chroma_dc(Some(&l.cr)));
            let (h_cb, h_cr) = (chroma_h(&l.cb), chroma_h(&l.cr));
            if sad(&src_cb, &h_cb) + sad(&src_cr, &h_cr)
                < sad(&src_cb, &dc_cb) + sad(&src_cr, &dc_cr)
            {
                (1u32, h_cb, h_cr)
            } else {
                (0, dc_cb, dc_cr)
            }
        }
        None => (0, chroma_dc(None), chroma_dc(None)),
    };

    // Luma residual: 16 4x4 transforms, their DCs through the Hadamard.
    let qbits = 15 + qp / 6;
    let f = (1 << qbits) / 3;
    let mut dc_in = [0i32; 16];
    let mut ac = [[0i32; 16]; 16];
    for by in 0..4 {
        for bx in 0..4 {
            let mut x = [0i32; 16];
            for i in 0..16 {
                let (r, c) = (by * 4 + i / 4, bx * 4 + i % 4);
                x[i] = src_y[r * 16 + c] - pred_y[r * 16 + c];
            }
            let t = forward4x4(&x);
            dc_in[by * 4 + bx] = t[0];
            for i in 1..16 {
                ac[by * 4 + bx][i] = quant(t[i], MF[(qp % 6) as usize][pos_class(i)], qbits, f);
            }
        }
    }
    let hd = hadamard4x4(&dc_in);
    let mut dc_levels = [0i32; 16];
    for i in 0..16 {
        let v = (hd[i] + if hd[i] >= 0 { 1 } else { -1 }) / 2;
        dc_levels[i] = quant(v, MF[(qp % 6) as usize][0], qbits + 1, 2 * f);
    }
    let luma_ac = ac.iter().any(|b| b[1..].iter().any(|&l| l != 0));

    // Chroma residual per component: 4 transforms, their DCs through the 2x2 Hadamard.
    let qbits_c = 15 + qpc / 6;
    let fc = (1 << qbits_c) / 3;
    let chroma = |src: &[i32], pred: &[i32]| -> ([i32; 4], [[i32; 16]; 4]) {
        let mut dcs = [0i32; 4];
        let mut acs = [[0i32; 16]; 4];
        for b in 0..4 {
            let (bx, by) = (b % 2, b / 2);
            let mut x = [0i32; 16];
            for i in 0..16 {
                let (r, c) = (by * 4 + i / 4, bx * 4 + i % 4);
                x[i] = src[r * 8 + c] - pred[r * 8 + c];
            }
            let t = forward4x4(&x);
            dcs[b] = t[0];
            for i in 1..16 {
                acs[b][i] = quant(t[i], MF[(qpc % 6) as usize][pos_class(i)], qbits_c, fc);
            }
        }
        let had = [
            dcs[0] + dcs[1] + dcs[2] + dcs[3],
            dcs[0] - dcs[1] + dcs[2] - dcs[3],
            dcs[0] + dcs[1] - dcs[2] - dcs[3],
            dcs[0] - dcs[1] - dcs[2] + dcs[3],
        ];
        let mut dl = [0i32; 4];
        for i in 0..4 {
            dl[i] = quant(had[i], MF[(qpc % 6) as usize][0], qbits_c + 1, 2 * fc);
        }
        (dl, acs)
    };
    let (cb_dc, cb_ac) = chroma(&src_cb, &pred_cb);
    let (cr_dc, cr_ac) = chroma(&src_cr, &pred_cr);
    let has_ac = |acs: &[[i32; 16]; 4]| acs.iter().any(|b| b[1..].iter().any(|&l| l != 0));
    let cbp_chroma = if has_ac(&cb_ac) || has_ac(&cr_ac) {
        2
    } else if cb_dc.iter().chain(&cr_dc).any(|&l| l != 0) {
        1
    } else {
        0
    };

    // macroblock_layer: mb_skip_run 0, mb_type, intra_chroma_pred_mode, mb_qp_delta.
    bw.ue(0);
    let imb = 1 + mode + 4 * cbp_chroma + if luma_ac { 12 } else { 0 };
    bw.ue(5 + imb);
    bw.ue(chroma_mode);
    bw.se(0);

    let nc_of = |a: Option<u8>, b: Option<u8>| -> i32 {
        match (a, b) {
            (Some(a), Some(b)) => (i32::from(a) + i32::from(b) + 1) >> 1,
            (Some(a), None) => i32::from(a),
            (None, Some(b)) => i32::from(b),
            (None, None) => 0,
        }
    };
    let mut luma_tc = [[0u8; 4]; 4];
    let left_luma = |by: usize| left.map(|l| l.luma_nc[by]);
    // Intra16x16DCLevel takes block 0's nC.
    let mut dc_scan = [0i32; 16];
    for (k, &p) in ZIGZAG.iter().enumerate() {
        dc_scan[k] = dc_levels[p];
    }
    cavlc_block(bw, &dc_scan, nc_of(left_luma(0), None));
    if luma_ac {
        for blk in 0..16 {
            let (b8, b4) = (blk / 4, blk % 4);
            let bx = (b8 % 2) * 2 + b4 % 2;
            let by = (b8 / 2) * 2 + b4 / 2;
            let na = if bx > 0 {
                Some(luma_tc[by][bx - 1])
            } else {
                left_luma(by)
            };
            let nb = (by > 0).then(|| luma_tc[by - 1][bx]);
            let mut scan = [0i32; 15];
            for k in 1..16 {
                scan[k - 1] = ac[by * 4 + bx][ZIGZAG[k]];
            }
            luma_tc[by][bx] = cavlc_block(bw, &scan, nc_of(na, nb));
        }
    }
    let mut cb_tc = [[0u8; 2]; 2];
    let mut cr_tc = [[0u8; 2]; 2];
    if cbp_chroma > 0 {
        cavlc_block(bw, &cb_dc, -1);
        cavlc_block(bw, &cr_dc, -1);
    }
    if cbp_chroma == 2 {
        for (acs, tc, lnc) in [
            (&cb_ac, &mut cb_tc, left.map(|l| l.cb_nc)),
            (&cr_ac, &mut cr_tc, left.map(|l| l.cr_nc)),
        ] {
            for b in 0..4 {
                let (bx, by) = (b % 2, b / 2);
                let na = if bx > 0 {
                    Some(tc[by][0])
                } else {
                    lnc.map(|n| n[by])
                };
                let nb = (by > 0).then(|| tc[0][bx]);
                let mut scan = [0i32; 15];
                for k in 1..16 {
                    scan[k - 1] = acs[b][ZIGZAG[k]];
                }
                tc[by][bx] = cavlc_block(bw, &scan, nc_of(na, nb));
            }
        }
    }

    // Reconstruction, exactly as the decoder does it (8.5.10 to 8.5.12).
    let (qd, qm) = (qp / 6, (qp % 6) as usize);
    let f_dc = hadamard4x4(&dc_levels);
    let ls00 = 16 * V[qm][0];
    let dcy: Vec<i32> = f_dc
        .iter()
        .map(|&v| {
            if qp >= 36 {
                (v * ls00) << (qd - 6)
            } else {
                (v * ls00 + (1 << (5 - qd))) >> (6 - qd)
            }
        })
        .collect();
    let mut recon_y = vec![0i32; 256];
    for by in 0..4 {
        for bx in 0..4 {
            let mut d = [0i32; 16];
            d[0] = dcy[by * 4 + bx];
            if luma_ac {
                for i in 1..16 {
                    d[i] = (ac[by * 4 + bx][i] * V[qm][pos_class(i)]) << qd;
                }
            }
            let r = inverse4x4(&d);
            for i in 0..16 {
                let (row, col) = (by * 4 + i / 4, bx * 4 + i % 4);
                recon_y[row * 16 + col] = (pred_y[row * 16 + col] + r[i]).clamp(0, 255);
            }
        }
    }
    let (qcd, qcm) = (qpc / 6, (qpc % 6) as usize);
    let recon_c = |dl: &[i32; 4], acs: &[[i32; 16]; 4], pred: &[i32]| -> Vec<i32> {
        let fh = [
            dl[0] + dl[1] + dl[2] + dl[3],
            dl[0] - dl[1] + dl[2] - dl[3],
            dl[0] + dl[1] - dl[2] - dl[3],
            dl[0] - dl[1] - dl[2] + dl[3],
        ];
        let mut out = vec![0i32; 64];
        for b in 0..4 {
            let (bx, by) = (b % 2, b / 2);
            let mut d = [0i32; 16];
            d[0] = if cbp_chroma > 0 {
                ((fh[b] * 16 * V[qcm][0]) << qcd) >> 5
            } else {
                0
            };
            if cbp_chroma == 2 {
                for i in 1..16 {
                    d[i] = (acs[b][i] * V[qcm][pos_class(i)]) << qcd;
                }
            }
            let r = inverse4x4(&d);
            for i in 0..16 {
                let (row, col) = (by * 4 + i / 4, bx * 4 + i % 4);
                out[row * 8 + col] = (pred[row * 8 + col] + r[i]).clamp(0, 255);
            }
        }
        out
    };
    let recon_cb = recon_c(&cb_dc, &cb_ac, &pred_cb);
    let recon_cr = recon_c(&cr_dc, &cr_ac, &pred_cr);

    let mut next = Left {
        y: [0; 16],
        cb: [0; 8],
        cr: [0; 8],
        luma_nc: [0; 4],
        cb_nc: [0; 2],
        cr_nc: [0; 2],
    };
    for r in 0..16 {
        next.y[r] = recon_y[r * 16 + 15];
    }
    for r in 0..8 {
        next.cb[r] = recon_cb[r * 8 + 7];
        next.cr[r] = recon_cr[r * 8 + 7];
    }
    for by in 0..4 {
        next.luma_nc[by] = luma_tc[by][3];
    }
    for by in 0..2 {
        next.cb_nc[by] = cb_tc[by][1];
        next.cr_nc[by] = cr_tc[by][1];
    }
    Coded {
        left: next,
        recon: (recon_y, recon_cb, recon_cr),
    }
}

/// A macroblock's decoded samples: its index, 256 luma, 64 Cb, 64 Cr.
#[cfg(test)]
pub type Recon = (usize, Vec<i32>, Vec<i32>, Vec<i32>);

/// A coded band: its NAL units (Annex B order, no start codes), the macroblocks it refined and their decoded samples.
pub struct Band {
    pub nals: Vec<Vec<u8>>,
    pub mbs: Vec<usize>,
    pub bytes: usize,
    #[cfg(test)]
    pub recon: Vec<Recon>,
}

/// Codes the `candidates` macroblocks (raster order) of `aux` at `qp` until about `byte_cap` bytes, as a non-reference
/// P picture following `last`: one skip slice per gap, one Intra16x16 slice per run. `None` once `stop` says so.
pub fn encode_band(
    p: &StreamParams,
    last: RefPicture,
    aux: &Nv12,
    candidates: &[usize],
    qp: i32,
    byte_cap: usize,
    stop: &dyn Fn() -> bool,
) -> Option<Band> {
    let qp = qp.clamp(10, 51);
    let qpc = chroma_qp(qp, p.chroma_qp_offset);
    let total_mbs = p.width_mbs * p.height_mbs;
    let frame_num = (last.frame_num + 1) % (1 << p.log2_max_frame_num);
    let header = |bw: &mut BitWriter, first_mb: usize| {
        bw.ue(first_mb as u32);
        bw.ue(0);
        bw.ue(p.pps_id);
        bw.u(p.log2_max_frame_num, frame_num);
        if let Some(bits) = p.log2_max_poc_lsb {
            // The last reference picture's POC: the next one VideoToolbox codes is later.
            bw.u(bits, last.poc_lsb);
            if p.bottom_field_poc {
                bw.se(0);
            }
        }
        bw.u(1, 0);
        bw.u(1, 0);
        bw.se(qp - p.pic_init_qp);
        bw.ue(1);
    };
    let mut nals = Vec::new();
    let mut bytes = 0;
    let mut coded = Vec::new();
    #[cfg(test)]
    let mut recon = Vec::new();
    let mut next_mb = 0;
    let skip_to = |nals: &mut Vec<Vec<u8>>, bytes: &mut usize, from: usize, to: usize| {
        if to > from {
            let mut bw = BitWriter::new();
            header(&mut bw, from);
            bw.ue((to - from) as u32);
            let nal = bw.into_nal(0x01);
            *bytes += nal.len() + 4;
            nals.push(nal);
        }
    };
    let mut i = 0;
    while i < candidates.len() && bytes < byte_cap {
        let start = candidates[i];
        if start >= total_mbs || start < next_mb {
            i += 1;
            continue;
        }
        let row = start / p.width_mbs;
        skip_to(&mut nals, &mut bytes, next_mb, start);
        let mut bw = BitWriter::new();
        header(&mut bw, start);
        let mut left: Option<Left> = None;
        let mut mb = start;
        while i < candidates.len() && candidates[i] == mb && mb / p.width_mbs == row {
            if stop() {
                return None;
            }
            let c = code_mb(&mut bw, aux, mb % p.width_mbs, row, qp, qpc, left.as_ref());
            left = Some(c.left);
            #[cfg(test)]
            recon.push((mb, c.recon.0, c.recon.1, c.recon.2));
            coded.push(mb);
            mb += 1;
            i += 1;
            if bytes + bw.bits() / 8 >= byte_cap {
                break;
            }
        }
        let nal = bw.into_nal(0x01);
        bytes += nal.len() + 4;
        nals.push(nal);
        next_mb = mb;
    }
    skip_to(&mut nals, &mut bytes, next_mb, total_mbs);
    Some(Band {
        nals,
        mbs: coded,
        bytes,
        #[cfg(test)]
        recon,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_round_trips() {
        let mut bw = BitWriter::new();
        for v in [0u32, 1, 2, 7, 255, 4096] {
            bw.ue(v);
        }
        for v in [0i32, 1, -1, 5, -77] {
            bw.se(v);
        }
        let nal = bw.into_nal(0x01);
        let d = rbsp(&nal);
        let mut r = BitReader { data: &d, pos: 0 };
        for v in [0u32, 1, 2, 7, 255, 4096] {
            assert_eq!(r.ue(), Some(v));
        }
        for v in [0i32, 1, -1, 5, -77] {
            assert_eq!(r.se(), Some(v));
        }
        assert!(!r.more_data());
    }

    #[test]
    fn emulation_prevention_is_inserted_and_removed() {
        let mut bw = BitWriter::new();
        bw.u(24, 0x000001);
        bw.u(8, 0xff);
        let nal = bw.into_nal(0x01);
        assert_eq!(&nal[1..6], &[0, 0, 3, 1, 0xff]);
        assert_eq!(&rbsp(&nal)[..4], &[0, 0, 1, 0xff]);
    }

    #[test]
    fn transforms_invert() {
        // A flat residual of 10 survives the forward transform, DC dequantisation at qp 24 and the inverse.
        let x = [10i32; 16];
        let t = forward4x4(&x);
        assert_eq!(t[0], 160);
        assert!(t[1..].iter().all(|&v| v == 0));
        let h = hadamard4x4(&[1; 16]);
        assert_eq!(h[0], 16);
        assert!(h[1..].iter().all(|&v| v == 0));
        let mut d = [0i32; 16];
        d[0] = 640;
        assert_eq!(inverse4x4(&d), [10; 16]);
    }
}
