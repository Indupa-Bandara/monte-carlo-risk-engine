//! Philox4x32-10 counter-based RNG (Salmon et al., SC'11).
//! Designed for reproducibility: same (seed, counter) -> same output
//! regardless of thread count or backend.

const M0: u32 = 0xD2511F53;
const M1: u32 = 0xCD9E8D57;
const W0: u32 = 0x9E3779B9;
const W1: u32 = 0xBB67AE85;

#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64) * (b as u64);
    ((p >> 32) as u32, p as u32)
}

#[inline]
fn round(ctr: &mut [u32; 4], key: &mut [u32; 2]) {
    let (hi0, lo0) = mulhilo(M0, ctr[0]);
    let (hi1, lo1) = mulhilo(M1, ctr[2]);
    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
    key[0] = key[0].wrapping_add(W0);
    key[1] = key[1].wrapping_add(W1);
}

/// Philox stream. Counter layout:
///   ctr[0..2] = scenario index (u64, little-endian u32 pair)
///   ctr[2..4] = draw index within scenario (advances per next_u32x4 call)
///
/// This means scenario `s` and scenario `s'` never collide as long as s != s',
/// because their scenario-index halves differ.
pub struct Philox {
    key: [u32; 2],
    ctr: [u32; 4],
}

impl Philox {
    pub fn from_scenario(seed: u64, scenario: u64) -> Self {
        Self {
            key: [seed as u32, (seed >> 32) as u32],
            ctr: [scenario as u32, (scenario >> 32) as u32, 0, 0],
        }
    }

    pub fn next_u32x4(&mut self) -> [u32; 4] {
        let mut ctr = self.ctr;
        let mut key = self.key;
        for _ in 0..10 {
            round(&mut ctr, &mut key);
        }
        // Advance the "draw index within scenario": ctr[2] with carry into ctr[3].
        self.ctr[2] = self.ctr[2].wrapping_add(1);
        if self.ctr[2] == 0 {
            self.ctr[3] = self.ctr[3].wrapping_add(1);
        }
        ctr
    }
}

#[inline]
fn u32_to_open01(v: u32) -> f64 {
    // (v + 0.5) / 2^32 keeps us strictly inside (0, 1), avoiding log(0).
    (v as f64 + 0.5) / (u32::MAX as f64 + 1.0)
}

/// Fill `out` with i.i.d. standard normal samples via Box-Muller.
/// Four uniforms per Philox call -> two independent normal pairs -> four normals.
pub fn fill_normals(rng: &mut Philox, out: &mut [f64]) {
    let mut i = 0;
    while i + 3 < out.len() {
        let v = rng.next_u32x4();
        for k in 0..2 {
            let a = u32_to_open01(v[2 * k]);
            let b = u32_to_open01(v[2 * k + 1]);
            let r = (-2.0 * a.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * b;
            out[i + 2 * k] = r * theta.cos();
            out[i + 2 * k + 1] = r * theta.sin();
        }
        i += 4;
    }
    while i < out.len() {
        let v = rng.next_u32x4();
        let a = u32_to_open01(v[0]);
        let b = u32_to_open01(v[1]);
        let r = (-2.0 * a.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * b;
        out[i] = r * theta.cos();
        i += 1;
    }
}
