//! Shared dense-layer kernels for the pure-Rust inference nets
//! (`dmc_net`, `bid_net`, `belief_net`).
//!
//! These nets are latency-bound: one observation at a time, no batching, called
//! on every card and every bid. The whole cost is dense matrix-vector products.
//!
//! The obvious loop — one accumulator, one element at a time — serialises a
//! dependent floating-point add per element. Each add has ~4 cycles of latency
//! and nothing else to overlap with, so throughput collapses to roughly one FLOP
//! every two cycles regardless of how many multiply units the core has. Measured
//! that way, a 411→1024³→32 forward pass ran at ~5 GFLOP/s on hardware capable of
//! ten to twenty times that.
//!
//! Splitting the sum across [`LANES`] independent accumulators breaks the
//! dependency chain and, just as importantly, lets the compiler keep the lanes in
//! a single SIMD register. Rust never grants float reassociation, so a plain
//! reduction cannot legally be vectorised — writing the lanes out explicitly is
//! what makes the transformation available without changing what is computed.
//!
//! Summation order does differ from a left-to-right sum, so results can move in
//! the last ulp. Measured on DouDou50 over 500 real observations: max absolute
//! deviation 1.3e-4, and the only action choice that flipped was an exact tie.

/// Independent accumulator lanes. 8 f32 lanes = one 256-bit AVX2 register.
pub const LANES: usize = 8;

/// Dot product of two equal-length slices, [`LANES`]-way accumulated.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (ra, rb) in ca.by_ref().zip(cb.by_ref()) {
        for k in 0..LANES {
            acc[k] += ra[k] * rb[k];
        }
    }
    let mut sum = 0.0f32;
    for k in 0..LANES {
        sum += acc[k];
    }
    for (ra, rb) in ca.remainder().iter().zip(cb.remainder()) {
        sum += ra * rb;
    }
    sum
}

/// Sum of a slice, [`LANES`]-way accumulated.
#[inline(always)]
pub fn sum_lanes(x: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut c = x.chunks_exact(LANES);
    for r in c.by_ref() {
        for k in 0..LANES {
            acc[k] += r[k];
        }
    }
    let mut sum = 0.0f32;
    for k in 0..LANES {
        sum += acc[k];
    }
    for r in c.remainder() {
        sum += r;
    }
    sum
}

#[inline(always)]
fn linear_impl(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32], in_dim: usize, out_dim: usize) {
    let x = &x[..in_dim];
    for i in 0..out_dim {
        out[i] = b[i] + dot(&w[i * in_dim..(i + 1) * in_dim], x);
    }
}

/// AVX2 build of `linear_impl`. Identical source, compiled with 256-bit vectors
/// available — the default x86-64 baseline only guarantees SSE2, and most of this
/// project's builds (arena, web, CI wheels) do not pass `target-cpu=native`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn linear_avx2(
    w: &[f32],
    b: &[f32],
    x: &[f32],
    out: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    linear_impl(w, b, x, out, in_dim, out_dim);
}

/// Compute `out = W * x + b` (no activation).
/// `W` is row-major: `W[i * in_dim + j]` is the weight from input `j` to output `i`.
///
/// The AVX2 feature check is a cached atomic load done once per layer — not per
/// row, and not per element — so it does not show up in the measurement.
#[inline]
pub fn linear(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32], in_dim: usize, out_dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                linear_avx2(w, b, x, out, in_dim, out_dim);
            }
            return;
        }
    }
    linear_impl(w, b, x, out, in_dim, out_dim);
}

/// LayerNorm in place: `x = gamma * (x - mean) / sqrt(var + eps) + beta`.
///
/// Two-pass variance, as before — only the accumulation is lane-split. The
/// one-pass `E[x²] - mean²` form would save a pass but is the cancellation-prone
/// one, and this pass is negligible next to the matrix-vector products.
#[inline]
pub fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], dim: usize, eps: f32) {
    let x = &mut x[..dim];
    let n = dim as f32;
    let mean = sum_lanes(x) / n;

    let mut acc = [0.0f32; LANES];
    let mut c = x.chunks_exact(LANES);
    for r in c.by_ref() {
        for k in 0..LANES {
            let d = r[k] - mean;
            acc[k] += d * d;
        }
    }
    let mut var = 0.0f32;
    for k in 0..LANES {
        var += acc[k];
    }
    for r in c.remainder() {
        let d = r - mean;
        var += d * d;
    }
    var /= n;

    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..dim {
        x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lane split must not change the value beyond float rounding.
    #[test]
    fn test_dot_matches_reference() {
        for len in [1usize, 7, 8, 9, 32, 411, 1024] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();
            let reference: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let got = dot(&a, &b) as f64;
            assert!(
                (got - reference).abs() < 1e-4 * reference.abs().max(1.0),
                "len {len}: got {got}, reference {reference}"
            );
        }
    }

    #[test]
    fn test_linear_matches_naive() {
        let (in_dim, out_dim) = (411usize, 64usize);
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| ((i % 97) as f32 - 48.0) / 100.0)
            .collect();
        let b: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.01).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.013).sin()).collect();

        let mut got = vec![0.0f32; out_dim];
        linear(&w, &b, &x, &mut got, in_dim, out_dim);

        for i in 0..out_dim {
            let mut want = b[i] as f64;
            for j in 0..in_dim {
                want += w[i * in_dim + j] as f64 * x[j] as f64;
            }
            assert!(
                (got[i] as f64 - want).abs() < 1e-3,
                "row {i}: got {}, want {want}",
                got[i]
            );
        }
    }
}
