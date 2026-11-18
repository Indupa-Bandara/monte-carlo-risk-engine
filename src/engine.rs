use crate::data::{AlignedReturns, DataError, Positions};
use crate::linalg::cholesky;
use crate::rng::{fill_normals, Philox};
use rayon::prelude::*;
use std::fmt;

pub trait EngineBackend: Sync {
    /// Generate `n_scenarios` portfolio P&L values.
    ///
    /// `weights` contains the position notionals, length `n_assets`.
    /// Simulated returns are log returns; each asset contributes
    /// `notional * (exp(log_return) - 1)` to portfolio P&L.
    /// `cholesky_l` is the row-major Cholesky factor of the log-return
    /// covariance, length `n_assets * n_assets`.
    fn simulate_pnl(
        &self,
        weights: &[f64],
        cholesky_l: &[f64],
        n_scenarios: u64,
        seed: u64,
    ) -> Vec<f64>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationError {
    ZeroScenarios,
    ZeroChunkSize,
    DegreesOfFreedomMustExceedTwo(u32),
    InvalidDimensions,
}

impl fmt::Display for SimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroScenarios => write!(f, "scenario count must be greater than zero"),
            Self::ZeroChunkSize => write!(f, "chunk size must be greater than zero"),
            Self::DegreesOfFreedomMustExceedTwo(degrees_of_freedom) => write!(
                f,
                "Student-t degrees of freedom must exceed 2, got {degrees_of_freedom}"
            ),
            Self::InvalidDimensions => write!(f, "simulation arrays have incompatible dimensions"),
        }
    }
}

impl std::error::Error for SimulationError {}

/// Apply a row-major lower-triangular Cholesky factor to a flat batch of normal vectors.
pub fn correlate_batch(
    cholesky_l: &[f64],
    normals: &[f64],
    correlated: &mut [f64],
    n_assets: usize,
) -> Result<(), SimulationError> {
    correlate_batch_with(cholesky_l, normals, correlated, n_assets, correlate_one)
}

/// Apply a Cholesky factor to a batch using AVX2 when supported, otherwise scalar code.
pub fn correlate_batch_simd(
    cholesky_l: &[f64],
    normals: &[f64],
    correlated: &mut [f64],
    n_assets: usize,
) -> Result<(), SimulationError> {
    correlate_batch_with(
        cholesky_l,
        normals,
        correlated,
        n_assets,
        simd_correlate_one(),
    )
}

fn correlate_batch_with(
    cholesky_l: &[f64],
    normals: &[f64],
    correlated: &mut [f64],
    n_assets: usize,
    correlate: CorrelateOne,
) -> Result<(), SimulationError> {
    let Some(matrix_len) = n_assets.checked_mul(n_assets) else {
        return Err(SimulationError::InvalidDimensions);
    };
    if n_assets == 0
        || cholesky_l.len() != matrix_len
        || normals.len() != correlated.len()
        || !normals.len().is_multiple_of(n_assets)
    {
        return Err(SimulationError::InvalidDimensions);
    }

    for (normal_vector, correlated_vector) in normals
        .chunks_exact(n_assets)
        .zip(correlated.chunks_exact_mut(n_assets))
    {
        correlate(cholesky_l, normal_vector, correlated_vector);
    }
    Ok(())
}

/// Revalue a flat batch of log-return vectors using simple-return P&L.
pub fn value_pnl_batch(
    weights: &[f64],
    log_returns: &[f64],
    pnl: &mut [f64],
) -> Result<(), SimulationError> {
    let Some(expected_len) = pnl.len().checked_mul(weights.len()) else {
        return Err(SimulationError::InvalidDimensions);
    };
    if weights.is_empty() || log_returns.len() != expected_len {
        return Err(SimulationError::InvalidDimensions);
    }
    for (returns, value) in log_returns.chunks_exact(weights.len()).zip(pnl.iter_mut()) {
        *value = value_pnl_one(weights, returns);
    }
    Ok(())
}

#[inline]
fn correlate_one(cholesky_l: &[f64], normals: &[f64], correlated: &mut [f64]) {
    let n_assets = normals.len();
    for (row, correlated_value) in correlated.iter_mut().enumerate().take(n_assets) {
        let mut acc = 0.0f64;
        let base = row * n_assets;
        for col in 0..=row {
            acc += cholesky_l[base + col] * normals[col];
        }
        *correlated_value = acc;
    }
}

type CorrelateOne = fn(&[f64], &[f64], &mut [f64]);

fn simd_correlate_one() -> CorrelateOne {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return correlate_one_avx2_dispatch;
    }
    correlate_one
}

#[cfg(target_arch = "x86_64")]
fn correlate_one_avx2_dispatch(cholesky_l: &[f64], normals: &[f64], correlated: &mut [f64]) {
    // This function is selected only after runtime AVX2 detection.
    unsafe { correlate_one_avx2(cholesky_l, normals, correlated) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn correlate_one_avx2(cholesky_l: &[f64], normals: &[f64], correlated: &mut [f64]) {
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    };

    let n_assets = normals.len();
    for (row, correlated_value) in correlated.iter_mut().enumerate().take(n_assets) {
        let base = row * n_assets;
        let end = row + 1;
        let mut col = 0;
        let mut sums: __m256d = _mm256_setzero_pd();
        while col + 4 <= end {
            let coefficients = _mm256_loadu_pd(cholesky_l.as_ptr().add(base + col));
            let values = _mm256_loadu_pd(normals.as_ptr().add(col));
            sums = _mm256_add_pd(sums, _mm256_mul_pd(coefficients, values));
            col += 4;
        }

        let mut lanes = [0.0; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), sums);
        let mut acc = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        while col < end {
            acc += cholesky_l[base + col] * normals[col];
            col += 1;
        }
        *correlated_value = acc;
    }
}

#[inline]
fn value_pnl_one(weights: &[f64], log_returns: &[f64]) -> f64 {
    weights
        .iter()
        .zip(log_returns)
        .map(|(weight, log_return)| weight * (log_return.exp() - 1.0))
        .sum()
}

/// Shared inner loop. Generates scenarios `[start, start+n)`.
/// Purely a function of (seed, start, n, inputs), so it is safe to call
/// in parallel across disjoint ranges.
fn simulate_range(
    weights: &[f64],
    cholesky_l: &[f64],
    start: u64,
    n: u64,
    seed: u64,
    student_t_degrees_of_freedom: Option<u32>,
    correlate: CorrelateOne,
) -> Vec<f64> {
    let n_assets = weights.len();
    let mut out = Vec::with_capacity(n as usize);
    let mut z = vec![0.0f64; n_assets];
    let mut r = vec![0.0f64; n_assets];
    for i in 0..n {
        let scenario = start + i;
        let mut rng = Philox::from_scenario(seed, scenario);
        fill_normals(&mut rng, &mut z);
        let scale = if let Some(degrees_of_freedom) = student_t_degrees_of_freedom {
            let mut chi_square = 0.0;
            let mut remaining = degrees_of_freedom;
            let mut chi_square_normals = [0.0; 4];
            while remaining > 0 {
                let count = remaining.min(4) as usize;
                fill_normals(&mut rng, &mut chi_square_normals[..count]);
                chi_square += chi_square_normals[..count]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>();
                remaining -= count as u32;
            }
            // This scale makes the multivariate t draw's covariance match the input covariance.
            ((degrees_of_freedom - 2) as f64 / chi_square).sqrt()
        } else {
            1.0
        };
        if scale == 1.0 {
            correlate(cholesky_l, &z, &mut r);
        } else {
            for value in &mut z[..n_assets] {
                *value *= scale;
            }
            correlate(cholesky_l, &z[..n_assets], &mut r);
        }
        out.push(value_pnl_one(weights, &r));
    }
    out
}

fn simulate_chunked(
    weights: &[f64],
    cholesky_l: &[f64],
    n_scenarios: u64,
    seed: u64,
    chunk_size: u64,
    student_t_degrees_of_freedom: Option<u32>,
    correlate: CorrelateOne,
) -> Vec<f64> {
    let n_chunks = n_scenarios.div_ceil(chunk_size) as usize;
    let chunks: Vec<Vec<f64>> = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_index| {
            let start = (chunk_index as u64) * chunk_size;
            let count = chunk_size.min(n_scenarios - start);
            simulate_range(
                weights,
                cholesky_l,
                start,
                count,
                seed,
                student_t_degrees_of_freedom,
                correlate,
            )
        })
        .collect();
    let mut output = Vec::with_capacity(n_scenarios as usize);
    for chunk in chunks {
        output.extend_from_slice(&chunk);
    }
    output
}

pub struct ScalarEngine;

impl EngineBackend for ScalarEngine {
    fn simulate_pnl(
        &self,
        weights: &[f64],
        cholesky_l: &[f64],
        n_scenarios: u64,
        seed: u64,
    ) -> Vec<f64> {
        simulate_range(
            weights,
            cholesky_l,
            0,
            n_scenarios,
            seed,
            None,
            correlate_one,
        )
    }
}

pub struct SimdEngine;

impl EngineBackend for SimdEngine {
    fn simulate_pnl(
        &self,
        weights: &[f64],
        cholesky_l: &[f64],
        n_scenarios: u64,
        seed: u64,
    ) -> Vec<f64> {
        simulate_range(
            weights,
            cholesky_l,
            0,
            n_scenarios,
            seed,
            None,
            simd_correlate_one(),
        )
    }
}

pub struct ParallelEngine {
    pub chunk_size: u64,
}

impl ParallelEngine {
    pub fn new(chunk_size: u64) -> Self {
        assert!(chunk_size > 0);
        Self { chunk_size }
    }

    pub fn simulate_student_t_pnl(
        &self,
        weights: &[f64],
        cholesky_l: &[f64],
        n_scenarios: u64,
        seed: u64,
        degrees_of_freedom: u32,
    ) -> Result<Vec<f64>, SimulationError> {
        if n_scenarios == 0 {
            return Err(SimulationError::ZeroScenarios);
        }
        if self.chunk_size == 0 {
            return Err(SimulationError::ZeroChunkSize);
        }
        if degrees_of_freedom <= 2 {
            return Err(SimulationError::DegreesOfFreedomMustExceedTwo(
                degrees_of_freedom,
            ));
        }
        Ok(simulate_chunked(
            weights,
            cholesky_l,
            n_scenarios,
            seed,
            self.chunk_size,
            Some(degrees_of_freedom),
            correlate_one,
        ))
    }
}

impl EngineBackend for ParallelEngine {
    fn simulate_pnl(
        &self,
        weights: &[f64],
        cholesky_l: &[f64],
        n_scenarios: u64,
        seed: u64,
    ) -> Vec<f64> {
        simulate_chunked(
            weights,
            cholesky_l,
            n_scenarios,
            seed,
            self.chunk_size,
            None,
            correlate_one,
        )
    }
}

pub struct ParallelSimdEngine {
    pub chunk_size: u64,
}

impl ParallelSimdEngine {
    pub fn new(chunk_size: u64) -> Self {
        assert!(chunk_size > 0);
        Self { chunk_size }
    }
}

impl EngineBackend for ParallelSimdEngine {
    fn simulate_pnl(
        &self,
        weights: &[f64],
        cholesky_l: &[f64],
        n_scenarios: u64,
        seed: u64,
    ) -> Vec<f64> {
        simulate_chunked(
            weights,
            cholesky_l,
            n_scenarios,
            seed,
            self.chunk_size,
            None,
            simd_correlate_one(),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum PrepareError {
    Data(DataError),
    CovarianceNotPositiveDefinite,
}

impl fmt::Display for PrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(error) => write!(f, "invalid portfolio inputs: {error}"),
            Self::CovarianceNotPositiveDefinite => {
                write!(f, "sample covariance is not positive definite")
            }
        }
    }
}

impl std::error::Error for PrepareError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Data(error) => Some(error),
            Self::CovarianceNotPositiveDefinite => None,
        }
    }
}

impl From<DataError> for PrepareError {
    fn from(error: DataError) -> Self {
        Self::Data(error)
    }
}

#[derive(Debug)]
pub struct PreparedPortfolio {
    /// Position notionals ordered to match the returns asset order.
    pub notionals: Vec<f64>,
    /// Row-major Cholesky factor of the sample log-return covariance.
    pub cholesky_l: Vec<f64>,
}

/// Align positions and estimate the covariance needed by the simulation engine.
pub fn prepare_portfolio(
    returns: &AlignedReturns,
    positions: &Positions,
) -> Result<PreparedPortfolio, PrepareError> {
    let notionals = positions.align_to(returns)?;
    let covariance = returns.sample_covariance()?;
    let cholesky_l = cholesky(&covariance, returns.asset_names().len())
        .ok_or(PrepareError::CovarianceNotPositiveDefinite)?;

    Ok(PreparedPortfolio {
        notionals,
        cholesky_l,
    })
}

/// Convenience: build the Cholesky factor from a covariance matrix.
pub fn prepare(weights: &[f64], cov: &[f64]) -> Vec<f64> {
    let n = weights.len();
    cholesky(cov, n).expect("covariance must be SPD")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_kernels_apply_correlation_and_simple_return_valuation() {
        let cholesky_l = [1.0, 0.0, 0.5, 2.0];
        let normals = [1.0, 2.0, 3.0, 4.0];
        let mut correlated = [0.0; 4];
        correlate_batch(&cholesky_l, &normals, &mut correlated, 2).unwrap();
        assert_eq!(correlated, [1.0, 4.5, 3.0, 9.5]);

        let weights = [10.0, 20.0];
        let log_returns = [0.0, 2.0_f64.ln(), 2.0_f64.ln(), 0.0];
        let mut pnl = [0.0; 2];
        value_pnl_batch(&weights, &log_returns, &mut pnl).unwrap();
        assert_eq!(pnl, [20.0, 10.0]);
    }

    #[test]
    fn simd_correlation_matches_scalar_within_rounding_tolerance() {
        let n_assets = 9;
        let cholesky_l: Vec<f64> = (0..n_assets)
            .flat_map(|row| {
                (0..n_assets).map(move |col| {
                    if row == col {
                        1.0 + row as f64 * 0.1
                    } else if col < row {
                        (row as f64 - col as f64) * 0.01
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let normals: Vec<f64> = (0..n_assets * 7)
            .map(|index| (index as f64 - 17.0) * 0.03)
            .collect();
        let mut scalar = vec![0.0; normals.len()];
        let mut simd = vec![0.0; normals.len()];

        correlate_batch(&cholesky_l, &normals, &mut scalar, n_assets).unwrap();
        correlate_batch_simd(&cholesky_l, &normals, &mut simd, n_assets).unwrap();

        for (expected, actual) in scalar.iter().zip(simd) {
            assert!((expected - actual).abs() < 1e-12);
        }
    }

    #[test]
    fn simd_engines_produce_close_scalar_pnl() {
        let n_assets = 9;
        let weights = vec![1_000.0; n_assets];
        let cholesky_l: Vec<f64> = (0..n_assets)
            .flat_map(|row| {
                (0..n_assets).map(move |col| {
                    if row == col {
                        0.002 + row as f64 * 0.0001
                    } else if col < row {
                        (row as f64 - col as f64) * 0.00002
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let expected = ScalarEngine.simulate_pnl(&weights, &cholesky_l, 128, 42);
        let simd = SimdEngine.simulate_pnl(&weights, &cholesky_l, 128, 42);
        let parallel_simd =
            ParallelSimdEngine::new(17).simulate_pnl(&weights, &cholesky_l, 128, 42);

        for actual in [simd, parallel_simd] {
            assert_eq!(actual.len(), expected.len());
            for (expected, actual) in expected.iter().zip(actual) {
                assert!((expected - actual).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn batch_kernels_reject_incompatible_shapes() {
        assert_eq!(
            correlate_batch(&[1.0], &[1.0], &mut [0.0], 2),
            Err(SimulationError::InvalidDimensions)
        );
        assert_eq!(
            value_pnl_batch(&[1.0, 2.0], &[0.0], &mut [0.0]),
            Err(SimulationError::InvalidDimensions)
        );
    }
}
