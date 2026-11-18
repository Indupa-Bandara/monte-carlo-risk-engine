use crate::engine::EngineBackend;
use rayon::prelude::*;
use std::convert::TryFrom;
use std::mem::size_of;

const SCRATCH_BUDGET_BYTES: usize = 32 * 1024;
const MAX_BLOCK_PATHS: usize = 256;
pub const DEFAULT_CPP_PATH_CHUNK_SIZE: usize = 10_000;

unsafe extern "C" {
    fn risk_engine_cpp_simulate_block(
        weights: *const f64,
        cholesky_l: *const f64,
        n_assets: usize,
        seed: u64,
        first_path: u64,
        path_count: usize,
        scratch_capacity: usize,
        shocks: *mut f32,
        correlated: *mut f64,
        pnl: *mut f64,
    ) -> i32;

    fn risk_engine_cpp_uses_avx2() -> i32;
}

pub fn cpp_uses_avx2() -> bool {
    unsafe { risk_engine_cpp_uses_avx2() == 1 }
}

pub struct CppEngine;

impl EngineBackend for CppEngine {
    fn simulate_pnl(
        &self,
        weights: &[f64],
        cholesky_l: &[f64],
        n_scenarios: u64,
        seed: u64,
    ) -> Vec<f64> {
        let output_len =
            usize::try_from(n_scenarios).expect("scenario count exceeds addressable memory");
        let mut output = vec![0.0; output_len];
        if !output.is_empty() {
            simulate_paths(weights, cholesky_l, seed, 0, &mut output)
                .unwrap_or_else(|status| panic!("C++ simulation failed with status {status}"));
        }
        output
    }
}

pub struct ParallelCppEngine {
    chunk_size: usize,
}

impl ParallelCppEngine {
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size > 0);
        Self { chunk_size }
    }
}

impl Default for ParallelCppEngine {
    fn default() -> Self {
        Self::new(DEFAULT_CPP_PATH_CHUNK_SIZE)
    }
}

impl EngineBackend for ParallelCppEngine {
    fn simulate_pnl(
        &self,
        weights: &[f64],
        cholesky_l: &[f64],
        n_scenarios: u64,
        seed: u64,
    ) -> Vec<f64> {
        let output_len =
            usize::try_from(n_scenarios).expect("scenario count exceeds addressable memory");
        let mut output = vec![0.0; output_len];
        if output.is_empty() {
            return output;
        }

        output
            .par_chunks_mut(self.chunk_size)
            .enumerate()
            .try_for_each(|(chunk_index, chunk)| {
                let first_path =
                    u64::try_from(chunk_index * self.chunk_size).expect("path index exceeds u64");
                simulate_paths(weights, cholesky_l, seed, first_path, chunk)
            })
            .unwrap_or_else(|status| panic!("C++ simulation failed with status {status}"));
        output
    }
}

fn simulate_paths(
    weights: &[f64],
    cholesky_l: &[f64],
    seed: u64,
    first_path: u64,
    output: &mut [f64],
) -> Result<(), i32> {
    let n_assets = weights.len();
    let matrix_len = n_assets
        .checked_mul(n_assets)
        .expect("asset matrix dimensions overflow usize");
    assert!(n_assets > 0, "C++ simulation requires at least one asset");
    assert_eq!(
        cholesky_l.len(),
        matrix_len,
        "Cholesky matrix dimensions do not match the position count"
    );

    let paths_per_block = block_path_count(n_assets);
    let scratch_len = paths_per_block
        .checked_mul(n_assets)
        .expect("simulation scratch dimensions overflow usize");
    let mut shocks = vec![0.0_f32; scratch_len];
    let mut correlated = vec![0.0_f64; scratch_len];
    let mut completed = 0;

    while completed < output.len() {
        let path_count = paths_per_block.min(output.len() - completed);
        let first = first_path
            .checked_add(u64::try_from(completed).expect("path index exceeds u64"))
            .expect("path index overflows u64");
        let status = unsafe {
            risk_engine_cpp_simulate_block(
                weights.as_ptr(),
                cholesky_l.as_ptr(),
                n_assets,
                seed,
                first,
                path_count,
                paths_per_block,
                shocks.as_mut_ptr(),
                correlated.as_mut_ptr(),
                output.as_mut_ptr().add(completed),
            )
        };
        if status != 0 {
            return Err(status);
        }
        completed += path_count;
    }
    Ok(())
}

fn block_path_count(n_assets: usize) -> usize {
    let bytes_per_path = n_assets
        .checked_mul(size_of::<f32>() + size_of::<f64>())
        .expect("simulation scratch dimensions overflow usize");
    (SCRATCH_BUDGET_BYTES / bytes_per_path.max(1)).clamp(1, MAX_BLOCK_PATHS)
}
