use rayon::ThreadPoolBuilder;
use risk_engine::cpp_engine::{CppEngine, ParallelCppEngine};
use risk_engine::engine::{EngineBackend, ScalarEngine};

fn test_inputs(n_assets: usize) -> (Vec<f64>, Vec<f64>) {
    let weights = vec![750.0; n_assets];
    let mut cholesky_l = vec![0.0; n_assets * n_assets];
    for row in 0..n_assets {
        for col in 0..=row {
            cholesky_l[row * n_assets + col] = if row == col {
                0.002 + row as f64 * 0.00005
            } else {
                (row as f64 - col as f64) * 0.00001
            };
        }
    }
    (weights, cholesky_l)
}

#[test]
fn cpp_backend_matches_rust_scalar_within_documented_tolerance() {
    let (weights, cholesky_l) = test_inputs(50);
    let expected = ScalarEngine.simulate_pnl(&weights, &cholesky_l, 257, 42);
    let actual = CppEngine.simulate_pnl(&weights, &cholesky_l, 257, 42);

    for (reference, cpp_value) in expected.iter().zip(actual) {
        let tolerance = 1e-4 + reference.abs() * 1e-6;
        assert!(
            (reference - cpp_value).abs() <= tolerance,
            "Rust={reference}, C++={cpp_value}, tolerance={tolerance}"
        );
    }
}

#[test]
fn cpp_backend_is_path_deterministic_across_thread_and_chunk_counts() {
    let (weights, cholesky_l) = test_inputs(13);
    let serial = CppEngine.simulate_pnl(&weights, &cholesky_l, 509, 71);
    let single_thread_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let four_thread_pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    let one_thread = single_thread_pool
        .install(|| ParallelCppEngine::new(53).simulate_pnl(&weights, &cholesky_l, 509, 71));
    let four_threads = four_thread_pool
        .install(|| ParallelCppEngine::new(127).simulate_pnl(&weights, &cholesky_l, 509, 71));

    assert_eq!(serial, one_thread);
    assert_eq!(one_thread, four_threads);
}
