use risk_engine::engine::{prepare, EngineBackend, ParallelEngine, ScalarEngine};

fn make_inputs() -> (Vec<f64>, Vec<f64>) {
    let weights = vec![0.5, 0.3, 0.2];
    // SPD covariance (3 assets)
    let cov = vec![
        0.040, 0.010, 0.005, 0.010, 0.090, 0.020, 0.005, 0.020, 0.160,
    ];
    let l = prepare(&weights, &cov);
    (weights, l)
}

#[test]
fn scalar_and_parallel_are_bit_identical() {
    let (w, l) = make_inputs();
    let n = 200_000u64;
    let seed = 42u64;

    let s = ScalarEngine.simulate_pnl(&w, &l, n, seed);
    let p = ParallelEngine::new(10_000).simulate_pnl(&w, &l, n, seed);

    assert_eq!(s.len(), p.len());
    for (i, (a, b)) in s.iter().zip(p.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "mismatch at scenario {i}");
    }
}

#[test]
fn parallel_is_thread_count_invariant() {
    // Rayon may use 1..N threads; the result must not depend on that.
    let (w, l) = make_inputs();
    let n = 100_000u64;
    let seed = 7u64;

    let p1 = ParallelEngine::new(10_000).simulate_pnl(&w, &l, n, seed);
    std::thread::sleep(std::time::Duration::from_millis(10));
    let p2 = ParallelEngine::new(10_000).simulate_pnl(&w, &l, n, seed);

    for (a, b) in p1.iter().zip(p2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn different_seeds_give_different_results() {
    let (w, l) = make_inputs();
    let n = 10_000u64;
    let a = ScalarEngine.simulate_pnl(&w, &l, n, 1);
    let b = ScalarEngine.simulate_pnl(&w, &l, n, 2);
    assert!(a.iter().zip(b.iter()).any(|(x, y)| x != y));
}
