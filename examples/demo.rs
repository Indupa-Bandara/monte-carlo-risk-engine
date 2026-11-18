//! End-to-end demo: generate correlated scenarios, compute VaR/ES,
//! verify scalar vs parallel agreement, and run a rolling backtest.

use risk_engine::backtest::{christoffersen_independence, kupiec_pof};
use risk_engine::engine::{prepare, EngineBackend, ParallelEngine, ScalarEngine};
use risk_engine::risk::var_es;

fn main() {
    // 1. Three-asset portfolio
    let weights = vec![0.5, 0.3, 0.2];
    let cov = vec![
        0.040, 0.010, 0.005, 0.010, 0.090, 0.020, 0.005, 0.020, 0.160,
    ];
    let l = prepare(&weights, &cov);

    let n = 1_000_000u64;
    let seed = 42u64;

    // 2. Monte Carlo
    let t0 = std::time::Instant::now();
    let mut pnl_seq = ScalarEngine.simulate_pnl(&weights, &l, n, seed);
    let t_seq = t0.elapsed();

    let t0 = std::time::Instant::now();
    let pnl_par = ParallelEngine::new(10_000).simulate_pnl(&weights, &l, n, seed);
    let t_par = t0.elapsed();

    let identical = pnl_seq
        .iter()
        .zip(pnl_par.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());

    // 3. Risk measures
    let (var99, es99) = var_es(&mut pnl_seq, 0.99);
    let (var95, es95) = var_es(&mut pnl_seq, 0.95);

    println!("Scenarios: {n}");
    println!("Seed: {seed}");
    println!("Scalar:   {:?}", t_seq);
    println!("Parallel: {:?}", t_par);
    println!("Bit-identical scalar vs parallel: {identical}");
    println!();
    println!("VaR 99%:  {:>10.4}", var99);
    println!("ES  99%:  {:>10.4}", es99);
    println!("VaR 95%:  {:>10.4}", var95);
    println!("ES  95%:  {:>10.4}", es95);
    println!();

    // 4. Synthetic rolling backtest
    //    Generate 1000 i.i.d. N(0, 0.01^2) returns, then test whether
    //    a 99% VaR model calibrated to sigma = 0.01 has correct coverage.
    use risk_engine::rng::{fill_normals, Philox};
    let mut rng = Philox::from_scenario(1, 0);
    let mut z = vec![0.0f64; 1000];
    fill_normals(&mut rng, &mut z);
    let sigma = 0.01f64;
    let returns: Vec<f64> = z.iter().map(|v| v * sigma).collect();

    // Known sigma -> analytic 99% VaR = 2.3263 * sigma (loss).
    let var_threshold = 2.3263478740408408 * sigma;
    let exceptions: Vec<bool> = returns.iter().map(|r| -r > var_threshold).collect();
    let n_obs = exceptions.len();
    let n_exc = exceptions.iter().filter(|b| **b).count();

    println!("Backtest (analytic VaR):");
    println!("  observations: {n_obs}");
    println!("  exceptions:   {n_exc}");
    let k = kupiec_pof(n_obs, n_exc, 0.01);
    println!(
        "  Kupiec LR:    {:.4}  p={:.4}  rejected={}",
        k.lr, k.p_value, k.rejected
    );
    let c = christoffersen_independence(&exceptions);
    println!(
        "  Christoffersen LR: {:.4}  p={:.4}  rejected={}",
        c.lr, c.p_value, c.rejected
    );

    // 5. Deliberately understate VaR by 2x and show Kupiec rejects.
    let var_bad = var_threshold * 0.5;
    let bad_exceptions: Vec<bool> = returns.iter().map(|r| -r > var_bad).collect();
    let bad_exc = bad_exceptions.iter().filter(|b| **b).count();
    let k_bad = kupiec_pof(n_obs, bad_exc, 0.01);
    println!();
    println!("Backtest (VaR understated 2x):");
    println!("  exceptions:   {bad_exc}");
    println!(
        "  Kupiec LR:    {:.4}  p={:.4}  rejected={}",
        k_bad.lr, k_bad.p_value, k_bad.rejected
    );
}
