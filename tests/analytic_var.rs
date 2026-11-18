use risk_engine::engine::{prepare, EngineBackend, ScalarEngine};
use risk_engine::risk::var_es_with_ci;

/// A unit long position with normally distributed log returns and sigma = 1.
/// Its simple return is lognormal, so VaR and ES must match those analytic
/// simple-return values rather than the linear normal-return approximation.
#[test]
fn single_asset_matches_lognormal_simple_return_var_es() {
    let sigma = 1.0f64;
    let weights = vec![1.0];
    let cov = vec![sigma * sigma];
    let l = prepare(&weights, &cov);

    let n = 1_000_000u64;
    let mut pnl = ScalarEngine.simulate_pnl(&weights, &l, n, 123);
    let estimate = var_es_with_ci(&mut pnl, 0.99);

    let analytic_var = 1.0 - (-2.3263478740408408 * sigma).exp();
    // ES = 1 - exp(sigma^2 / 2) * Phi(-z - sigma) / (1 - alpha)
    let analytic_es = 0.9274628292191839;

    assert!(
        (estimate.var - analytic_var).abs() < 0.002,
        "VaR: mc={} analytic={analytic_var}",
        estimate.var
    );
    assert!(
        (estimate.es - analytic_es).abs() < 0.002,
        "ES: mc={} analytic={analytic_es}",
        estimate.es
    );
    let var_ci = estimate.var_ci_95.unwrap();
    assert!(var_ci.lower <= analytic_var && analytic_var <= var_ci.upper);
    let es_ci = estimate.es_ci_95.unwrap();
    assert!(es_ci.lower <= analytic_es && analytic_es <= es_ci.upper);
}

#[test]
fn omits_confidence_intervals_for_too_few_tail_scenarios() {
    let mut pnl: Vec<f64> = (0..100).map(|value| value as f64).collect();
    let estimate = var_es_with_ci(&mut pnl, 0.99);

    assert!(estimate.var_ci_95.is_none());
    assert!(estimate.es_ci_95.is_none());
}

/// Two-asset portfolio: verify that generated returns reconstruct the
/// target covariance to within Monte Carlo noise.
#[test]
fn reconstructed_covariance_matches_target() {
    use risk_engine::rng::{fill_normals, Philox};
    let sigma = [0.05, 0.10];
    let rho = 0.4;
    let c00 = sigma[0] * sigma[0];
    let c11 = sigma[1] * sigma[1];
    let c01 = rho * sigma[0] * sigma[1];
    let cov = vec![c00, c01, c01, c11];
    let l = risk_engine::linalg::cholesky(&cov, 2).unwrap();

    let n = 500_000u64;
    let mut s00 = 0.0f64;
    let mut s11 = 0.0f64;
    let mut s01 = 0.0f64;
    let mut z = [0.0f64; 2];
    let mut r = [0.0f64; 2];
    for i in 0..n {
        let mut rng = Philox::from_scenario(999, i);
        fill_normals(&mut rng, &mut z);
        r[0] = l[0] * z[0];
        r[1] = l[2] * z[0] + l[3] * z[1];
        s00 += r[0] * r[0];
        s11 += r[1] * r[1];
        s01 += r[0] * r[1];
    }
    let nf = n as f64;
    assert!((s00 / nf - c00).abs() < 1e-4);
    assert!((s11 / nf - c11).abs() < 4e-4);
    assert!((s01 / nf - c01).abs() < 4e-4);
}
