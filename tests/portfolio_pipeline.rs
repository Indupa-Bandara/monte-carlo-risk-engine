use risk_engine::data::{AlignedReturns, DataError, Positions};
use risk_engine::engine::{prepare_portfolio, EngineBackend, PrepareError, ScalarEngine};
use risk_engine::risk::var_es;

#[test]
fn prepared_inputs_run_through_monte_carlo() {
    let returns = AlignedReturns::new(
        vec!["A".into(), "B".into()],
        vec![
            "2026-01-01".into(),
            "2026-01-02".into(),
            "2026-01-03".into(),
        ],
        vec![vec![0.01, 0.01], vec![0.0, -0.01], vec![-0.01, 0.0]],
    )
    .unwrap();
    let positions = Positions::new(vec!["B".into(), "A".into()], vec![20_000.0, 10_000.0]).unwrap();

    let prepared = prepare_portfolio(&returns, &positions).unwrap();
    assert_eq!(prepared.notionals, vec![10_000.0, 20_000.0]);

    let mut pnl = ScalarEngine.simulate_pnl(&prepared.notionals, &prepared.cholesky_l, 20_000, 7);
    let (var, es) = var_es(&mut pnl, 0.99);
    assert!(var > 0.0);
    assert!(es >= var);
}

#[test]
fn preparation_reports_singular_sample_covariance() {
    let returns = AlignedReturns::new(
        vec!["A".into()],
        vec!["2026-01-01".into(), "2026-01-02".into()],
        vec![vec![0.01], vec![0.01]],
    )
    .unwrap();
    let positions = Positions::new(vec!["A".into()], vec![10_000.0]).unwrap();

    assert_eq!(
        prepare_portfolio(&returns, &positions).unwrap_err(),
        PrepareError::CovarianceNotPositiveDefinite
    );
}

#[test]
fn preparation_propagates_data_validation_errors() {
    let returns = AlignedReturns::new(
        vec!["A".into()],
        vec!["2026-01-01".into()],
        vec![vec![0.01]],
    )
    .unwrap();
    let positions = Positions::new(vec!["A".into()], vec![10_000.0]).unwrap();

    assert_eq!(
        prepare_portfolio(&returns, &positions).unwrap_err(),
        PrepareError::Data(DataError::InsufficientObservations { actual: 1 })
    );
}
