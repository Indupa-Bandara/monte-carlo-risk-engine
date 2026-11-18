use risk_engine::data::{AlignedReturns, DataError};

#[test]
fn sample_covariance_matches_known_matrix() {
    let returns = AlignedReturns::new(
        vec!["A".into(), "B".into()],
        vec![
            "2026-01-01".into(),
            "2026-01-02".into(),
            "2026-01-03".into(),
        ],
        vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 2.0]],
    )
    .unwrap();

    let covariance = returns.sample_covariance().unwrap();
    assert_eq!(covariance, vec![1.0, 0.5, 0.5, 1.0]);
    assert!(risk_engine::linalg::cholesky(&covariance, 2).is_some());
}

#[test]
fn sample_covariance_requires_at_least_two_observations() {
    let returns = AlignedReturns::new(
        vec!["A".into()],
        vec!["2026-01-01".into()],
        vec![vec![0.01]],
    )
    .unwrap();

    assert_eq!(
        returns.sample_covariance().unwrap_err(),
        DataError::InsufficientObservations { actual: 1 }
    );
}
