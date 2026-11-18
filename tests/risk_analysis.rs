use risk_engine::attribution::historical_component_es;
use risk_engine::data::{AlignedReturns, Positions};
use risk_engine::stress::{historical_stress_replay, hypothetical_stress, StressError};

fn sample_returns() -> AlignedReturns {
    AlignedReturns::new(
        vec!["A".into(), "B".into()],
        vec![
            "2026-01-01".into(),
            "2026-01-02".into(),
            "2026-01-03".into(),
        ],
        vec![
            vec![0.8_f64.ln(), 1.1_f64.ln()],
            vec![0.0, 0.0],
            vec![1.1_f64.ln(), 1.0_f64.ln()],
        ],
    )
    .unwrap()
}

#[test]
fn component_es_adds_to_portfolio_es_and_keeps_hedge_contributions_signed() {
    let returns = sample_returns();
    let positions = Positions::new(vec!["B".into(), "A".into()], vec![100.0, 100.0]).unwrap();

    let result = historical_component_es(&returns, &positions, 0.66).unwrap();
    let contribution_sum = result
        .contributions
        .iter()
        .map(|contribution| contribution.loss_contribution)
        .sum::<f64>();

    assert_eq!(result.tail_scenarios, 1);
    assert_eq!(result.contributions[0].asset_name, "A");
    assert_eq!(result.contributions[1].asset_name, "B");
    assert!((contribution_sum - result.es).abs() < 1e-12);
    assert!((result.contributions[0].loss_contribution - 20.0).abs() < 1e-10);
    assert!((result.contributions[1].loss_contribution + 10.0).abs() < 1e-10);
}

#[test]
fn historical_replay_orders_worst_loss_first() {
    let returns = sample_returns();
    let positions = Positions::new(vec!["A".into(), "B".into()], vec![100.0, 100.0]).unwrap();

    let scenarios = historical_stress_replay(&returns, &positions).unwrap();
    assert_eq!(scenarios[0].date.as_deref(), Some("2026-01-01"));
    assert!((scenarios[0].loss - 10.0).abs() < 1e-12);
    assert!(scenarios
        .windows(2)
        .all(|pair| pair[0].loss >= pair[1].loss));
}

#[test]
fn hypothetical_stress_uses_simple_return_shocks_and_validates_inputs() {
    let returns = sample_returns();
    let positions = Positions::new(vec!["A".into(), "B".into()], vec![100.0, 200.0]).unwrap();

    let scenario =
        hypothetical_stress(&returns, &positions, "market crash", &[-0.2, -0.1]).unwrap();
    assert_eq!(scenario.pnl, -40.0);
    assert_eq!(scenario.loss, 40.0);
    assert_eq!(scenario.asset_impacts[0].pnl, -20.0);
    assert_eq!(scenario.asset_impacts[1].pnl, -20.0);
    assert_eq!(
        hypothetical_stress(&returns, &positions, "bad width", &[-0.2]).unwrap_err(),
        StressError::InvalidShockCount {
            expected: 2,
            actual: 1
        }
    );
    assert!(matches!(
        hypothetical_stress(&returns, &positions, "invalid", &[-1.01, 0.0]),
        Err(StressError::InvalidSimpleReturn { .. })
    ));
}
