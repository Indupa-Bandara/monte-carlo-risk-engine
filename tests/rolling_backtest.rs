use risk_engine::backtest::{
    rolling_backtest_all_models, rolling_monte_carlo_var_es, RollingBacktestConfig,
    RollingBacktestError,
};
use risk_engine::data::{AlignedReturns, Positions};
use risk_engine::models::RiskModel;

fn make_returns(forecast_log_return: f64) -> AlignedReturns {
    AlignedReturns::new(
        vec!["A".into()],
        vec![
            "2026-01-01".into(),
            "2026-01-02".into(),
            "2026-01-03".into(),
            "2026-01-04".into(),
            "2026-01-05".into(),
        ],
        vec![
            vec![0.01],
            vec![0.0],
            vec![-0.01],
            vec![forecast_log_return],
            vec![0.0],
        ],
    )
    .unwrap()
}

fn config() -> RollingBacktestConfig {
    RollingBacktestConfig {
        window_size: 3,
        n_scenarios: 20_000,
        seed: 42,
        confidence: 0.99,
        chunk_size: 1_000,
    }
}

#[test]
fn forecasts_use_only_prior_observations_and_detect_large_loss() {
    let positions = Positions::new(vec!["A".into()], vec![10_000.0]).unwrap();
    let loss_returns = make_returns(-0.10);
    let profit_returns = make_returns(0.10);

    let loss_backtest = rolling_monte_carlo_var_es(&loss_returns, &positions, config()).unwrap();
    let profit_backtest =
        rolling_monte_carlo_var_es(&profit_returns, &positions, config()).unwrap();
    let loss_forecast = &loss_backtest.forecasts[0];
    let profit_forecast = &profit_backtest.forecasts[0];

    assert_eq!(loss_forecast.date, "2026-01-04");
    assert_eq!(loss_forecast.var, profit_forecast.var);
    assert!(loss_forecast.exception);
    assert!(!profit_forecast.exception);
    assert_eq!(loss_backtest.forecasts.len(), 2);
    assert_eq!(loss_backtest.kupiec.n, 2);
    assert_eq!(
        loss_backtest.kupiec.x,
        loss_backtest
            .forecasts
            .iter()
            .filter(|forecast| forecast.exception)
            .count()
    );
}

#[test]
fn rejects_window_without_out_of_sample_observations() {
    let returns = make_returns(-0.10);
    let positions = Positions::new(vec!["A".into()], vec![10_000.0]).unwrap();
    let mut invalid_config = config();
    invalid_config.window_size = 5;

    assert_eq!(
        rolling_monte_carlo_var_es(&returns, &positions, invalid_config).unwrap_err(),
        RollingBacktestError::InvalidWindow {
            window_size: 5,
            observations: 5,
        }
    );
}

#[test]
fn all_models_produce_rolling_forecasts_from_the_prior_window() {
    let returns = make_returns(-0.10);
    let positions = Positions::new(vec!["A".into()], vec![10_000.0]).unwrap();
    let results = rolling_backtest_all_models(&returns, &positions, config()).unwrap();

    assert_eq!(
        results
            .iter()
            .map(|result| result.model)
            .collect::<Vec<_>>(),
        RiskModel::ALL
    );
    for result in results {
        assert_eq!(result.forecasts.len(), 2);
        assert_eq!(result.forecasts[0].date, "2026-01-04");
        assert_eq!(result.kupiec.n, 2);
    }
}
