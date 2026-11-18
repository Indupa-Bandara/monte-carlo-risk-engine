//! Coverage backtests for VaR models. All tests use the asymptotic
//! chi-squared distribution: df = 1 for POF and independence,
//! df = 2 for conditional coverage.

use crate::data::{AlignedReturns, DataError, Positions};
use crate::models::{estimate_var_es, ModelError, RiskModel};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct KupiecResult {
    pub n: usize,
    pub x: usize,
    pub p: f64,         // expected exception rate
    pub pi: f64,        // observed
    pub lr: f64,        // likelihood-ratio statistic
    pub p_value: f64,   // chi2(1) survival function at lr
    pub rejected: bool, // at 5%
}

#[derive(Debug, Clone, Copy)]
pub struct RollingBacktestConfig {
    pub window_size: usize,
    pub n_scenarios: u64,
    pub seed: u64,
    pub confidence: f64,
    pub chunk_size: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RollingForecast {
    pub date: String,
    pub var: f64,
    pub es: f64,
    pub realized_loss: f64,
    pub exception: bool,
}

#[derive(Debug)]
pub struct RollingBacktestResult {
    pub model: RiskModel,
    pub forecasts: Vec<RollingForecast>,
    pub kupiec: KupiecResult,
    pub christoffersen: ChristoffersenResult,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RollingBacktestError {
    InvalidWindow {
        window_size: usize,
        observations: usize,
    },
    ZeroScenarios,
    ZeroChunkSize,
    InvalidConfidence,
    CovarianceNotPositiveDefinite {
        date: String,
    },
    Model {
        date: String,
        error: ModelError,
    },
    NonFiniteRealizedPnl {
        date: String,
    },
    Data(DataError),
}

impl fmt::Display for RollingBacktestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWindow {
                window_size,
                observations,
            } => write!(
                f,
                "window size {window_size} must be at least 2 and less than {observations}"
            ),
            Self::ZeroScenarios => write!(f, "scenario count must be greater than zero"),
            Self::ZeroChunkSize => write!(f, "chunk size must be greater than zero"),
            Self::InvalidConfidence => {
                write!(f, "confidence must be greater than 0 and less than 1")
            }
            Self::CovarianceNotPositiveDefinite { date } => {
                write!(f, "model covariance is not positive definite before {date}")
            }
            Self::Model { date, error } => {
                write!(f, "could not estimate risk before {date}: {error}")
            }
            Self::NonFiniteRealizedPnl { date } => {
                write!(f, "realized P&L is not finite for {date}")
            }
            Self::Data(error) => write!(f, "invalid backtest data: {error}"),
        }
    }
}

impl std::error::Error for RollingBacktestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Data(error) => Some(error),
            Self::Model { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl From<DataError> for RollingBacktestError {
    fn from(error: DataError) -> Self {
        Self::Data(error)
    }
}

/// Run rolling normal Monte Carlo forecasts using only prior observations.
pub fn rolling_monte_carlo_var_es(
    returns: &AlignedReturns,
    positions: &Positions,
    config: RollingBacktestConfig,
) -> Result<RollingBacktestResult, RollingBacktestError> {
    rolling_model_backtest(returns, positions, config, RiskModel::MonteCarloNormal)
}

pub fn rolling_model_backtest(
    returns: &AlignedReturns,
    positions: &Positions,
    config: RollingBacktestConfig,
    model: RiskModel,
) -> Result<RollingBacktestResult, RollingBacktestError> {
    let n_observations = returns.values().len();
    if config.window_size < 2 || config.window_size >= n_observations {
        return Err(RollingBacktestError::InvalidWindow {
            window_size: config.window_size,
            observations: n_observations,
        });
    }
    if config.n_scenarios == 0 {
        return Err(RollingBacktestError::ZeroScenarios);
    }
    if config.chunk_size == 0 {
        return Err(RollingBacktestError::ZeroChunkSize);
    }
    if !config.confidence.is_finite() || !(0.0..1.0).contains(&config.confidence) {
        return Err(RollingBacktestError::InvalidConfidence);
    }

    let notionals = positions.align_to(returns)?;
    let mut forecasts = Vec::with_capacity(n_observations - config.window_size);
    for forecast_index in config.window_size..n_observations {
        let window_start = forecast_index - config.window_size;
        let date = returns.dates()[forecast_index].clone();
        let estimate = estimate_var_es(
            model,
            &returns.values()[window_start..forecast_index],
            &notionals,
            config.confidence,
            config.n_scenarios,
            config.seed.wrapping_add(forecast_index as u64),
            config.chunk_size,
        )
        .map_err(|error| match error {
            ModelError::Data(data_error) => RollingBacktestError::Data(data_error),
            ModelError::CovarianceNotPositiveDefinite => {
                RollingBacktestError::CovarianceNotPositiveDefinite { date: date.clone() }
            }
            error => RollingBacktestError::Model {
                date: date.clone(),
                error,
            },
        })?;
        let realized_pnl = notionals
            .iter()
            .zip(&returns.values()[forecast_index])
            .map(|(notional, log_return)| notional * (log_return.exp() - 1.0))
            .sum::<f64>();
        if !realized_pnl.is_finite() {
            return Err(RollingBacktestError::NonFiniteRealizedPnl { date });
        }
        let realized_loss = -realized_pnl;
        forecasts.push(RollingForecast {
            date,
            var: estimate.var,
            es: estimate.es,
            realized_loss,
            exception: realized_loss > estimate.var,
        });
    }

    let exceptions: Vec<bool> = forecasts
        .iter()
        .map(|forecast| forecast.exception)
        .collect();
    let exception_count = exceptions.iter().filter(|exception| **exception).count();
    let kupiec = kupiec_pof(forecasts.len(), exception_count, 1.0 - config.confidence);
    let christoffersen = christoffersen_independence(&exceptions);
    Ok(RollingBacktestResult {
        model,
        forecasts,
        kupiec,
        christoffersen,
    })
}

pub fn rolling_backtest_all_models(
    returns: &AlignedReturns,
    positions: &Positions,
    config: RollingBacktestConfig,
) -> Result<Vec<RollingBacktestResult>, RollingBacktestError> {
    RiskModel::ALL
        .into_iter()
        .map(|model| rolling_model_backtest(returns, positions, config, model))
        .collect()
}

pub fn kupiec_pof(n: usize, x: usize, p: f64) -> KupiecResult {
    assert!(n > 0 && x <= n);
    let pi = x as f64 / n as f64;
    let lr = if x == 0 {
        -2.0 * (n as f64) * (1.0 - p).ln()
    } else if x == n {
        -2.0 * (n as f64) * p.ln()
    } else {
        let ll0 = (n - x) as f64 * (1.0 - p).ln() + (x as f64) * p.ln();
        let ll1 = (n - x) as f64 * (1.0 - pi).ln() + (x as f64) * pi.ln();
        -2.0 * (ll0 - ll1)
    };
    let p_value = chi2_1_sf(lr);
    KupiecResult {
        n,
        x,
        p,
        pi,
        lr,
        p_value,
        rejected: p_value < 0.05,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChristoffersenResult {
    pub n00: usize,
    pub n01: usize,
    pub n10: usize,
    pub n11: usize,
    pub pi01: f64,
    pub pi11: f64,
    pub pi: f64,
    pub lr: f64,
    pub p_value: f64,
    pub rejected: bool,
}

pub fn christoffersen_independence(exceptions: &[bool]) -> ChristoffersenResult {
    let mut n00 = 0usize;
    let mut n01 = 0usize;
    let mut n10 = 0usize;
    let mut n11 = 0usize;
    for w in exceptions.windows(2) {
        match (w[0], w[1]) {
            (false, false) => n00 += 1,
            (false, true) => n01 += 1,
            (true, false) => n10 += 1,
            (true, true) => n11 += 1,
        }
    }
    let n0 = n00 + n01;
    let n1 = n10 + n11;
    let pi01 = if n0 > 0 { n01 as f64 / n0 as f64 } else { 0.0 };
    let pi11 = if n1 > 0 { n11 as f64 / n1 as f64 } else { 0.0 };
    let total = n0 + n1;
    let pi = if total > 0 {
        (n01 + n11) as f64 / total as f64
    } else {
        0.0
    };
    let ll_null = ln_bern(n00 + n10, n01 + n11, 1.0 - pi, pi);
    let ll_alt = ln_bern(n00, n01, 1.0 - pi01, pi01) + ln_bern(n10, n11, 1.0 - pi11, pi11);
    let lr = -2.0 * (ll_null - ll_alt);
    let p_value = chi2_1_sf(lr);
    ChristoffersenResult {
        n00,
        n01,
        n10,
        n11,
        pi01,
        pi11,
        pi,
        lr,
        p_value,
        rejected: p_value < 0.05,
    }
}

#[inline]
fn ln_bern(k0: usize, k1: usize, p0: f64, p1: f64) -> f64 {
    let mut s = 0.0f64;
    if k0 > 0 {
        s += (k0 as f64) * p0.ln();
    }
    if k1 > 0 {
        s += (k1 as f64) * p1.ln();
    }
    s
}

pub fn chi2_1_sf(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    erfc((x / 2.0).sqrt())
}

pub fn chi2_2_sf(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    (-x / 2.0).exp()
}

/// Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly =
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592;
    sign * (1.0 - poly * t * (-x * x).exp())
}

fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}
