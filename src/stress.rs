use crate::data::{AlignedReturns, DataError, Positions};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct AssetStressImpact {
    pub asset_name: String,
    pub pnl: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StressScenarioResult {
    pub name: String,
    pub date: Option<String>,
    pub pnl: f64,
    pub loss: f64,
    pub asset_impacts: Vec<AssetStressImpact>,
}

#[derive(Debug, PartialEq)]
pub enum StressError {
    InvalidShockCount { expected: usize, actual: usize },
    InvalidSimpleReturn { asset: String, value: f64 },
    NonFinitePnl { asset: Option<String> },
    Data(DataError),
}

impl fmt::Display for StressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShockCount { expected, actual } => write!(
                f,
                "hypothetical scenario has {actual} shocks; expected {expected}"
            ),
            Self::InvalidSimpleReturn { asset, value } => write!(
                f,
                "simple-return shock for {asset} must be finite and at least -1, got {value}"
            ),
            Self::NonFinitePnl { asset } => match asset {
                Some(asset) => write!(f, "stress P&L is not finite for asset {asset}"),
                None => write!(f, "portfolio stress P&L is not finite"),
            },
            Self::Data(error) => write!(f, "invalid stress-test data: {error}"),
        }
    }
}

impl std::error::Error for StressError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Data(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DataError> for StressError {
    fn from(error: DataError) -> Self {
        Self::Data(error)
    }
}

/// Replay each observed return row as a one-day historical stress scenario.
/// Results are sorted from the largest loss to the largest gain.
pub fn historical_stress_replay(
    returns: &AlignedReturns,
    positions: &Positions,
) -> Result<Vec<StressScenarioResult>, StressError> {
    let notionals = positions.align_to(returns)?;
    let mut results = Vec::with_capacity(returns.values().len());
    for (date, observation) in returns.dates().iter().zip(returns.values()) {
        let asset_returns: Vec<f64> = observation
            .iter()
            .map(|log_return| log_return.exp() - 1.0)
            .collect();
        results.push(evaluate_scenario(
            "historical replay".into(),
            Some(date.clone()),
            returns,
            &notionals,
            &asset_returns,
        )?);
    }
    results.sort_by(|left, right| right.loss.total_cmp(&left.loss));
    Ok(results)
}

/// Evaluate hypothetical simple-return shocks in `returns` asset-column order.
pub fn hypothetical_stress(
    returns: &AlignedReturns,
    positions: &Positions,
    name: impl Into<String>,
    simple_returns: &[f64],
) -> Result<StressScenarioResult, StressError> {
    if simple_returns.len() != returns.asset_names().len() {
        return Err(StressError::InvalidShockCount {
            expected: returns.asset_names().len(),
            actual: simple_returns.len(),
        });
    }
    for (asset_name, &value) in returns.asset_names().iter().zip(simple_returns) {
        if !value.is_finite() || value < -1.0 {
            return Err(StressError::InvalidSimpleReturn {
                asset: asset_name.clone(),
                value,
            });
        }
    }

    let notionals = positions.align_to(returns)?;
    evaluate_scenario(name.into(), None, returns, &notionals, simple_returns)
}

fn evaluate_scenario(
    name: String,
    date: Option<String>,
    returns: &AlignedReturns,
    notionals: &[f64],
    simple_returns: &[f64],
) -> Result<StressScenarioResult, StressError> {
    let mut pnl = 0.0;
    let mut asset_impacts = Vec::with_capacity(notionals.len());
    for ((asset_name, &notional), &simple_return) in returns
        .asset_names()
        .iter()
        .zip(notionals)
        .zip(simple_returns)
    {
        let asset_pnl = notional * simple_return;
        if !asset_pnl.is_finite() {
            return Err(StressError::NonFinitePnl {
                asset: Some(asset_name.clone()),
            });
        }
        pnl += asset_pnl;
        asset_impacts.push(AssetStressImpact {
            asset_name: asset_name.clone(),
            pnl: asset_pnl,
        });
    }
    if !pnl.is_finite() {
        return Err(StressError::NonFinitePnl { asset: None });
    }
    Ok(StressScenarioResult {
        name,
        date,
        pnl,
        loss: -pnl,
        asset_impacts,
    })
}
