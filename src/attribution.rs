use crate::data::{AlignedReturns, DataError, Positions};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct AssetEsContribution {
    pub asset_name: String,
    pub loss_contribution: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalComponentEs {
    pub confidence: f64,
    pub var: f64,
    pub es: f64,
    pub tail_scenarios: usize,
    pub contributions: Vec<AssetEsContribution>,
}

#[derive(Debug, PartialEq)]
pub enum AttributionError {
    InvalidConfidence,
    NonFinitePnl { row: usize, asset: Option<String> },
    Data(DataError),
}

impl fmt::Display for AttributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfidence => {
                write!(f, "confidence must be greater than 0 and less than 1")
            }
            Self::NonFinitePnl { row, asset } => match asset {
                Some(asset) => write!(f, "P&L is not finite for asset {asset} at row {row}"),
                None => write!(f, "portfolio P&L is not finite at row {row}"),
            },
            Self::Data(error) => write!(f, "invalid attribution data: {error}"),
        }
    }
}

impl std::error::Error for AttributionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Data(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DataError> for AttributionError {
    fn from(error: DataError) -> Self {
        Self::Data(error)
    }
}

/// Attribute empirical historical ES to assets in the portfolio loss tail.
///
/// Contributions are signed losses, so hedges may contribute negatively. Their
/// sum equals the reported portfolio ES, up to floating-point rounding.
pub fn historical_component_es(
    returns: &AlignedReturns,
    positions: &Positions,
    confidence: f64,
) -> Result<HistoricalComponentEs, AttributionError> {
    if !confidence.is_finite() || !(0.0..1.0).contains(&confidence) {
        return Err(AttributionError::InvalidConfidence);
    }

    let notionals = positions.align_to(returns)?;
    let n_assets = returns.asset_names().len();
    let mut asset_pnls = Vec::with_capacity(returns.values().len());
    let mut portfolio_pnls = Vec::with_capacity(returns.values().len());
    for (row, observation) in returns.values().iter().enumerate() {
        let mut row_pnls = Vec::with_capacity(n_assets);
        let mut portfolio_pnl = 0.0;
        for (asset, (&notional, &log_return)) in notionals.iter().zip(observation).enumerate() {
            let pnl = notional * (log_return.exp() - 1.0);
            if !pnl.is_finite() {
                return Err(AttributionError::NonFinitePnl {
                    row,
                    asset: Some(returns.asset_names()[asset].clone()),
                });
            }
            row_pnls.push(pnl);
            portfolio_pnl += pnl;
        }
        if !portfolio_pnl.is_finite() {
            return Err(AttributionError::NonFinitePnl { row, asset: None });
        }
        asset_pnls.push(row_pnls);
        portfolio_pnls.push(portfolio_pnl);
    }

    let tail_scenarios = (((1.0 - confidence) * portfolio_pnls.len() as f64).floor() as usize)
        .max(1)
        .min(portfolio_pnls.len());
    let mut scenario_order: Vec<usize> = (0..portfolio_pnls.len()).collect();
    scenario_order.sort_by(|&left, &right| portfolio_pnls[left].total_cmp(&portfolio_pnls[right]));
    let var = -portfolio_pnls[scenario_order[tail_scenarios - 1]];

    let mut contribution_totals = vec![0.0; n_assets];
    let mut tail_pnl = 0.0;
    for &scenario in &scenario_order[..tail_scenarios] {
        tail_pnl += portfolio_pnls[scenario];
        for (asset, pnl) in asset_pnls[scenario].iter().enumerate() {
            contribution_totals[asset] -= pnl;
        }
    }
    let es = -tail_pnl / tail_scenarios as f64;
    let contributions = returns
        .asset_names()
        .iter()
        .zip(contribution_totals)
        .map(|(asset_name, loss_contribution)| AssetEsContribution {
            asset_name: asset_name.clone(),
            loss_contribution: loss_contribution / tail_scenarios as f64,
        })
        .collect();

    Ok(HistoricalComponentEs {
        confidence,
        var,
        es,
        tail_scenarios,
        contributions,
    })
}
