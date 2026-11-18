use crate::data::DataError;
use crate::engine::{EngineBackend, ParallelEngine, SimulationError};
use crate::linalg::cholesky;
use crate::risk::{var_es, var_es_with_ci, RiskEstimate};
use std::fmt;

pub const EWMA_LAMBDA: f64 = 0.94;
pub const STUDENT_T_DEGREES_OF_FREEDOM: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskModel {
    HistoricalSimulation,
    EwmaNormal,
    MonteCarloNormal,
    MonteCarloStudentT,
}

impl RiskModel {
    pub const ALL: [Self; 4] = [
        Self::HistoricalSimulation,
        Self::EwmaNormal,
        Self::MonteCarloNormal,
        Self::MonteCarloStudentT,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::HistoricalSimulation => "Historical simulation",
            Self::EwmaNormal => "EWMA normal (lambda=0.94)",
            Self::MonteCarloNormal => "Normal Monte Carlo (Ledoit-Wolf diagonal)",
            Self::MonteCarloStudentT => "Student-t Monte Carlo (nu=5)",
        }
    }

    fn uses_monte_carlo(self) -> bool {
        matches!(self, Self::MonteCarloNormal | Self::MonteCarloStudentT)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ModelError {
    InvalidConfidence,
    EmptyObservations,
    InvalidDimensions,
    NonFinitePnl,
    CovarianceNotPositiveDefinite,
    InvalidCovariance,
    Simulation(SimulationError),
    Data(DataError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfidence => {
                write!(f, "confidence must be greater than 0 and less than 1")
            }
            Self::EmptyObservations => write!(f, "at least two observations are required"),
            Self::InvalidDimensions => {
                write!(
                    f,
                    "return observations and position notionals have incompatible sizes"
                )
            }
            Self::NonFinitePnl => write!(f, "portfolio P&L contains a non-finite value"),
            Self::CovarianceNotPositiveDefinite => {
                write!(f, "estimated covariance is not positive definite")
            }
            Self::InvalidCovariance => {
                write!(f, "estimated portfolio variance is invalid")
            }
            Self::Simulation(error) => write!(f, "Monte Carlo simulation failed: {error}"),
            Self::Data(error) => write!(f, "invalid model input data: {error}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Simulation(error) => Some(error),
            Self::Data(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SimulationError> for ModelError {
    fn from(error: SimulationError) -> Self {
        Self::Simulation(error)
    }
}

impl From<DataError> for ModelError {
    fn from(error: DataError) -> Self {
        Self::Data(error)
    }
}

pub fn estimate_var_es(
    model: RiskModel,
    observations: &[Vec<f64>],
    notionals: &[f64],
    confidence: f64,
    n_scenarios: u64,
    seed: u64,
    chunk_size: u64,
) -> Result<RiskEstimate, ModelError> {
    if !confidence.is_finite() || !(0.0..1.0).contains(&confidence) {
        return Err(ModelError::InvalidConfidence);
    }
    if observations.len() < 2 {
        return Err(ModelError::EmptyObservations);
    }
    if notionals.is_empty()
        || observations
            .iter()
            .any(|observation| observation.len() != notionals.len())
    {
        return Err(ModelError::InvalidDimensions);
    }
    for (index, notional) in notionals.iter().enumerate() {
        if !notional.is_finite() {
            return Err(ModelError::Data(DataError::NonFiniteNotional { index }));
        }
    }
    for (row, observation) in observations.iter().enumerate() {
        for (asset, value) in observation.iter().enumerate() {
            if !value.is_finite() {
                return Err(ModelError::Data(DataError::NonFiniteReturn { row, asset }));
            }
        }
    }
    if model.uses_monte_carlo() {
        if n_scenarios == 0 {
            return Err(ModelError::Simulation(SimulationError::ZeroScenarios));
        }
        if chunk_size == 0 {
            return Err(ModelError::Simulation(SimulationError::ZeroChunkSize));
        }
    }

    match model {
        RiskModel::HistoricalSimulation => {
            let mut pnl = historical_pnl(observations, notionals);
            if pnl.iter().any(|value| !value.is_finite()) {
                return Err(ModelError::NonFinitePnl);
            }
            let (var, es) = var_es(&mut pnl, confidence);
            Ok(RiskEstimate {
                var,
                es,
                var_ci_95: None,
                es_ci_95: None,
            })
        }
        RiskModel::EwmaNormal => ewma_normal_estimate(observations, notionals, confidence),
        RiskModel::MonteCarloNormal | RiskModel::MonteCarloStudentT => {
            let covariance = ledoit_wolf_diagonal_covariance(observations, notionals.len())?;
            let cholesky_l = cholesky(&covariance, notionals.len())
                .ok_or(ModelError::CovarianceNotPositiveDefinite)?;
            let engine = ParallelEngine::new(chunk_size);
            let mut pnl = match model {
                RiskModel::MonteCarloNormal => {
                    engine.simulate_pnl(notionals, &cholesky_l, n_scenarios, seed)
                }
                RiskModel::MonteCarloStudentT => engine.simulate_student_t_pnl(
                    notionals,
                    &cholesky_l,
                    n_scenarios,
                    seed,
                    STUDENT_T_DEGREES_OF_FREEDOM,
                )?,
                _ => unreachable!(),
            };
            if pnl.iter().any(|value| !value.is_finite()) {
                return Err(ModelError::NonFinitePnl);
            }
            Ok(var_es_with_ci(&mut pnl, confidence))
        }
    }
}

fn historical_pnl(observations: &[Vec<f64>], notionals: &[f64]) -> Vec<f64> {
    observations
        .iter()
        .map(|observation| {
            notionals
                .iter()
                .zip(observation)
                .map(|(notional, log_return)| notional * (log_return.exp() - 1.0))
                .sum()
        })
        .collect()
}

fn ewma_normal_estimate(
    observations: &[Vec<f64>],
    notionals: &[f64],
    confidence: f64,
) -> Result<RiskEstimate, ModelError> {
    let covariance = ewma_covariance(observations, notionals.len());
    let n_assets = notionals.len();
    // Moment-match exact lognormal P&L moments to a normal portfolio P&L.
    let mut mean_pnl = 0.0;
    let mut variance_pnl = 0.0;
    let mut absolute_variance_terms = 0.0;
    for row in 0..n_assets {
        let variance = covariance[row * n_assets + row];
        mean_pnl += notionals[row] * (0.5 * variance).exp_m1();
        for col in 0..n_assets {
            let covariance_ij = covariance[row * n_assets + col];
            let simple_return_covariance = (0.5 * (variance + covariance[col * n_assets + col]))
                .exp()
                * covariance_ij.exp_m1();
            let term = notionals[row] * notionals[col] * simple_return_covariance;
            variance_pnl += term;
            absolute_variance_terms += term.abs();
        }
    }
    if !mean_pnl.is_finite()
        || !variance_pnl.is_finite()
        || variance_pnl < -1e-12 * absolute_variance_terms
    {
        return Err(ModelError::InvalidCovariance);
    }

    let standard_deviation = variance_pnl.max(0.0).sqrt();
    let z = inverse_standard_normal(confidence);
    let tail_probability = 1.0 - confidence;
    let var = z * standard_deviation - mean_pnl;
    let es = standard_deviation * standard_normal_pdf(z) / tail_probability - mean_pnl;
    if !var.is_finite() || !es.is_finite() {
        return Err(ModelError::InvalidCovariance);
    }

    Ok(RiskEstimate {
        var,
        es,
        var_ci_95: None,
        es_ci_95: None,
    })
}

fn ewma_covariance(observations: &[Vec<f64>], n_assets: usize) -> Vec<f64> {
    let mut covariance = vec![0.0; n_assets * n_assets];
    let mut weight = 1.0;
    let mut total_weight = 0.0;
    for observation in observations.iter().rev() {
        for row in 0..n_assets {
            for col in 0..=row {
                covariance[row * n_assets + col] += weight * observation[row] * observation[col];
            }
        }
        total_weight += weight;
        weight *= EWMA_LAMBDA;
    }
    for row in 0..n_assets {
        for col in 0..=row {
            let value = covariance[row * n_assets + col] / total_weight;
            covariance[row * n_assets + col] = value;
            covariance[col * n_assets + row] = value;
        }
    }
    covariance
}

/// Estimate covariance with Ledoit-Wolf shrinkage toward a diagonal target.
fn ledoit_wolf_diagonal_covariance(
    observations: &[Vec<f64>],
    n_assets: usize,
) -> Result<Vec<f64>, ModelError> {
    let n_observations = observations.len();
    if n_observations < 2 {
        return Err(ModelError::EmptyObservations);
    }
    let mut means = vec![0.0; n_assets];
    for observation in observations {
        for (mean, value) in means.iter_mut().zip(observation) {
            *mean += value / n_observations as f64;
        }
    }
    let centered: Vec<Vec<f64>> = observations
        .iter()
        .map(|observation| {
            observation
                .iter()
                .zip(&means)
                .map(|(value, mean)| value - mean)
                .collect()
        })
        .collect();

    let mut covariance = vec![0.0; n_assets * n_assets];
    for observation in &centered {
        for row in 0..n_assets {
            for col in 0..=row {
                covariance[row * n_assets + col] +=
                    observation[row] * observation[col] / n_observations as f64;
            }
        }
    }
    for row in 0..n_assets {
        for col in 0..=row {
            let value = covariance[row * n_assets + col];
            covariance[col * n_assets + row] = value;
        }
    }

    let mut delta = 0.0;
    let mut beta = 0.0;
    for row in 0..n_assets {
        for col in 0..n_assets {
            if row == col {
                continue;
            }
            let value = covariance[row * n_assets + col];
            delta += value * value;
            for observation in &centered {
                let deviation = observation[row] * observation[col] - value;
                beta += deviation * deviation;
            }
        }
    }
    beta /= (n_observations * n_observations) as f64;
    let shrinkage = if delta > 0.0 {
        (beta / delta).clamp(0.0, 1.0)
    } else {
        0.0
    };
    for row in 0..n_assets {
        for col in 0..n_assets {
            if row != col {
                covariance[row * n_assets + col] *= 1.0 - shrinkage;
            }
        }
    }
    if covariance.iter().all(|value| value.is_finite()) {
        Ok(covariance)
    } else {
        Err(ModelError::InvalidCovariance)
    }
}

fn inverse_standard_normal(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOWER_BOUND: f64 = 0.02425;

    if probability < LOWER_BOUND {
        let q = (-2.0 * probability.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if probability > 1.0 - LOWER_BOUND {
        let q = (-2.0 * (1.0 - probability).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = probability - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

fn standard_normal_pdf(value: f64) -> f64 {
    (-0.5 * value * value).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ParallelEngine;

    #[test]
    fn inverse_normal_matches_common_confidence_levels() {
        assert!((inverse_standard_normal(0.95) - 1.644_853_626_951_472_2).abs() < 1e-8);
        assert!((inverse_standard_normal(0.975) - 1.959_963_984_540_054).abs() < 1e-8);
        assert!((inverse_standard_normal(0.99) - 2.326_347_874_040_840_8).abs() < 1e-8);
    }

    #[test]
    fn diagonal_shrinkage_regularizes_collinear_observations() {
        let observations = vec![
            vec![0.01, 0.02],
            vec![0.02, 0.04],
            vec![0.03, 0.06],
            vec![0.04, 0.08],
        ];
        let covariance = ledoit_wolf_diagonal_covariance(&observations, 2).unwrap();
        assert!(cholesky(&covariance, 2).is_some());
    }

    #[test]
    fn student_t_engine_has_heavier_tail_than_normal_engine() {
        let returns = vec![vec![-0.02], vec![-0.01], vec![0.0], vec![0.01], vec![0.02]];
        let notionals = [10_000.0];
        let normal = estimate_var_es(
            RiskModel::MonteCarloNormal,
            &returns,
            &notionals,
            0.99,
            100_000,
            17,
            2_000,
        )
        .unwrap();
        let student_t = estimate_var_es(
            RiskModel::MonteCarloStudentT,
            &returns,
            &notionals,
            0.99,
            100_000,
            17,
            2_000,
        )
        .unwrap();
        assert!(student_t.var > normal.var);
        assert!(student_t.es > normal.es);
    }

    #[test]
    fn student_t_simulation_is_chunk_invariant() {
        let weights = [10_000.0];
        let cholesky_l = [0.01];
        let one_scenario_per_chunk = ParallelEngine::new(1)
            .simulate_student_t_pnl(&weights, &cholesky_l, 128, 42, 5)
            .unwrap();
        let larger_chunks = ParallelEngine::new(31)
            .simulate_student_t_pnl(&weights, &cholesky_l, 128, 42, 5)
            .unwrap();
        assert_eq!(one_scenario_per_chunk, larger_chunks);
    }

    #[test]
    fn historical_simulation_uses_observed_simple_return_pnl() {
        let observations = vec![vec![-0.1], vec![0.01], vec![0.02]];
        let estimate = estimate_var_es(
            RiskModel::HistoricalSimulation,
            &observations,
            &[1_000.0],
            0.67,
            0,
            0,
            0,
        )
        .unwrap();
        let expected_loss = 1_000.0 * (1.0 - (-0.1_f64).exp());
        assert!((estimate.var - expected_loss).abs() < 1e-10);
        assert!((estimate.es - expected_loss).abs() < 1e-10);
    }

    #[test]
    fn ewma_normal_places_more_weight_on_recent_volatility() {
        let recent_shock = vec![vec![0.001], vec![0.001], vec![0.02]];
        let older_shock = vec![vec![0.02], vec![0.001], vec![0.001]];
        let recent_shock_estimate = estimate_var_es(
            RiskModel::EwmaNormal,
            &recent_shock,
            &[10_000.0],
            0.99,
            0,
            0,
            0,
        )
        .unwrap();
        let older_shock_estimate = estimate_var_es(
            RiskModel::EwmaNormal,
            &older_shock,
            &[10_000.0],
            0.99,
            0,
            0,
            0,
        )
        .unwrap();
        assert!(recent_shock_estimate.var > older_shock_estimate.var);
    }
}
