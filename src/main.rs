use clap::{Parser, ValueEnum};
use risk_engine::attribution::historical_component_es;
use risk_engine::backtest::{rolling_model_backtest, RollingBacktestConfig};
use risk_engine::data::{AlignedReturns, Positions};
use risk_engine::models::{estimate_var_es, RiskModel};
use risk_engine::stress::{historical_stress_replay, hypothetical_stress};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelChoice {
    All,
    Historical,
    EwmaNormal,
    MonteCarloNormal,
    MonteCarloStudentT,
}

impl ModelChoice {
    fn risk_models(self) -> Vec<RiskModel> {
        match self {
            Self::All => RiskModel::ALL.to_vec(),
            Self::Historical => vec![RiskModel::HistoricalSimulation],
            Self::EwmaNormal => vec![RiskModel::EwmaNormal],
            Self::MonteCarloNormal => vec![RiskModel::MonteCarloNormal],
            Self::MonteCarloStudentT => vec![RiskModel::MonteCarloStudentT],
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Estimate portfolio VaR and ES from Parquet returns and CSV positions."
)]
struct Args {
    #[arg(
        value_name = "RETURNS.parquet",
        help = "Wide Parquet table with a string or native date `date` column and Float32/Float64 asset columns"
    )]
    returns: PathBuf,
    #[arg(
        value_name = "POSITIONS.csv",
        help = "CSV file with `asset` and `notional` headers"
    )]
    positions: PathBuf,
    #[arg(
        long,
        default_value_t = 1_000_000,
        help = "Number of Monte Carlo scenarios"
    )]
    scenarios: u64,
    #[arg(long, default_value_t = 42, help = "Seed for reproducible simulation")]
    seed: u64,
    #[arg(long, default_value_t = 10_000, help = "Parallel work chunk size")]
    chunk_size: u64,
    #[arg(
        long,
        default_value = "0.99",
        value_parser = parse_confidence,
        help = "VaR/ES confidence level, strictly between 0 and 1"
    )]
    confidence: f64,
    #[arg(
        long,
        value_name = "OBSERVATIONS",
        help = "Run a rolling out-of-sample backtest with this lookback window"
    )]
    backtest_window: Option<usize>,
    #[arg(
        long,
        value_enum,
        default_value = "monte-carlo-normal",
        help = "Risk model to run; `all` compares all four models"
    )]
    model: ModelChoice,
    #[arg(long, help = "Print historical component ES contributions by asset")]
    contributions: bool,
    #[arg(
        long,
        help = "Print the five worst historical days and uniform -20%/-40% hypothetical shocks"
    )]
    stress: bool,
}

fn parse_confidence(value: &str) -> Result<f64, String> {
    let confidence = value
        .parse::<f64>()
        .map_err(|_| "confidence must be a number between 0 and 1".to_owned())?;
    if confidence.is_finite() && (0.0..1.0).contains(&confidence) {
        Ok(confidence)
    } else {
        Err("confidence must be greater than 0 and less than 1".to_owned())
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.scenarios == 0 || args.chunk_size == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "scenarios and chunk-size must be greater than zero",
        )
        .into());
    }

    let returns = AlignedReturns::from_parquet_path(&args.returns)?;
    let positions = Positions::from_csv_path(&args.positions)?;
    let models = args.model.risk_models();
    let notionals = positions.align_to(&returns)?;
    if let Some(window_size) = args.backtest_window {
        for model in models {
            let result = rolling_model_backtest(
                &returns,
                &positions,
                RollingBacktestConfig {
                    window_size,
                    n_scenarios: args.scenarios,
                    seed: args.seed,
                    confidence: args.confidence,
                    chunk_size: args.chunk_size,
                },
                model,
            )?;
            println!("\nModel: {}", result.model.name());
            println!("Rolling out-of-sample forecasts:");
            for forecast in &result.forecasts {
                println!(
                    "{} VaR={:.6} ES={:.6} realized_loss={:.6} exception={}",
                    forecast.date,
                    forecast.var,
                    forecast.es,
                    forecast.realized_loss,
                    forecast.exception
                );
            }
            println!("Exceptions: {}/{}", result.kupiec.x, result.kupiec.n);
            println!(
                "Kupiec POF p-value: {:.6} rejected={}",
                result.kupiec.p_value, result.kupiec.rejected
            );
            println!(
                "Christoffersen independence p-value: {:.6} rejected={}",
                result.christoffersen.p_value, result.christoffersen.rejected
            );
        }
        print_portfolio_analysis(
            &returns,
            &positions,
            args.confidence,
            args.contributions,
            args.stress,
        )?;
        return Ok(());
    }

    for model in models {
        let estimate = estimate_var_es(
            model,
            returns.values(),
            &notionals,
            args.confidence,
            args.scenarios,
            args.seed,
            args.chunk_size,
        )?;

        println!("\nModel: {}", model.name());
        println!("Assets: {}", notionals.len());
        println!("Historical observations: {}", returns.values().len());
        if matches!(
            model,
            RiskModel::MonteCarloNormal | RiskModel::MonteCarloStudentT
        ) {
            println!("Scenarios: {}", args.scenarios);
            println!("Seed: {}", args.seed);
        }
        println!("Confidence: {:.2}%", args.confidence * 100.0);
        println!("VaR: {:.6}", estimate.var);
        println!("ES:  {:.6}", estimate.es);
        if let Some(ci) = estimate.var_ci_95 {
            println!(
                "VaR 95% CI (order-statistic approximation): [{:.6}, {:.6}]",
                ci.lower, ci.upper
            );
        } else if matches!(
            model,
            RiskModel::MonteCarloNormal | RiskModel::MonteCarloStudentT
        ) {
            println!("VaR 95% CI: unavailable (fewer than 30 tail scenarios)");
        } else {
            println!("VaR 95% CI: not estimated for this model");
        }
        if let Some(ci) = estimate.es_ci_95 {
            println!(
                "ES 95% CI (tail-mean approximation): [{:.6}, {:.6}]",
                ci.lower, ci.upper
            );
        } else if matches!(
            model,
            RiskModel::MonteCarloNormal | RiskModel::MonteCarloStudentT
        ) {
            println!("ES 95% CI: unavailable (fewer than 30 tail scenarios)");
        } else {
            println!("ES 95% CI: not estimated for this model");
        }
    }

    print_portfolio_analysis(
        &returns,
        &positions,
        args.confidence,
        args.contributions,
        args.stress,
    )?;

    Ok(())
}

fn print_portfolio_analysis(
    returns: &AlignedReturns,
    positions: &Positions,
    confidence: f64,
    print_contributions: bool,
    print_stress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if print_contributions {
        let attribution = historical_component_es(returns, positions, confidence)?;
        println!(
            "\nHistorical component ES at {:.2}% confidence ({} tail observations):",
            confidence * 100.0,
            attribution.tail_scenarios
        );
        println!("Portfolio VaR: {:.6}", attribution.var);
        println!("Portfolio ES:  {:.6}", attribution.es);
        for contribution in attribution.contributions {
            println!(
                "{}: {:.6}",
                contribution.asset_name, contribution.loss_contribution
            );
        }
    }

    if print_stress {
        println!("\nWorst historical one-day replay scenarios:");
        for scenario in historical_stress_replay(returns, positions)?.iter().take(5) {
            println!(
                "{} loss={:.6} P&L={:.6}",
                scenario.date.as_deref().unwrap_or("unknown"),
                scenario.loss,
                scenario.pnl
            );
        }
        for (name, simple_return) in [("uniform -20% shock", -0.20), ("uniform -40% shock", -0.40)]
        {
            let shocks = vec![simple_return; returns.asset_names().len()];
            let scenario = hypothetical_stress(returns, positions, name, &shocks)?;
            println!(
                "{} loss={:.6} P&L={:.6}",
                scenario.name, scenario.loss, scenario.pnl
            );
        }
    }

    Ok(())
}
