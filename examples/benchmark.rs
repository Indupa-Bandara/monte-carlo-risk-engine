use clap::Parser;
use rayon::ThreadPoolBuilder;
use risk_engine::cpp_engine::{cpp_uses_avx2, CppEngine, ParallelCppEngine};
use risk_engine::engine::{
    correlate_batch, correlate_batch_simd, value_pnl_batch, EngineBackend, ParallelEngine,
    ParallelSimdEngine, ScalarEngine, SimdEngine,
};
use risk_engine::risk::var_es;
use risk_engine::rng::{fill_normals, Philox};
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Debug, Parser)]
#[command(about = "Repeatable CPU benchmark for the portfolio Monte Carlo engine.")]
struct Args {
    #[arg(long, default_value_t = 1_000_000, help = "Scenarios per workload")]
    scenarios: u64,
    #[arg(long, default_value_t = 50, help = "Assets per workload")]
    assets: usize,
    #[arg(
        long,
        value_delimiter = ',',
        help = "Parallel thread counts, e.g. 1,2,4,8"
    )]
    threads: Vec<usize>,
    #[arg(long, default_value_t = 10_000, help = "Scenarios per parallel chunk")]
    chunk_size: u64,
    #[arg(long, default_value_t = 2, help = "Untimed warmup runs")]
    warmups: usize,
    #[arg(long, default_value_t = 5, help = "Timed repetitions")]
    repetitions: usize,
    #[arg(long, default_value_t = 42, help = "Deterministic scenario seed")]
    seed: u64,
    #[arg(
        long,
        default_value = "0.99",
        help = "Confidence level for VaR/ES selection"
    )]
    confidence: f64,
    #[arg(
        long,
        default_value = "not-specified",
        help = "CPU model to record in metadata"
    )]
    cpu_model: String,
    #[arg(
        long,
        help = "Sweep asset counts 2, 10, 20, 50 and scenario counts 100k, 1M"
    )]
    sweep: bool,
}

#[derive(Clone, Copy)]
struct TimingSummary {
    mean_ms: f64,
    std_dev_ms: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.scenarios == 0
        || args.assets == 0
        || args.chunk_size == 0
        || args.repetitions == 0
        || !args.confidence.is_finite()
        || !(0.0..1.0).contains(&args.confidence)
    {
        return Err("scenarios, assets, chunk-size, and repetitions must be positive; confidence must be between 0 and 1".into());
    }

    let available_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let mut thread_counts = if args.threads.is_empty() {
        vec![1, 2, 4, available_threads]
    } else {
        args.threads.clone()
    };
    if thread_counts.contains(&0) {
        return Err("thread counts must be greater than zero".into());
    }
    thread_counts.sort_unstable();
    thread_counts.dedup();

    println!("# cpu_model={}", args.cpu_model);
    println!(
        "# os={} arch={} logical_threads={available_threads}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("# simd_backend={}", simd_backend_name());
    println!(
        "# cpp_backend={}",
        if cpp_uses_avx2() {
            "avx2"
        } else {
            "scalar-fallback"
        }
    );
    println!(
        "# seed={} chunk_size={} warmups={} repetitions={} confidence={}",
        args.seed, args.chunk_size, args.warmups, args.repetitions, args.confidence
    );
    println!("record_type,scenarios,assets,backend,threads,mean_simulation_ms,stddev_simulation_ms,mean_total_ms,stddev_total_ms,paths_per_second,speedup_vs_scalar,max_abs_pnl_error,var,es,component,mean_component_ms,stddev_component_ms,ns_per_scenario");

    let workloads = if args.sweep {
        [100_000, 1_000_000]
            .into_iter()
            .flat_map(|scenarios| [2, 10, 20, 50].map(move |assets| (scenarios, assets)))
            .collect::<Vec<_>>()
    } else {
        vec![(args.scenarios, args.assets)]
    };

    for (scenarios, n_assets) in workloads {
        let (weights, cholesky_l) = make_inputs(n_assets);
        let stage_scenarios = scenarios.min(10_000) as usize;
        benchmark_stages(
            stage_scenarios,
            n_assets,
            &weights,
            &cholesky_l,
            args.seed,
            args.warmups,
            args.repetitions,
            args.confidence,
        )?;

        let scalar = benchmark_backend(
            args.warmups,
            args.repetitions,
            args.confidence,
            || ScalarEngine.simulate_pnl(&weights, &cholesky_l, scenarios, args.seed),
            None,
        );
        let scalar_mean_ms = scalar.simulation.mean_ms;
        print_result(scenarios, n_assets, "scalar", 1, &scalar, scalar_mean_ms);

        let simd = benchmark_backend(
            args.warmups,
            args.repetitions,
            args.confidence,
            || SimdEngine.simulate_pnl(&weights, &cholesky_l, scenarios, args.seed),
            Some(&scalar.reference),
        );
        print_result(scenarios, n_assets, "simd", 1, &simd, scalar_mean_ms);

        let cpp = benchmark_backend(
            args.warmups,
            args.repetitions,
            args.confidence,
            || CppEngine.simulate_pnl(&weights, &cholesky_l, scenarios, args.seed),
            Some(&scalar.reference),
        );
        print_result(scenarios, n_assets, "cpp", 1, &cpp, scalar_mean_ms);

        for &threads in &thread_counts {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            let engine = ParallelEngine::new(args.chunk_size);
            let parallel = benchmark_backend(
                args.warmups,
                args.repetitions,
                args.confidence,
                || {
                    pool.install(|| {
                        engine.simulate_pnl(&weights, &cholesky_l, scenarios, args.seed)
                    })
                },
                Some(&scalar.reference),
            );
            print_result(
                scenarios,
                n_assets,
                "rayon",
                threads,
                &parallel,
                scalar_mean_ms,
            );

            let simd_engine = ParallelSimdEngine::new(args.chunk_size);
            let parallel_simd = benchmark_backend(
                args.warmups,
                args.repetitions,
                args.confidence,
                || {
                    pool.install(|| {
                        simd_engine.simulate_pnl(&weights, &cholesky_l, scenarios, args.seed)
                    })
                },
                Some(&scalar.reference),
            );
            print_result(
                scenarios,
                n_assets,
                "rayon_simd",
                threads,
                &parallel_simd,
                scalar_mean_ms,
            );

            let cpp_engine = ParallelCppEngine::new(
                usize::try_from(args.chunk_size).expect("chunk size exceeds addressable memory"),
            );
            let parallel_cpp = benchmark_backend(
                args.warmups,
                args.repetitions,
                args.confidence,
                || {
                    pool.install(|| {
                        cpp_engine.simulate_pnl(&weights, &cholesky_l, scenarios, args.seed)
                    })
                },
                Some(&scalar.reference),
            );
            print_result(
                scenarios,
                n_assets,
                "rayon_cpp",
                threads,
                &parallel_cpp,
                scalar_mean_ms,
            );
        }
    }
    Ok(())
}

fn simd_backend_name() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return "avx2";
    }
    "scalar-fallback"
}

struct BackendResult {
    simulation: TimingSummary,
    total: TimingSummary,
    reference: Vec<f64>,
    max_abs_error: f64,
    var: f64,
    es: f64,
}

fn benchmark_backend<F>(
    warmups: usize,
    repetitions: usize,
    confidence: f64,
    mut simulate: F,
    reference: Option<&[f64]>,
) -> BackendResult
where
    F: FnMut() -> Vec<f64>,
{
    for _ in 0..warmups {
        let mut pnl = simulate();
        let estimate = var_es(&mut pnl, confidence);
        black_box(estimate);
    }

    let mut simulation_times = Vec::with_capacity(repetitions);
    let mut total_times = Vec::with_capacity(repetitions);
    let mut first_output = None;
    let mut max_abs_error: f64 = 0.0;
    let mut var = 0.0;
    let mut es = 0.0;
    for _ in 0..repetitions {
        let started = Instant::now();
        let mut pnl = simulate();
        let simulation_time = started.elapsed();
        if let Some(expected) = reference {
            max_abs_error = max_abs_error.max(
                pnl.iter()
                    .zip(expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0, f64::max),
            );
        } else if first_output.is_none() {
            first_output = Some(pnl.clone());
        }

        let selection_started = Instant::now();
        (var, es) = var_es(&mut pnl, confidence);
        let selection_time = selection_started.elapsed();
        simulation_times.push(simulation_time);
        total_times.push(simulation_time + selection_time);
        black_box((var, es));
    }

    BackendResult {
        simulation: summarize(&simulation_times),
        total: summarize(&total_times),
        reference: reference
            .map(<[f64]>::to_vec)
            .or(first_output)
            .expect("at least one output is generated"),
        max_abs_error,
        var,
        es,
    }
}

#[allow(clippy::too_many_arguments)]
fn benchmark_stages(
    scenarios: usize,
    n_assets: usize,
    weights: &[f64],
    cholesky_l: &[f64],
    seed: u64,
    warmups: usize,
    repetitions: usize,
    confidence: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut normals = vec![0.0; scenarios * n_assets];
    let mut correlated = vec![0.0; normals.len()];
    let mut pnl = vec![0.0; scenarios];
    let mut normal_vector = vec![0.0; n_assets];
    let rng_times = measure_stage(warmups, repetitions, || {
        for scenario in 0..scenarios {
            let mut rng = Philox::from_scenario(seed, scenario as u64);
            fill_normals(&mut rng, &mut normal_vector);
            black_box(normal_vector[scenario % n_assets]);
        }
    });

    fill_normal_batch(seed, n_assets, &mut normals);
    let correlation_times = measure_stage(warmups, repetitions, || {
        correlate_batch(cholesky_l, &normals, &mut correlated, n_assets).unwrap();
        black_box(&correlated);
    });
    let simd_correlation_times = measure_stage(warmups, repetitions, || {
        correlate_batch_simd(cholesky_l, &normals, &mut correlated, n_assets).unwrap();
        black_box(&correlated);
    });
    let valuation_times = measure_stage(warmups, repetitions, || {
        value_pnl_batch(weights, &correlated, &mut pnl).unwrap();
        black_box(&pnl);
    });
    let selection_times = measure_stage(warmups, repetitions, || {
        let mut sample = pnl.clone();
        black_box(var_es(&mut sample, confidence));
    });

    for (name, times) in [
        ("rng", rng_times),
        ("correlation", correlation_times),
        ("correlation_simd", simd_correlation_times),
        ("valuation", valuation_times),
        ("quantile_es", selection_times),
    ] {
        let summary = summarize(&times);
        let ns_per_scenario = summary.mean_ms * 1_000_000.0 / scenarios as f64;
        let mut row = vec![String::new(); 18];
        row[0] = "component".into();
        row[1] = scenarios.to_string();
        row[2] = n_assets.to_string();
        row[14] = name.into();
        row[15] = format!("{:.6}", summary.mean_ms);
        row[16] = format!("{:.6}", summary.std_dev_ms);
        row[17] = format!("{ns_per_scenario:.3}");
        println!("{}", row.join(","));
    }
    Ok(())
}

fn fill_normal_batch(seed: u64, n_assets: usize, output: &mut [f64]) {
    for (scenario, normal_vector) in output.chunks_exact_mut(n_assets).enumerate() {
        let mut rng = Philox::from_scenario(seed, scenario as u64);
        fill_normals(&mut rng, normal_vector);
    }
}

fn measure_stage<F>(warmups: usize, repetitions: usize, mut stage: F) -> Vec<Duration>
where
    F: FnMut(),
{
    for _ in 0..warmups {
        stage();
    }
    (0..repetitions)
        .map(|_| {
            let started = Instant::now();
            stage();
            started.elapsed()
        })
        .collect()
}

fn make_inputs(n_assets: usize) -> (Vec<f64>, Vec<f64>) {
    let weights = vec![10_000.0 / n_assets as f64; n_assets];
    let mut cholesky_l = vec![0.0; n_assets * n_assets];
    for row in 0..n_assets {
        for col in 0..=row {
            cholesky_l[row * n_assets + col] = if row == col {
                0.005 + (row % 7) as f64 * 0.0001
            } else {
                (((row * 17 + col * 13) % 11) as f64 - 5.0) * 0.00002
            };
        }
    }
    (weights, cholesky_l)
}

fn print_result(
    scenarios: u64,
    n_assets: usize,
    backend: &str,
    threads: usize,
    result: &BackendResult,
    scalar_mean_ms: f64,
) {
    let paths_per_second = scenarios as f64 / (result.simulation.mean_ms / 1_000.0);
    let speedup = scalar_mean_ms / result.simulation.mean_ms;
    let mut row = vec![String::new(); 18];
    row[0] = "engine".into();
    row[1] = scenarios.to_string();
    row[2] = n_assets.to_string();
    row[3] = backend.into();
    row[4] = threads.to_string();
    row[5] = format!("{:.6}", result.simulation.mean_ms);
    row[6] = format!("{:.6}", result.simulation.std_dev_ms);
    row[7] = format!("{:.6}", result.total.mean_ms);
    row[8] = format!("{:.6}", result.total.std_dev_ms);
    row[9] = format!("{paths_per_second:.0}");
    row[10] = format!("{speedup:.3}");
    row[11] = format!("{:.12}", result.max_abs_error);
    row[12] = format!("{:.6}", result.var);
    row[13] = format!("{:.6}", result.es);
    println!("{}", row.join(","));
}

fn summarize(durations: &[Duration]) -> TimingSummary {
    let values: Vec<f64> = durations
        .iter()
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect();
    let mean_ms = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let deviation = value - mean_ms;
            deviation * deviation
        })
        .sum::<f64>()
        / values.len() as f64;
    TimingSummary {
        mean_ms,
        std_dev_ms: variance.sqrt(),
    }
}
