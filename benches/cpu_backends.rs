use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rayon::ThreadPoolBuilder;
use risk_engine::cpp_engine::{cpp_uses_avx2, CppEngine, ParallelCppEngine};
use risk_engine::engine::{EngineBackend, ParallelEngine, ScalarEngine};
use std::time::Duration;

unsafe extern "C" {
    fn risk_engine_cpp_fill_shocks(
        seed: u64,
        first_path: u64,
        n_assets: usize,
        path_count: usize,
        scratch_capacity: usize,
        shocks: *mut f32,
    ) -> i32;

    fn risk_engine_cpp_correlate_batch(
        cholesky_l: *const f64,
        n_assets: usize,
        path_count: usize,
        scratch_capacity: usize,
        shocks: *const f32,
        correlated: *mut f64,
    ) -> i32;

    fn risk_engine_cpp_value_batch(
        weights: *const f64,
        n_assets: usize,
        path_count: usize,
        scratch_capacity: usize,
        correlated: *const f64,
        pnl: *mut f64,
    ) -> i32;
}

fn cpu_backends(criterion: &mut Criterion) {
    let scenarios = 100_000;
    let n_assets = 20;
    let seed = 42;
    let weights = vec![500.0; n_assets];
    let mut cholesky_l = vec![0.0; n_assets * n_assets];
    for row in 0..n_assets {
        for col in 0..=row {
            cholesky_l[row * n_assets + col] = if row == col {
                0.003 + row as f64 * 0.00005
            } else {
                (row as f64 - col as f64) * 0.00001
            };
        }
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("failed to create Rayon benchmark pool");
    let rayon_engine = ParallelEngine::new(10_000);
    let cpp_engine = ParallelCppEngine::new(10_000);
    let cpp_name = if cpp_uses_avx2() {
        "cpp_avx2"
    } else {
        "cpp_scalar_fallback"
    };

    let mut group = criterion.benchmark_group("cpu_backends_100k_paths_20_assets");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(scenarios));

    group.bench_function(BenchmarkId::new("rust_scalar", scenarios), |bencher| {
        bencher
            .iter(|| black_box(ScalarEngine.simulate_pnl(&weights, &cholesky_l, scenarios, seed)));
    });
    group.bench_function(BenchmarkId::new("rayon_8_threads", scenarios), |bencher| {
        bencher.iter(|| {
            black_box(
                pool.install(|| rayon_engine.simulate_pnl(&weights, &cholesky_l, scenarios, seed)),
            )
        });
    });
    group.bench_function(BenchmarkId::new(cpp_name, scenarios), |bencher| {
        bencher.iter(|| black_box(CppEngine.simulate_pnl(&weights, &cholesky_l, scenarios, seed)));
    });
    group.bench_function(
        BenchmarkId::new("rayon_cpp_8_threads", scenarios),
        |bencher| {
            bencher.iter(|| {
                black_box(
                    pool.install(|| {
                        cpp_engine.simulate_pnl(&weights, &cholesky_l, scenarios, seed)
                    }),
                )
            });
        },
    );
    group.finish();

    benchmark_cpp_stages(criterion, seed, &weights, &cholesky_l, n_assets);
}

fn benchmark_cpp_stages(
    criterion: &mut Criterion,
    seed: u64,
    weights: &[f64],
    cholesky_l: &[f64],
    n_assets: usize,
) {
    let paths = 10_000;
    let mut rust_normals = vec![0.0_f64; paths * n_assets];
    let mut rust_normal_row = vec![0.0_f64; n_assets];
    let mut cpp_shocks = vec![0.0_f32; paths * n_assets];
    let mut rust_correlated = vec![0.0_f64; paths * n_assets];
    let mut cpp_correlated = vec![0.0_f64; paths * n_assets];
    let mut rust_pnl = vec![0.0_f64; paths];
    let mut cpp_pnl = vec![0.0_f64; paths];

    let cpp_status = unsafe {
        risk_engine_cpp_fill_shocks(seed, 0, n_assets, paths, paths, cpp_shocks.as_mut_ptr())
    };
    assert_eq!(cpp_status, 0, "C++ shock generation failed");
    for (rust_row, cpp_row) in rust_normals
        .chunks_exact_mut(n_assets)
        .zip(cpp_shocks.chunks_exact(n_assets))
    {
        for (rust_value, cpp_value) in rust_row.iter_mut().zip(cpp_row) {
            *rust_value = f64::from(*cpp_value);
        }
    }

    let cpp_status = unsafe {
        risk_engine_cpp_correlate_batch(
            cholesky_l.as_ptr(),
            n_assets,
            paths,
            paths,
            cpp_shocks.as_ptr(),
            cpp_correlated.as_mut_ptr(),
        )
    };
    assert_eq!(cpp_status, 0, "C++ correlation failed");
    let cpp_status = unsafe {
        risk_engine_cpp_value_batch(
            weights.as_ptr(),
            n_assets,
            paths,
            paths,
            cpp_correlated.as_ptr(),
            cpp_pnl.as_mut_ptr(),
        )
    };
    assert_eq!(cpp_status, 0, "C++ valuation failed");

    let mut group = criterion.benchmark_group("backend_stages_10k_paths_20_assets");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("rust_philox_box_muller", |bencher| {
        bencher.iter(|| {
            for path in 0..paths {
                let mut rng = risk_engine::rng::Philox::from_scenario(seed, path as u64);
                risk_engine::rng::fill_normals(&mut rng, &mut rust_normal_row);
                rust_normals[path * n_assets..(path + 1) * n_assets]
                    .copy_from_slice(&rust_normal_row);
            }
            black_box(&rust_normals);
        });
    });
    group.bench_function("cpp_philox_box_muller", |bencher| {
        bencher.iter(|| {
            let status = unsafe {
                risk_engine_cpp_fill_shocks(
                    seed,
                    0,
                    n_assets,
                    paths,
                    paths,
                    cpp_shocks.as_mut_ptr(),
                )
            };
            assert_eq!(status, 0);
            black_box(&cpp_shocks);
        });
    });
    group.bench_function("rust_avx2_correlation", |bencher| {
        bencher.iter(|| {
            risk_engine::engine::correlate_batch_simd(
                cholesky_l,
                &rust_normals,
                &mut rust_correlated,
                n_assets,
            )
            .expect("Rust correlation failed");
            black_box(&rust_correlated);
        });
    });
    group.bench_function("cpp_runtime_correlation", |bencher| {
        bencher.iter(|| {
            let status = unsafe {
                risk_engine_cpp_correlate_batch(
                    cholesky_l.as_ptr(),
                    n_assets,
                    paths,
                    paths,
                    cpp_shocks.as_ptr(),
                    cpp_correlated.as_mut_ptr(),
                )
            };
            assert_eq!(status, 0);
            black_box(&cpp_correlated);
        });
    });
    group.bench_function("rust_valuation", |bencher| {
        bencher.iter(|| {
            risk_engine::engine::value_pnl_batch(weights, &cpp_correlated, &mut rust_pnl)
                .expect("Rust valuation failed");
            black_box(&rust_pnl);
        });
    });
    group.bench_function("cpp_valuation", |bencher| {
        bencher.iter(|| {
            let status = unsafe {
                risk_engine_cpp_value_batch(
                    weights.as_ptr(),
                    n_assets,
                    paths,
                    paths,
                    cpp_correlated.as_ptr(),
                    cpp_pnl.as_mut_ptr(),
                )
            };
            assert_eq!(status, 0);
            black_box(&cpp_pnl);
        });
    });
    group.finish();
}

criterion_group!(benches, cpu_backends);
criterion_main!(benches);
