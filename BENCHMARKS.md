# CPU benchmark suite

The benchmark uses a deterministic synthetic Cholesky factor and equal-notional
portfolio. It measures scalar, AVX2-accelerated (with runtime scalar fallback),
Rayon, and the C++20 FFI backend, plus separate RNG, correlation, valuation,
and VaR/ES selection stages. The component probes use up to 10,000 scenarios;
full-workload simulation throughput and total simulation-plus-selection time
are reported separately.

Run the fixed 1-million-path, 50-asset workload in release mode:

```powershell
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
cargo run --release --quiet --example benchmark -- --scenarios 1000000 --assets 50 --threads 1,2,4,8 --warmups 2 --repetitions 5 --chunk-size 10000 --seed 42 --cpu-model "$cpu" | Tee-Object .\benchmark-results.csv
```

To sweep the prescribed path and asset counts:

```powershell
cargo run --release --quiet --example benchmark -- --sweep --threads 1,2,4,8 --warmups 2 --repetitions 5 --chunk-size 10000 --seed 42
```

The Criterion benchmark compares the Rust scalar, Rayon, C++ runtime-dispatched,
and Rayon+C++ backends:

```powershell
cargo bench --bench cpu_backends
```

The C++ backend requires CMake 3.20+ and a C++20 compiler. On Windows, use the
MSVC C++ workload; on Linux, use GCC or Clang. `build.rs` invokes CMake, which
builds the scalar implementation everywhere and compiles the AVX2 translation
unit only for x86 targets. The build script finds CMake on `PATH` or in a
Visual Studio installation. Runtime feature detection selects AVX2 when
supported and otherwise uses the scalar fallback. The FFI passes preallocated
scratch buffers and executes in bounded cache-sized blocks without C++ heap
allocation in the path loop.

The output includes per-run mean and standard deviation, paths per second,
speedup against the scalar implementation, and maximum absolute P&L difference
from the scalar output. Scalar and Rayon backends should report zero difference;
SIMD and C++ backends may have small floating-point rounding differences.
The C++ path uses f32 shocks and f64 accumulation; tests enforce an absolute
tolerance of `1e-4 + 1e-6 * abs(reference_pnl)` against Rust scalar output.
Metadata lines beginning with `#` record the CPU model, OS, architecture,
logical processor count, and selected SIMD implementations. The component
measurements are microbenchmarks, so use the end-to-end rows for performance
comparisons. For defensible comparisons, run on an otherwise idle machine and
record its power/performance settings.
