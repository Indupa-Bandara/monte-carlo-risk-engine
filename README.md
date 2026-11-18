# Risk Engine

A Rust portfolio risk engine with four VaR/ES models, rolling backtests,
historical risk attribution, stress scenarios, and scalar, Rayon, SIMD, and
C++20 simulation backends.

## Risk conventions and models

Asset inputs are aligned daily log returns and positions are signed notionals.
Portfolio P&L is `notional * (exp(log_return) - 1)`. VaR and ES are reported as
positive loss magnitudes. The available models are:

- Historical simulation.
- EWMA normal (`lambda = 0.94`).
- Normal Monte Carlo with diagonal covariance shrinkage.
- Student-t Monte Carlo (`nu = 5`).

Rolling forecasts use only observations before the forecast date. The rolling
backtest reports Kupiec coverage and Christoffersen exception-independence
tests; a non-rejection is not proof that a model is correct.

## Attribution and stress analysis

`--contributions` reports empirical historical component ES. Each asset's
signed loss contribution is averaged over the portfolio's historical ES tail;
hedges can therefore have negative contributions, and contributions sum to
portfolio ES up to floating-point rounding.

`--stress` replays each historical return row and reports the five worst days,
then applies uniform hypothetical simple-return shocks of -20% and -40% to all
held assets. The library API also accepts a per-asset simple-return shock
vector in returns-column order.

Example:

```powershell
cargo run --release -- .\examples\real_data\binance_20_assets\returns.parquet .\examples\real_data\binance_20_assets\positions.csv --model historical --confidence 0.99 --contributions --stress
```

## Real-data rolling backtest

The bundled dataset contains 1,825 aligned daily return observations for 20
USDT-quoted Binance spot assets from 2021-11-18 through 2026-11-16. Positions
are long 10,000 USDT per asset. The 99% backtest used a 250-observation lookback
and 5,000 Monte Carlo scenarios per forecast, producing 1,575 out-of-sample
forecasts per model:

| Model | Exceptions | Kupiec p-value | Kupiec rejects | Christoffersen p-value | Independence rejects |
|---|---:|---:|:---:|---:|:---:|
| Historical simulation | 12 | 0.321557 | No | 0.077865 | No |
| EWMA normal | 39 | 0.000001 | Yes | 0.342871 | No |
| Normal Monte Carlo | 32 | 0.000305 | Yes | 0.027852 | Yes |
| Student-t Monte Carlo | 27 | 0.009711 | Yes | 0.085306 | No |

These are historical diagnostics for this particular asset universe and period,
not evidence that the historical model will remain calibrated or that the
Binance sample represents a real holdings book.

## CPU benchmark

`benchmark-results.csv` contains the final 1-million-path, 50-asset run on an
11th Gen Intel Core i5-1135G7 Windows x86-64 system, with AVX2 selected and
seed 42. Mean simulation times:

| Backend | Threads | Mean time |
|---|---:|---:|
| Rust scalar | 1 | 1,895 ms |
| Rust SIMD | 1 | 1,571 ms |
| Rayon scalar | 8 | 538 ms |
| Rayon SIMD | 8 | 430 ms |
| C++ AVX2 | 1 | 5,846 ms |
| Rayon + C++ AVX2 | 8 | 1,673 ms |

The C++ backend matched Rust scalar within about `1.02e-6` maximum absolute
P&L error, but it was slower on this machine; it is retained as an experimental
backend, not as the recommended performance path. Benchmark results depend
on power settings and system load. See [BENCHMARKS.md](BENCHMARKS.md) for the
reproducible command and methodology.

## Build, test, and run

Requirements: Rust stable, CMake 3.20+, and a C++20 compiler. Windows/MSVC was
validated. The C++ build includes runtime AVX2 detection and a scalar fallback.
Linux GCC/Clang validation could not be completed in the current environment:
there is no installed Linux distribution or active Docker engine.

```powershell
cargo test
cargo run --release -- .\examples\real_data\binance_20_assets\returns.parquet .\examples\real_data\binance_20_assets\positions.csv --model all --confidence 0.99
cargo run --release --example benchmark -- --scenarios 1000000 --assets 50 --threads 1,2,4,8 --warmups 2 --repetitions 5 --chunk-size 10000 --seed 42
```
