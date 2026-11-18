#ifndef RISK_ENGINE_CPP_H
#define RISK_ENGINE_CPP_H

#include <cstddef>
#include <cstdint>

extern "C" {

int risk_engine_cpp_simulate_block(
    const double* weights,
    const double* cholesky_l,
    std::size_t n_assets,
    std::uint64_t seed,
    std::uint64_t first_path,
    std::size_t path_count,
    std::size_t scratch_capacity,
    float* shocks,
    double* correlated,
    double* pnl);

int risk_engine_cpp_fill_shocks(
    std::uint64_t seed,
    std::uint64_t first_path,
    std::size_t n_assets,
    std::size_t path_count,
    std::size_t scratch_capacity,
    float* shocks);

int risk_engine_cpp_correlate_batch(
    const double* cholesky_l,
    std::size_t n_assets,
    std::size_t path_count,
    std::size_t scratch_capacity,
    const float* shocks,
    double* correlated);

int risk_engine_cpp_value_batch(
    const double* weights,
    std::size_t n_assets,
    std::size_t path_count,
    std::size_t scratch_capacity,
    const double* correlated,
    double* pnl);

int risk_engine_cpp_uses_avx2();

}

#endif
