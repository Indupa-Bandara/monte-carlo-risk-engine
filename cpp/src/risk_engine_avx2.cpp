#include "risk_engine.h"

#include "kernel_common.h"

#include <immintrin.h>

namespace risk_engine_internal {

void correlate_one_avx2(const double* cholesky_l, const float* shocks,
                        double* correlated, std::size_t n_assets) {
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        correlated[asset] = static_cast<double>(shocks[asset]);
    }

    // Descending rows keeps each output from overwriting inputs needed by later rows.
    for (std::size_t row = n_assets; row-- > 0;) {
        const std::size_t base = row * n_assets;
        const std::size_t end = row + 1;
        std::size_t col = 0;
        __m256d sums = _mm256_setzero_pd();
        for (; col + 4 <= end; col += 4) {
            const __m256d shock_f64 = _mm256_loadu_pd(correlated + col);
            const __m256d coefficients = _mm256_loadu_pd(cholesky_l + base + col);
            sums = _mm256_add_pd(sums, _mm256_mul_pd(coefficients, shock_f64));
        }

        alignas(32) double lanes[4];
        _mm256_store_pd(lanes, sums);
        double sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for (; col < end; ++col) {
            sum += cholesky_l[base + col] * correlated[col];
        }
        correlated[row] = sum;
    }
}

void simulate_block_avx2(
    const double* weights,
    const double* cholesky_l,
    std::size_t n_assets,
    std::uint64_t seed,
    std::uint64_t first_path,
    std::size_t path_count,
    float* shocks,
    double* correlated,
    double* pnl) {
    for (std::size_t path = 0; path < path_count; ++path) {
        float* path_shocks = shocks + path * n_assets;
        double* path_correlated = correlated + path * n_assets;
        risk_engine_internal::fill_shocks(seed, first_path + path, path_shocks, n_assets);
        correlate_one_avx2(cholesky_l, path_shocks, path_correlated, n_assets);
        pnl[path] = risk_engine_internal::value_pnl(weights, path_correlated, n_assets);
    }
}

}
