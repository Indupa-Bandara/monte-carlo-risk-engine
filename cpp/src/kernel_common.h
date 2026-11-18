#ifndef RISK_ENGINE_KERNEL_COMMON_H
#define RISK_ENGINE_KERNEL_COMMON_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace risk_engine_internal {

constexpr std::uint32_t philox_m0 = 0xD2511F53U;
constexpr std::uint32_t philox_m1 = 0xCD9E8D57U;
constexpr std::uint32_t philox_w0 = 0x9E3779B9U;
constexpr std::uint32_t philox_w1 = 0xBB67AE85U;

struct Philox {
    std::uint32_t key[2];
    std::uint32_t counter[4];

    Philox(std::uint64_t seed, std::uint64_t path)
        : key{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)},
          counter{static_cast<std::uint32_t>(path), static_cast<std::uint32_t>(path >> 32), 0, 0} {}

    static void multiply_high_low(std::uint32_t left, std::uint32_t right,
                                  std::uint32_t& high, std::uint32_t& low) {
        const auto product = static_cast<std::uint64_t>(left) * right;
        high = static_cast<std::uint32_t>(product >> 32);
        low = static_cast<std::uint32_t>(product);
    }

    static void round(std::uint32_t (&words)[4], std::uint32_t (&round_key)[2]) {
        std::uint32_t high0;
        std::uint32_t low0;
        std::uint32_t high1;
        std::uint32_t low1;
        multiply_high_low(philox_m0, words[0], high0, low0);
        multiply_high_low(philox_m1, words[2], high1, low1);
        words[0] = high1 ^ words[1] ^ round_key[0];
        words[1] = low1;
        words[2] = high0 ^ words[3] ^ round_key[1];
        words[3] = low0;
        round_key[0] += philox_w0;
        round_key[1] += philox_w1;
    }

    void next(std::uint32_t (&output)[4]) {
        std::uint32_t words[4] = {counter[0], counter[1], counter[2], counter[3]};
        std::uint32_t round_key[2] = {key[0], key[1]};
        for (int index = 0; index < 10; ++index) {
            round(words, round_key);
        }
        for (int index = 0; index < 4; ++index) {
            output[index] = words[index];
        }
        ++counter[2];
        if (counter[2] == 0) {
            ++counter[3];
        }
    }
};

inline double open_unit(std::uint32_t value) {
    return (static_cast<double>(value) + 0.5) / 4294967296.0;
}

inline void normal_pair(std::uint32_t first, std::uint32_t second,
                        float& output_first, float& output_second) {
    const double u = open_unit(first);
    const double v = open_unit(second);
    const double radius = std::sqrt(-2.0 * std::log(u));
    const double angle = 2.0 * std::numbers::pi_v<double> * v;
    output_first = static_cast<float>(radius * std::cos(angle));
    output_second = static_cast<float>(radius * std::sin(angle));
}

inline void fill_shocks(std::uint64_t seed, std::uint64_t path, float* output,
                        std::size_t n_assets) {
    Philox rng(seed, path);
    std::size_t index = 0;
    while (index + 3 < n_assets) {
        std::uint32_t values[4];
        rng.next(values);
        normal_pair(values[0], values[1], output[index], output[index + 1]);
        normal_pair(values[2], values[3], output[index + 2], output[index + 3]);
        index += 4;
    }
    while (index < n_assets) {
        std::uint32_t values[4];
        rng.next(values);
        float unused;
        normal_pair(values[0], values[1], output[index], unused);
        ++index;
    }
}

inline double value_pnl(const double* weights, const double* correlated,
                        std::size_t n_assets) {
    double pnl = 0.0;
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        pnl += weights[asset] * (std::exp(correlated[asset]) - 1.0);
    }
    return pnl;
}

inline void correlate_one_scalar(const double* cholesky_l, const float* shocks,
                                 double* correlated, std::size_t n_assets) {
    for (std::size_t row = 0; row < n_assets; ++row) {
        double sum = 0.0;
        const std::size_t base = row * n_assets;
        for (std::size_t col = 0; col <= row; ++col) {
            sum += cholesky_l[base + col] * static_cast<double>(shocks[col]);
        }
        correlated[row] = sum;
    }
}

inline void simulate_block_scalar(
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
        fill_shocks(seed, first_path + path, path_shocks, n_assets);
        correlate_one_scalar(cholesky_l, path_shocks, path_correlated, n_assets);
        pnl[path] = value_pnl(weights, path_correlated, n_assets);
    }
}

}

#endif
