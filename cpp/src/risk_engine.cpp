#include "risk_engine.h"

#include "kernel_common.h"

#include <limits>

#if defined(RISK_ENGINE_HAS_AVX2_KERNEL) && defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#elif defined(RISK_ENGINE_HAS_AVX2_KERNEL)
#include <cpuid.h>
#endif

namespace risk_engine_internal {

#if defined(RISK_ENGINE_HAS_AVX2_KERNEL)
void simulate_block_avx2(
    const double* weights,
    const double* cholesky_l,
    std::size_t n_assets,
    std::uint64_t seed,
    std::uint64_t first_path,
    std::size_t path_count,
    float* shocks,
    double* correlated,
    double* pnl);
void correlate_one_avx2(const double* cholesky_l, const float* shocks,
                        double* correlated, std::size_t n_assets);
#endif

}

namespace {

bool detect_avx2() {
#if defined(RISK_ENGINE_HAS_AVX2_KERNEL) && defined(_MSC_VER)
    int registers[4];
    __cpuid(registers, 1);
    constexpr int osxsave_bit = 1 << 27;
    constexpr int avx_bit = 1 << 28;
    if ((registers[2] & (osxsave_bit | avx_bit)) != (osxsave_bit | avx_bit)) {
        return false;
    }
    if ((GetEnabledXStateFeatures() & 0x6) != 0x6) {
        return false;
    }
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#elif defined(RISK_ENGINE_HAS_AVX2_KERNEL)
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
    if (__get_cpuid_max(0, nullptr) < 7 ||
        !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    constexpr unsigned int osxsave_bit = 1U << 27;
    constexpr unsigned int avx_bit = 1U << 28;
    if ((ecx & (osxsave_bit | avx_bit)) != (osxsave_bit | avx_bit)) {
        return false;
    }
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

bool avx2_available() {
    static const bool available = detect_avx2();
    return available;
}

bool valid_batch(std::size_t n_assets, std::size_t path_count,
                 std::size_t scratch_capacity) {
    return n_assets != 0 && path_count <= scratch_capacity &&
           n_assets <= std::numeric_limits<std::size_t>::max() / n_assets &&
           path_count <= std::numeric_limits<std::size_t>::max() / n_assets;
}

bool valid_path_range(std::uint64_t first_path, std::size_t path_count) {
    return path_count == 0 ||
           path_count - 1 <= std::numeric_limits<std::uint64_t>::max() - first_path;
}

void correlate_one_dispatch(const double* cholesky_l, const float* shocks,
                            double* correlated, std::size_t n_assets,
                            bool use_avx2) {
#if defined(RISK_ENGINE_HAS_AVX2_KERNEL)
    if (use_avx2) {
        risk_engine_internal::correlate_one_avx2(cholesky_l, shocks, correlated, n_assets);
        return;
    }
#else
    (void)use_avx2;
#endif
    risk_engine_internal::correlate_one_scalar(cholesky_l, shocks, correlated, n_assets);
}

}

extern "C" int risk_engine_cpp_simulate_block(
    const double* weights,
    const double* cholesky_l,
    std::size_t n_assets,
    std::uint64_t seed,
    std::uint64_t first_path,
    std::size_t path_count,
    std::size_t scratch_capacity,
    float* shocks,
    double* correlated,
    double* pnl) {
    if (!valid_batch(n_assets, path_count, scratch_capacity) ||
        !valid_path_range(first_path, path_count) ||
        weights == nullptr || cholesky_l == nullptr || shocks == nullptr ||
        correlated == nullptr || pnl == nullptr) {
        return 1;
    }

#if defined(RISK_ENGINE_HAS_AVX2_KERNEL)
    if (avx2_available()) {
        risk_engine_internal::simulate_block_avx2(
            weights, cholesky_l, n_assets, seed, first_path, path_count, shocks,
            correlated, pnl);
        return 0;
    }
#endif
    risk_engine_internal::simulate_block_scalar(
        weights, cholesky_l, n_assets, seed, first_path, path_count, shocks,
        correlated, pnl);
    return 0;
}

extern "C" int risk_engine_cpp_fill_shocks(
    std::uint64_t seed,
    std::uint64_t first_path,
    std::size_t n_assets,
    std::size_t path_count,
    std::size_t scratch_capacity,
    float* shocks) {
    if (!valid_batch(n_assets, path_count, scratch_capacity) ||
        !valid_path_range(first_path, path_count) || shocks == nullptr) {
        return 1;
    }
    for (std::size_t path = 0; path < path_count; ++path) {
        risk_engine_internal::fill_shocks(seed, first_path + path, shocks + path * n_assets,
                                          n_assets);
    }
    return 0;
}

extern "C" int risk_engine_cpp_correlate_batch(
    const double* cholesky_l,
    std::size_t n_assets,
    std::size_t path_count,
    std::size_t scratch_capacity,
    const float* shocks,
    double* correlated) {
    if (!valid_batch(n_assets, path_count, scratch_capacity) || cholesky_l == nullptr ||
        shocks == nullptr || correlated == nullptr) {
        return 1;
    }
    const bool use_avx2 = avx2_available();
    for (std::size_t path = 0; path < path_count; ++path) {
        correlate_one_dispatch(cholesky_l, shocks + path * n_assets,
                               correlated + path * n_assets, n_assets, use_avx2);
    }
    return 0;
}

extern "C" int risk_engine_cpp_value_batch(
    const double* weights,
    std::size_t n_assets,
    std::size_t path_count,
    std::size_t scratch_capacity,
    const double* correlated,
    double* pnl) {
    if (!valid_batch(n_assets, path_count, scratch_capacity) || weights == nullptr ||
        correlated == nullptr || pnl == nullptr) {
        return 1;
    }
    for (std::size_t path = 0; path < path_count; ++path) {
        pnl[path] = risk_engine_internal::value_pnl(weights, correlated + path * n_assets,
                                                    n_assets);
    }
    return 0;
}

extern "C" int risk_engine_cpp_uses_avx2() {
    return avx2_available() ? 1 : 0;
}
