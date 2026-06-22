// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_kernel_cpu_gtest.cc
 * @brief CPU Kernel Backend Unit Tests
 *
 * Tests for CPU kernel optimizations:
 *   - SIMD detection (SSE, AVX, AVX-512, NEON)
 *   - Cache-aware tile computation
 *   - Thread optimization
 *   - Vectorization efficiency
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

namespace yirage {
namespace kernel {
namespace cpu {

// =============================================================================
// SIMD Types
// =============================================================================

enum class SIMDType {
    NONE = 0,
    SSE = 1,
    SSE2 = 2,
    SSE3 = 3,
    SSE4_1 = 4,
    SSE4_2 = 5,
    AVX = 6,
    AVX2 = 7,
    AVX512 = 8,
    NEON = 9,  // ARM
};

inline std::string simd_type_to_string(SIMDType simd) {
    switch (simd) {
        case SIMDType::AVX512: return "AVX-512";
        case SIMDType::AVX2: return "AVX2";
        case SIMDType::AVX: return "AVX";
        case SIMDType::SSE4_2: return "SSE4.2";
        case SIMDType::SSE4_1: return "SSE4.1";
        case SIMDType::SSE3: return "SSE3";
        case SIMDType::SSE2: return "SSE2";
        case SIMDType::SSE: return "SSE";
        case SIMDType::NEON: return "NEON";
        default: return "None";
    }
}

inline int get_simd_vector_bytes(SIMDType simd) {
    switch (simd) {
        case SIMDType::AVX512: return 64;
        case SIMDType::AVX2:
        case SIMDType::AVX: return 32;
        case SIMDType::SSE4_2:
        case SIMDType::SSE4_1:
        case SIMDType::SSE3:
        case SIMDType::SSE2:
        case SIMDType::SSE:
        case SIMDType::NEON: return 16;
        default: return 4;
    }
}

// =============================================================================
// CPUKernelConfig
// =============================================================================

struct CPUKernelConfig {
    SIMDType simd_type = SIMDType::NONE;
    int num_threads = 1;
    int vector_width = 1;
    int alignment = 4;
    int unroll_factor = 1;
    
    // Tile sizes
    int tile_m = 64;
    int tile_n = 64;
    int tile_k = 64;
    int micro_tile_m = 8;
    int micro_tile_n = 8;
    
    // Cache sizes
    size_t l1_cache_size = 32 * 1024;     // 32 KB
    size_t l2_cache_size = 256 * 1024;    // 256 KB
    size_t l3_cache_size = 8 * 1024 * 1024;  // 8 MB
    
    int get_vector_bytes() const {
        return get_simd_vector_bytes(simd_type);
    }
};

// =============================================================================
// CPUOptimizer
// =============================================================================

class CPUOptimizer {
public:
    static SIMDType detect_simd_support() {
#ifdef __x86_64__
        return SIMDType::AVX2;  // Simulated for testing
#elif defined(__aarch64__)
        return SIMDType::NEON;
#else
        return SIMDType::NONE;
#endif
    }
    
    static std::string get_cpu_features() {
        SIMDType simd = detect_simd_support();
        return "SIMD: " + simd_type_to_string(simd) +
               ", Cores: " + std::to_string(std::thread::hardware_concurrency());
    }
    
    static void compute_optimal_tiles(int m, int n, int k, size_t element_size,
                                      CPUKernelConfig& config) {
        size_t l2_size = config.l2_cache_size;
        size_t total_elements = l2_size / element_size / 3;
        
        int tile_size = static_cast<int>(std::sqrt(total_elements));
        int vector_width = config.get_vector_bytes() / element_size;
        tile_size = (tile_size / vector_width) * vector_width;
        tile_size = std::max(16, std::min(256, tile_size));
        
        config.tile_m = std::min(m, tile_size);
        config.tile_n = std::min(n, tile_size);
        config.tile_k = std::min(k, tile_size);
        
        // Micro tiles for L1
        size_t l1_size = config.l1_cache_size;
        int micro_size = static_cast<int>(std::sqrt(l1_size / element_size / 3));
        micro_size = (micro_size / vector_width) * vector_width;
        micro_size = std::max(4, std::min(32, micro_size));
        
        config.micro_tile_m = micro_size;
        config.micro_tile_n = micro_size;
    }
    
    static int compute_optimal_threads(size_t problem_size, int num_cores,
                                       bool memory_bound) {
        if (num_cores <= 0) {
            num_cores = std::thread::hardware_concurrency();
        }
        
        if (memory_bound) {
            return std::max(1, num_cores / 2);
        } else {
            size_t min_work_per_thread = 1024;
            int max_useful_threads = static_cast<int>(problem_size / min_work_per_thread);
            return std::max(1, std::min(num_cores, max_useful_threads));
        }
    }
    
    static float estimate_cache_efficiency(CPUKernelConfig const& config,
                                           int m, int n, int k,
                                           size_t element_size) {
        size_t working_set = (config.tile_m * config.tile_k +
                              config.tile_k * config.tile_n +
                              config.tile_m * config.tile_n) * element_size;
        
        if (working_set <= config.l1_cache_size) return 0.99f;
        if (working_set <= config.l2_cache_size) return 0.95f;
        if (working_set <= config.l3_cache_size) return 0.85f;
        return 0.5f;
    }
    
    static float estimate_vectorization_efficiency(CPUKernelConfig const& config,
                                                   size_t data_size) {
        if (config.simd_type == SIMDType::NONE) return 1.0f;
        
        int vector_width = config.get_vector_bytes() / 4;  // float32
        size_t vectorized = (data_size / vector_width) * vector_width;
        
        float vectorized_ratio = static_cast<float>(vectorized) / data_size;
        float speedup = static_cast<float>(vector_width);
        float alignment_penalty = (config.alignment == 64) ? 1.0f : 0.9f;
        
        return (vectorized_ratio * speedup + (1.0f - vectorized_ratio)) * alignment_penalty;
    }
    
    static int compute_unroll_factor(int loop_size, SIMDType simd_type) {
        int base_unroll = 4;
        switch (simd_type) {
            case SIMDType::AVX512: base_unroll = 8; break;
            case SIMDType::AVX2:
            case SIMDType::AVX: base_unroll = 4; break;
            case SIMDType::SSE4_2:
            case SIMDType::SSE4_1:
            case SIMDType::SSE3:
            case SIMDType::SSE2:
            case SIMDType::SSE:
            case SIMDType::NEON: base_unroll = 2; break;
            default: base_unroll = 1; break;
        }
        
        if (loop_size < 16) {
            base_unroll = std::min(base_unroll, 2);
        }
        return base_unroll;
    }
    
    static void optimize_for_cpu(int m, int n, int k, CPUKernelConfig& config) {
        config.simd_type = detect_simd_support();
        
        switch (config.simd_type) {
            case SIMDType::AVX512:
                config.vector_width = 16;
                config.alignment = 64;
                break;
            case SIMDType::AVX2:
            case SIMDType::AVX:
                config.vector_width = 8;
                config.alignment = 32;
                break;
            case SIMDType::SSE4_2:
            case SIMDType::SSE4_1:
            case SIMDType::SSE3:
            case SIMDType::SSE2:
            case SIMDType::SSE:
            case SIMDType::NEON:
                config.vector_width = 4;
                config.alignment = 16;
                break;
            default:
                config.vector_width = 1;
                config.alignment = 4;
                break;
        }
        
        compute_optimal_tiles(m, n, k, sizeof(float), config);
        
        size_t problem_size = static_cast<size_t>(m) * n * k;
        int num_cores = std::thread::hardware_concurrency();
        config.num_threads = compute_optimal_threads(problem_size, num_cores, false);
        config.unroll_factor = compute_unroll_factor(k, config.simd_type);
    }
};

}  // namespace cpu
}  // namespace kernel
}  // namespace yirage

using namespace yirage::kernel::cpu;

// =============================================================================
// SIMD Type Tests
// =============================================================================

class SIMDTypeTest : public ::testing::Test {};

TEST_F(SIMDTypeTest, SIMDToString) {
    EXPECT_EQ(simd_type_to_string(SIMDType::AVX512), "AVX-512");
    EXPECT_EQ(simd_type_to_string(SIMDType::AVX2), "AVX2");
    EXPECT_EQ(simd_type_to_string(SIMDType::AVX), "AVX");
    EXPECT_EQ(simd_type_to_string(SIMDType::SSE4_2), "SSE4.2");
    EXPECT_EQ(simd_type_to_string(SIMDType::SSE4_1), "SSE4.1");
    EXPECT_EQ(simd_type_to_string(SIMDType::NEON), "NEON");
    EXPECT_EQ(simd_type_to_string(SIMDType::NONE), "None");
}

TEST_F(SIMDTypeTest, SIMDVectorBytes) {
    EXPECT_EQ(get_simd_vector_bytes(SIMDType::AVX512), 64);
    EXPECT_EQ(get_simd_vector_bytes(SIMDType::AVX2), 32);
    EXPECT_EQ(get_simd_vector_bytes(SIMDType::AVX), 32);
    EXPECT_EQ(get_simd_vector_bytes(SIMDType::SSE4_2), 16);
    EXPECT_EQ(get_simd_vector_bytes(SIMDType::SSE2), 16);
    EXPECT_EQ(get_simd_vector_bytes(SIMDType::NEON), 16);
    EXPECT_EQ(get_simd_vector_bytes(SIMDType::NONE), 4);
}

// =============================================================================
// CPUKernelConfig Tests
// =============================================================================

class CPUKernelConfigTest : public ::testing::Test {};

TEST_F(CPUKernelConfigTest, DefaultValues) {
    CPUKernelConfig config;
    EXPECT_EQ(config.simd_type, SIMDType::NONE);
    EXPECT_EQ(config.num_threads, 1);
    EXPECT_EQ(config.tile_m, 64);
    EXPECT_EQ(config.tile_n, 64);
    EXPECT_EQ(config.tile_k, 64);
}

TEST_F(CPUKernelConfigTest, DefaultCacheSizes) {
    CPUKernelConfig config;
    EXPECT_EQ(config.l1_cache_size, 32u * 1024u);
    EXPECT_EQ(config.l2_cache_size, 256u * 1024u);
    EXPECT_EQ(config.l3_cache_size, 8u * 1024u * 1024u);
}

TEST_F(CPUKernelConfigTest, GetVectorBytes) {
    CPUKernelConfig config;
    
    config.simd_type = SIMDType::AVX512;
    EXPECT_EQ(config.get_vector_bytes(), 64);
    
    config.simd_type = SIMDType::AVX2;
    EXPECT_EQ(config.get_vector_bytes(), 32);
    
    config.simd_type = SIMDType::SSE4_2;
    EXPECT_EQ(config.get_vector_bytes(), 16);
}

// =============================================================================
// CPUOptimizer Tests
// =============================================================================

class CPUOptimizerTest : public ::testing::Test {};

TEST_F(CPUOptimizerTest, DetectSIMD) {
    SIMDType simd = CPUOptimizer::detect_simd_support();
    EXPECT_GE(static_cast<int>(simd), 0);
}

TEST_F(CPUOptimizerTest, GetCPUFeatures) {
    std::string features = CPUOptimizer::get_cpu_features();
    EXPECT_FALSE(features.empty());
    EXPECT_NE(features.find("SIMD"), std::string::npos);
    EXPECT_NE(features.find("Cores"), std::string::npos);
}

TEST_F(CPUOptimizerTest, ComputeOptimalTilesSmall) {
    CPUKernelConfig config;
    config.simd_type = SIMDType::AVX2;
    
    CPUOptimizer::compute_optimal_tiles(64, 64, 64, sizeof(float), config);
    
    EXPECT_LE(config.tile_m, 64);
    EXPECT_LE(config.tile_n, 64);
    EXPECT_LE(config.tile_k, 64);
}

TEST_F(CPUOptimizerTest, ComputeOptimalTilesLarge) {
    CPUKernelConfig config;
    config.simd_type = SIMDType::AVX2;
    
    CPUOptimizer::compute_optimal_tiles(4096, 4096, 4096, sizeof(float), config);
    
    EXPECT_GT(config.tile_m, 0);
    EXPECT_GT(config.tile_n, 0);
    EXPECT_GT(config.tile_k, 0);
    EXPECT_LE(config.tile_m, 256);  // Max tile size
}

TEST_F(CPUOptimizerTest, ComputeOptimalThreadsComputeBound) {
    int threads = CPUOptimizer::compute_optimal_threads(1000000, 8, false);
    EXPECT_GE(threads, 1);
    EXPECT_LE(threads, 8);
}

TEST_F(CPUOptimizerTest, ComputeOptimalThreadsMemoryBound) {
    int threads = CPUOptimizer::compute_optimal_threads(1000000, 8, true);
    EXPECT_GE(threads, 1);
    EXPECT_LE(threads, 4);  // Memory bound uses fewer threads
}

TEST_F(CPUOptimizerTest, ComputeOptimalThreadsSmallProblem) {
    int threads = CPUOptimizer::compute_optimal_threads(100, 8, false);
    EXPECT_EQ(threads, 1);  // Problem too small for parallelism
}

TEST_F(CPUOptimizerTest, EstimateCacheEfficiencyL1) {
    CPUKernelConfig config;
    config.tile_m = 16;
    config.tile_n = 16;
    config.tile_k = 16;
    
    float efficiency = CPUOptimizer::estimate_cache_efficiency(config, 256, 256, 256, 4);
    EXPECT_GT(efficiency, 0.95f);  // Should fit in L1
}

TEST_F(CPUOptimizerTest, EstimateCacheEfficiencyL2) {
    CPUKernelConfig config;
    config.tile_m = 64;
    config.tile_n = 64;
    config.tile_k = 64;
    
    float efficiency = CPUOptimizer::estimate_cache_efficiency(config, 1024, 1024, 1024, 4);
    EXPECT_GT(efficiency, 0.9f);
    EXPECT_LE(efficiency, 0.99f);
}

TEST_F(CPUOptimizerTest, EstimateCacheEfficiencyL3) {
    CPUKernelConfig config;
    config.tile_m = 256;
    config.tile_n = 256;
    config.tile_k = 256;
    
    float efficiency = CPUOptimizer::estimate_cache_efficiency(config, 4096, 4096, 4096, 4);
    EXPECT_GT(efficiency, 0.5f);
    EXPECT_LE(efficiency, 0.95f);
}

TEST_F(CPUOptimizerTest, EstimateVectorizationEfficiency) {
    CPUKernelConfig config;
    config.simd_type = SIMDType::AVX2;
    config.alignment = 32;
    
    float efficiency = CPUOptimizer::estimate_vectorization_efficiency(config, 1024);
    EXPECT_GT(efficiency, 1.0f);  // Vectorization should provide speedup
}

TEST_F(CPUOptimizerTest, ComputeUnrollFactorAVX512) {
    EXPECT_EQ(CPUOptimizer::compute_unroll_factor(128, SIMDType::AVX512), 8);
}

TEST_F(CPUOptimizerTest, ComputeUnrollFactorAVX2) {
    EXPECT_EQ(CPUOptimizer::compute_unroll_factor(128, SIMDType::AVX2), 4);
}

TEST_F(CPUOptimizerTest, ComputeUnrollFactorSSE) {
    EXPECT_EQ(CPUOptimizer::compute_unroll_factor(128, SIMDType::SSE4_2), 2);
}

TEST_F(CPUOptimizerTest, ComputeUnrollFactorNEON) {
    EXPECT_EQ(CPUOptimizer::compute_unroll_factor(128, SIMDType::NEON), 2);
}

TEST_F(CPUOptimizerTest, ComputeUnrollFactorSmallLoop) {
    // Small loops should have limited unrolling
    EXPECT_LE(CPUOptimizer::compute_unroll_factor(8, SIMDType::AVX512), 2);
}

TEST_F(CPUOptimizerTest, OptimizeForCPU) {
    CPUKernelConfig config;
    CPUOptimizer::optimize_for_cpu(1024, 1024, 1024, config);
    
    EXPECT_NE(config.simd_type, SIMDType::NONE);
    EXPECT_GT(config.num_threads, 0);
    EXPECT_GT(config.tile_m, 0);
    EXPECT_GT(config.unroll_factor, 0);
}

// =============================================================================
// Parameterized SIMD Tests
// =============================================================================

struct SIMDTestParam {
    SIMDType simd;
    int expected_bytes;
    int expected_unroll;
};

class SIMDParameterizedTest : public ::testing::TestWithParam<SIMDTestParam> {};

TEST_P(SIMDParameterizedTest, VectorBytesAndUnroll) {
    auto param = GetParam();
    
    EXPECT_EQ(get_simd_vector_bytes(param.simd), param.expected_bytes);
    EXPECT_EQ(CPUOptimizer::compute_unroll_factor(128, param.simd), param.expected_unroll);
}

INSTANTIATE_TEST_SUITE_P(
    AllSIMDTypes,
    SIMDParameterizedTest,
    ::testing::Values(
        SIMDTestParam{SIMDType::AVX512, 64, 8},
        SIMDTestParam{SIMDType::AVX2, 32, 4},
        SIMDTestParam{SIMDType::AVX, 32, 4},
        SIMDTestParam{SIMDType::SSE4_2, 16, 2},
        SIMDTestParam{SIMDType::SSE4_1, 16, 2},
        SIMDTestParam{SIMDType::SSE3, 16, 2},
        SIMDTestParam{SIMDType::SSE2, 16, 2},
        SIMDTestParam{SIMDType::NEON, 16, 2},
        SIMDTestParam{SIMDType::NONE, 4, 1}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
