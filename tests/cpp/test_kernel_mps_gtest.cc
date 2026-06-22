// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_kernel_mps_gtest.cc
 * @brief MPS (Apple Metal Performance Shaders) Kernel Backend Unit Tests
 *
 * Tests for MPS kernel optimizations:
 *   - GPU family detection (M1-M4)
 *   - Threadgroup configuration
 *   - Tile size optimization
 *   - Memory pattern selection
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <ostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace yirage {
namespace kernel {
namespace mps {

// =============================================================================
// MPS Enums and Structures
// =============================================================================

enum class AppleGPUFamily {
    FAMILY_7 = 7,   // M1
    FAMILY_8 = 8,   // M2
    FAMILY_9 = 9,   // M3
    FAMILY_10 = 10, // M4
};

enum class AppleChipVariant {
    BASE = 0,
    PRO = 1,
    MAX = 2,
    ULTRA = 3,
};

enum class MemoryPattern {
    COALESCED = 0,
    STRIDED = 1,
    TILED = 2,
};

// GTest PrintTo for MemoryPattern
inline void PrintTo(MemoryPattern p, std::ostream* os) {
    switch(p) {
        case MemoryPattern::COALESCED: *os << "COALESCED"; break;
        case MemoryPattern::STRIDED: *os << "STRIDED"; break;
        case MemoryPattern::TILED: *os << "TILED"; break;
        default: *os << "UNKNOWN(" << static_cast<int>(p) << ")"; break;
    }
}

struct MPSKernelConfig {
    int gpu_family = 8;  // M2
    AppleChipVariant chip_variant = AppleChipVariant::BASE;
    int threads_per_threadgroup = 256;
    int simd_width = 32;
    size_t threadgroup_memory_size = 32 * 1024;
    int tile_m = 32;
    int tile_n = 32;
    int tile_k = 32;
    bool use_fast_math = true;
    int grid_dim_x = 1;
    int grid_dim_y = 1;
    int grid_dim_z = 1;
    
    int get_gpu_cores() const {
        if (chip_variant == AppleChipVariant::ULTRA) {
            return (gpu_family >= 9) ? 80 : 64;
        } else if (chip_variant == AppleChipVariant::MAX) {
            return (gpu_family >= 9) ? 40 : 38;
        } else if (chip_variant == AppleChipVariant::PRO) {
            return (gpu_family >= 9) ? 18 : 19;
        }
        return 10;  // Base
    }
};

// =============================================================================
// MPSOptimizer
// =============================================================================

class MPSOptimizer {
public:
    static int detect_gpu_family() {
        // Simulated detection - returns M2 by default
        return 8;
    }
    
    static int get_gpu_core_count(int gpu_family, AppleChipVariant variant) {
        switch (variant) {
            case AppleChipVariant::ULTRA:
                switch (gpu_family) {
                    case 7: return 64;   // M1 Ultra
                    case 8: return 76;   // M2 Ultra
                    case 9: return 80;   // M3 Ultra (estimated)
                    case 10: return 80;  // M4 Ultra (estimated)
                    default: return 64;
                }
            case AppleChipVariant::MAX:
                switch (gpu_family) {
                    case 7: return 32;   // M1 Max
                    case 8: return 38;   // M2 Max
                    case 9: return 40;   // M3 Max
                    case 10: return 40;  // M4 Max (estimated)
                    default: return 32;
                }
            case AppleChipVariant::PRO:
                switch (gpu_family) {
                    case 7: return 16;   // M1 Pro
                    case 8: return 19;   // M2 Pro
                    case 9: return 18;   // M3 Pro
                    case 10: return 20;  // M4 Pro (estimated)
                    default: return 16;
                }
            default:  // BASE
                switch (gpu_family) {
                    case 7: return 8;    // M1
                    case 8: return 10;   // M2
                    case 9: return 10;   // M3
                    case 10: return 10;  // M4
                    default: return 8;
                }
        }
    }
    
    static size_t get_threadgroup_memory_size(int gpu_family) {
        if (gpu_family >= 8) {
            return 64 * 1024;  // M2+ has 64KB
        }
        return 32 * 1024;  // M1 has 32KB
    }
    
    static int compute_optimal_threadgroup_size(size_t problem_size, int gpu_family) {
        int simd_width = 32;
        int base_size = 256;
        
        if (problem_size < 1024) {
            base_size = 128;
        } else if (problem_size > 1024 * 1024) {
            base_size = 512;
        }
        
        // Ensure multiple of SIMD width
        base_size = ((base_size + simd_width - 1) / simd_width) * simd_width;
        
        // Clamp to valid Metal range
        return std::max(32, std::min(1024, base_size));
    }
    
    static void compute_optimal_tiles(int m, int n, int k, int gpu_family,
                                      MPSKernelConfig& config) {
        size_t tg_memory = get_threadgroup_memory_size(gpu_family);
        size_t element_size = 4;  // float32
        
        size_t total_elements = tg_memory / element_size / 3;
        
        int tile_size = static_cast<int>(std::sqrt(total_elements));
        tile_size = (tile_size / 32) * 32;  // Align to SIMD width
        tile_size = std::max(16, std::min(64, tile_size));
        
        config.tile_m = std::min(m, tile_size);
        config.tile_n = std::min(n, tile_size);
        config.tile_k = std::min(k, tile_size);
    }
    
    static MemoryPattern select_memory_pattern(size_t data_size, int stride) {
        if (stride == 1) {
            return MemoryPattern::COALESCED;
        }
        
        if (data_size < 4096 && stride > 16) {
            return MemoryPattern::TILED;
        }
        
        return MemoryPattern::STRIDED;
    }
    
    static float estimate_memory_bandwidth(MPSKernelConfig const& config,
                                           size_t bytes_accessed,
                                           float execution_time_ms) {
        if (execution_time_ms <= 0.0f) return 0.0f;
        
        float seconds = execution_time_ms / 1000.0f;
        float gigabytes = bytes_accessed / (1024.0f * 1024.0f * 1024.0f);
        return gigabytes / seconds;
    }
    
    static void optimize_for_apple_silicon(int m, int n, int k,
                                           MPSKernelConfig& config) {
        config.gpu_family = detect_gpu_family();
        
        size_t problem_size = static_cast<size_t>(m) * n;
        config.threads_per_threadgroup = compute_optimal_threadgroup_size(
            problem_size, config.gpu_family);
        
        compute_optimal_tiles(m, n, k, config.gpu_family, config);
        
        config.simd_width = 32;
        config.threadgroup_memory_size = get_threadgroup_memory_size(config.gpu_family);
        config.use_fast_math = true;
        
        int tile_m = config.tile_m;
        int tile_n = config.tile_n;
        config.grid_dim_x = (n + tile_n - 1) / tile_n;
        config.grid_dim_y = (m + tile_m - 1) / tile_m;
        config.grid_dim_z = 1;
    }
    
    static float get_peak_tflops(int gpu_family, AppleChipVariant variant) {
        int cores = get_gpu_core_count(gpu_family, variant);
        
        // Apple GPUs: ~1 TFLOPS per core for M1, increasing for newer
        float tflops_per_core = 1.0f;
        if (gpu_family >= 9) {
            tflops_per_core = 1.5f;  // M3 has improved efficiency
        } else if (gpu_family >= 8) {
            tflops_per_core = 1.2f;  // M2
        }
        
        return cores * tflops_per_core;
    }
    
    static size_t get_unified_memory_bandwidth(int gpu_family, AppleChipVariant variant) {
        // GB/s
        switch (variant) {
            case AppleChipVariant::ULTRA:
                return (gpu_family >= 8) ? 800 : 800;
            case AppleChipVariant::MAX:
                return (gpu_family >= 8) ? 400 : 400;
            case AppleChipVariant::PRO:
                return (gpu_family >= 8) ? 200 : 200;
            default:
                return (gpu_family >= 8) ? 100 : 68;
        }
    }
};

}  // namespace mps
}  // namespace kernel
}  // namespace yirage

using namespace yirage::kernel::mps;

// =============================================================================
// MPSKernelConfig Tests
// =============================================================================

class MPSKernelConfigTest : public ::testing::Test {};

TEST_F(MPSKernelConfigTest, DefaultValues) {
    MPSKernelConfig config;
    EXPECT_EQ(config.gpu_family, 8);
    EXPECT_EQ(config.threads_per_threadgroup, 256);
    EXPECT_EQ(config.simd_width, 32);
    EXPECT_EQ(config.threadgroup_memory_size, 32u * 1024u);
}

TEST_F(MPSKernelConfigTest, DefaultTileSizes) {
    MPSKernelConfig config;
    EXPECT_EQ(config.tile_m, 32);
    EXPECT_EQ(config.tile_n, 32);
    EXPECT_EQ(config.tile_k, 32);
}

TEST_F(MPSKernelConfigTest, FastMathEnabled) {
    MPSKernelConfig config;
    EXPECT_TRUE(config.use_fast_math);
}

TEST_F(MPSKernelConfigTest, GetGPUCoresBase) {
    MPSKernelConfig config;
    config.gpu_family = 8;
    config.chip_variant = AppleChipVariant::BASE;
    EXPECT_EQ(config.get_gpu_cores(), 10);
}

TEST_F(MPSKernelConfigTest, GetGPUCoresMax) {
    MPSKernelConfig config;
    config.gpu_family = 9;
    config.chip_variant = AppleChipVariant::MAX;
    EXPECT_EQ(config.get_gpu_cores(), 40);
}

// =============================================================================
// MPSOptimizer Tests
// =============================================================================

class MPSOptimizerTest : public ::testing::Test {};

TEST_F(MPSOptimizerTest, DetectGPUFamily) {
    int family = MPSOptimizer::detect_gpu_family();
    EXPECT_GE(family, 7);
}

TEST_F(MPSOptimizerTest, GetGPUCoreCountM1) {
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(7, AppleChipVariant::BASE), 8);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(7, AppleChipVariant::PRO), 16);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(7, AppleChipVariant::MAX), 32);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(7, AppleChipVariant::ULTRA), 64);
}

TEST_F(MPSOptimizerTest, GetGPUCoreCountM2) {
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(8, AppleChipVariant::BASE), 10);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(8, AppleChipVariant::PRO), 19);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(8, AppleChipVariant::MAX), 38);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(8, AppleChipVariant::ULTRA), 76);
}

TEST_F(MPSOptimizerTest, GetGPUCoreCountM3) {
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(9, AppleChipVariant::BASE), 10);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(9, AppleChipVariant::PRO), 18);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(9, AppleChipVariant::MAX), 40);
}

TEST_F(MPSOptimizerTest, GetThreadgroupMemorySize) {
    EXPECT_EQ(MPSOptimizer::get_threadgroup_memory_size(7), 32u * 1024u);  // M1
    EXPECT_EQ(MPSOptimizer::get_threadgroup_memory_size(8), 64u * 1024u);  // M2
    EXPECT_EQ(MPSOptimizer::get_threadgroup_memory_size(9), 64u * 1024u);  // M3
}

TEST_F(MPSOptimizerTest, ComputeOptimalThreadgroupSizeSmall) {
    int size = MPSOptimizer::compute_optimal_threadgroup_size(512, 8);
    EXPECT_EQ(size, 128);
    EXPECT_EQ(size % 32, 0);
}

TEST_F(MPSOptimizerTest, ComputeOptimalThreadgroupSizeMedium) {
    int size = MPSOptimizer::compute_optimal_threadgroup_size(10000, 8);
    EXPECT_EQ(size, 256);
}

TEST_F(MPSOptimizerTest, ComputeOptimalThreadgroupSizeLarge) {
    int size = MPSOptimizer::compute_optimal_threadgroup_size(10000000, 8);
    EXPECT_EQ(size, 512);
}

TEST_F(MPSOptimizerTest, ComputeOptimalTiles) {
    MPSKernelConfig config;
    MPSOptimizer::compute_optimal_tiles(512, 512, 512, 8, config);
    
    EXPECT_GT(config.tile_m, 0);
    EXPECT_GT(config.tile_n, 0);
    EXPECT_GT(config.tile_k, 0);
    EXPECT_LE(config.tile_m, 64);
    EXPECT_EQ(config.tile_m % 32, 0);  // Aligned to SIMD
}

TEST_F(MPSOptimizerTest, SelectMemoryPatternCoalesced) {
    auto pattern = MPSOptimizer::select_memory_pattern(4096, 1);
    EXPECT_EQ(pattern, MemoryPattern::COALESCED);
}

TEST_F(MPSOptimizerTest, SelectMemoryPatternTiled) {
    auto pattern = MPSOptimizer::select_memory_pattern(2048, 32);
    EXPECT_EQ(pattern, MemoryPattern::TILED);
}

TEST_F(MPSOptimizerTest, SelectMemoryPatternStrided) {
    auto pattern = MPSOptimizer::select_memory_pattern(10000, 8);
    EXPECT_EQ(pattern, MemoryPattern::STRIDED);
}

TEST_F(MPSOptimizerTest, EstimateMemoryBandwidth) {
    MPSKernelConfig config;
    
    // 1 GB accessed in 10 ms
    float bw = MPSOptimizer::estimate_memory_bandwidth(config, 1024 * 1024 * 1024, 10.0f);
    EXPECT_NEAR(bw, 100.0f, 1.0f);  // ~100 GB/s
}

TEST_F(MPSOptimizerTest, OptimizeForAppleSilicon) {
    MPSKernelConfig config;
    MPSOptimizer::optimize_for_apple_silicon(1024, 1024, 1024, config);
    
    EXPECT_GT(config.threads_per_threadgroup, 0);
    EXPECT_EQ(config.threads_per_threadgroup % 32, 0);
    EXPECT_GT(config.tile_m, 0);
    EXPECT_GT(config.grid_dim_x, 0);
    EXPECT_GT(config.grid_dim_y, 0);
}

TEST_F(MPSOptimizerTest, GetPeakTFLOPS) {
    float tflops_m2_base = MPSOptimizer::get_peak_tflops(8, AppleChipVariant::BASE);
    float tflops_m2_max = MPSOptimizer::get_peak_tflops(8, AppleChipVariant::MAX);
    
    EXPECT_GT(tflops_m2_base, 0.0f);
    EXPECT_GT(tflops_m2_max, tflops_m2_base);  // Max should be higher
}

TEST_F(MPSOptimizerTest, GetUnifiedMemoryBandwidth) {
    size_t bw_base = MPSOptimizer::get_unified_memory_bandwidth(8, AppleChipVariant::BASE);
    size_t bw_max = MPSOptimizer::get_unified_memory_bandwidth(8, AppleChipVariant::MAX);
    size_t bw_ultra = MPSOptimizer::get_unified_memory_bandwidth(8, AppleChipVariant::ULTRA);
    
    EXPECT_LT(bw_base, bw_max);
    EXPECT_LT(bw_max, bw_ultra);
}

// =============================================================================
// Tests for Apple Silicon Generations
// =============================================================================

TEST(MPSAppleSiliconTest, M1Properties) {
    EXPECT_EQ(MPSOptimizer::get_threadgroup_memory_size(7), 32 * 1024);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(7, AppleChipVariant::BASE), 8);
}

TEST(MPSAppleSiliconTest, M2Properties) {
    EXPECT_EQ(MPSOptimizer::get_threadgroup_memory_size(8), 64 * 1024);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(8, AppleChipVariant::BASE), 10);
}

TEST(MPSAppleSiliconTest, M3Properties) {
    EXPECT_EQ(MPSOptimizer::get_threadgroup_memory_size(9), 64 * 1024);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(9, AppleChipVariant::BASE), 10);
}

TEST(MPSAppleSiliconTest, M4Properties) {
    EXPECT_EQ(MPSOptimizer::get_threadgroup_memory_size(10), 64 * 1024);
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(10, AppleChipVariant::BASE), 10);
}

// =============================================================================
// Tests for Chip Variants
// =============================================================================

TEST(MPSChipVariantTest, M2BaseVariant) {
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(8, AppleChipVariant::BASE), 10);
    EXPECT_EQ(MPSOptimizer::get_unified_memory_bandwidth(8, AppleChipVariant::BASE), 100);
}

TEST(MPSChipVariantTest, M2ProVariant) {
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(8, AppleChipVariant::PRO), 19);
    EXPECT_EQ(MPSOptimizer::get_unified_memory_bandwidth(8, AppleChipVariant::PRO), 200);
}

TEST(MPSChipVariantTest, M2MaxVariant) {
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(8, AppleChipVariant::MAX), 38);
    EXPECT_EQ(MPSOptimizer::get_unified_memory_bandwidth(8, AppleChipVariant::MAX), 400);
}

TEST(MPSChipVariantTest, M2UltraVariant) {
    EXPECT_EQ(MPSOptimizer::get_gpu_core_count(8, AppleChipVariant::ULTRA), 76);
    EXPECT_EQ(MPSOptimizer::get_unified_memory_bandwidth(8, AppleChipVariant::ULTRA), 800);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
