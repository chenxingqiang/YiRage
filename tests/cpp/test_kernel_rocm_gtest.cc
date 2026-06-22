// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_kernel_rocm_gtest.cc
 * @brief ROCm (AMD HIP) Kernel Backend Unit Tests
 *
 * Tests for ROCm kernel optimizations:
 *   - Wavefront configuration (CDNA vs RDNA)
 *   - LDS (Local Data Share) optimization
 *   - Matrix Cores (MFMA) configuration
 *   - Occupancy estimation
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace yirage {
namespace kernel {
namespace rocm {

// =============================================================================
// ROCm Enums and Structures
// =============================================================================

enum class ROCmArch {
    CDNA1 = 0,  // MI100 (gfx908)
    CDNA2 = 1,  // MI200 (gfx90a)
    CDNA3 = 2,  // MI300 (gfx942)
    RDNA2 = 3,  // RX 6000 (gfx1030)
    RDNA3 = 4,  // RX 7000 (gfx1100)
};

enum class LDSLayout {
    LINEAR = 0,
    SWIZZLED = 1,
    TILED = 2,
};

struct ROCmArchParams {
    int wavefront_size;
    int max_threads_per_block;
    int max_wavefronts_per_cu;
    int max_blocks_per_cu;
    int vgprs_per_cu;
    int sgprs_per_cu;
    int lds_per_cu;
    int cu_count;
    int mfma_m, mfma_n, mfma_k;
    bool has_fp8;
    bool has_sparsity;
};

inline ROCmArchParams get_arch_params(ROCmArch arch) {
    ROCmArchParams params;
    params.wavefront_size = 64;
    params.max_threads_per_block = 1024;
    params.sgprs_per_cu = 102;
    
    switch (arch) {
        case ROCmArch::CDNA1:
            params.max_wavefronts_per_cu = 40;
            params.max_blocks_per_cu = 16;
            params.vgprs_per_cu = 256;
            params.lds_per_cu = 65536;
            params.cu_count = 120;
            params.mfma_m = 32;
            params.mfma_n = 32;
            params.mfma_k = 8;
            params.has_fp8 = false;
            params.has_sparsity = false;
            break;
            
        case ROCmArch::CDNA2:
            params.max_wavefronts_per_cu = 40;
            params.max_blocks_per_cu = 16;
            params.vgprs_per_cu = 256;
            params.lds_per_cu = 65536;
            params.cu_count = 220;
            params.mfma_m = 32;
            params.mfma_n = 32;
            params.mfma_k = 8;
            params.has_fp8 = false;
            params.has_sparsity = false;
            break;
            
        case ROCmArch::CDNA3:
            params.max_wavefronts_per_cu = 40;
            params.max_blocks_per_cu = 16;
            params.vgprs_per_cu = 512;
            params.lds_per_cu = 65536;
            params.cu_count = 304;
            params.mfma_m = 32;
            params.mfma_n = 32;
            params.mfma_k = 16;
            params.has_fp8 = true;
            params.has_sparsity = true;
            break;
            
        case ROCmArch::RDNA2:
        case ROCmArch::RDNA3:
            params.wavefront_size = 32;
            params.max_wavefronts_per_cu = 32;
            params.max_blocks_per_cu = 16;
            params.vgprs_per_cu = 256;
            params.lds_per_cu = 65536;
            params.cu_count = 80;
            params.mfma_m = 0;
            params.mfma_n = 0;
            params.mfma_k = 0;
            params.has_fp8 = false;
            params.has_sparsity = false;
            break;
    }
    
    return params;
}

struct ROCmKernelConfig {
    ROCmArch arch = ROCmArch::CDNA3;
    int num_wavefronts = 4;
    int num_threads = 256;
    size_t lds_size = 64 * 1024;
    bool use_matrix_core = false;
    int mfma_m = 32, mfma_n = 32, mfma_k = 16;
    int forall_dim[3] = {128, 128, 1};
    int imap_dim[3] = {1, 1, 1};
    
    bool has_matrix_cores() const {
        return arch == ROCmArch::CDNA1 || arch == ROCmArch::CDNA2 ||
               arch == ROCmArch::CDNA3;
    }
    
    int get_wavefront_size() const {
        if (arch == ROCmArch::RDNA2 || arch == ROCmArch::RDNA3) {
            return 32;
        }
        return 64;
    }
};

// =============================================================================
// ROCmOptimizer
// =============================================================================

class ROCmOptimizer {
public:
    static int compute_optimal_wavefronts(size_t problem_size, ROCmArch arch) {
        ROCmArchParams params = get_arch_params(arch);
        
        size_t elements_per_wave = params.wavefront_size;
        int waves_needed = static_cast<int>((problem_size + elements_per_wave - 1) /
                                            elements_per_wave);
        
        int optimal_waves = 1;
        while (optimal_waves < waves_needed &&
               optimal_waves < params.max_wavefronts_per_cu) {
            optimal_waves *= 2;
        }
        
        return std::min(optimal_waves, params.max_wavefronts_per_cu);
    }
    
    static size_t compute_optimal_lds(size_t data_size, LDSLayout layout, int padding) {
        size_t lds_size = data_size;
        constexpr int LDS_BANKS = 32;
        
        switch (layout) {
            case LDSLayout::SWIZZLED: {
                int row_size = static_cast<int>(std::sqrt(data_size / sizeof(float)));
                int padded_row = ((row_size + LDS_BANKS - 1) / LDS_BANKS) * LDS_BANKS + padding;
                lds_size = padded_row * row_size * sizeof(float);
                break;
            }
            case LDSLayout::TILED:
                lds_size += padding * sizeof(float);
                break;
            default:
                lds_size += padding;
                break;
        }
        
        // Align to 256 bytes
        lds_size = ((lds_size + 255) / 256) * 256;
        
        return lds_size;
    }
    
    static bool has_bank_conflict(LDSLayout layout, int stride, int bank_size) {
        constexpr int LDS_BANKS = 32;
        
        if (layout == LDSLayout::SWIZZLED) {
            return false;
        }
        
        int stride_in_banks = stride / bank_size;
        return (stride_in_banks % LDS_BANKS) == 0;
    }
    
    static float estimate_occupancy(ROCmKernelConfig const& config,
                                    int vgprs_used, int sgprs_used) {
        ROCmArchParams params = get_arch_params(config.arch);
        
        int waves_by_vgprs = params.vgprs_per_cu / std::max(vgprs_used, 1);
        int waves_by_sgprs = params.sgprs_per_cu / std::max(sgprs_used, 1);
        
        int lds_per_block = static_cast<int>(config.lds_size);
        int blocks_by_lds = params.lds_per_cu / std::max(lds_per_block, 1);
        int waves_per_block = config.num_wavefronts;
        int waves_by_lds = blocks_by_lds * waves_per_block;
        
        int achievable_waves = std::min({waves_by_vgprs, waves_by_sgprs,
                                         waves_by_lds, params.max_wavefronts_per_cu});
        
        return static_cast<float>(achievable_waves) / params.max_wavefronts_per_cu;
    }
    
    static bool select_matrix_core_config(int m, int n, int k, ROCmArch arch,
                                          ROCmKernelConfig& config) {
        if (!config.has_matrix_cores()) {
            return false;
        }
        
        ROCmArchParams params = get_arch_params(arch);
        
        config.use_matrix_core = true;
        config.mfma_m = params.mfma_m;
        config.mfma_n = params.mfma_n;
        config.mfma_k = params.mfma_k;
        
        if (m >= 256 && n >= 256) {
            config.forall_dim[0] = 256;
            config.forall_dim[1] = 256;
            config.forall_dim[2] = 64;
        } else if (m >= 128 && n >= 128) {
            config.forall_dim[0] = 128;
            config.forall_dim[1] = 128;
            config.forall_dim[2] = 32;
        } else {
            config.forall_dim[0] = 64;
            config.forall_dim[1] = 64;
            config.forall_dim[2] = 32;
        }
        
        return true;
    }
    
    static void optimize_grid_block_dims(int problem_m, int problem_n, int problem_k,
                                         ROCmArch arch, ROCmKernelConfig& config) {
        ROCmArchParams params = get_arch_params(arch);
        
        int tile_m = 128, tile_n = 128;
        
        if (problem_m >= 4096 && problem_n >= 4096) {
            tile_m = tile_n = 256;
        } else if (problem_m < 256 || problem_n < 256) {
            tile_m = tile_n = 64;
        }
        
        int grid_m = (problem_m + tile_m - 1) / tile_m;
        int grid_n = (problem_n + tile_n - 1) / tile_n;
        
        config.forall_dim[0] = tile_m;
        config.forall_dim[1] = tile_n;
        config.forall_dim[2] = 1;
        
        config.num_threads = 256;
        config.num_wavefronts = config.num_threads / params.wavefront_size;
        
        config.imap_dim[0] = grid_m;
        config.imap_dim[1] = grid_n;
        config.imap_dim[2] = 1;
    }
    
    static std::string get_arch_string(ROCmArch arch) {
        switch (arch) {
            case ROCmArch::CDNA1: return "gfx908";
            case ROCmArch::CDNA2: return "gfx90a";
            case ROCmArch::CDNA3: return "gfx942";
            case ROCmArch::RDNA2: return "gfx1030";
            case ROCmArch::RDNA3: return "gfx1100";
            default: return "gfx942";
        }
    }
    
    static bool should_use_async_copy(ROCmArch arch, size_t transfer_size) {
        if (arch == ROCmArch::CDNA1 || arch == ROCmArch::RDNA2 ||
            arch == ROCmArch::RDNA3) {
            return false;
        }
        return transfer_size >= 4096;
    }
    
    static int get_recommended_stages(ROCmArch arch, size_t lds_available,
                                      size_t lds_per_stage) {
        int max_stages = static_cast<int>(lds_available / lds_per_stage);
        
        switch (arch) {
            case ROCmArch::CDNA3: return std::min(max_stages, 4);
            case ROCmArch::CDNA2: return std::min(max_stages, 3);
            default: return std::min(max_stages, 2);
        }
    }
};

}  // namespace rocm
}  // namespace kernel
}  // namespace yirage

using namespace yirage::kernel::rocm;

// =============================================================================
// ROCmArchParams Tests
// =============================================================================

class ROCmArchParamsTest : public ::testing::Test {};

TEST_F(ROCmArchParamsTest, CDNA1Params) {
    auto params = get_arch_params(ROCmArch::CDNA1);
    EXPECT_EQ(params.wavefront_size, 64);
    EXPECT_EQ(params.cu_count, 120);
    EXPECT_EQ(params.mfma_m, 32);
    EXPECT_FALSE(params.has_fp8);
}

TEST_F(ROCmArchParamsTest, CDNA2Params) {
    auto params = get_arch_params(ROCmArch::CDNA2);
    EXPECT_EQ(params.wavefront_size, 64);
    EXPECT_EQ(params.cu_count, 220);
    EXPECT_EQ(params.vgprs_per_cu, 256);
}

TEST_F(ROCmArchParamsTest, CDNA3Params) {
    auto params = get_arch_params(ROCmArch::CDNA3);
    EXPECT_EQ(params.wavefront_size, 64);
    EXPECT_EQ(params.cu_count, 304);
    EXPECT_EQ(params.vgprs_per_cu, 512);
    EXPECT_EQ(params.mfma_k, 16);
    EXPECT_TRUE(params.has_fp8);
    EXPECT_TRUE(params.has_sparsity);
}

TEST_F(ROCmArchParamsTest, RDNA2Params) {
    auto params = get_arch_params(ROCmArch::RDNA2);
    EXPECT_EQ(params.wavefront_size, 32);  // RDNA uses 32
    EXPECT_EQ(params.mfma_m, 0);  // No MFMA
}

TEST_F(ROCmArchParamsTest, RDNA3Params) {
    auto params = get_arch_params(ROCmArch::RDNA3);
    EXPECT_EQ(params.wavefront_size, 32);
}

// =============================================================================
// ROCmKernelConfig Tests
// =============================================================================

class ROCmKernelConfigTest : public ::testing::Test {};

TEST_F(ROCmKernelConfigTest, DefaultValues) {
    ROCmKernelConfig config;
    EXPECT_EQ(config.arch, ROCmArch::CDNA3);
    EXPECT_EQ(config.num_threads, 256);
    EXPECT_EQ(config.lds_size, 64u * 1024u);
}

TEST_F(ROCmKernelConfigTest, HasMatrixCores) {
    ROCmKernelConfig config;
    
    config.arch = ROCmArch::CDNA3;
    EXPECT_TRUE(config.has_matrix_cores());
    
    config.arch = ROCmArch::RDNA3;
    EXPECT_FALSE(config.has_matrix_cores());
}

TEST_F(ROCmKernelConfigTest, WavefrontSize) {
    ROCmKernelConfig config;
    
    config.arch = ROCmArch::CDNA3;
    EXPECT_EQ(config.get_wavefront_size(), 64);
    
    config.arch = ROCmArch::RDNA3;
    EXPECT_EQ(config.get_wavefront_size(), 32);
}

// =============================================================================
// ROCmOptimizer Tests
// =============================================================================

class ROCmOptimizerTest : public ::testing::Test {};

TEST_F(ROCmOptimizerTest, ComputeOptimalWavefrontsCDNA) {
    int waves = ROCmOptimizer::compute_optimal_wavefronts(1000000, ROCmArch::CDNA3);
    EXPECT_GE(waves, 1);
    EXPECT_LE(waves, 40);
}

TEST_F(ROCmOptimizerTest, ComputeOptimalWavefrontsRDNA) {
    int waves = ROCmOptimizer::compute_optimal_wavefronts(1000000, ROCmArch::RDNA3);
    EXPECT_GE(waves, 1);
    EXPECT_LE(waves, 32);
}

TEST_F(ROCmOptimizerTest, ComputeOptimalLDSLinear) {
    size_t lds = ROCmOptimizer::compute_optimal_lds(4096, LDSLayout::LINEAR, 64);
    EXPECT_GT(lds, 4096u);
    EXPECT_EQ(lds % 256, 0u);  // Aligned
}

TEST_F(ROCmOptimizerTest, ComputeOptimalLDSSwizzled) {
    size_t lds = ROCmOptimizer::compute_optimal_lds(4096, LDSLayout::SWIZZLED, 8);
    EXPECT_GT(lds, 0u);
    EXPECT_EQ(lds % 256, 0u);
}

TEST_F(ROCmOptimizerTest, BankConflictSwizzled) {
    EXPECT_FALSE(ROCmOptimizer::has_bank_conflict(LDSLayout::SWIZZLED, 128, 4));
}

TEST_F(ROCmOptimizerTest, BankConflictLinear) {
    EXPECT_TRUE(ROCmOptimizer::has_bank_conflict(LDSLayout::LINEAR, 128, 4));
}

TEST_F(ROCmOptimizerTest, EstimateOccupancy) {
    ROCmKernelConfig config;
    config.arch = ROCmArch::CDNA3;
    config.lds_size = 32 * 1024;
    
    float occupancy = ROCmOptimizer::estimate_occupancy(config, 64, 32);
    EXPECT_GT(occupancy, 0.0f);
    EXPECT_LE(occupancy, 1.0f);
}

TEST_F(ROCmOptimizerTest, SelectMatrixCoreConfigCDNA) {
    ROCmKernelConfig config;
    config.arch = ROCmArch::CDNA3;
    
    bool result = ROCmOptimizer::select_matrix_core_config(1024, 1024, 1024, ROCmArch::CDNA3, config);
    EXPECT_TRUE(result);
    EXPECT_TRUE(config.use_matrix_core);
    EXPECT_EQ(config.mfma_m, 32);
}

TEST_F(ROCmOptimizerTest, SelectMatrixCoreConfigRDNA) {
    ROCmKernelConfig config;
    config.arch = ROCmArch::RDNA3;
    
    bool result = ROCmOptimizer::select_matrix_core_config(1024, 1024, 1024, ROCmArch::RDNA3, config);
    EXPECT_FALSE(result);  // RDNA doesn't have MFMA
}

TEST_F(ROCmOptimizerTest, OptimizeGridBlockDims) {
    ROCmKernelConfig config;
    ROCmOptimizer::optimize_grid_block_dims(2048, 2048, 2048, ROCmArch::CDNA3, config);
    
    EXPECT_GT(config.imap_dim[0], 0);
    EXPECT_GT(config.imap_dim[1], 0);
    EXPECT_EQ(config.num_threads, 256);
}

TEST_F(ROCmOptimizerTest, GetArchString) {
    EXPECT_EQ(ROCmOptimizer::get_arch_string(ROCmArch::CDNA1), "gfx908");
    EXPECT_EQ(ROCmOptimizer::get_arch_string(ROCmArch::CDNA2), "gfx90a");
    EXPECT_EQ(ROCmOptimizer::get_arch_string(ROCmArch::CDNA3), "gfx942");
    EXPECT_EQ(ROCmOptimizer::get_arch_string(ROCmArch::RDNA2), "gfx1030");
    EXPECT_EQ(ROCmOptimizer::get_arch_string(ROCmArch::RDNA3), "gfx1100");
}

TEST_F(ROCmOptimizerTest, ShouldUseAsyncCopy) {
    EXPECT_FALSE(ROCmOptimizer::should_use_async_copy(ROCmArch::CDNA1, 8192));
    EXPECT_TRUE(ROCmOptimizer::should_use_async_copy(ROCmArch::CDNA2, 8192));
    EXPECT_TRUE(ROCmOptimizer::should_use_async_copy(ROCmArch::CDNA3, 8192));
    EXPECT_FALSE(ROCmOptimizer::should_use_async_copy(ROCmArch::CDNA3, 1024));  // Too small
}

TEST_F(ROCmOptimizerTest, GetRecommendedStages) {
    EXPECT_EQ(ROCmOptimizer::get_recommended_stages(ROCmArch::CDNA3, 65536, 8192), 4);
    EXPECT_EQ(ROCmOptimizer::get_recommended_stages(ROCmArch::CDNA2, 65536, 8192), 3);
    EXPECT_EQ(ROCmOptimizer::get_recommended_stages(ROCmArch::CDNA1, 65536, 8192), 2);
}

// =============================================================================
// Parameterized Tests for Architectures
// =============================================================================

struct ROCmArchTestParam {
    ROCmArch arch;
    std::string expected_string;
    int wavefront_size;
    bool has_mfma;
};

class ROCmArchParameterizedTest : public ::testing::TestWithParam<ROCmArchTestParam> {};

TEST_P(ROCmArchParameterizedTest, ArchProperties) {
    auto param = GetParam();
    
    EXPECT_EQ(ROCmOptimizer::get_arch_string(param.arch), param.expected_string);
    
    auto arch_params = get_arch_params(param.arch);
    EXPECT_EQ(arch_params.wavefront_size, param.wavefront_size);
    EXPECT_EQ(arch_params.mfma_m > 0, param.has_mfma);
}

INSTANTIATE_TEST_SUITE_P(
    AllROCmArchs,
    ROCmArchParameterizedTest,
    ::testing::Values(
        ROCmArchTestParam{ROCmArch::CDNA1, "gfx908", 64, true},
        ROCmArchTestParam{ROCmArch::CDNA2, "gfx90a", 64, true},
        ROCmArchTestParam{ROCmArch::CDNA3, "gfx942", 64, true},
        ROCmArchTestParam{ROCmArch::RDNA2, "gfx1030", 32, false},
        ROCmArchTestParam{ROCmArch::RDNA3, "gfx1100", 32, false}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
