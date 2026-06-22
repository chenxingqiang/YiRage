// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_kernel_cuda_gtest.cc
 * @brief CUDA Kernel Backend Unit Tests
 *
 * Tests for CUDA kernel optimizations:
 *   - Warp configuration
 *   - Shared memory optimization
 *   - Bank conflict detection
 *   - Tensor Core configuration
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
namespace cuda {

// =============================================================================
// CUDA Enums and Structures
// =============================================================================

enum class SmemLayout {
    LINEAR = 0,
    SWIZZLED = 1,
    TILED = 2,
};

enum class ComputeCapability {
    PASCAL = 60,      // GTX 10xx
    VOLTA = 70,       // V100
    TURING = 75,      // RTX 20xx
    AMPERE = 80,      // A100, RTX 30xx
    ADA = 89,         // RTX 40xx
    HOPPER = 90,      // H100
    BLACKWELL = 100,  // B100
};

struct CUDAKernelConfig {
    int compute_capability = 80;
    int block_dim_x = 256;
    int block_dim_y = 1;
    int block_dim_z = 1;
    int grid_dim_x = 1;
    int grid_dim_y = 1;
    int grid_dim_z = 1;
    int num_warps = 4;
    size_t shared_memory_size = 48 * 1024;
    SmemLayout smem_layout = SmemLayout::SWIZZLED;
    
    // Tensor Core config
    bool use_tensor_core = false;
    int mma_m = 16;
    int mma_n = 8;
    int mma_k = 16;
    
    int get_total_threads() const {
        return block_dim_x * block_dim_y * block_dim_z;
    }
    
    int get_total_blocks() const {
        return grid_dim_x * grid_dim_y * grid_dim_z;
    }
};

// =============================================================================
// CUDAOptimizer
// =============================================================================

class CUDAOptimizer {
public:
    static int compute_optimal_warps(size_t problem_size, int compute_capability) {
        int max_warps_per_sm = 64;
        
        if (compute_capability >= 90) {
            max_warps_per_sm = 64;  // Hopper
        } else if (compute_capability >= 80) {
            max_warps_per_sm = 64;  // Ampere
        } else if (compute_capability >= 70) {
            max_warps_per_sm = 64;  // Volta/Turing
        }
        
        int warps_needed = static_cast<int>((problem_size + 1023) / 1024);
        
        int optimal_warps = 1;
        while (optimal_warps < warps_needed && optimal_warps < max_warps_per_sm) {
            optimal_warps *= 2;
        }
        
        return std::min(optimal_warps, max_warps_per_sm);
    }
    
    static size_t compute_optimal_smem(size_t data_size, SmemLayout layout, int padding) {
        size_t smem_size = data_size;
        
        if (layout == SmemLayout::SWIZZLED) {
            int num_banks = 32;
            int elements_per_row = static_cast<int>(std::sqrt(data_size / 4));
            int padded_elements = ((elements_per_row + num_banks - 1) / num_banks) *
                                  num_banks + padding;
            smem_size = data_size + padding * sizeof(float);
        } else {
            smem_size += padding;
        }
        
        // Align to 128 bytes
        smem_size = ((smem_size + 127) / 128) * 128;
        
        return smem_size;
    }
    
    static bool has_bank_conflict(SmemLayout layout, int stride, int bank_size) {
        if (layout == SmemLayout::SWIZZLED) {
            return false;
        }
        
        int num_banks = 32;
        if (stride % (num_banks * bank_size) == 0) {
            return true;
        }
        
        return false;
    }
    
    static float estimate_occupancy(CUDAKernelConfig const& config,
                                    int registers_per_thread) {
        int threads_per_block = config.get_total_threads();
        int warps_per_block = (threads_per_block + 31) / 32;
        
        // SM limits based on compute capability
        int max_threads_per_sm = 2048;
        int max_warps_per_sm = 64;
        int max_blocks_per_sm = 32;
        int max_registers_per_sm = 65536;
        size_t max_smem_per_sm = 164 * 1024;
        
        if (config.compute_capability == 75) {
            max_threads_per_sm = 1024;
            max_warps_per_sm = 32;
            max_blocks_per_sm = 16;
            max_smem_per_sm = 64 * 1024;
        } else if (config.compute_capability >= 90) {
            max_smem_per_sm = 228 * 1024;  // Hopper
        }
        
        int blocks_by_threads = max_threads_per_sm / threads_per_block;
        int blocks_by_warps = max_warps_per_sm / warps_per_block;
        int blocks_by_regs = max_registers_per_sm / (registers_per_thread * threads_per_block);
        int blocks_by_smem = static_cast<int>(max_smem_per_sm / config.shared_memory_size);
        
        int blocks_per_sm = std::min({blocks_by_threads, blocks_by_warps,
                                      blocks_by_regs, blocks_by_smem,
                                      max_blocks_per_sm});
        blocks_per_sm = std::max(1, blocks_per_sm);
        
        int active_warps = blocks_per_sm * warps_per_block;
        return static_cast<float>(active_warps) / max_warps_per_sm;
    }
    
    static bool select_tensor_core_config(int m, int n, int k,
                                          int compute_capability,
                                          CUDAKernelConfig& config) {
        if (compute_capability < 70) {
            return false;  // No Tensor Cores
        }
        
        if (compute_capability >= 90) {
            // Hopper
            config.mma_m = 16;
            config.mma_n = 8;
            config.mma_k = 16;
        } else if (compute_capability >= 80) {
            // Ampere
            config.mma_m = 16;
            config.mma_n = 8;
            config.mma_k = 16;
        } else {
            // Volta/Turing
            config.mma_m = 16;
            config.mma_n = 16;
            config.mma_k = 16;
        }
        
        bool large_enough = (m >= config.mma_m * 4) &&
                           (n >= config.mma_n * 4) &&
                           (k >= config.mma_k * 4);
        
        config.use_tensor_core = large_enough;
        return large_enough;
    }
    
    static void optimize_grid_block_dims(int problem_m, int problem_n, int problem_k,
                                         int compute_capability,
                                         CUDAKernelConfig& config) {
        select_tensor_core_config(problem_m, problem_n, problem_k,
                                  compute_capability, config);
        
        if (config.use_tensor_core) {
            int tile_m = config.mma_m * 4;
            int tile_n = config.mma_n * 4;
            
            config.grid_dim_x = (problem_n + tile_n - 1) / tile_n;
            config.grid_dim_y = (problem_m + tile_m - 1) / tile_m;
            config.grid_dim_z = 1;
            
            config.block_dim_x = 32;  // Warp size
            config.block_dim_y = 4;   // 4 warps
            config.block_dim_z = 1;
            config.num_warps = 4;
        } else {
            int block_size = 256;
            int tile_size = 32;
            
            config.block_dim_x = block_size;
            config.block_dim_y = 1;
            config.block_dim_z = 1;
            
            config.grid_dim_x = (problem_n + tile_size - 1) / tile_size;
            config.grid_dim_y = (problem_m + tile_size - 1) / tile_size;
            config.grid_dim_z = 1;
            
            config.num_warps = (block_size + 31) / 32;
        }
    }
    
    static float estimate_memory_bandwidth(CUDAKernelConfig const& config,
                                           size_t bytes_accessed,
                                           float execution_time_ms) {
        if (execution_time_ms <= 0.0f) return 0.0f;
        
        float seconds = execution_time_ms / 1000.0f;
        float gigabytes = bytes_accessed / (1024.0f * 1024.0f * 1024.0f);
        return gigabytes / seconds;
    }
    
    static float estimate_compute_throughput(CUDAKernelConfig const& config,
                                             size_t num_operations,
                                             float execution_time_ms) {
        if (execution_time_ms <= 0.0f) return 0.0f;
        
        float seconds = execution_time_ms / 1000.0f;
        float tflops = (num_operations / 1e12f) / seconds;
        return tflops;
    }
    
    static size_t get_max_shared_memory(int compute_capability) {
        if (compute_capability >= 90) return 228 * 1024;  // Hopper
        if (compute_capability >= 86) return 100 * 1024;  // Ampere A10
        if (compute_capability >= 80) return 164 * 1024;  // Ampere A100
        if (compute_capability >= 75) return 64 * 1024;   // Turing
        return 48 * 1024;  // Volta
    }
};

}  // namespace cuda
}  // namespace kernel
}  // namespace yirage

using namespace yirage::kernel::cuda;

// =============================================================================
// CUDAKernelConfig Tests
// =============================================================================

class CUDAKernelConfigTest : public ::testing::Test {};

TEST_F(CUDAKernelConfigTest, DefaultValues) {
    CUDAKernelConfig config;
    EXPECT_EQ(config.compute_capability, 80);
    EXPECT_EQ(config.block_dim_x, 256);
    EXPECT_EQ(config.num_warps, 4);
    EXPECT_EQ(config.shared_memory_size, 48u * 1024u);
}

TEST_F(CUDAKernelConfigTest, TotalThreads) {
    CUDAKernelConfig config;
    config.block_dim_x = 128;
    config.block_dim_y = 2;
    config.block_dim_z = 1;
    
    EXPECT_EQ(config.get_total_threads(), 256);
}

TEST_F(CUDAKernelConfigTest, TotalBlocks) {
    CUDAKernelConfig config;
    config.grid_dim_x = 64;
    config.grid_dim_y = 32;
    config.grid_dim_z = 1;
    
    EXPECT_EQ(config.get_total_blocks(), 2048);
}

// =============================================================================
// CUDAOptimizer Tests
// =============================================================================

class CUDAOptimizerTest : public ::testing::Test {};

TEST_F(CUDAOptimizerTest, ComputeOptimalWarpsSmall) {
    int warps = CUDAOptimizer::compute_optimal_warps(1024, 80);
    EXPECT_GE(warps, 1);
    EXPECT_LE(warps, 64);
}

TEST_F(CUDAOptimizerTest, ComputeOptimalWarpsLarge) {
    int warps = CUDAOptimizer::compute_optimal_warps(1000000, 80);
    EXPECT_GE(warps, 1);
    EXPECT_LE(warps, 64);
}

TEST_F(CUDAOptimizerTest, ComputeOptimalSmemLinear) {
    size_t smem = CUDAOptimizer::compute_optimal_smem(4096, SmemLayout::LINEAR, 16);
    EXPECT_GT(smem, 4096u);
    EXPECT_EQ(smem % 128, 0u);  // Aligned
}

TEST_F(CUDAOptimizerTest, ComputeOptimalSmemSwizzled) {
    size_t smem = CUDAOptimizer::compute_optimal_smem(4096, SmemLayout::SWIZZLED, 8);
    EXPECT_GT(smem, 4096u);
    EXPECT_EQ(smem % 128, 0u);
}

TEST_F(CUDAOptimizerTest, BankConflictSwizzled) {
    EXPECT_FALSE(CUDAOptimizer::has_bank_conflict(SmemLayout::SWIZZLED, 128, 4));
    EXPECT_FALSE(CUDAOptimizer::has_bank_conflict(SmemLayout::SWIZZLED, 256, 4));
}

TEST_F(CUDAOptimizerTest, BankConflictLinear) {
    EXPECT_TRUE(CUDAOptimizer::has_bank_conflict(SmemLayout::LINEAR, 128, 4));
    EXPECT_FALSE(CUDAOptimizer::has_bank_conflict(SmemLayout::LINEAR, 132, 4));
}

TEST_F(CUDAOptimizerTest, EstimateOccupancyHigh) {
    CUDAKernelConfig config;
    config.block_dim_x = 128;
    config.compute_capability = 80;
    config.shared_memory_size = 16 * 1024;
    
    float occupancy = CUDAOptimizer::estimate_occupancy(config, 32);
    EXPECT_GT(occupancy, 0.5f);
    EXPECT_LE(occupancy, 1.0f);
}

TEST_F(CUDAOptimizerTest, EstimateOccupancyLow) {
    CUDAKernelConfig config;
    config.block_dim_x = 1024;
    config.compute_capability = 80;
    config.shared_memory_size = 128 * 1024;
    
    float occupancy = CUDAOptimizer::estimate_occupancy(config, 128);
    EXPECT_GT(occupancy, 0.0f);
    EXPECT_LE(occupancy, 1.0f);
}

TEST_F(CUDAOptimizerTest, SelectTensorCoreConfigLarge) {
    CUDAKernelConfig config;
    
    bool use_tc = CUDAOptimizer::select_tensor_core_config(1024, 1024, 1024, 80, config);
    EXPECT_TRUE(use_tc);
    EXPECT_TRUE(config.use_tensor_core);
    EXPECT_EQ(config.mma_m, 16);
    EXPECT_EQ(config.mma_n, 8);
    EXPECT_EQ(config.mma_k, 16);
}

TEST_F(CUDAOptimizerTest, SelectTensorCoreConfigSmall) {
    CUDAKernelConfig config;
    
    bool use_tc = CUDAOptimizer::select_tensor_core_config(16, 16, 16, 80, config);
    EXPECT_FALSE(use_tc);
    EXPECT_FALSE(config.use_tensor_core);
}

TEST_F(CUDAOptimizerTest, SelectTensorCoreConfigOldGPU) {
    CUDAKernelConfig config;
    
    bool use_tc = CUDAOptimizer::select_tensor_core_config(1024, 1024, 1024, 60, config);
    EXPECT_FALSE(use_tc);  // Pascal doesn't have Tensor Cores
}

TEST_F(CUDAOptimizerTest, SelectTensorCoreConfigHopper) {
    CUDAKernelConfig config;
    
    CUDAOptimizer::select_tensor_core_config(1024, 1024, 1024, 90, config);
    EXPECT_EQ(config.mma_m, 16);
    EXPECT_EQ(config.mma_n, 8);
    EXPECT_EQ(config.mma_k, 16);
}

TEST_F(CUDAOptimizerTest, OptimizeGridBlockDimsTC) {
    CUDAKernelConfig config;
    config.compute_capability = 80;
    
    CUDAOptimizer::optimize_grid_block_dims(2048, 2048, 2048, 80, config);
    
    EXPECT_TRUE(config.use_tensor_core);
    EXPECT_GT(config.grid_dim_x, 0);
    EXPECT_GT(config.grid_dim_y, 0);
}

TEST_F(CUDAOptimizerTest, OptimizeGridBlockDimsNoTC) {
    CUDAKernelConfig config;
    config.compute_capability = 60;  // Pascal
    
    CUDAOptimizer::optimize_grid_block_dims(2048, 2048, 2048, 60, config);
    
    EXPECT_FALSE(config.use_tensor_core);
    EXPECT_EQ(config.block_dim_x, 256);
}

TEST_F(CUDAOptimizerTest, EstimateMemoryBandwidth) {
    CUDAKernelConfig config;
    
    // 1 GB accessed in 10 ms
    float bw = CUDAOptimizer::estimate_memory_bandwidth(config, 1024 * 1024 * 1024, 10.0f);
    EXPECT_NEAR(bw, 100.0f, 1.0f);  // ~100 GB/s
}

TEST_F(CUDAOptimizerTest, EstimateComputeThroughput) {
    CUDAKernelConfig config;
    
    // 1 trillion ops in 1 second
    float tflops = CUDAOptimizer::estimate_compute_throughput(config, 1e12, 1000.0f);
    EXPECT_NEAR(tflops, 1.0f, 0.01f);
}

TEST_F(CUDAOptimizerTest, GetMaxSharedMemory) {
    EXPECT_EQ(CUDAOptimizer::get_max_shared_memory(90), 228u * 1024u);  // Hopper
    EXPECT_EQ(CUDAOptimizer::get_max_shared_memory(80), 164u * 1024u);  // Ampere
    EXPECT_EQ(CUDAOptimizer::get_max_shared_memory(75), 64u * 1024u);   // Turing
}

// =============================================================================
// Parameterized Tests for Compute Capability
// =============================================================================

struct CCTestParam {
    int cc;
    size_t expected_max_smem;
    bool has_tensor_cores;
};

class ComputeCapabilityTest : public ::testing::TestWithParam<CCTestParam> {};

TEST_P(ComputeCapabilityTest, MaxSharedMemory) {
    auto param = GetParam();
    size_t max_smem = CUDAOptimizer::get_max_shared_memory(param.cc);
    EXPECT_EQ(max_smem, param.expected_max_smem);
}

TEST_P(ComputeCapabilityTest, TensorCoreSupport) {
    auto param = GetParam();
    CUDAKernelConfig config;
    
    bool has_tc = CUDAOptimizer::select_tensor_core_config(1024, 1024, 1024, param.cc, config);
    EXPECT_EQ(has_tc && config.use_tensor_core, param.has_tensor_cores);
}

INSTANTIATE_TEST_SUITE_P(
    AllComputeCapabilities,
    ComputeCapabilityTest,
    ::testing::Values(
        CCTestParam{60, 48 * 1024, false},   // Pascal
        CCTestParam{70, 48 * 1024, true},    // Volta
        CCTestParam{75, 64 * 1024, true},    // Turing
        CCTestParam{80, 164 * 1024, true},   // Ampere
        CCTestParam{90, 228 * 1024, true}    // Hopper
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
