// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_kernel_other_backends_gtest.cc
 * @brief Other Kernel Backends Unit Tests
 *
 * Tests for additional kernel backends:
 *   - Triton (OpenAI)
 *   - MKL (Intel Math Kernel Library)
 *   - NKI (AWS Neuron Kernel Interface)
 *   - TPU (Google Tensor Processing Unit)
 *   - XPU (Intel oneAPI)
 *   - FPGA
 *   - MACA (Moore Threads)
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

namespace yirage {
namespace kernel {

// =============================================================================
// Triton Backend (OpenAI)
// =============================================================================

namespace triton {

struct TritonKernelConfig {
    int block_m = 128;
    int block_n = 256;
    int block_k = 64;
    int num_stages = 3;
    int num_warps = 8;
    bool enable_fp8 = false;
    bool enable_tma = false;  // Hopper TMA
    
    size_t get_smem_size() const {
        return (block_m * block_k + block_k * block_n) * sizeof(float) * num_stages;
    }
};

class TritonOptimizer {
public:
    static void select_block_sizes(int m, int n, int k, TritonKernelConfig& config) {
        if (m >= 4096 && n >= 4096) {
            config.block_m = 256;
            config.block_n = 256;
            config.block_k = 64;
        } else if (m >= 1024 && n >= 1024) {
            config.block_m = 128;
            config.block_n = 256;
            config.block_k = 64;
        } else {
            config.block_m = 64;
            config.block_n = 128;
            config.block_k = 32;
        }
    }
    
    static int select_num_stages(size_t smem_available, TritonKernelConfig const& config) {
        size_t per_stage = (config.block_m * config.block_k + 
                           config.block_k * config.block_n) * sizeof(float);
        int max_stages = static_cast<int>(smem_available / per_stage);
        return std::min(std::max(2, max_stages), 5);
    }
    
    static int select_num_warps(int block_m, int block_n) {
        int threads_needed = (block_m / 16) * (block_n / 16);
        int warps = (threads_needed + 31) / 32;
        return std::min(std::max(4, warps), 16);
    }
};

}  // namespace triton

// =============================================================================
// MKL Backend (Intel)
// =============================================================================

namespace mkl {

enum class MKLThreadingLayer {
    SEQUENTIAL = 0,
    OPENMP = 1,
    TBB = 2,
};

inline void PrintTo(MKLThreadingLayer t, std::ostream* os) {
    switch(t) {
        case MKLThreadingLayer::SEQUENTIAL: *os << "SEQUENTIAL"; break;
        case MKLThreadingLayer::OPENMP: *os << "OPENMP"; break;
        case MKLThreadingLayer::TBB: *os << "TBB"; break;
        default: *os << "UNKNOWN"; break;
    }
}

struct MKLKernelConfig {
    int num_threads = 0;  // 0 = auto
    bool use_avx512 = true;
    bool use_amx = false;
    MKLThreadingLayer threading = MKLThreadingLayer::OPENMP;
    int packed_layout = 1;  // 0=none, 1=A, 2=B, 3=both
    
    int get_effective_threads() const {
        if (num_threads == 0) {
            return std::thread::hardware_concurrency();
        }
        return num_threads;
    }
};

class MKLOptimizer {
public:
    static bool detect_avx512_support() {
        return true;  // Simulated
    }
    
    static bool detect_amx_support() {
        return false;  // Simulated - requires Sapphire Rapids+
    }
    
    static int compute_optimal_threads(size_t problem_size) {
        int cores = std::thread::hardware_concurrency();
        if (problem_size < 1024 * 1024) {
            return std::max(1, cores / 2);
        }
        return cores;
    }
    
    static void optimize_for_intel(int m, int n, int k, MKLKernelConfig& config) {
        config.use_avx512 = detect_avx512_support();
        config.use_amx = detect_amx_support();
        config.num_threads = compute_optimal_threads(static_cast<size_t>(m) * n * k);
        
        // Use packed layout for large matrices
        if (m * k > 1024 * 1024) config.packed_layout |= 1;
        if (k * n > 1024 * 1024) config.packed_layout |= 2;
    }
};

}  // namespace mkl

// =============================================================================
// NKI Backend (AWS Neuron)
// =============================================================================

namespace nki {

struct NKIKernelConfig {
    int partition_dim = 128;
    int free_dim = 512;
    int nc_count = 2;  // Neuron cores
    bool use_psum = true;
    bool use_transpose_engine = true;
    
    size_t get_sbuf_size() const {
        return partition_dim * free_dim * sizeof(float);
    }
};

class NKIOptimizer {
public:
    static int compute_partition_dim(int tensor_size) {
        // Neuron partition dim is typically 128
        return std::min(128, tensor_size);
    }
    
    static int compute_free_dim(int tensor_size, int partition_dim) {
        // Free dim is limited to fit in SBUF
        constexpr size_t SBUF_SIZE = 2 * 1024 * 1024;  // 2MB
        int max_free = static_cast<int>(SBUF_SIZE / (partition_dim * sizeof(float)));
        return std::min(max_free, tensor_size);
    }
    
    static void optimize_for_neuron(int m, int n, int k, NKIKernelConfig& config) {
        config.partition_dim = compute_partition_dim(m);
        config.free_dim = compute_free_dim(n, config.partition_dim);
        config.use_psum = (k > 128);
        config.use_transpose_engine = (m != n);
    }
};

}  // namespace nki

// =============================================================================
// TPU Backend (Google)
// =============================================================================

namespace tpu {

enum class TPUGeneration {
    TPU_V2 = 2,
    TPU_V3 = 3,
    TPU_V4 = 4,
    TPU_V5E = 5,
    TPU_V5P = 6,
};

inline void PrintTo(TPUGeneration g, std::ostream* os) {
    switch(g) {
        case TPUGeneration::TPU_V2: *os << "TPU_V2"; break;
        case TPUGeneration::TPU_V3: *os << "TPU_V3"; break;
        case TPUGeneration::TPU_V4: *os << "TPU_V4"; break;
        case TPUGeneration::TPU_V5E: *os << "TPU_V5E"; break;
        case TPUGeneration::TPU_V5P: *os << "TPU_V5P"; break;
        default: *os << "UNKNOWN"; break;
    }
}

struct TPUKernelConfig {
    TPUGeneration generation = TPUGeneration::TPU_V4;
    int mxu_size = 128;
    size_t vmem_size = 32 * 1024 * 1024;
    int replicas = 1;
    bool use_bfloat16 = true;
    
    int get_mxu_flops_per_cycle() const {
        return mxu_size * mxu_size * 2;  // matmul FLOPs
    }
};

class TPUOptimizer {
public:
    static int get_mxu_size(TPUGeneration gen) {
        switch (gen) {
            case TPUGeneration::TPU_V2:
            case TPUGeneration::TPU_V3: return 128;
            case TPUGeneration::TPU_V4:
            case TPUGeneration::TPU_V5E:
            case TPUGeneration::TPU_V5P: return 128;
            default: return 128;
        }
    }
    
    static size_t get_vmem_size(TPUGeneration gen) {
        switch (gen) {
            case TPUGeneration::TPU_V2: return 8 * 1024 * 1024;
            case TPUGeneration::TPU_V3: return 16 * 1024 * 1024;
            case TPUGeneration::TPU_V4: return 32 * 1024 * 1024;
            case TPUGeneration::TPU_V5E:
            case TPUGeneration::TPU_V5P: return 64 * 1024 * 1024;
            default: return 32 * 1024 * 1024;
        }
    }
    
    static float get_peak_tflops(TPUGeneration gen) {
        switch (gen) {
            case TPUGeneration::TPU_V2: return 45.0f;
            case TPUGeneration::TPU_V3: return 123.0f;
            case TPUGeneration::TPU_V4: return 275.0f;
            case TPUGeneration::TPU_V5E: return 200.0f;
            case TPUGeneration::TPU_V5P: return 459.0f;
            default: return 275.0f;
        }
    }
    
    static void optimize_for_tpu(int m, int n, int k, TPUKernelConfig& config) {
        config.mxu_size = get_mxu_size(config.generation);
        config.vmem_size = get_vmem_size(config.generation);
        config.use_bfloat16 = true;  // TPU prefers bfloat16
    }
};

}  // namespace tpu

// =============================================================================
// XPU Backend (Intel oneAPI)
// =============================================================================

namespace xpu {

enum class XPUArch {
    ARC_A770 = 0,
    DATA_CENTER_GPU_MAX = 1,  // Ponte Vecchio
    DATA_CENTER_GPU_FLEX = 2,
};

struct XPUKernelConfig {
    XPUArch arch = XPUArch::DATA_CENTER_GPU_MAX;
    int subgroup_size = 32;
    int work_group_size = 256;
    size_t slm_size = 64 * 1024;  // Shared Local Memory
    bool use_systolic = true;
    
    int get_eu_count() const {
        switch (arch) {
            case XPUArch::ARC_A770: return 512;
            case XPUArch::DATA_CENTER_GPU_MAX: return 896;
            case XPUArch::DATA_CENTER_GPU_FLEX: return 448;
            default: return 512;
        }
    }
};

class XPUOptimizer {
public:
    static int get_subgroup_size(XPUArch arch) {
        return 32;  // Standard for Intel GPUs
    }
    
    static int compute_optimal_work_group(size_t problem_size, XPUArch arch) {
        int base = 256;
        if (problem_size < 4096) base = 128;
        else if (problem_size > 1000000) base = 512;
        return base;
    }
    
    static void optimize_for_xpu(int m, int n, int k, XPUKernelConfig& config) {
        config.subgroup_size = get_subgroup_size(config.arch);
        config.work_group_size = compute_optimal_work_group(
            static_cast<size_t>(m) * n, config.arch);
        config.use_systolic = (m >= 32 && n >= 32 && k >= 32);
    }
};

}  // namespace xpu

// =============================================================================
// FPGA Backend
// =============================================================================

namespace fpga {

enum class FPGAVendor {
    XILINX = 0,
    INTEL_FPGA = 1,
};

struct FPGAKernelConfig {
    FPGAVendor vendor = FPGAVendor::XILINX;
    int systolic_array_size = 16;
    int pipeline_depth = 8;
    int dsp_count = 2000;
    int bram_kb = 2048;
    bool use_hbm = false;
    
    int get_peak_gops() const {
        return systolic_array_size * systolic_array_size * 2 * 200;  // @ 200 MHz
    }
};

class FPGAOptimizer {
public:
    static int compute_systolic_size(int dsp_count, int target_util) {
        int max_size = static_cast<int>(std::sqrt(dsp_count * target_util / 100));
        return std::min(std::max(8, max_size), 64);
    }
    
    static int compute_pipeline_depth(int k, int systolic_size) {
        return std::min(16, std::max(4, k / systolic_size));
    }
    
    static void optimize_for_fpga(int m, int n, int k, FPGAKernelConfig& config) {
        config.systolic_array_size = compute_systolic_size(config.dsp_count, 80);
        config.pipeline_depth = compute_pipeline_depth(k, config.systolic_array_size);
        config.use_hbm = (m * n * sizeof(float) > config.bram_kb * 1024);
    }
};

}  // namespace fpga

// =============================================================================
// MACA Backend (Moore Threads)
// =============================================================================

namespace maca {

struct MACAKernelConfig {
    int block_size = 256;
    int stream_count = 2;
    size_t shared_memory = 48 * 1024;
    bool use_tensor_core = false;
    
    int get_total_threads() const {
        return block_size * stream_count;
    }
};

class MACAOptimizer {
public:
    static int compute_optimal_block_size(size_t problem_size) {
        if (problem_size < 4096) return 128;
        if (problem_size < 65536) return 256;
        return 512;
    }
    
    static void optimize_for_maca(int m, int n, int k, MACAKernelConfig& config) {
        config.block_size = compute_optimal_block_size(
            static_cast<size_t>(m) * n);
        config.use_tensor_core = (m >= 16 && n >= 16 && k >= 16);
    }
};

}  // namespace maca

}  // namespace kernel
}  // namespace yirage

using namespace yirage::kernel;

// =============================================================================
// Triton Tests
// =============================================================================

class TritonTest : public ::testing::Test {};

TEST_F(TritonTest, ConfigDefaults) {
    triton::TritonKernelConfig config;
    EXPECT_EQ(config.block_m, 128);
    EXPECT_EQ(config.block_n, 256);
    EXPECT_EQ(config.num_stages, 3);
    EXPECT_EQ(config.num_warps, 8);
}

TEST_F(TritonTest, SelectBlockSizesLarge) {
    triton::TritonKernelConfig config;
    triton::TritonOptimizer::select_block_sizes(4096, 4096, 4096, config);
    EXPECT_EQ(config.block_m, 256);
    EXPECT_EQ(config.block_n, 256);
}

TEST_F(TritonTest, SelectBlockSizesSmall) {
    triton::TritonKernelConfig config;
    triton::TritonOptimizer::select_block_sizes(256, 256, 256, config);
    EXPECT_EQ(config.block_m, 64);
    EXPECT_EQ(config.block_n, 128);
}

TEST_F(TritonTest, SelectNumStages) {
    triton::TritonKernelConfig config;
    int stages = triton::TritonOptimizer::select_num_stages(164 * 1024, config);
    EXPECT_GE(stages, 2);
    EXPECT_LE(stages, 5);
}

TEST_F(TritonTest, SelectNumWarps) {
    int warps = triton::TritonOptimizer::select_num_warps(128, 256);
    EXPECT_GE(warps, 4);
    EXPECT_LE(warps, 16);
}

// =============================================================================
// MKL Tests
// =============================================================================

class MKLTest : public ::testing::Test {};

TEST_F(MKLTest, ConfigDefaults) {
    mkl::MKLKernelConfig config;
    EXPECT_EQ(config.num_threads, 0);
    EXPECT_TRUE(config.use_avx512);
    EXPECT_EQ(config.threading, mkl::MKLThreadingLayer::OPENMP);
}

TEST_F(MKLTest, GetEffectiveThreadsAuto) {
    mkl::MKLKernelConfig config;
    config.num_threads = 0;
    EXPECT_GT(config.get_effective_threads(), 0);
}

TEST_F(MKLTest, GetEffectiveThreadsManual) {
    mkl::MKLKernelConfig config;
    config.num_threads = 4;
    EXPECT_EQ(config.get_effective_threads(), 4);
}

TEST_F(MKLTest, ComputeOptimalThreads) {
    int threads = mkl::MKLOptimizer::compute_optimal_threads(1000000);
    EXPECT_GT(threads, 0);
}

TEST_F(MKLTest, OptimizeForIntel) {
    mkl::MKLKernelConfig config;
    mkl::MKLOptimizer::optimize_for_intel(1024, 1024, 1024, config);
    EXPECT_TRUE(config.use_avx512);
    EXPECT_GT(config.num_threads, 0);
}

// =============================================================================
// NKI Tests
// =============================================================================

class NKITest : public ::testing::Test {};

TEST_F(NKITest, ConfigDefaults) {
    nki::NKIKernelConfig config;
    EXPECT_EQ(config.partition_dim, 128);
    EXPECT_EQ(config.free_dim, 512);
    EXPECT_TRUE(config.use_psum);
}

TEST_F(NKITest, ComputePartitionDim) {
    EXPECT_EQ(nki::NKIOptimizer::compute_partition_dim(256), 128);
    EXPECT_EQ(nki::NKIOptimizer::compute_partition_dim(64), 64);
}

TEST_F(NKITest, ComputeFreeDim) {
    int free = nki::NKIOptimizer::compute_free_dim(1024, 128);
    EXPECT_GT(free, 0);
    EXPECT_LE(free, 1024);
}

TEST_F(NKITest, OptimizeForNeuron) {
    nki::NKIKernelConfig config;
    nki::NKIOptimizer::optimize_for_neuron(256, 512, 256, config);
    EXPECT_EQ(config.partition_dim, 128);
    EXPECT_TRUE(config.use_psum);
}

// =============================================================================
// TPU Tests
// =============================================================================

class TPUTest : public ::testing::Test {};

TEST_F(TPUTest, ConfigDefaults) {
    tpu::TPUKernelConfig config;
    EXPECT_EQ(config.generation, tpu::TPUGeneration::TPU_V4);
    EXPECT_EQ(config.mxu_size, 128);
    EXPECT_TRUE(config.use_bfloat16);
}

TEST_F(TPUTest, GetMXUSize) {
    EXPECT_EQ(tpu::TPUOptimizer::get_mxu_size(tpu::TPUGeneration::TPU_V2), 128);
    EXPECT_EQ(tpu::TPUOptimizer::get_mxu_size(tpu::TPUGeneration::TPU_V4), 128);
}

TEST_F(TPUTest, GetVMEMSize) {
    EXPECT_EQ(tpu::TPUOptimizer::get_vmem_size(tpu::TPUGeneration::TPU_V2), 8u * 1024u * 1024u);
    EXPECT_EQ(tpu::TPUOptimizer::get_vmem_size(tpu::TPUGeneration::TPU_V4), 32u * 1024u * 1024u);
    EXPECT_EQ(tpu::TPUOptimizer::get_vmem_size(tpu::TPUGeneration::TPU_V5P), 64u * 1024u * 1024u);
}

TEST_F(TPUTest, GetPeakTFLOPS) {
    EXPECT_NEAR(tpu::TPUOptimizer::get_peak_tflops(tpu::TPUGeneration::TPU_V2), 45.0f, 1.0f);
    EXPECT_NEAR(tpu::TPUOptimizer::get_peak_tflops(tpu::TPUGeneration::TPU_V4), 275.0f, 1.0f);
    EXPECT_NEAR(tpu::TPUOptimizer::get_peak_tflops(tpu::TPUGeneration::TPU_V5P), 459.0f, 1.0f);
}

TEST_F(TPUTest, MXUFlopsPerCycle) {
    tpu::TPUKernelConfig config;
    config.mxu_size = 128;
    EXPECT_EQ(config.get_mxu_flops_per_cycle(), 128 * 128 * 2);
}

// =============================================================================
// XPU Tests
// =============================================================================

class XPUTest : public ::testing::Test {};

TEST_F(XPUTest, ConfigDefaults) {
    xpu::XPUKernelConfig config;
    EXPECT_EQ(config.subgroup_size, 32);
    EXPECT_EQ(config.work_group_size, 256);
    EXPECT_TRUE(config.use_systolic);
}

TEST_F(XPUTest, GetEUCount) {
    xpu::XPUKernelConfig config;
    
    config.arch = xpu::XPUArch::ARC_A770;
    EXPECT_EQ(config.get_eu_count(), 512);
    
    config.arch = xpu::XPUArch::DATA_CENTER_GPU_MAX;
    EXPECT_EQ(config.get_eu_count(), 896);
}

TEST_F(XPUTest, ComputeOptimalWorkGroup) {
    int wg = xpu::XPUOptimizer::compute_optimal_work_group(100000, xpu::XPUArch::DATA_CENTER_GPU_MAX);
    EXPECT_EQ(wg, 256);
}

TEST_F(XPUTest, OptimizeForXPU) {
    xpu::XPUKernelConfig config;
    xpu::XPUOptimizer::optimize_for_xpu(256, 256, 256, config);
    EXPECT_TRUE(config.use_systolic);
}

// =============================================================================
// FPGA Tests
// =============================================================================

class FPGATest : public ::testing::Test {};

TEST_F(FPGATest, ConfigDefaults) {
    fpga::FPGAKernelConfig config;
    EXPECT_EQ(config.systolic_array_size, 16);
    EXPECT_EQ(config.pipeline_depth, 8);
    EXPECT_FALSE(config.use_hbm);
}

TEST_F(FPGATest, ComputeSystolicSize) {
    int size = fpga::FPGAOptimizer::compute_systolic_size(2000, 80);
    EXPECT_GE(size, 8);
    EXPECT_LE(size, 64);
}

TEST_F(FPGATest, ComputePipelineDepth) {
    int depth = fpga::FPGAOptimizer::compute_pipeline_depth(256, 16);
    EXPECT_GE(depth, 4);
    EXPECT_LE(depth, 16);
}

TEST_F(FPGATest, GetPeakGOPS) {
    fpga::FPGAKernelConfig config;
    config.systolic_array_size = 16;
    int gops = config.get_peak_gops();
    EXPECT_GT(gops, 0);
}

// =============================================================================
// MACA Tests
// =============================================================================

class MACATest : public ::testing::Test {};

TEST_F(MACATest, ConfigDefaults) {
    maca::MACAKernelConfig config;
    EXPECT_EQ(config.block_size, 256);
    EXPECT_EQ(config.stream_count, 2);
    EXPECT_FALSE(config.use_tensor_core);
}

TEST_F(MACATest, GetTotalThreads) {
    maca::MACAKernelConfig config;
    EXPECT_EQ(config.get_total_threads(), 512);
}

TEST_F(MACATest, ComputeOptimalBlockSize) {
    EXPECT_EQ(maca::MACAOptimizer::compute_optimal_block_size(1000), 128);
    EXPECT_EQ(maca::MACAOptimizer::compute_optimal_block_size(10000), 256);
    EXPECT_EQ(maca::MACAOptimizer::compute_optimal_block_size(1000000), 512);
}

TEST_F(MACATest, OptimizeForMACA) {
    maca::MACAKernelConfig config;
    maca::MACAOptimizer::optimize_for_maca(64, 64, 64, config);
    EXPECT_TRUE(config.use_tensor_core);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
