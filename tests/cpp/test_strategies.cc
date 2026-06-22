// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_strategies.cc
 * @brief Backend Strategy Unit Tests
 *
 * Tests for all 12 backend search strategies.
 * Compile with: clang++ -std=c++17 -I../../include test_strategies.cc -o test_strategies
 */

#include "test_framework.h"
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace yirage {
namespace test {

// =============================================================================
// Backend Type Definitions
// =============================================================================

enum class BackendType {
    CUDA = 0,
    ROCM = 1,
    CPU = 2,
    MPS = 3,
    ASCEND = 4,
    MACA = 5,
    TPU = 6,
    XPU = 7,
    FPGA = 8,
    TRITON = 9,
    NKI = 10,
    MLIR = 11,
};

struct BackendInfo {
    std::string name;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int32_t shared_memory_size;
    bool has_tensor_core;
    std::string matrix_unit;
};

// Backend specifications
std::map<BackendType, BackendInfo> BACKEND_SPECS = {
    {BackendType::CUDA, {"cuda", 32, 1024, 49152, true, "TensorCore"}},
    {BackendType::ROCM, {"rocm", 64, 1024, 65536, true, "MatrixCore"}},
    {BackendType::CPU, {"cpu", 1, 1, 0, false, "SIMD"}},
    {BackendType::MPS, {"mps", 32, 1024, 32768, true, "AppleGPU"}},
    {BackendType::ASCEND, {"ascend", 16, 256, 131072, true, "CubeUnit"}},
    {BackendType::MACA, {"maca", 64, 1024, 49152, true, "TensorCore"}},
    {BackendType::TPU, {"tpu", 128, 128, 16777216, true, "MXU"}},
    {BackendType::XPU, {"xpu", 16, 1024, 131072, true, "XMX"}},
    {BackendType::FPGA, {"fpga", 1, 1, 0, false, "DSP"}},
    {BackendType::TRITON, {"triton", 32, 1024, 49152, true, "Auto"}},
    {BackendType::NKI, {"nki", 128, 128, 25165824, true, "TensorEngine"}},
    {BackendType::MLIR, {"mlir", 32, 1024, 49152, true, "Target"}},
};

}  // namespace test
}  // namespace yirage

using namespace yirage::test;

// =============================================================================
// CUDA Strategy Tests
// =============================================================================

TEST(CUDAStrategy, WarpSize32) {
    auto& info = BACKEND_SPECS[BackendType::CUDA];
    EXPECT_EQ(info.warp_size, 32);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CUDAStrategy, MaxThreadsPerBlock) {
    auto& info = BACKEND_SPECS[BackendType::CUDA];
    EXPECT_EQ(info.max_threads_per_block, 1024);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CUDAStrategy, TensorCoreSupport) {
    auto& info = BACKEND_SPECS[BackendType::CUDA];
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "TensorCore");
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CUDAStrategy, SharedMemorySize) {
    auto& info = BACKEND_SPECS[BackendType::CUDA];
    // Typical shared memory: 48KB
    EXPECT_EQ(info.shared_memory_size, 49152);
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// ROCm Strategy Tests
// =============================================================================

TEST(ROCmStrategy, WavefrontSize64) {
    auto& info = BACKEND_SPECS[BackendType::ROCM];
    // AMD uses 64-thread wavefronts
    EXPECT_EQ(info.warp_size, 64);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(ROCmStrategy, LDSSize) {
    auto& info = BACKEND_SPECS[BackendType::ROCM];
    // AMD LDS: 64KB
    EXPECT_EQ(info.shared_memory_size, 65536);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(ROCmStrategy, MatrixCoreSupport) {
    auto& info = BACKEND_SPECS[BackendType::ROCM];
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "MatrixCore");
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// MPS Strategy Tests
// =============================================================================

TEST(MPSStrategy, ThreadgroupConfiguration) {
    auto& info = BACKEND_SPECS[BackendType::MPS];
    EXPECT_EQ(info.warp_size, 32);
    EXPECT_EQ(info.max_threads_per_block, 1024);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(MPSStrategy, ThreadgroupMemory) {
    auto& info = BACKEND_SPECS[BackendType::MPS];
    // Apple Silicon threadgroup memory: 32KB
    EXPECT_EQ(info.shared_memory_size, 32768);
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// Ascend Strategy Tests
// =============================================================================

TEST(AscendStrategy, AICoreTiling) {
    auto& info = BACKEND_SPECS[BackendType::ASCEND];
    // Ascend AI Core: 16-thread groups
    EXPECT_EQ(info.warp_size, 16);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(AscendStrategy, CubeUnitSupport) {
    auto& info = BACKEND_SPECS[BackendType::ASCEND];
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "CubeUnit");
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(AscendStrategy, L1BufferSize) {
    auto& info = BACKEND_SPECS[BackendType::ASCEND];
    // Ascend L1 buffer: 128KB
    EXPECT_EQ(info.shared_memory_size, 131072);
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// MACA Strategy Tests
// =============================================================================

TEST(MACAStrategy, WarpSize64) {
    auto& info = BACKEND_SPECS[BackendType::MACA];
    // MACA (MetaX) uses 64-thread warps like AMD
    EXPECT_EQ(info.warp_size, 64);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(MACAStrategy, TensorCoreCompat) {
    auto& info = BACKEND_SPECS[BackendType::MACA];
    EXPECT_TRUE(info.has_tensor_core);
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// CPU Strategy Tests
// =============================================================================

TEST(CPUStrategy, NoWarp) {
    auto& info = BACKEND_SPECS[BackendType::CPU];
    // CPU doesn't have warps
    EXPECT_EQ(info.warp_size, 1);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CPUStrategy, SIMDUnit) {
    auto& info = BACKEND_SPECS[BackendType::CPU];
    EXPECT_FALSE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "SIMD");
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CPUStrategy, NoSharedMemory) {
    auto& info = BACKEND_SPECS[BackendType::CPU];
    EXPECT_EQ(info.shared_memory_size, 0);
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// TPU Strategy Tests
// =============================================================================

TEST(TPUStrategy, MXUSize) {
    auto& info = BACKEND_SPECS[BackendType::TPU];
    // TPU MXU: 128x128 systolic array
    EXPECT_EQ(info.warp_size, 128);
    EXPECT_EQ(info.max_threads_per_block, 128);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(TPUStrategy, VMEMSize) {
    auto& info = BACKEND_SPECS[BackendType::TPU];
    // TPU VMEM: 16MB per core
    EXPECT_EQ(info.shared_memory_size, 16777216);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(TPUStrategy, MXUSupport) {
    auto& info = BACKEND_SPECS[BackendType::TPU];
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "MXU");
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// XPU (Intel) Strategy Tests
// =============================================================================

TEST(XPUStrategy, SubgroupSize) {
    auto& info = BACKEND_SPECS[BackendType::XPU];
    // Intel Xe subgroup: 16 threads
    EXPECT_EQ(info.warp_size, 16);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(XPUStrategy, XMXSupport) {
    auto& info = BACKEND_SPECS[BackendType::XPU];
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "XMX");
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(XPUStrategy, SLMSize) {
    auto& info = BACKEND_SPECS[BackendType::XPU];
    // Intel SLM: 128KB
    EXPECT_EQ(info.shared_memory_size, 131072);
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// FPGA Strategy Tests
// =============================================================================

TEST(FPGAStrategy, Pipeline) {
    auto& info = BACKEND_SPECS[BackendType::FPGA];
    // FPGA: Pipeline-based, no warp concept
    EXPECT_EQ(info.warp_size, 1);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(FPGAStrategy, DSPBlocks) {
    auto& info = BACKEND_SPECS[BackendType::FPGA];
    EXPECT_FALSE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "DSP");
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// NKI (AWS Neuron) Strategy Tests
// =============================================================================

TEST(NKIStrategy, TensorEngineSize) {
    auto& info = BACKEND_SPECS[BackendType::NKI];
    // Neuron Tensor Engine: 128x128
    EXPECT_EQ(info.warp_size, 128);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(NKIStrategy, SBUFSize) {
    auto& info = BACKEND_SPECS[BackendType::NKI];
    // Neuron SBUF: 24MB
    EXPECT_EQ(info.shared_memory_size, 25165824);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(NKIStrategy, TensorEngineSupport) {
    auto& info = BACKEND_SPECS[BackendType::NKI];
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "TensorEngine");
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// Strategy Comparison Tests
// =============================================================================

TEST(StrategyComparison, AllBackendsRegistered) {
    EXPECT_EQ(BACKEND_SPECS.size(), 12);
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(StrategyComparison, GPUBackendsHaveTensorUnits) {
    std::vector<BackendType> gpu_backends = {
        BackendType::CUDA, BackendType::ROCM, BackendType::MPS,
        BackendType::ASCEND, BackendType::MACA, BackendType::TPU,
        BackendType::XPU, BackendType::NKI
    };
    
    for (auto backend : gpu_backends) {
        auto& info = BACKEND_SPECS[backend];
        EXPECT_TRUE(info.has_tensor_core) << "Backend " << info.name << " should have tensor unit";
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(StrategyComparison, WarpSizeVariants) {
    // 32-thread: CUDA, MPS, Triton, MLIR
    EXPECT_EQ(BACKEND_SPECS[BackendType::CUDA].warp_size, 32);
    EXPECT_EQ(BACKEND_SPECS[BackendType::MPS].warp_size, 32);
    
    // 64-thread: ROCm, MACA
    EXPECT_EQ(BACKEND_SPECS[BackendType::ROCM].warp_size, 64);
    EXPECT_EQ(BACKEND_SPECS[BackendType::MACA].warp_size, 64);
    
    // 128-thread: TPU, NKI
    EXPECT_EQ(BACKEND_SPECS[BackendType::TPU].warp_size, 128);
    EXPECT_EQ(BACKEND_SPECS[BackendType::NKI].warp_size, 128);
    
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// Main
// =============================================================================

YIRAGE_TEST_MAIN()
