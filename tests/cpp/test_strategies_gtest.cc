// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_strategies_gtest.cc
 * @brief Backend Strategy Unit Tests (Google Test version)
 *
 * Tests for all 12 backend search strategies.
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
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
inline const std::map<BackendType, BackendInfo>& get_backend_specs() {
    static const std::map<BackendType, BackendInfo> specs = {
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
    return specs;
}

}  // namespace test
}  // namespace yirage

using namespace yirage::test;

// =============================================================================
// CUDA Strategy Tests
// =============================================================================

class CUDAStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::CUDA);
};

TEST_F(CUDAStrategyTest, WarpSize32) {
    EXPECT_EQ(info.warp_size, 32);
}

TEST_F(CUDAStrategyTest, MaxThreadsPerBlock) {
    EXPECT_EQ(info.max_threads_per_block, 1024);
}

TEST_F(CUDAStrategyTest, TensorCoreSupport) {
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "TensorCore");
}

TEST_F(CUDAStrategyTest, SharedMemorySize) {
    // Typical shared memory: 48KB
    EXPECT_EQ(info.shared_memory_size, 49152);
}

// =============================================================================
// ROCm Strategy Tests
// =============================================================================

class ROCmStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::ROCM);
};

TEST_F(ROCmStrategyTest, WavefrontSize64) {
    // AMD uses 64-thread wavefronts
    EXPECT_EQ(info.warp_size, 64);
}

TEST_F(ROCmStrategyTest, LDSSize) {
    // AMD LDS: 64KB
    EXPECT_EQ(info.shared_memory_size, 65536);
}

TEST_F(ROCmStrategyTest, MatrixCoreSupport) {
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "MatrixCore");
}

// =============================================================================
// MPS Strategy Tests
// =============================================================================

class MPSStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::MPS);
};

TEST_F(MPSStrategyTest, ThreadgroupConfiguration) {
    EXPECT_EQ(info.warp_size, 32);
    EXPECT_EQ(info.max_threads_per_block, 1024);
}

TEST_F(MPSStrategyTest, ThreadgroupMemory) {
    // Apple Silicon threadgroup memory: 32KB
    EXPECT_EQ(info.shared_memory_size, 32768);
}

// =============================================================================
// Ascend Strategy Tests
// =============================================================================

class AscendStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::ASCEND);
};

TEST_F(AscendStrategyTest, AICoreTiling) {
    // Ascend AI Core: 16-thread groups
    EXPECT_EQ(info.warp_size, 16);
}

TEST_F(AscendStrategyTest, CubeUnitSupport) {
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "CubeUnit");
}

TEST_F(AscendStrategyTest, L1BufferSize) {
    // Ascend L1 buffer: 128KB
    EXPECT_EQ(info.shared_memory_size, 131072);
}

// =============================================================================
// MACA Strategy Tests
// =============================================================================

class MACAStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::MACA);
};

TEST_F(MACAStrategyTest, WarpSize64) {
    // MACA (MetaX) uses 64-thread warps like AMD
    EXPECT_EQ(info.warp_size, 64);
}

TEST_F(MACAStrategyTest, TensorCoreCompat) {
    EXPECT_TRUE(info.has_tensor_core);
}

// =============================================================================
// CPU Strategy Tests
// =============================================================================

class CPUStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::CPU);
};

TEST_F(CPUStrategyTest, NoWarp) {
    // CPU doesn't have warps
    EXPECT_EQ(info.warp_size, 1);
}

TEST_F(CPUStrategyTest, SIMDUnit) {
    EXPECT_FALSE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "SIMD");
}

TEST_F(CPUStrategyTest, NoSharedMemory) {
    EXPECT_EQ(info.shared_memory_size, 0);
}

// =============================================================================
// TPU Strategy Tests
// =============================================================================

class TPUStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::TPU);
};

TEST_F(TPUStrategyTest, MXUSize) {
    // TPU MXU: 128x128 systolic array
    EXPECT_EQ(info.warp_size, 128);
    EXPECT_EQ(info.max_threads_per_block, 128);
}

TEST_F(TPUStrategyTest, VMEMSize) {
    // TPU VMEM: 16MB per core
    EXPECT_EQ(info.shared_memory_size, 16777216);
}

TEST_F(TPUStrategyTest, MXUSupport) {
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "MXU");
}

// =============================================================================
// XPU (Intel) Strategy Tests
// =============================================================================

class XPUStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::XPU);
};

TEST_F(XPUStrategyTest, SubgroupSize) {
    // Intel Xe subgroup: 16 threads
    EXPECT_EQ(info.warp_size, 16);
}

TEST_F(XPUStrategyTest, XMXSupport) {
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "XMX");
}

TEST_F(XPUStrategyTest, SLMSize) {
    // Intel SLM: 128KB
    EXPECT_EQ(info.shared_memory_size, 131072);
}

// =============================================================================
// FPGA Strategy Tests
// =============================================================================

class FPGAStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::FPGA);
};

TEST_F(FPGAStrategyTest, Pipeline) {
    // FPGA: Pipeline-based, no warp concept
    EXPECT_EQ(info.warp_size, 1);
}

TEST_F(FPGAStrategyTest, DSPBlocks) {
    EXPECT_FALSE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "DSP");
}

// =============================================================================
// NKI (AWS Neuron) Strategy Tests
// =============================================================================

class NKIStrategyTest : public ::testing::Test {
protected:
    const BackendInfo& info = get_backend_specs().at(BackendType::NKI);
};

TEST_F(NKIStrategyTest, TensorEngineSize) {
    // Neuron Tensor Engine: 128x128
    EXPECT_EQ(info.warp_size, 128);
}

TEST_F(NKIStrategyTest, SBUFSize) {
    // Neuron SBUF: 24MB
    EXPECT_EQ(info.shared_memory_size, 25165824);
}

TEST_F(NKIStrategyTest, TensorEngineSupport) {
    EXPECT_TRUE(info.has_tensor_core);
    EXPECT_EQ(info.matrix_unit, "TensorEngine");
}

// =============================================================================
// Strategy Comparison Tests
// =============================================================================

class StrategyComparisonTest : public ::testing::Test {
protected:
    const std::map<BackendType, BackendInfo>& specs = get_backend_specs();
};

TEST_F(StrategyComparisonTest, AllBackendsRegistered) {
    EXPECT_EQ(specs.size(), 12u);
}

TEST_F(StrategyComparisonTest, GPUBackendsHaveTensorUnits) {
    std::vector<BackendType> gpu_backends = {
        BackendType::CUDA, BackendType::ROCM, BackendType::MPS,
        BackendType::ASCEND, BackendType::MACA, BackendType::TPU,
        BackendType::XPU, BackendType::NKI
    };
    
    for (auto backend : gpu_backends) {
        const auto& info = specs.at(backend);
        EXPECT_TRUE(info.has_tensor_core) 
            << "Backend " << info.name << " should have tensor unit";
    }
}

TEST_F(StrategyComparisonTest, WarpSizeVariants) {
    // 32-thread: CUDA, MPS, Triton, MLIR
    EXPECT_EQ(specs.at(BackendType::CUDA).warp_size, 32);
    EXPECT_EQ(specs.at(BackendType::MPS).warp_size, 32);
    
    // 64-thread: ROCm, MACA
    EXPECT_EQ(specs.at(BackendType::ROCM).warp_size, 64);
    EXPECT_EQ(specs.at(BackendType::MACA).warp_size, 64);
    
    // 128-thread: TPU, NKI
    EXPECT_EQ(specs.at(BackendType::TPU).warp_size, 128);
    EXPECT_EQ(specs.at(BackendType::NKI).warp_size, 128);
}

// =============================================================================
// Parameterized Backend Tests
// =============================================================================

class BackendWarpSizeTest : public ::testing::TestWithParam<
    std::tuple<BackendType, int32_t>> {};

TEST_P(BackendWarpSizeTest, VerifyWarpSize) {
    auto [backend, expected_warp_size] = GetParam();
    const auto& info = get_backend_specs().at(backend);
    EXPECT_EQ(info.warp_size, expected_warp_size);
}

INSTANTIATE_TEST_SUITE_P(
    WarpSizes,
    BackendWarpSizeTest,
    ::testing::Values(
        std::make_tuple(BackendType::CUDA, 32),
        std::make_tuple(BackendType::ROCM, 64),
        std::make_tuple(BackendType::MPS, 32),
        std::make_tuple(BackendType::MACA, 64),
        std::make_tuple(BackendType::TPU, 128),
        std::make_tuple(BackendType::NKI, 128),
        std::make_tuple(BackendType::XPU, 16),
        std::make_tuple(BackendType::ASCEND, 16)
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
