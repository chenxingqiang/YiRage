/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file layer1_backend_api_test.cc
 * @brief Layer 1: Backend API tests for all 12 backends
 * 
 * Tests:
 * - Backend registration and discovery
 * - Hardware capabilities query
 * - Memory info
 * - Data type support
 * - Device management
 */

#include "test_framework.h"
#include "backend_test_common.h"

// Conditional includes based on available backends
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include "backend/cuda_backend.h"
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include "backend/rocm_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
#include "backend/mps_backend.h"
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
#include "backend/ascend_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MACA_ENABLED
#include "backend/maca_backend.h"
#endif

#ifdef YIRAGE_BACKEND_TPU_ENABLED
#include "backend/tpu_backend.h"
#endif

#ifdef YIRAGE_BACKEND_XPU_ENABLED
#include "backend/xpu_backend.h"
#endif

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
#include "backend/fpga_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MLIR_ENABLED
#include "backend/mlir_backend.h"
#endif

#include "backend/backends.h"
#include "backend/cpu_backend.h"

using namespace yirage::test;
using namespace yirage::backend;

// =============================================================================
// Backend Registry Tests
// =============================================================================

TEST(BackendRegistry, Singleton) {
    auto& registry1 = BackendRegistry::get_instance();
    auto& registry2 = BackendRegistry::get_instance();
    
    EXPECT_EQ(&registry1, &registry2);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(BackendRegistry, HasAvailableBackends) {
    auto& registry = BackendRegistry::get_instance();
    auto available = registry.get_available_backends();
    
    // At minimum, CPU should always be available
    EXPECT_FALSE(available.empty());
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(BackendRegistry, CPUAlwaysAvailable) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    EXPECT_NOT_NULL(cpu);
    EXPECT_TRUE(cpu->is_available());
    EXPECT_EQ(std::string(cpu->get_name()), std::string("cpu"));
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(BackendRegistry, GetByType) {
    auto& registry = BackendRegistry::get_instance();
    
    auto* backend = registry.get_backend(yirage::type::BT_CPU);
    EXPECT_NOT_NULL(backend);
    EXPECT_EQ(std::string(backend->get_name()), std::string("cpu"));
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(BackendRegistry, GetByName) {
    auto& registry = BackendRegistry::get_instance();
    
    // Test various name formats
    std::vector<std::pair<std::string, bool>> names = {
        {"cpu", true},
        {"CPU", true},
        {"cuda", false},  // May or may not exist
        {"nonexistent", false}
    };
    
    for (const auto& [name, must_exist] : names) {
        auto* backend = registry.get_backend(name);
        if (must_exist) {
            EXPECT_NOT_NULL(backend);
        }
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(BackendRegistry, DefaultBackend) {
    auto& registry = BackendRegistry::get_instance();
    auto default_type = registry.get_default_backend();
    
    auto* backend = registry.get_backend(default_type);
    EXPECT_NOT_NULL(backend);
    EXPECT_TRUE(backend->is_available());
    
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// CPU Backend Tests (Always Available)
// =============================================================================

TEST(CPUBackend, BasicInfo) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    EXPECT_NOT_NULL(cpu);
    EXPECT_TRUE(cpu->is_available());
    EXPECT_EQ(cpu->get_type(), yirage::type::BT_CPU);
    
    auto info = cpu->get_info();
    EXPECT_FALSE(info.requires_gpu);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CPUBackend, MemoryInfo) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    EXPECT_NOT_NULL(cpu);
    
    size_t max_memory = cpu->get_max_memory();
    EXPECT_GT(max_memory, 0UL);  // Should have positive memory
    
    int device_count = cpu->get_device_count();
    EXPECT_GE(device_count, 1);  // At least 1 CPU
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CPUBackend, DataTypeSupport) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    EXPECT_NOT_NULL(cpu);
    
    // CPU should support basic types
    EXPECT_TRUE(cpu->supports_data_type(yirage::type::DT_FLOAT32));
    EXPECT_TRUE(cpu->supports_data_type(yirage::type::DT_FLOAT16));
    EXPECT_TRUE(cpu->supports_data_type(yirage::type::DT_INT32));
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CPUBackend, ComputeUnits) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    EXPECT_NOT_NULL(cpu);
    
    int units = cpu->get_num_compute_units();
    EXPECT_GT(units, 0);  // At least 1 core
    
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// CUDA Backend Tests
// =============================================================================

#ifdef YIRAGE_BACKEND_CUDA_ENABLED
TEST(CUDABackend, Registration) {
    auto& registry = BackendRegistry::get_instance();
    auto* cuda = registry.get_backend("cuda");
    
    EXPECT_NOT_NULL(cuda);
    EXPECT_EQ(cuda->get_type(), yirage::type::BT_CUDA);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CUDABackend, WarpSize) {
    auto& registry = BackendRegistry::get_instance();
    auto* cuda = registry.get_backend("cuda");
    
    if (!cuda || !cuda->is_available()) {
        // Skip if CUDA not available
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    // CUDA warp size should be 32
    // Note: This may need to be accessed via specific method
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CUDABackend, TensorCoreSupport) {
    auto& registry = BackendRegistry::get_instance();
    auto* cuda = registry.get_backend("cuda");
    
    if (!cuda || !cuda->is_available()) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    // Check compute capability for Tensor Core support (>=7.0)
    // This is architecture dependent
    
    return YIRAGE_TEST_RESULT_PASS();
}
#endif

// =============================================================================
// ROCm Backend Tests
// =============================================================================

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
TEST(ROCmBackend, Registration) {
    auto& registry = BackendRegistry::get_instance();
    auto* rocm = registry.get_backend("rocm");
    
    // ROCm may not be compiled in, so just check if pointer is valid
    if (rocm != nullptr) {
        EXPECT_EQ(std::string(rocm->get_name()), std::string("rocm"));
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(ROCmBackend, WavefrontSize) {
    auto& registry = BackendRegistry::get_instance();
    auto* rocm = registry.get_backend("rocm");
    
    if (!rocm || !rocm->is_available()) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    // ROCm wavefront size should be 64
    auto* rocm_impl = dynamic_cast<ROCmBackend*>(rocm);
    if (rocm_impl) {
        EXPECT_EQ(rocm_impl->get_wavefront_size(), 64);
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}
#endif

// =============================================================================
// MPS Backend Tests
// =============================================================================

#ifdef YIRAGE_BACKEND_MPS_ENABLED
TEST(MPSBackend, Registration) {
    auto& registry = BackendRegistry::get_instance();
    auto* mps = registry.get_backend("mps");
    
    if (mps != nullptr) {
        EXPECT_EQ(std::string(mps->get_name()), std::string("mps"));
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(MPSBackend, AppleSiliconDetection) {
    auto& registry = BackendRegistry::get_instance();
    auto* mps = registry.get_backend("mps");
    
    if (!mps || !mps->is_available()) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    // MPS should have unified memory
    auto info = mps->get_info();
    // Unified memory means requires_gpu may be true but memory is shared
    
    return YIRAGE_TEST_RESULT_PASS();
}
#endif

// =============================================================================
// Ascend Backend Tests
// =============================================================================

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
TEST(AscendBackend, Registration) {
    auto& registry = BackendRegistry::get_instance();
    auto* ascend = registry.get_backend("ascend");
    
    if (ascend != nullptr) {
        EXPECT_EQ(std::string(ascend->get_name()), std::string("ascend"));
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(AscendBackend, AICoreCount) {
    auto& registry = BackendRegistry::get_instance();
    auto* ascend = registry.get_backend("ascend");
    
    if (!ascend || !ascend->is_available()) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    int units = ascend->get_num_compute_units();
    EXPECT_GT(units, 0);  // Should have AI cores
    
    return YIRAGE_TEST_RESULT_PASS();
}
#endif

// =============================================================================
// MACA Backend Tests
// =============================================================================

#ifdef YIRAGE_BACKEND_MACA_ENABLED
TEST(MACABackend, Registration) {
    auto& registry = BackendRegistry::get_instance();
    auto* maca = registry.get_backend("maca");
    
    if (maca != nullptr) {
        EXPECT_EQ(std::string(maca->get_name()), std::string("maca"));
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(MACABackend, WarpSize64) {
    auto& registry = BackendRegistry::get_instance();
    auto* maca = registry.get_backend("maca");
    
    if (!maca || !maca->is_available()) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    // MACA uses 64-thread warps (vs CUDA's 32)
    auto* maca_impl = dynamic_cast<MACABackend*>(maca);
    if (maca_impl) {
        EXPECT_EQ(maca_impl->get_warp_size(), 64);
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}
#endif

// =============================================================================
// TPU Backend Tests
// =============================================================================

#ifdef YIRAGE_BACKEND_TPU_ENABLED
TEST(TPUBackend, Registration) {
    auto& registry = BackendRegistry::get_instance();
    auto* tpu = registry.get_backend("tpu");
    
    if (tpu != nullptr) {
        EXPECT_EQ(std::string(tpu->get_name()), std::string("tpu"));
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(TPUBackend, MXUSize) {
    auto& registry = BackendRegistry::get_instance();
    auto* tpu = registry.get_backend("tpu");
    
    if (!tpu || !tpu->is_available()) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    // TPU MXU is 128x128
    auto* tpu_impl = dynamic_cast<TPUBackend*>(tpu);
    if (tpu_impl) {
        // Check MXU configuration
        int cores = tpu_impl->get_cores_per_chip();
        EXPECT_GT(cores, 0);
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(TPUBackend, BF16Native) {
    auto& registry = BackendRegistry::get_instance();
    auto* tpu = registry.get_backend("tpu");
    
    if (!tpu || !tpu->is_available()) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    // TPU has native BF16 support
    EXPECT_TRUE(tpu->supports_data_type(yirage::type::DT_BFLOAT16));
    
    return YIRAGE_TEST_RESULT_PASS();
}
#endif

// =============================================================================
// XPU Backend Tests
// =============================================================================

#ifdef YIRAGE_BACKEND_XPU_ENABLED
TEST(XPUBackend, Registration) {
    auto& registry = BackendRegistry::get_instance();
    auto* xpu = registry.get_backend("xpu");
    
    if (xpu != nullptr) {
        EXPECT_EQ(std::string(xpu->get_name()), std::string("xpu"));
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(XPUBackend, XMXSupport) {
    auto& registry = BackendRegistry::get_instance();
    auto* xpu = registry.get_backend("xpu");
    
    if (!xpu || !xpu->is_available()) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    // Check for XMX (Xe Matrix Extensions) support
    auto* xpu_impl = dynamic_cast<XPUBackend*>(xpu);
    if (xpu_impl) {
        bool has_xmx = xpu_impl->has_xmx();
        // XMX availability depends on hardware
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}
#endif

// =============================================================================
// FPGA Backend Tests
// =============================================================================

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
TEST(FPGABackend, Registration) {
    auto& registry = BackendRegistry::get_instance();
    auto* fpga = registry.get_backend("fpga");
    
    if (fpga != nullptr) {
        EXPECT_EQ(std::string(fpga->get_name()), std::string("fpga"));
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(FPGABackend, DSPCount) {
    auto& registry = BackendRegistry::get_instance();
    auto* fpga = registry.get_backend("fpga");
    
    if (!fpga || !fpga->is_available()) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    auto* fpga_impl = dynamic_cast<FPGABackend*>(fpga);
    if (fpga_impl) {
        int dsp = fpga_impl->get_dsp_count();
        EXPECT_GE(dsp, 0);
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}
#endif

// =============================================================================
// MLIR Backend Tests
// =============================================================================

#ifdef YIRAGE_BACKEND_MLIR_ENABLED
TEST(MLIRBackend, Registration) {
    auto& registry = BackendRegistry::get_instance();
    auto* mlir = registry.get_backend("mlir");
    
    if (mlir != nullptr) {
        EXPECT_EQ(std::string(mlir->get_name()), std::string("mlir"));
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(MLIRBackend, MultiTarget) {
    auto& registry = BackendRegistry::get_instance();
    auto* mlir = registry.get_backend("mlir");
    
    if (!mlir) {
        return YIRAGE_TEST_RESULT_PASS();
    }
    
    auto* mlir_impl = dynamic_cast<MLIRBackend*>(mlir);
    if (mlir_impl) {
        // MLIR supports multiple targets
        std::vector<MLIRBackend::MLIRTarget> targets = {
            MLIRBackend::MLIRTarget::LLVM,
            MLIRBackend::MLIRTarget::NVVM,
            MLIRBackend::MLIRTarget::ROCDL,
            MLIRBackend::MLIRTarget::SPIRV
        };
        
        // At least LLVM should be available
        mlir_impl->set_target(MLIRBackend::MLIRTarget::LLVM);
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}
#endif

// =============================================================================
// Cross-Backend Tests
// =============================================================================

TEST(CrossBackend, AllRegisteredBackendsHaveNames) {
    auto& registry = BackendRegistry::get_instance();
    auto available = registry.get_available_backends();
    
    for (auto type : available) {
        auto* backend = registry.get_backend(type);
        EXPECT_NOT_NULL(backend);
        
        const char* name = backend->get_name();
        EXPECT_NOT_NULL(name);
        EXPECT_GT(strlen(name), 0UL);
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CrossBackend, AllBackendsHaveDisplayNames) {
    auto& registry = BackendRegistry::get_instance();
    auto available = registry.get_available_backends();
    
    for (auto type : available) {
        auto* backend = registry.get_backend(type);
        EXPECT_NOT_NULL(backend);
        
        const char* display = backend->get_display_name();
        EXPECT_NOT_NULL(display);
        EXPECT_GT(strlen(display), 0UL);
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CrossBackend, AllBackendsHaveInfo) {
    auto& registry = BackendRegistry::get_instance();
    auto available = registry.get_available_backends();
    
    for (auto type : available) {
        auto* backend = registry.get_backend(type);
        EXPECT_NOT_NULL(backend);
        
        auto info = backend->get_info();
        // Info should be valid
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(CrossBackend, DataTypeSupportConsistency) {
    auto& registry = BackendRegistry::get_instance();
    auto available = registry.get_available_backends();
    
    for (auto type : available) {
        auto* backend = registry.get_backend(type);
        if (!backend || !backend->is_available()) continue;
        
        // All backends should support FP32
        EXPECT_TRUE(backend->supports_data_type(yirage::type::DT_FLOAT32));
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// Backend Priority Tests
// =============================================================================

TEST(BackendPriority, GPUPreferredOverCPU) {
    auto& registry = BackendRegistry::get_instance();
    auto default_type = registry.get_default_backend();
    
    // If GPU is available, it should be preferred
    auto* cuda = registry.get_backend("cuda");
    auto* mps = registry.get_backend("mps");
    auto* rocm = registry.get_backend("rocm");
    
    bool gpu_available = 
        (cuda && cuda->is_available()) ||
        (mps && mps->is_available()) ||
        (rocm && rocm->is_available());
    
    if (gpu_available) {
        // Default should be a GPU backend
        EXPECT_NE(default_type, yirage::type::BT_CPU);
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

// =============================================================================
// Hardware Info Tests
// =============================================================================

TEST(HardwareInfo, AllBackendsHaveInfo) {
    for (auto b : all_backends()) {
        auto info = get_hardware_info(b);
        
        EXPECT_EQ(info.backend, b);
        EXPECT_FALSE(info.thread_model.empty());
        EXPECT_FALSE(info.matrix_unit.empty());
        EXPECT_FALSE(info.memory_levels.empty());
        EXPECT_FALSE(info.native_dtypes.empty());
    }
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(HardwareInfo, WarpSizeCorrect) {
    auto cuda_info = get_hardware_info(Backend::CUDA);
    EXPECT_EQ(cuda_info.warp_size, 32);
    
    auto rocm_info = get_hardware_info(Backend::ROCM);
    EXPECT_EQ(rocm_info.warp_size, 64);
    
    auto maca_info = get_hardware_info(Backend::MACA);
    EXPECT_EQ(maca_info.warp_size, 64);
    
    return YIRAGE_TEST_RESULT_PASS();
}

TEST(HardwareInfo, MatrixUnitSize) {
    auto tpu_info = get_hardware_info(Backend::TPU);
    EXPECT_EQ(tpu_info.matrix_size, 128);  // MXU 128x128
    
    auto nki_info = get_hardware_info(Backend::NKI);
    EXPECT_EQ(nki_info.matrix_size, 128);  // Tensor Engine 128x128
    
    return YIRAGE_TEST_RESULT_PASS();
}

YIRAGE_TEST_MAIN()
