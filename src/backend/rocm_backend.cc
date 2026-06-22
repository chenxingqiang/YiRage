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
 *
 * ROCm Backend Implementation for AMD GPUs
 */

#include "backend/rocm_backend.h"
#include "backend/backend_registry.h"

#include <cstdlib>
#include <iostream>
#include <cstring>

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

namespace yirage {
namespace backend {

// =============================================================================
// Constructor / Destructor
// =============================================================================

ROCmBackend::ROCmBackend()
    : is_available_(false), current_device_(0), device_count_(0) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    rocblas_handle_ = nullptr;
#endif
    is_available_ = check_rocm_availability();
    if (is_available_) {
        query_device_properties();
    }
}

ROCmBackend::~ROCmBackend() {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    if (rocblas_handle_) {
        rocblas_destroy_handle(rocblas_handle_);
    }
#endif
}

// =============================================================================
// Availability Check
// =============================================================================

bool ROCmBackend::check_rocm_availability() {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipError_t err = hipGetDeviceCount(&device_count_);
    if (err != hipSuccess || device_count_ == 0) {
        return false;
    }

    // Try to get properties of device 0
    err = hipGetDeviceProperties(&device_prop_, 0);
    if (err != hipSuccess) {
        return false;
    }

    // Initialize rocBLAS handle
    rocblas_status status = rocblas_create_handle(&rocblas_handle_);
    if (status != rocblas_status_success) {
        std::cerr << "Warning: Failed to create rocBLAS handle" << std::endl;
    }

    return true;
#else
    return false;
#endif
}

void ROCmBackend::query_device_properties() {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    if (device_count_ > 0) {
        hipGetDevice(&current_device_);
        hipGetDeviceProperties(&device_prop_, current_device_);
    }
#endif
}

// =============================================================================
// Backend Information
// =============================================================================

type::BackendType ROCmBackend::get_type() const {
    return type::BT_ROCM;
}

std::string ROCmBackend::get_name() const {
    return "rocm";
}

std::string ROCmBackend::get_display_name() const {
    return "ROCm (AMD GPU)";
}

bool ROCmBackend::is_available() const {
    return is_available_;
}

type::BackendInfo ROCmBackend::get_info() const {
    type::BackendInfo info;
    info.type = type::BT_ROCM;
    info.name = "rocm";
    info.display_name = "ROCm (AMD GPU)";
    info.requires_gpu = true;
    info.required_libs = {"amdhip64", "rocblas", "hipblas"};
    return info;
}

// =============================================================================
// Compilation
// =============================================================================

bool ROCmBackend::compile(CompileContext const& ctx) {
    // HIP compilation is typically handled by hipcc/CMake
    // Runtime compilation would use hiprtc
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    // Could implement hiprtc compilation here
    return true;
#else
    return false;
#endif
}

std::string ROCmBackend::get_compile_flags() const {
    std::string flags = "-std=c++17 -O3 -fPIC";
    
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    // Add architecture-specific flags
    flags += " --offload-arch=" + get_arch_name();
#endif
    
    return flags;
}

std::vector<std::string> ROCmBackend::get_include_dirs() const {
    std::vector<std::string> dirs;
    
    const char* rocm_path = getenv("ROCM_PATH");
    if (!rocm_path) {
        rocm_path = "/opt/rocm";  // Default ROCm installation path
    }
    
    dirs.push_back(std::string(rocm_path) + "/include");
    dirs.push_back(std::string(rocm_path) + "/include/hip");
    dirs.push_back(std::string(rocm_path) + "/include/rocblas");
    
    return dirs;
}

std::vector<std::string> ROCmBackend::get_library_dirs() const {
    std::vector<std::string> dirs;
    
    const char* rocm_path = getenv("ROCM_PATH");
    if (!rocm_path) {
        rocm_path = "/opt/rocm";
    }
    
    dirs.push_back(std::string(rocm_path) + "/lib");
    dirs.push_back(std::string(rocm_path) + "/lib64");
    
    return dirs;
}

std::vector<std::string> ROCmBackend::get_link_libraries() const {
    return {"amdhip64", "rocblas", "hipblas", "roctx64", "roctracer64"};
}

// =============================================================================
// Memory Management
// =============================================================================

void* ROCmBackend::allocate_memory(size_t size) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    void* ptr = nullptr;
    hipError_t err = hipMalloc(&ptr, size);
    if (err != hipSuccess) {
        std::cerr << "ROCm: Failed to allocate " << size << " bytes: "
                  << hipGetErrorString(err) << std::endl;
        return nullptr;
    }
    return ptr;
#else
    return nullptr;
#endif
}

void ROCmBackend::free_memory(void* ptr) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    if (ptr) {
        hipFree(ptr);
    }
#endif
}

bool ROCmBackend::copy_to_device(void* dst, void const* src, size_t size) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipError_t err = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    return err == hipSuccess;
#else
    return false;
#endif
}

bool ROCmBackend::copy_to_host(void* dst, void const* src, size_t size) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipError_t err = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    return err == hipSuccess;
#else
    return false;
#endif
}

bool ROCmBackend::copy_device_to_device(void* dst, void const* src, size_t size) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipError_t err = hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
    return err == hipSuccess;
#else
    return false;
#endif
}

// =============================================================================
// Synchronization
// =============================================================================

void ROCmBackend::synchronize() {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipDeviceSynchronize();
#endif
}

// =============================================================================
// Capability Query
// =============================================================================

size_t ROCmBackend::get_max_memory() const {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    return device_prop_.totalGlobalMem;
#else
    return 0;
#endif
}

size_t ROCmBackend::get_max_shared_memory() const {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    return device_prop_.sharedMemPerBlock;
#else
    return 0;
#endif
}

bool ROCmBackend::supports_data_type(type::DataType dt) const {
    switch (dt) {
        case type::DT_FLOAT32:
        case type::DT_FLOAT16:
        case type::DT_BFLOAT16:
        case type::DT_INT32:
        case type::DT_INT8:
            return true;
        case type::DT_DOUBLE:
            return true;  // AMD GPUs have good FP64
        default:
            return false;
    }
}

int ROCmBackend::get_compute_capability() const {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    // AMD uses GCN architecture version
    // Convert to a comparable format (e.g., gfx90a -> 900)
    return device_prop_.gcnArch;
#else
    return 0;
#endif
}

int ROCmBackend::get_num_compute_units() const {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    return device_prop_.multiProcessorCount;
#else
    return 0;
#endif
}

// =============================================================================
// Device Management
// =============================================================================

bool ROCmBackend::set_device(int device_id) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    if (device_id >= 0 && device_id < device_count_) {
        hipError_t err = hipSetDevice(device_id);
        if (err == hipSuccess) {
            current_device_ = device_id;
            query_device_properties();
            return true;
        }
    }
    return false;
#else
    return false;
#endif
}

int ROCmBackend::get_device() const {
    return current_device_;
}

int ROCmBackend::get_device_count() const {
    return device_count_;
}

// =============================================================================
// ROCm-specific
// =============================================================================

std::string ROCmBackend::get_arch_name() const {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    return device_prop_.gcnArchName;
#else
    return "gfx000";
#endif
}

bool ROCmBackend::has_matrix_cores() const {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    // CDNA architecture (MI100+) has matrix cores
    // gfx908 = MI100, gfx90a = MI200, gfx942 = MI300
    int arch = device_prop_.gcnArch;
    return arch >= 908;
#else
    return false;
#endif
}

size_t ROCmBackend::get_lds_size() const {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    return device_prop_.sharedMemPerBlock;  // LDS = Local Data Share
#else
    return 0;
#endif
}

// =============================================================================
// Helper Functions
// =============================================================================

AMDGPUInfo get_amd_gpu_info(int device_id) {
    AMDGPUInfo info;
    
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device_id) == hipSuccess) {
        info.arch_name = prop.gcnArchName;
        info.compute_units = prop.multiProcessorCount;
        info.wavefront_size = prop.warpSize;  // 64 for AMD
        info.global_memory = prop.totalGlobalMem;
        info.lds_size = prop.sharedMemPerBlock;
        
        // Detect matrix core generation
        int arch = prop.gcnArch;
        if (arch >= 942) {
            info.has_matrix_cores = true;
            info.matrix_core_gen = 3;  // MI300
        } else if (arch >= 90) {
            info.has_matrix_cores = true;
            info.matrix_core_gen = 2;  // MI200
        } else if (arch >= 908) {
            info.has_matrix_cores = true;
            info.matrix_core_gen = 1;  // MI100
        } else {
            info.has_matrix_cores = false;
            info.matrix_core_gen = 0;
        }
    }
#else
    info.arch_name = "unknown";
    info.compute_units = 0;
    info.wavefront_size = 64;
    info.global_memory = 0;
    info.lds_size = 0;
    info.has_matrix_cores = false;
    info.matrix_core_gen = 0;
#endif
    
    return info;
}

// =============================================================================
// Backend Registration
// =============================================================================

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
namespace {
    struct ROCmBackendRegistrar {
        ROCmBackendRegistrar() {
            auto backend = std::make_shared<ROCmBackend>();
            BackendRegistry::instance().register_backend(
                type::BT_ROCM, backend);
        }
    };
    static ROCmBackendRegistrar rocm_registrar;
}
#endif

}  // namespace backend
}  // namespace yirage
