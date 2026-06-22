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
 * Triton Persistent Kernel Backend Implementation
 */

#include "persistent_kernel/backends/triton_pk_backend.h"

#include <iostream>
#include <cstdlib>

#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include <hip/hip_runtime.h>
#endif

namespace yirage {
namespace pk {

// =============================================================================
// Constructor / Destructor
// =============================================================================

TritonPKBackend::TritonPKBackend()
    : is_initialized_(false), device_id_(0), gpu_context_(nullptr) {}

TritonPKBackend::~TritonPKBackend() {
    shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool TritonPKBackend::initialize(int device_id) {
    if (is_initialized_) {
        return true;
    }
    
    device_id_ = device_id;
    
    // Detect GPU backend and target architecture
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        target_arch_ = "sm_" + std::to_string(prop.major * 10 + prop.minor);
        is_initialized_ = true;
        return true;
    }
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device_id) == hipSuccess) {
        target_arch_ = prop.gcnArchName;
        is_initialized_ = true;
        return true;
    }
#endif
    
    return false;
}

void TritonPKBackend::shutdown() {
    is_initialized_ = false;
    gpu_context_ = nullptr;
}

bool TritonPKBackend::is_initialized() const {
    return is_initialized_;
}

// =============================================================================
// Memory Management
// =============================================================================

void* TritonPKBackend::allocate_memory(size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, size) == cudaSuccess) {
        return ptr;
    }
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size) == hipSuccess) {
        return ptr;
    }
#endif
    
    return nullptr;
}

void TritonPKBackend::free_memory(void* ptr) {
    if (!ptr) return;
    
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaFree(ptr);
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipFree(ptr);
#endif
}

bool TritonPKBackend::copy_to_device(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess;
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    return hipMemcpy(dst, src, size, hipMemcpyHostToDevice) == hipSuccess;
#endif
    
    return false;
}

bool TritonPKBackend::copy_to_host(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess;
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    return hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) == hipSuccess;
#endif
    
    return false;
}

// =============================================================================
// Kernel Execution
// =============================================================================

bool TritonPKBackend::launch_kernel(const PKKernelConfig& config) {
    // Standard kernel launch - would use compiled Triton binary
    return true;
}

void TritonPKBackend::synchronize() {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaDeviceSynchronize();
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipDeviceSynchronize();
#endif
}

// =============================================================================
// Triton-specific
// =============================================================================

bool TritonPKBackend::compile_kernel(const std::string& kernel_code,
                                     const std::string& kernel_name) {
    // This would invoke Triton compiler via Python subprocess
    // or using Triton's C++ compilation API
    
    // 1. Write kernel code to temporary file
    // 2. Invoke: triton-jit --target=<arch> kernel.py
    // 3. Load the resulting PTX/HSACO
    
    threadblock::triton::TritonKernelRegistry::instance()
        .register_kernel(kernel_name, kernel_code);
    
    return threadblock::triton::TritonKernelRegistry::instance()
        .compile_kernel(kernel_name, target_arch_);
}

bool TritonPKBackend::launch_triton_kernel(
    const std::string& kernel_name,
    void** args,
    int num_args,
    const threadblock::triton::TritonTileConfig& config
) {
    // Calculate grid dimensions
    // This would use the compiled Triton kernel binary
    
    // For CUDA:
    // cuLaunchKernel(kernel, grid_x, grid_y, grid_z, block_x, block_y, block_z, ...)
    
    return true;
}

std::string TritonPKBackend::get_target_arch() const {
    return target_arch_;
}

// =============================================================================
// Factory Registration
// =============================================================================

namespace {
    struct TritonPKBackendRegistrar {
        TritonPKBackendRegistrar() {
            // PKBackendFactory::register_backend(PKBackendType::TRITON, ...);
        }
    };
    static TritonPKBackendRegistrar triton_pk_registrar;
}

}  // namespace pk
}  // namespace yirage
