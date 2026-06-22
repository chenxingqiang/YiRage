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
 * Intel XPU Persistent Kernel Backend Implementation
 */

#include "persistent_kernel/backends/xpu_pk_backend.h"

#include <iostream>

#ifdef YIRAGE_BACKEND_XPU_ENABLED
#include <sycl/sycl.hpp>
#endif

namespace yirage {
namespace pk {

// =============================================================================
// Constructor / Destructor
// =============================================================================

XPUPKBackend::XPUPKBackend()
    : is_initialized_(false), device_id_(0) {}

XPUPKBackend::~XPUPKBackend() {
    shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool XPUPKBackend::initialize(int device_id) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    if (is_initialized_) {
        return true;
    }
    
    try {
        // Find Intel GPU
        auto gpu_selector = sycl::gpu_selector_v;
        device_ = sycl::device(gpu_selector);
        
        std::string vendor = device_.get_info<sycl::info::device::vendor>();
        if (vendor.find("Intel") == std::string::npos) {
            return false;
        }
        
        queue_ = std::make_unique<sycl::queue>(device_);
        device_id_ = device_id;
        is_initialized_ = true;
        return true;
        
    } catch (const sycl::exception& e) {
        std::cerr << "XPU PK init error: " << e.what() << std::endl;
        return false;
    }
#else
    return false;
#endif
}

void XPUPKBackend::shutdown() {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    if (queue_) {
        queue_->wait();
    }
    queue_.reset();
    is_initialized_ = false;
#endif
}

bool XPUPKBackend::is_initialized() const {
    return is_initialized_;
}

// =============================================================================
// Memory Management
// =============================================================================

void* XPUPKBackend::allocate_device(size_t size) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    return sycl::malloc_device(size, *queue_);
#else
    return nullptr;
#endif
}

void* XPUPKBackend::allocate_shared(size_t size) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    return sycl::malloc_shared(size, *queue_);
#else
    return nullptr;
#endif
}

void XPUPKBackend::free_memory(void* ptr) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    if (ptr && queue_) {
        sycl::free(ptr, *queue_);
    }
#endif
}

bool XPUPKBackend::copy_to_device(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    try {
        queue_->memcpy(dst, src, size).wait();
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool XPUPKBackend::copy_to_host(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    try {
        queue_->memcpy(dst, src, size).wait();
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

// =============================================================================
// Kernel Execution
// =============================================================================

bool XPUPKBackend::launch_kernel(const PKKernelConfig& config) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    // Launch SYCL kernel
    // This would be implemented with queue_->submit([&](sycl::handler& h) {...})
    return true;
#else
    return false;
#endif
}

void XPUPKBackend::synchronize() {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    if (queue_) {
        queue_->wait();
    }
#endif
}

// =============================================================================
// XPU-specific
// =============================================================================

int XPUPKBackend::get_subgroup_size() const {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    auto sizes = device_.get_info<sycl::info::device::sub_group_sizes>();
    // Prefer 16 for Arc, 32 for Max
    return sizes.empty() ? 16 : sizes.back();
#else
    return 16;
#endif
}

bool XPUPKBackend::has_xmx() const {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    std::string name = device_.get_info<sycl::info::device::name>();
    return name.find("Arc") != std::string::npos ||
           name.find("Max") != std::string::npos;
#else
    return false;
#endif
}

size_t XPUPKBackend::get_slm_size() const {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    return device_.get_info<sycl::info::device::local_mem_size>();
#else
    return 128 * 1024;  // 128KB typical
#endif
}

#ifdef YIRAGE_BACKEND_XPU_ENABLED
sycl::queue& XPUPKBackend::get_queue() {
    return *queue_;
}
#endif

// =============================================================================
// Factory Registration
// =============================================================================

#ifdef YIRAGE_BACKEND_XPU_ENABLED
namespace {
    struct XPUPKBackendRegistrar {
        XPUPKBackendRegistrar() {
            // PKBackendFactory::register_backend(PKBackendType::XPU, ...);
        }
    };
    static XPUPKBackendRegistrar xpu_pk_registrar;
}
#endif

}  // namespace pk
}  // namespace yirage
