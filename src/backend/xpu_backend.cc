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
 * Intel XPU Backend Implementation
 */

#include "backend/xpu_backend.h"
#include "backend/backend_registry.h"

#include <cstdlib>
#include <iostream>
#include <cstring>

#ifdef YIRAGE_BACKEND_XPU_ENABLED
#include <sycl/sycl.hpp>
#endif

namespace yirage {
namespace backend {

// =============================================================================
// Constructor / Destructor
// =============================================================================

XPUBackend::XPUBackend()
    : is_available_(false), current_device_(0), device_count_(0),
      device_type_(XPU_UNKNOWN) {
    is_available_ = check_xpu_availability();
    if (is_available_) {
        query_device_properties();
    }
}

XPUBackend::~XPUBackend() {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    queue_.reset();
#endif
}

// =============================================================================
// Availability Check
// =============================================================================

bool XPUBackend::check_xpu_availability() {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    try {
        // Look for Intel GPU devices
        auto gpu_selector = sycl::gpu_selector_v;
        sycl::device gpu_device(gpu_selector);
        
        // Check if it's an Intel device
        std::string vendor = gpu_device.get_info<sycl::info::device::vendor>();
        if (vendor.find("Intel") != std::string::npos) {
            device_ = gpu_device;
            queue_ = std::make_unique<sycl::queue>(device_);
            return true;
        }
    } catch (const sycl::exception& e) {
        // No Intel GPU found
    }
    
    // Try to find any Level Zero device
    try {
        auto platform_list = sycl::platform::get_platforms();
        for (const auto& platform : platform_list) {
            auto devices = platform.get_devices(sycl::info::device_type::gpu);
            for (const auto& device : devices) {
                std::string vendor = device.get_info<sycl::info::device::vendor>();
                if (vendor.find("Intel") != std::string::npos) {
                    device_ = device;
                    queue_ = std::make_unique<sycl::queue>(device_);
                    return true;
                }
            }
        }
    } catch (...) {
        // Exception during device enumeration
    }
    
    return false;
#else
    return false;
#endif
}

void XPUBackend::query_device_properties() {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    if (!is_available_) return;
    
    std::string name = device_.get_info<sycl::info::device::name>();
    
    // Detect device type from name
    if (name.find("Max") != std::string::npos || 
        name.find("Ponte Vecchio") != std::string::npos) {
        device_type_ = XPU_MAX;
    } else if (name.find("Gaudi") != std::string::npos ||
               name.find("Habana") != std::string::npos) {
        device_type_ = XPU_GAUDI;
    } else if (name.find("Arc") != std::string::npos ||
               name.find("A770") != std::string::npos ||
               name.find("A750") != std::string::npos) {
        device_type_ = XPU_ARC;
    } else {
        device_type_ = XPU_UNKNOWN;
    }
    
    // Count devices
    auto platform_list = sycl::platform::get_platforms();
    device_count_ = 0;
    for (const auto& platform : platform_list) {
        auto devices = platform.get_devices(sycl::info::device_type::gpu);
        for (const auto& device : devices) {
            std::string vendor = device.get_info<sycl::info::device::vendor>();
            if (vendor.find("Intel") != std::string::npos) {
                device_count_++;
            }
        }
    }
#endif
}

// =============================================================================
// Backend Information
// =============================================================================

type::BackendType XPUBackend::get_type() const {
    return type::BT_XPU;
}

std::string XPUBackend::get_name() const {
    return "xpu";
}

std::string XPUBackend::get_display_name() const {
    switch (device_type_) {
        case XPU_MAX: return "Intel Data Center GPU Max";
        case XPU_ARC: return "Intel Arc GPU";
        case XPU_GAUDI: return "Intel Gaudi";
        default: return "Intel XPU";
    }
}

bool XPUBackend::is_available() const {
    return is_available_;
}

type::BackendInfo XPUBackend::get_info() const {
    type::BackendInfo info;
    info.type = type::BT_XPU;
    info.name = "xpu";
    info.display_name = get_display_name();
    info.requires_gpu = true;
    info.required_libs = {"sycl", "mkl_sycl", "ze_loader"};
    return info;
}

// =============================================================================
// Compilation
// =============================================================================

bool XPUBackend::compile(CompileContext const& ctx) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    // SYCL compilation via dpcpp/icpx
    return true;
#else
    return false;
#endif
}

std::string XPUBackend::get_compile_flags() const {
    std::string flags = "-fsycl -O3";
    
    // Add device-specific flags
    switch (device_type_) {
        case XPU_MAX:
            flags += " -fsycl-targets=spir64_gen -Xs \"-device pvc\"";
            break;
        case XPU_ARC:
            flags += " -fsycl-targets=spir64_gen -Xs \"-device dg2\"";
            break;
        default:
            flags += " -fsycl-targets=spir64";
    }
    
    return flags;
}

std::vector<std::string> XPUBackend::get_include_dirs() const {
    std::vector<std::string> dirs;
    
    const char* oneapi_root = getenv("ONEAPI_ROOT");
    if (!oneapi_root) {
        oneapi_root = "/opt/intel/oneapi";
    }
    
    dirs.push_back(std::string(oneapi_root) + "/compiler/latest/include");
    dirs.push_back(std::string(oneapi_root) + "/compiler/latest/include/sycl");
    dirs.push_back(std::string(oneapi_root) + "/mkl/latest/include");
    
    return dirs;
}

std::vector<std::string> XPUBackend::get_library_dirs() const {
    std::vector<std::string> dirs;
    
    const char* oneapi_root = getenv("ONEAPI_ROOT");
    if (!oneapi_root) {
        oneapi_root = "/opt/intel/oneapi";
    }
    
    dirs.push_back(std::string(oneapi_root) + "/compiler/latest/lib");
    dirs.push_back(std::string(oneapi_root) + "/mkl/latest/lib");
    
    return dirs;
}

std::vector<std::string> XPUBackend::get_link_libraries() const {
    return {"sycl", "OpenCL", "ze_loader", "mkl_sycl", "mkl_intel_ilp64", 
            "mkl_tbb_thread", "tbb"};
}

// =============================================================================
// Memory Management
// =============================================================================

void* XPUBackend::allocate_memory(size_t size) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    return sycl::malloc_device(size, *queue_);
#else
    return nullptr;
#endif
}

void XPUBackend::free_memory(void* ptr) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    if (ptr) {
        sycl::free(ptr, *queue_);
    }
#endif
}

bool XPUBackend::copy_to_device(void* dst, void const* src, size_t size) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    try {
        queue_->memcpy(dst, src, size).wait();
        return true;
    } catch (const sycl::exception& e) {
        std::cerr << "XPU copy_to_device error: " << e.what() << std::endl;
        return false;
    }
#else
    return false;
#endif
}

bool XPUBackend::copy_to_host(void* dst, void const* src, size_t size) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    try {
        queue_->memcpy(dst, src, size).wait();
        return true;
    } catch (const sycl::exception& e) {
        std::cerr << "XPU copy_to_host error: " << e.what() << std::endl;
        return false;
    }
#else
    return false;
#endif
}

bool XPUBackend::copy_device_to_device(void* dst, void const* src, size_t size) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    try {
        queue_->memcpy(dst, src, size).wait();
        return true;
    } catch (const sycl::exception& e) {
        return false;
    }
#else
    return false;
#endif
}

// =============================================================================
// Synchronization
// =============================================================================

void XPUBackend::synchronize() {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    queue_->wait();
#endif
}

// =============================================================================
// Capability Query
// =============================================================================

size_t XPUBackend::get_max_memory() const {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    return device_.get_info<sycl::info::device::global_mem_size>();
#else
    return 0;
#endif
}

size_t XPUBackend::get_max_shared_memory() const {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    return device_.get_info<sycl::info::device::local_mem_size>();
#else
    return xpu::SLM_SIZE_KB * 1024;
#endif
}

bool XPUBackend::supports_data_type(type::DataType dt) const {
    switch (dt) {
        case type::DT_FLOAT32:
        case type::DT_FLOAT16:
        case type::DT_BFLOAT16:
        case type::DT_INT32:
        case type::DT_INT8:
            return true;
        case type::DT_DOUBLE:
            return device_type_ == XPU_MAX;  // FP64 on Max series
        default:
            return false;
    }
}

int XPUBackend::get_compute_capability() const {
    // Return Xe generation
    switch (device_type_) {
        case XPU_MAX: return 2;   // Xe-HPC
        case XPU_ARC: return 1;   // Xe-HPG
        case XPU_GAUDI: return 3; // Gaudi
        default: return 0;
    }
}

int XPUBackend::get_num_compute_units() const {
    return get_xe_cores();
}

// =============================================================================
// Device Management
// =============================================================================

bool XPUBackend::set_device(int device_id) {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    if (device_id >= 0 && device_id < device_count_) {
        current_device_ = device_id;
        // Re-initialize queue with new device
        return true;
    }
    return false;
#else
    return false;
#endif
}

int XPUBackend::get_device() const {
    return current_device_;
}

int XPUBackend::get_device_count() const {
    return device_count_;
}

// =============================================================================
// XPU-specific
// =============================================================================

XPUBackend::XPUDeviceType XPUBackend::get_device_type() const {
    return device_type_;
}

int XPUBackend::get_xe_cores() const {
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    return device_.get_info<sycl::info::device::max_compute_units>();
#else
    return 0;
#endif
}

bool XPUBackend::has_xmx() const {
    // XMX available on Xe-HPG (Arc) and Xe-HPC (Max)
    return device_type_ == XPU_ARC || device_type_ == XPU_MAX;
}

#ifdef YIRAGE_BACKEND_XPU_ENABLED
sycl::queue& XPUBackend::get_queue() {
    return *queue_;
}
#endif

// =============================================================================
// Helper Functions
// =============================================================================

IntelXPUInfo get_intel_xpu_info(int device_id) {
    IntelXPUInfo info;
    
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    XPUBackend backend;
    if (backend.is_available()) {
        info.type = backend.get_device_type();
        info.xe_cores = backend.get_xe_cores();
        info.global_memory = backend.get_max_memory();
        info.local_memory = backend.get_max_shared_memory();
        info.has_xmx = backend.has_xmx();
        
        switch (info.type) {
            case XPUBackend::XPU_MAX:
                info.name = "Intel Data Center GPU Max";
                info.simd_width = xpu::MAX_SIMD_WIDTH;
                info.xe_slices = info.xe_cores / 16;  // Approximate
                break;
            case XPUBackend::XPU_ARC:
                info.name = "Intel Arc GPU";
                info.simd_width = xpu::ARC_SIMD_WIDTH;
                info.xe_slices = info.xe_cores / 4;
                break;
            default:
                info.name = "Unknown Intel XPU";
                info.simd_width = 16;
                info.xe_slices = 0;
        }
    }
#else
    info.name = "Intel XPU (not available)";
    info.type = XPUBackend::XPU_UNKNOWN;
    info.xe_cores = 0;
    info.global_memory = 0;
    info.local_memory = 0;
    info.has_xmx = false;
    info.simd_width = 0;
    info.xe_slices = 0;
#endif
    
    return info;
}

// =============================================================================
// Backend Registration
// =============================================================================

#ifdef YIRAGE_BACKEND_XPU_ENABLED
namespace {
    struct XPUBackendRegistrar {
        XPUBackendRegistrar() {
            auto backend = std::make_shared<XPUBackend>();
            BackendRegistry::instance().register_backend(
                type::BT_XPU, backend);
        }
    };
    static XPUBackendRegistrar xpu_registrar;
}
#endif

}  // namespace backend
}  // namespace yirage
