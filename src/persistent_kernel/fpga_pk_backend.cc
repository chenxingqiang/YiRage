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
 * FPGA Persistent Kernel Backend Implementation
 */

#include "persistent_kernel/backends/fpga_pk_backend.h"

#include <iostream>
#include <fstream>
#include <vector>

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace yirage {
namespace pk {

// =============================================================================
// Constructor / Destructor
// =============================================================================

FPGAPKBackend::FPGAPKBackend()
    : is_initialized_(false), device_id_(0), pipeline_depth_(16),
      ii_(1), clock_mhz_(300), cl_context_(nullptr), cl_queue_(nullptr),
      cl_kernel_(nullptr) {}

FPGAPKBackend::~FPGAPKBackend() {
    shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool FPGAPKBackend::initialize(int device_id) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    if (is_initialized_) {
        return true;
    }
    
    // Find FPGA platform and device
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        return false;
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    
    cl_device_id device = nullptr;
    cl_platform_id platform = nullptr;
    
    for (auto plat : platforms) {
        cl_uint num_devices;
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ACCELERATOR, 
                            0, nullptr, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> devices(num_devices);
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_ACCELERATOR,
                          num_devices, devices.data(), nullptr);
            
            if (device_id < static_cast<int>(num_devices)) {
                device = devices[device_id];
                platform = plat;
                break;
            }
        }
    }
    
    if (!device) {
        return false;
    }
    
    // Create context and queue
    cl_context context = clCreateContext(nullptr, 1, &device,
                                        nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    
    cl_command_queue queue = clCreateCommandQueue(
        context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        return false;
    }
    
    cl_context_ = context;
    cl_queue_ = queue;
    device_id_ = device_id;
    is_initialized_ = true;
    
    return true;
#else
    return false;
#endif
}

void FPGAPKBackend::shutdown() {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    if (cl_kernel_) {
        clReleaseKernel(static_cast<cl_kernel>(cl_kernel_));
        cl_kernel_ = nullptr;
    }
    if (cl_queue_) {
        clReleaseCommandQueue(static_cast<cl_command_queue>(cl_queue_));
        cl_queue_ = nullptr;
    }
    if (cl_context_) {
        clReleaseContext(static_cast<cl_context>(cl_context_));
        cl_context_ = nullptr;
    }
    is_initialized_ = false;
#endif
}

bool FPGAPKBackend::is_initialized() const {
    return is_initialized_;
}

// =============================================================================
// Memory Management
// =============================================================================

void* FPGAPKBackend::allocate_ddr(size_t size) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_int err;
    cl_mem buffer = clCreateBuffer(
        static_cast<cl_context>(cl_context_),
        CL_MEM_READ_WRITE,
        size, nullptr, &err);
    return (err == CL_SUCCESS) ? buffer : nullptr;
#else
    return nullptr;
#endif
}

void* FPGAPKBackend::allocate_bram(size_t size) {
    // BRAM allocation is typically done at compile time
    // This is a placeholder for runtime allocation if supported
    return allocate_ddr(size);
}

void FPGAPKBackend::free_memory(void* ptr) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    if (ptr) {
        clReleaseMemObject(static_cast<cl_mem>(ptr));
    }
#endif
}

bool FPGAPKBackend::copy_to_device(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_int err = clEnqueueWriteBuffer(
        static_cast<cl_command_queue>(cl_queue_),
        static_cast<cl_mem>(dst),
        CL_TRUE, 0, size, src,
        0, nullptr, nullptr);
    return err == CL_SUCCESS;
#else
    return false;
#endif
}

bool FPGAPKBackend::copy_to_host(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_int err = clEnqueueReadBuffer(
        static_cast<cl_command_queue>(cl_queue_),
        static_cast<cl_mem>(const_cast<void*>(src)),
        CL_TRUE, 0, size, dst,
        0, nullptr, nullptr);
    return err == CL_SUCCESS;
#else
    return false;
#endif
}

// =============================================================================
// Kernel Execution
// =============================================================================

bool FPGAPKBackend::launch_kernel(const PKKernelConfig& config) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    if (!cl_kernel_) {
        return false;
    }
    
    // Set kernel arguments
    // clSetKernelArg(...)
    
    // Launch kernel
    size_t global_work_size[] = {config.grid_dim_x, config.grid_dim_y, 1};
    size_t local_work_size[] = {config.block_dim_x, config.block_dim_y, 1};
    
    cl_int err = clEnqueueNDRangeKernel(
        static_cast<cl_command_queue>(cl_queue_),
        static_cast<cl_kernel>(cl_kernel_),
        2, nullptr,
        global_work_size,
        local_work_size,
        0, nullptr, nullptr);
    
    return err == CL_SUCCESS;
#else
    return false;
#endif
}

void FPGAPKBackend::synchronize() {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    clFinish(static_cast<cl_command_queue>(cl_queue_));
#endif
}

// =============================================================================
// FPGA-specific
// =============================================================================

bool FPGAPKBackend::load_bitstream(const std::string& path) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    // Read bitstream file
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> binary(size);
    file.read(reinterpret_cast<char*>(binary.data()), size);
    file.close();
    
    // Create program from binary
    // (Simplified - would need device handle)
    
    return true;
#else
    return false;
#endif
}

// =============================================================================
// Factory Registration
// =============================================================================

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
namespace {
    struct FPGAPKBackendRegistrar {
        FPGAPKBackendRegistrar() {
            // PKBackendFactory::register_backend(PKBackendType::FPGA, ...);
        }
    };
    static FPGAPKBackendRegistrar fpga_pk_registrar;
}
#endif

}  // namespace pk
}  // namespace yirage
