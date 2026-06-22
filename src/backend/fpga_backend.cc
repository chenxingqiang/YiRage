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
 * FPGA Backend Implementation
 */

#include "backend/fpga_backend.h"
#include "backend/backend_registry.h"

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <dlfcn.h>

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
// OpenCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace yirage {
namespace backend {

// =============================================================================
// Constructor / Destructor
// =============================================================================

FPGABackend::FPGABackend()
    : is_available_(false), current_device_(0), device_count_(0),
      vendor_(FPGA_UNKNOWN), cl_platform_(nullptr), cl_device_(nullptr),
      cl_context_(nullptr), cl_queue_(nullptr) {
    is_available_ = check_fpga_availability();
    if (is_available_) {
        query_device_properties();
    }
}

FPGABackend::~FPGABackend() {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    if (cl_queue_) {
        clReleaseCommandQueue(static_cast<cl_command_queue>(cl_queue_));
    }
    if (cl_context_) {
        clReleaseContext(static_cast<cl_context>(cl_context_));
    }
#endif
}

// =============================================================================
// Availability Check
// =============================================================================

bool FPGABackend::check_fpga_availability() {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        return false;
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    
    // Look for FPGA platforms (Intel, Xilinx)
    for (auto platform : platforms) {
        char vendor[256];
        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
        
        bool is_fpga = false;
        if (strstr(vendor, "Intel") && strstr(vendor, "FPGA")) {
            vendor_ = FPGA_INTEL;
            is_fpga = true;
        } else if (strstr(vendor, "Xilinx")) {
            vendor_ = FPGA_XILINX;
            is_fpga = true;
        }
        
        if (is_fpga) {
            cl_platform_ = platform;
            
            // Get FPGA devices
            cl_uint num_devices;
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 
                                0, nullptr, &num_devices);
            if (err == CL_SUCCESS && num_devices > 0) {
                device_count_ = num_devices;
                
                cl_device_id device;
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR,
                              1, &device, nullptr);
                cl_device_ = device;
                
                // Create context and queue
                cl_context context = clCreateContext(nullptr, 1, &device, 
                                                    nullptr, nullptr, &err);
                if (err == CL_SUCCESS) {
                    cl_context_ = context;
                    
                    cl_command_queue queue = clCreateCommandQueue(
                        context, device, CL_QUEUE_PROFILING_ENABLE, &err);
                    if (err == CL_SUCCESS) {
                        cl_queue_ = queue;
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
#else
    return false;
#endif
}

void FPGABackend::query_device_properties() {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    if (cl_device_) {
        char name[256];
        clGetDeviceInfo(static_cast<cl_device_id>(cl_device_), 
                       CL_DEVICE_NAME, sizeof(name), name, nullptr);
        fpga_name_ = name;
    }
#endif
}

// =============================================================================
// Backend Information
// =============================================================================

type::BackendType FPGABackend::get_type() const {
    return type::BT_FPGA;
}

std::string FPGABackend::get_name() const {
    return "fpga";
}

std::string FPGABackend::get_display_name() const {
    switch (vendor_) {
        case FPGA_INTEL: return "Intel FPGA";
        case FPGA_XILINX: return "Xilinx FPGA";
        case FPGA_LATTICE: return "Lattice FPGA";
        default: return "FPGA";
    }
}

bool FPGABackend::is_available() const {
    return is_available_;
}

type::BackendInfo FPGABackend::get_info() const {
    type::BackendInfo info;
    info.type = type::BT_FPGA;
    info.name = "fpga";
    info.display_name = get_display_name();
    info.requires_gpu = false;
    
    switch (vendor_) {
        case FPGA_INTEL:
            info.required_libs = {"OpenCL", "alteracl"};
            break;
        case FPGA_XILINX:
            info.required_libs = {"OpenCL", "xrt_core", "xrt_coreutil"};
            break;
        default:
            info.required_libs = {"OpenCL"};
    }
    
    return info;
}

// =============================================================================
// Compilation
// =============================================================================

bool FPGABackend::compile(CompileContext const& ctx) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    // FPGA compilation is typically done offline
    // This would invoke aoc (Intel) or v++ (Xilinx)
    return true;
#else
    return false;
#endif
}

std::string FPGABackend::get_compile_flags() const {
    switch (vendor_) {
        case FPGA_INTEL:
            return "-march=emulator -O2";  // For emulation; real: -board=...
        case FPGA_XILINX:
            return "-t hw -O2";
        default:
            return "";
    }
}

std::vector<std::string> FPGABackend::get_include_dirs() const {
    std::vector<std::string> dirs;
    
    if (vendor_ == FPGA_INTEL) {
        const char* intelfpga = getenv("INTELFPGAOCLSDKROOT");
        if (intelfpga) {
            dirs.push_back(std::string(intelfpga) + "/include");
        }
    } else if (vendor_ == FPGA_XILINX) {
        const char* xilinx = getenv("XILINX_XRT");
        if (xilinx) {
            dirs.push_back(std::string(xilinx) + "/include");
        }
    }
    
    return dirs;
}

std::vector<std::string> FPGABackend::get_library_dirs() const {
    std::vector<std::string> dirs;
    
    if (vendor_ == FPGA_INTEL) {
        const char* intelfpga = getenv("INTELFPGAOCLSDKROOT");
        if (intelfpga) {
            dirs.push_back(std::string(intelfpga) + "/lib64");
        }
    } else if (vendor_ == FPGA_XILINX) {
        const char* xilinx = getenv("XILINX_XRT");
        if (xilinx) {
            dirs.push_back(std::string(xilinx) + "/lib");
        }
    }
    
    return dirs;
}

std::vector<std::string> FPGABackend::get_link_libraries() const {
    if (vendor_ == FPGA_INTEL) {
        return {"OpenCL", "alteracl"};
    } else if (vendor_ == FPGA_XILINX) {
        return {"OpenCL", "xrt_core", "xrt_coreutil"};
    }
    return {"OpenCL"};
}

// =============================================================================
// Memory Management
// =============================================================================

void* FPGABackend::allocate_memory(size_t size) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_int err;
    cl_mem buffer = clCreateBuffer(
        static_cast<cl_context>(cl_context_),
        CL_MEM_READ_WRITE,
        size, nullptr, &err);
    
    if (err != CL_SUCCESS) {
        return nullptr;
    }
    return buffer;
#else
    return nullptr;
#endif
}

void FPGABackend::free_memory(void* ptr) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    if (ptr) {
        clReleaseMemObject(static_cast<cl_mem>(ptr));
    }
#endif
}

bool FPGABackend::copy_to_device(void* dst, void const* src, size_t size) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_int err = clEnqueueWriteBuffer(
        static_cast<cl_command_queue>(cl_queue_),
        static_cast<cl_mem>(dst),
        CL_TRUE,  // blocking
        0, size, src,
        0, nullptr, nullptr);
    return err == CL_SUCCESS;
#else
    return false;
#endif
}

bool FPGABackend::copy_to_host(void* dst, void const* src, size_t size) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_int err = clEnqueueReadBuffer(
        static_cast<cl_command_queue>(cl_queue_),
        static_cast<cl_mem>(const_cast<void*>(src)),
        CL_TRUE,  // blocking
        0, size, dst,
        0, nullptr, nullptr);
    return err == CL_SUCCESS;
#else
    return false;
#endif
}

bool FPGABackend::copy_device_to_device(void* dst, void const* src, size_t size) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_int err = clEnqueueCopyBuffer(
        static_cast<cl_command_queue>(cl_queue_),
        static_cast<cl_mem>(const_cast<void*>(src)),
        static_cast<cl_mem>(dst),
        0, 0, size,
        0, nullptr, nullptr);
    return err == CL_SUCCESS;
#else
    return false;
#endif
}

// =============================================================================
// Synchronization
// =============================================================================

void FPGABackend::synchronize() {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    clFinish(static_cast<cl_command_queue>(cl_queue_));
#endif
}

// =============================================================================
// Capability Query
// =============================================================================

size_t FPGABackend::get_max_memory() const {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_ulong size;
    clGetDeviceInfo(static_cast<cl_device_id>(cl_device_),
                   CL_DEVICE_GLOBAL_MEM_SIZE,
                   sizeof(size), &size, nullptr);
    return size;
#else
    return 0;
#endif
}

size_t FPGABackend::get_max_shared_memory() const {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_ulong size;
    clGetDeviceInfo(static_cast<cl_device_id>(cl_device_),
                   CL_DEVICE_LOCAL_MEM_SIZE,
                   sizeof(size), &size, nullptr);
    return size;
#else
    return 0;
#endif
}

bool FPGABackend::supports_data_type(type::DataType dt) const {
    switch (dt) {
        case type::DT_FLOAT32:
        case type::DT_FLOAT16:
        case type::DT_INT32:
        case type::DT_INT8:
            return true;
        case type::DT_DOUBLE:
            // Some FPGAs support FP64
            return vendor_ == FPGA_INTEL;
        default:
            return false;
    }
}

int FPGABackend::get_compute_capability() const {
    // Return vendor-specific capability
    switch (vendor_) {
        case FPGA_INTEL: return 1;
        case FPGA_XILINX: return 2;
        default: return 0;
    }
}

int FPGABackend::get_num_compute_units() const {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    cl_uint count;
    clGetDeviceInfo(static_cast<cl_device_id>(cl_device_),
                   CL_DEVICE_MAX_COMPUTE_UNITS,
                   sizeof(count), &count, nullptr);
    return count;
#else
    return 0;
#endif
}

// =============================================================================
// Device Management
// =============================================================================

bool FPGABackend::set_device(int device_id) {
    if (device_id >= 0 && device_id < device_count_) {
        current_device_ = device_id;
        // Re-initialize with new device
        return true;
    }
    return false;
}

int FPGABackend::get_device() const {
    return current_device_;
}

int FPGABackend::get_device_count() const {
    return device_count_;
}

// =============================================================================
// FPGA-specific
// =============================================================================

FPGABackend::FPGAVendor FPGABackend::get_vendor() const {
    return vendor_;
}

std::string FPGABackend::get_fpga_name() const {
    return fpga_name_;
}

bool FPGABackend::load_bitstream(const std::string& path) {
#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    // For Xilinx: Load xclbin
    // For Intel: Load aocx
    
    // Read bitstream file
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        return false;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    std::vector<unsigned char> binary(size);
    fread(binary.data(), 1, size, fp);
    fclose(fp);
    
    // Create program from binary
    const unsigned char* binaries[] = {binary.data()};
    size_t lengths[] = {size};
    cl_int binary_status;
    cl_int err;
    
    cl_program program = clCreateProgramWithBinary(
        static_cast<cl_context>(cl_context_),
        1, static_cast<cl_device_id*>(&cl_device_),
        lengths, binaries,
        &binary_status, &err);
    
    if (err != CL_SUCCESS || binary_status != CL_SUCCESS) {
        return false;
    }
    
    err = clBuildProgram(program, 1, 
                        static_cast<cl_device_id*>(&cl_device_),
                        "", nullptr, nullptr);
    
    return err == CL_SUCCESS;
#else
    return false;
#endif
}

int FPGABackend::get_dsp_count() const {
    // This is typically queried from device info or known from device type
    // Placeholder values
    if (fpga_name_.find("Stratix 10") != std::string::npos) {
        return 5760;  // Stratix 10 GX 2800
    } else if (fpga_name_.find("Alveo U250") != std::string::npos) {
        return 12288;  // U250
    }
    return 0;
}

size_t FPGABackend::get_onchip_memory() const {
    // BRAM + URAM
    if (fpga_name_.find("Stratix 10") != std::string::npos) {
        return 239ULL * 1024 * 1024;  // 239 Mb
    } else if (fpga_name_.find("Alveo U250") != std::string::npos) {
        return 360ULL * 1024 * 1024;  // ~360 Mb BRAM+URAM
    }
    return 0;
}

// =============================================================================
// Helper Functions
// =============================================================================

FPGAInfo get_fpga_info(int device_id) {
    FPGAInfo info;
    
    FPGABackend backend;
    if (backend.is_available()) {
        info.name = backend.get_fpga_name();
        info.vendor = backend.get_vendor();
        info.dsp_blocks = backend.get_dsp_count();
        info.bram_bytes = backend.get_onchip_memory();
        
        // Approximate values based on common devices
        info.clock_mhz = 300;  // Typical kernel clock
        info.ddr_bytes = backend.get_max_memory();
    }
    
    return info;
}

// =============================================================================
// Backend Registration
// =============================================================================

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
namespace {
    struct FPGABackendRegistrar {
        FPGABackendRegistrar() {
            auto backend = std::make_shared<FPGABackend>();
            BackendRegistry::instance().register_backend(
                type::BT_FPGA, backend);
        }
    };
    static FPGABackendRegistrar fpga_registrar;
}
#endif

}  // namespace backend
}  // namespace yirage
