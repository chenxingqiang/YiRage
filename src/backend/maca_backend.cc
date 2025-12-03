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
 * This file is part of YiRage (Yi Revolutionary AGile Engine)
 * 
 * MetaX MACA Backend Implementation
 * 
 * MACA (MetaX Architecture for Compute Acceleration) provides CUDA-compatible
 * APIs for MetaX GPU hardware. The programming model is nearly identical to
 * CUDA, allowing direct compilation of .cu files.
 * 
 * References:
 * - MetaX Developer Community: https://developer.metax-tech.com/
 * - vLLM-metax: https://github.com/MetaX-MACA/vLLM-metax
 * - mcPytorch: https://github.com/MetaX-MACA/mcPytorch
 */

#include "yirage/backend/maca_backend.h"

#ifdef YIRAGE_BACKEND_MACA_ENABLED

#include "yirage/backend/backend_registry.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace yirage {
namespace backend {

MACABackend::MACABackend()
    : is_available_(false), current_device_(0), device_count_(0) {
  maca_path_ = detect_maca_path();
  is_available_ = check_maca_availability();
  if (is_available_) {
    query_device_properties();
  }
}

std::string MACABackend::detect_maca_path() {
  // Check standard MACA environment variables
  const char *maca_home = std::getenv("MACA_HOME");
  if (maca_home && std::strlen(maca_home) > 0) {
    return std::string(maca_home);
  }

  const char *maca_path = std::getenv("MACA_PATH");
  if (maca_path && std::strlen(maca_path) > 0) {
    return std::string(maca_path);
  }

  // Check standard installation paths
  std::vector<std::string> standard_paths = {
      "/opt/maca",
      "/usr/local/maca",
      "/opt/metax/maca",
      "/usr/local/metax/maca"
  };

  for (const auto &path : standard_paths) {
    std::ifstream check(path + "/include/cuda_runtime.h");
    if (check.good()) {
      return path;
    }
  }

  return "";
}

bool MACABackend::check_maca_availability() {
  // First check if MACA SDK path exists
  if (maca_path_.empty()) {
    return false;
  }

  // Use MACA native runtime API (mc* prefix)
  mcError_t err = mcGetDeviceCount(&device_count_);
  if (err != mcSuccess || device_count_ == 0) {
    return false;
  }

  // Try to get properties of device 0
  err = mcGetDeviceProperties(&device_prop_, 0);
  if (err != mcSuccess) {
    return false;
  }

  // Verify this is actually a MetaX device by checking device name
  // MetaX devices typically have names like "MetaX C500" etc.
  std::string device_name(device_prop_.name);
  
  // Accept any device when MACA SDK is present
  // (in MACA environment, all GPUs are MetaX devices)
  return true;
}

void MACABackend::query_device_properties() {
  if (device_count_ > 0) {
    mcGetDevice(&current_device_);
    mcGetDeviceProperties(&device_prop_, current_device_);
  }
}

type::BackendType MACABackend::get_type() const {
  return type::BT_MACA;
}

std::string MACABackend::get_name() const {
  return "maca";
}

std::string MACABackend::get_display_name() const {
  return "MetaX MACA GPU";
}

bool MACABackend::is_available() const {
  return is_available_;
}

type::BackendInfo MACABackend::get_info() const {
  type::BackendInfo info;
  info.type = type::BT_MACA;
  info.name = "maca";
  info.display_name = "MetaX MACA GPU";
  info.requires_gpu = true;
  info.required_libs = {"libmaca_runtime.so", "libmcblas.so", "libmccl.so"};
  return info;
}

bool MACABackend::compile(CompileContext const &ctx) {
  // MACA uses mxcc compiler (CUDA-compatible)
  // Compilation is typically handled by CMake/mxcc
  // This is a placeholder for runtime compilation if needed
  return true;
}

std::string MACABackend::get_compile_flags() const {
  // MACA compiler flags (similar to nvcc)
  return "-std=c++17 -O2 -Xcompiler=-fPIC --gpu-architecture=compute_75";
}

std::vector<std::string> MACABackend::get_include_dirs() const {
  std::vector<std::string> dirs;
  
  if (!maca_path_.empty()) {
    dirs.push_back(maca_path_ + "/include");
    dirs.push_back(maca_path_ + "/include/thrust");
  }
  
  // Standard MACA paths
  dirs.push_back("/opt/maca/include");
  dirs.push_back("/usr/local/maca/include");
  
  return dirs;
}

std::vector<std::string> MACABackend::get_library_dirs() const {
  std::vector<std::string> dirs;
  
  if (!maca_path_.empty()) {
    dirs.push_back(maca_path_ + "/lib");
    dirs.push_back(maca_path_ + "/lib64");
  }
  
  // Standard MACA paths
  dirs.push_back("/opt/maca/lib");
  dirs.push_back("/opt/maca/lib64");
  dirs.push_back("/usr/local/maca/lib");
  dirs.push_back("/usr/local/maca/lib64");
  
  return dirs;
}

std::vector<std::string> MACABackend::get_link_libraries() const {
  // MACA libraries (CUDA-compatible naming)
  return {
      "maca_runtime",   // Runtime library (like cudart)
      "mcblas",         // BLAS library (like cublas)
      "mccl",           // Collective communication (like nccl)
      "mcrand",         // Random number generation (like curand)
      "mcsolver"        // Solver library (like cusolver)
  };
}

void *MACABackend::allocate_memory(size_t size) {
  void *ptr = nullptr;
  mcError_t err = mcMalloc(&ptr, size);
  if (err != mcSuccess) {
    std::cerr << "MACA malloc failed: " << mcGetErrorString(err) 
              << std::endl;
    return nullptr;
  }
  return ptr;
}

void MACABackend::free_memory(void *ptr) {
  if (ptr) {
    mcFree(ptr);
  }
}

bool MACABackend::copy_to_device(void *dst, void const *src, size_t size) {
  mcError_t err = mcMemcpy(dst, src, size, mcMemcpyHostToDevice);
  if (err != mcSuccess) {
    std::cerr << "MACA copy to device failed: " << mcGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

bool MACABackend::copy_to_host(void *dst, void const *src, size_t size) {
  mcError_t err = mcMemcpy(dst, src, size, mcMemcpyDeviceToHost);
  if (err != mcSuccess) {
    std::cerr << "MACA copy to host failed: " << mcGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

bool MACABackend::copy_device_to_device(void *dst, void const *src,
                                         size_t size) {
  mcError_t err = mcMemcpy(dst, src, size, mcMemcpyDeviceToDevice);
  if (err != mcSuccess) {
    std::cerr << "MACA device-to-device copy failed: "
              << mcGetErrorString(err) << std::endl;
    return false;
  }
  return true;
}

void MACABackend::synchronize() {
  mcDeviceSynchronize();
}

size_t MACABackend::get_max_memory() const {
  if (!is_available_) {
    return 0;
  }
  return device_prop_.totalGlobalMem;
}

size_t MACABackend::get_max_shared_memory() const {
  if (!is_available_) {
    return 0;
  }
  return device_prop_.sharedMemPerBlock;
}

bool MACABackend::supports_data_type(type::DataType dt) const {
  // MACA supports most common data types
  switch (dt) {
  case type::DT_FLOAT16:
  case type::DT_BFLOAT16:
  case type::DT_FLOAT32:
  case type::DT_DOUBLE:
  case type::DT_INT8:
  case type::DT_INT16:
  case type::DT_INT32:
  case type::DT_INT64:
  case type::DT_UINT8:
  case type::DT_UINT16:
  case type::DT_UINT32:
  case type::DT_UINT64:
    return true;
  case type::DT_FLOAT8:
  case type::DT_FLOAT4:
  case type::DT_INT4:
  case type::DT_UINT4:
    // Newer data types may require specific MACA hardware
    return device_prop_.major >= 8;
  default:
    return false;
  }
}

int MACABackend::get_compute_capability() const {
  if (!is_available_) {
    return 0;
  }
  return device_prop_.major * 10 + device_prop_.minor;
}

int MACABackend::get_num_compute_units() const {
  if (!is_available_) {
    return 0;
  }
  return device_prop_.multiProcessorCount;
}

bool MACABackend::set_device(int device_id) {
  if (device_id < 0 || device_id >= device_count_) {
    return false;
  }
  mcError_t err = mcSetDevice(device_id);
  if (err != mcSuccess) {
    return false;
  }
  current_device_ = device_id;
  query_device_properties();
  return true;
}

int MACABackend::get_device() const {
  return current_device_;
}

int MACABackend::get_device_count() const {
  return device_count_;
}

std::string MACABackend::get_maca_path() const {
  return maca_path_;
}

bool MACABackend::is_sdk_available() const {
  return !maca_path_.empty();
}

std::string MACABackend::get_driver_version() const {
  if (!is_available_) {
    return "N/A";
  }
  
  int driver_version = 0;
  mcError_t err = mcDriverGetVersion(&driver_version);
  if (err != mcSuccess) {
    return "Unknown";
  }
  
  std::ostringstream oss;
  oss << (driver_version / 1000) << "." << ((driver_version % 1000) / 10);
  return oss.str();
}

std::string MACABackend::get_runtime_version() const {
  if (!is_available_) {
    return "N/A";
  }
  
  int runtime_version = 0;
  mcError_t err = mcRuntimeGetVersion(&runtime_version);
  if (err != mcSuccess) {
    return "Unknown";
  }
  
  std::ostringstream oss;
  oss << (runtime_version / 1000) << "." << ((runtime_version % 1000) / 10);
  return oss.str();
}

// Register MACA backend
REGISTER_BACKEND(MACABackend);

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

