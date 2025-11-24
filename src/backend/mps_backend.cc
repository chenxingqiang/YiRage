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
 * This file is part of YiRage (Yi Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */


#include "yirage/backend/mps_backend.h"

#ifdef YIRAGE_BACKEND_MPS_ENABLED

#include "yirage/backend/backend_registry.h"
#include "yirage/kernel/mps/mps_kernel_config.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace yirage {
namespace backend {

MPSBackend::MPSBackend() : is_available_(false) {
#ifdef __APPLE__
  is_available_ = check_mps_availability();
#else
  is_available_ = false;
#endif
}

bool MPSBackend::check_mps_availability() {
#ifdef __APPLE__
  // TODO: Add actual Metal/MPS availability check
  // For now, assume available on macOS 12.3+
  return true;
#else
  return false;
#endif
}

type::BackendType MPSBackend::get_type() const {
  return type::BT_MPS;
}

std::string MPSBackend::get_name() const {
  return "mps";
}

std::string MPSBackend::get_display_name() const {
  return "Metal Performance Shaders";
}

bool MPSBackend::is_available() const {
  return is_available_;
}

type::BackendInfo MPSBackend::get_info() const {
  type::BackendInfo info;
  info.type = type::BT_MPS;
  info.name = "mps";
  info.display_name = "Metal Performance Shaders";
  info.requires_gpu = true;
  info.required_libs = {"Metal", "MetalPerformanceShaders"};
  return info;
}

bool MPSBackend::compile(CompileContext const &ctx) {
  // MPS compilation would involve Metal shader compilation
  std::cerr << "MPS backend compilation not yet implemented" << std::endl;
  return false;
}

std::string MPSBackend::get_compile_flags() const {
  return "-std=c++17 -O2 -framework Metal -framework MetalPerformanceShaders";
}

std::vector<std::string> MPSBackend::get_include_dirs() const {
  return {};
}

std::vector<std::string> MPSBackend::get_library_dirs() const {
  return {};
}

std::vector<std::string> MPSBackend::get_link_libraries() const {
  return {"Metal", "MetalPerformanceShaders"};
}

void *MPSBackend::allocate_memory(size_t size) {
  // Use system malloc for now (unified memory)
  // TODO: Use Metal buffer allocation via Objective-C++
  void *ptr = std::malloc(size);
  if (!ptr) {
    std::cerr << "MPS memory allocation failed for size " << size << std::endl;
  }
  return ptr;
}

void MPSBackend::free_memory(void *ptr) {
  if (ptr) {
    // TODO: Use Metal buffer deallocation
    std::free(ptr);
  }
}

bool MPSBackend::copy_to_device(void *dst, void const *src, size_t size) {
  // For unified memory architecture, this is just a memcpy
  std::memcpy(dst, src, size);
  return true;
}

bool MPSBackend::copy_to_host(void *dst, void const *src, size_t size) {
  // For unified memory architecture, this is just a memcpy
  std::memcpy(dst, src, size);
  return true;
}

bool MPSBackend::copy_device_to_device(void *dst, void const *src,
                                        size_t size) {
  std::memcpy(dst, src, size);
  return true;
}

void MPSBackend::synchronize() {
  // TODO: Metal command buffer synchronization
  // For now, no-op since we're using synchronous operations
}

size_t MPSBackend::get_max_memory() const {
  if (!is_available_) {
    return 0;
  }

#ifdef __APPLE__
  // Get total physical memory on macOS
  int64_t memsize = 0;
  size_t len = sizeof(memsize);
  if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
    // Apple Silicon uses unified memory
    // Typically allocate up to 75% for GPU workloads
    return static_cast<size_t>(memsize * 0.75);
  }
#endif

  // Default estimate
  return (size_t)16 * 1024 * 1024 * 1024; // 16 GB
}

size_t MPSBackend::get_max_shared_memory() const {
  if (!is_available_) {
    return 0;
  }

  // Threadgroup memory on all Apple Silicon GPUs
  // All M-series (M1/M2/M3/M4/M5, including Pro/Max/Ultra): 32 KB
  // Reference: Apple Metal Feature Set Tables
  return 32 * 1024; // 32 KB (accurate for all M-series)
}

bool MPSBackend::supports_data_type(type::DataType dt) const {
  // MPS supports common data types
  switch (dt) {
  case type::DT_FLOAT16:
  case type::DT_FLOAT32:
  case type::DT_INT8:
  case type::DT_INT16:
  case type::DT_INT32:
  case type::DT_UINT8:
  case type::DT_UINT16:
  case type::DT_UINT32:
    return true;
  case type::DT_BFLOAT16: {
    // BF16 support added in recent Apple GPUs (M2+)
    // Detect GPU family on demand
    int family = kernel::mps::MPSOptimizer::detect_gpu_family();
    return family >= 8;
  }
  default:
    return false;
  }
}

int MPSBackend::get_compute_capability() const {
  // Return Apple GPU family as "compute capability"
  return kernel::mps::MPSOptimizer::detect_gpu_family();
}

int MPSBackend::get_num_compute_units() const {
  if (!is_available_) {
    return 0;
  }

  // Use MPS optimizer to get GPU core count
  return kernel::mps::MPSOptimizer::get_gpu_core_count();
}

bool MPSBackend::set_device(int device_id) {
  // MPS typically has only one device
  return device_id == 0;
}

int MPSBackend::get_device() const {
  return 0;
}

int MPSBackend::get_device_count() const {
  return is_available_ ? 1 : 0;
}

// Register MPS backend
REGISTER_BACKEND(MPSBackend);

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_MPS_ENABLED

