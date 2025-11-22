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


#include "yirage/backend/cpu_backend.h"

#ifdef YIRAGE_BACKEND_CPU_ENABLED

#include "yirage/backend/backend_registry.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#elif __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace yirage {
namespace backend {

CPUBackend::CPUBackend()
    : num_cores_(0), total_memory_(0), cache_size_(0) {
  query_cpu_info();
}

void CPUBackend::query_cpu_info() {
  // Get number of cores
  num_cores_ = std::thread::hardware_concurrency();

#ifdef __linux__
  // Get total memory on Linux
  struct sysinfo si;
  if (sysinfo(&si) == 0) {
    total_memory_ = si.totalram;
  }
  
  // Try to get cache size from sysconf
  long cache = sysconf(_SC_LEVEL3_CACHE_SIZE);
  if (cache > 0) {
    cache_size_ = cache;
  } else {
    // Default to 8MB if can't detect
    cache_size_ = 8 * 1024 * 1024;
  }
#elif __APPLE__
  // Get total memory on macOS
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  size_t length = sizeof(total_memory_);
  sysctl(mib, 2, &total_memory_, &length, NULL, 0);
  
  // Get cache size
  size_t cache_size = 0;
  length = sizeof(cache_size);
  if (sysctlbyname("hw.l3cachesize", &cache_size, &length, NULL, 0) == 0) {
    cache_size_ = cache_size;
  } else if (sysctlbyname("hw.l2cachesize", &cache_size, &length, NULL, 0) == 0) {
    cache_size_ = cache_size;
  } else {
    cache_size_ = 8 * 1024 * 1024;  // Default
  }
#else
  // Default values for unknown platforms
  total_memory_ = (size_t)64 * 1024 * 1024 * 1024;  // 64 GB default
  cache_size_ = 8 * 1024 * 1024;  // 8 MB default
#endif
}

type::BackendType CPUBackend::get_type() const {
  return type::BT_CPU;
}

std::string CPUBackend::get_name() const {
  return "cpu";
}

std::string CPUBackend::get_display_name() const {
  return "CPU";
}

bool CPUBackend::is_available() const {
  return true;  // CPU is always available
}

type::BackendInfo CPUBackend::get_info() const {
  type::BackendInfo info;
  info.type = type::BT_CPU;
  info.name = "cpu";
  info.display_name = "CPU";
  info.requires_gpu = false;
  info.required_libs = {};
  return info;
}

bool CPUBackend::compile(CompileContext const &ctx) {
  // CPU compilation is handled by CMake/g++
  return true;
}

std::string CPUBackend::get_compile_flags() const {
  return "-std=c++17 -O2 -march=native -fopenmp";
}

std::vector<std::string> CPUBackend::get_include_dirs() const {
  return {};
}

std::vector<std::string> CPUBackend::get_library_dirs() const {
  return {};
}

std::vector<std::string> CPUBackend::get_link_libraries() const {
  return {"gomp", "pthread"};
}

void *CPUBackend::allocate_memory(size_t size) {
  void *ptr = std::malloc(size);
  if (!ptr) {
    std::cerr << "CPU malloc failed for size " << size << std::endl;
    return nullptr;
  }
  return ptr;
}

void CPUBackend::free_memory(void *ptr) {
  if (ptr) {
    std::free(ptr);
  }
}

bool CPUBackend::copy_to_device(void *dst, void const *src, size_t size) {
  // For CPU, device and host memory are the same
  std::memcpy(dst, src, size);
  return true;
}

bool CPUBackend::copy_to_host(void *dst, void const *src, size_t size) {
  // For CPU, device and host memory are the same
  std::memcpy(dst, src, size);
  return true;
}

bool CPUBackend::copy_device_to_device(void *dst, void const *src,
                                        size_t size) {
  std::memcpy(dst, src, size);
  return true;
}

void CPUBackend::synchronize() {
  // No-op for CPU - all operations are synchronous by default
}

size_t CPUBackend::get_max_memory() const {
  return total_memory_;
}

size_t CPUBackend::get_max_shared_memory() const {
  return cache_size_;
}

bool CPUBackend::supports_data_type(type::DataType dt) const {
  // CPU supports all standard data types
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
  default:
    return false;
  }
}

int CPUBackend::get_compute_capability() const {
  // Return a synthetic capability value based on CPU features
  return 100;  // Arbitrary value for CPU
}

int CPUBackend::get_num_compute_units() const {
  return num_cores_;
}

bool CPUBackend::set_device(int device_id) {
  // CPU backend only has one "device"
  return device_id == 0;
}

int CPUBackend::get_device() const {
  return 0;
}

int CPUBackend::get_device_count() const {
  return 1;
}

// Register CPU backend
REGISTER_BACKEND(CPUBackend);

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_CPU_ENABLED

