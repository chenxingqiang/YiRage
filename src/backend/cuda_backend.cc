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


#include "yirage/backend/cuda_backend.h"

#ifdef YIRAGE_BACKEND_CUDA_ENABLED

#include "yirage/backend/backend_registry.h"
#include <iostream>

namespace yirage {
namespace backend {

CUDABackend::CUDABackend()
    : is_available_(false), current_device_(0), device_count_(0) {
  is_available_ = check_cuda_availability();
  if (is_available_) {
    query_device_properties();
  }
}

bool CUDABackend::check_cuda_availability() {
  cudaError_t err = cudaGetDeviceCount(&device_count_);
  if (err != cudaSuccess || device_count_ == 0) {
    return false;
  }

  // Try to get properties of device 0
  err = cudaGetDeviceProperties(&device_prop_, 0);
  if (err != cudaSuccess) {
    return false;
  }

  return true;
}

void CUDABackend::query_device_properties() {
  if (device_count_ > 0) {
    cudaGetDevice(&current_device_);
    cudaGetDeviceProperties(&device_prop_, current_device_);
  }
}

type::BackendType CUDABackend::get_type() const {
  return type::BT_CUDA;
}

std::string CUDABackend::get_name() const {
  return "cuda";
}

std::string CUDABackend::get_display_name() const {
  return "CUDA";
}

bool CUDABackend::is_available() const {
  return is_available_;
}

type::BackendInfo CUDABackend::get_info() const {
  type::BackendInfo info;
  info.type = type::BT_CUDA;
  info.name = "cuda";
  info.display_name = "CUDA";
  info.requires_gpu = true;
  info.required_libs = {"cudart", "cuda", "cudadevrt"};
  return info;
}

bool CUDABackend::compile(CompileContext const &ctx) {
  // CUDA compilation is handled by CMake/nvcc
  // This is a placeholder for runtime compilation if needed
  return true;
}

std::string CUDABackend::get_compile_flags() const {
  return "-std=c++17 -O2 -Xcompiler=-fPIC";
}

std::vector<std::string> CUDABackend::get_include_dirs() const {
  std::vector<std::string> dirs;
  // Add CUDA include directory
  const char *cuda_home = getenv("CUDA_HOME");
  if (cuda_home) {
    dirs.push_back(std::string(cuda_home) + "/include");
  } else {
    dirs.push_back("/usr/local/cuda/include");
  }
  return dirs;
}

std::vector<std::string> CUDABackend::get_library_dirs() const {
  std::vector<std::string> dirs;
  const char *cuda_home = getenv("CUDA_HOME");
  if (cuda_home) {
    dirs.push_back(std::string(cuda_home) + "/lib64");
    dirs.push_back(std::string(cuda_home) + "/lib");
  } else {
    dirs.push_back("/usr/local/cuda/lib64");
    dirs.push_back("/usr/local/cuda/lib");
  }
  return dirs;
}

std::vector<std::string> CUDABackend::get_link_libraries() const {
  return {"cudart", "cuda", "cudadevrt", "cublas"};
}

void *CUDABackend::allocate_memory(size_t size) {
  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err)
              << std::endl;
    return nullptr;
  }
  return ptr;
}

void CUDABackend::free_memory(void *ptr) {
  if (ptr) {
    cudaFree(ptr);
  }
}

bool CUDABackend::copy_to_device(void *dst, void const *src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "CUDA copy to device failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

bool CUDABackend::copy_to_host(void *dst, void const *src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "CUDA copy to host failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

bool CUDABackend::copy_device_to_device(void *dst, void const *src,
                                         size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    std::cerr << "CUDA device-to-device copy failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }
  return true;
}

void CUDABackend::synchronize() {
  cudaDeviceSynchronize();
}

size_t CUDABackend::get_max_memory() const {
  if (!is_available_) {
    return 0;
  }
  return device_prop_.totalGlobalMem;
}

size_t CUDABackend::get_max_shared_memory() const {
  if (!is_available_) {
    return 0;
  }
  return device_prop_.sharedMemPerBlock;
}

bool CUDABackend::supports_data_type(type::DataType dt) const {
  // CUDA supports most data types
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
    // Require newer CUDA versions
    return device_prop_.major >= 8;
  default:
    return false;
  }
}

int CUDABackend::get_compute_capability() const {
  if (!is_available_) {
    return 0;
  }
  return device_prop_.major * 10 + device_prop_.minor;
}

int CUDABackend::get_num_compute_units() const {
  if (!is_available_) {
    return 0;
  }
  return device_prop_.multiProcessorCount;
}

bool CUDABackend::set_device(int device_id) {
  if (device_id < 0 || device_id >= device_count_) {
    return false;
  }
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    return false;
  }
  current_device_ = device_id;
  query_device_properties();
  return true;
}

int CUDABackend::get_device() const {
  return current_device_;
}

int CUDABackend::get_device_count() const {
  return device_count_;
}

// Register CUDA backend
REGISTER_BACKEND(CUDABackend);

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDA_ENABLED

