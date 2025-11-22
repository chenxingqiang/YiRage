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


#include "yirage/backend/triton_backend.h"

#ifdef YIRAGE_BACKEND_TRITON_ENABLED

#include "yirage/backend/backend_registry.h"
#include <iostream>

#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace yirage {
namespace backend {

TritonBackend::TritonBackend()
    : is_available_(false), compute_capability_(0) {
  is_available_ = check_triton_availability();
}

bool TritonBackend::check_triton_availability() {
  // Triton requires CUDA backend
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err == cudaSuccess && device_count > 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    compute_capability_ = prop.major * 10 + prop.minor;
    return true;
  }
#endif
  return false;
}

type::BackendType TritonBackend::get_type() const {
  return type::BT_TRITON;
}

std::string TritonBackend::get_name() const {
  return "triton";
}

std::string TritonBackend::get_display_name() const {
  return "Triton";
}

bool TritonBackend::is_available() const {
  return is_available_;
}

type::BackendInfo TritonBackend::get_info() const {
  type::BackendInfo info;
  info.type = type::BT_TRITON;
  info.name = "triton";
  info.display_name = "Triton";
  info.requires_gpu = true;
  info.required_libs = {"triton", "cuda", "cudart"};
  return info;
}

bool TritonBackend::compile(CompileContext const &ctx) {
  // Triton compilation handled by Python layer
  return true;
}

std::string TritonBackend::get_compile_flags() const {
  return "-std=c++17 -O2";
}

std::vector<std::string> TritonBackend::get_include_dirs() const {
  return {};
}

std::vector<std::string> TritonBackend::get_library_dirs() const {
  return {};
}

std::vector<std::string> TritonBackend::get_link_libraries() const {
  return {"cudart", "cuda"};
}

// Memory operations delegate to CUDA
void *TritonBackend::allocate_memory(size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  void *ptr = nullptr;
  cudaMalloc(&ptr, size);
  return ptr;
#else
  return nullptr;
#endif
}

void TritonBackend::free_memory(void *ptr) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  if (ptr) {
    cudaFree(ptr);
  }
#endif
}

bool TritonBackend::copy_to_device(void *dst, void const *src, size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess;
#else
  return false;
#endif
}

bool TritonBackend::copy_to_host(void *dst, void const *src, size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess;
#else
  return false;
#endif
}

bool TritonBackend::copy_device_to_device(void *dst, void const *src,
                                          size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) == cudaSuccess;
#else
  return false;
#endif
}

void TritonBackend::synchronize() {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  cudaDeviceSynchronize();
#endif
}

size_t TritonBackend::get_max_memory() const {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  if (!is_available_) return 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.totalGlobalMem;
#else
  return 0;
#endif
}

size_t TritonBackend::get_max_shared_memory() const {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  if (!is_available_) return 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.sharedMemPerBlock;
#else
  return 0;
#endif
}

bool TritonBackend::supports_data_type(type::DataType dt) const {
  // Triton supports most CUDA data types
  switch (dt) {
  case type::DT_FLOAT16:
  case type::DT_BFLOAT16:
  case type::DT_FLOAT32:
  case type::DT_INT8:
  case type::DT_INT32:
    return true;
  default:
    return false;
  }
}

int TritonBackend::get_compute_capability() const {
  return compute_capability_;
}

int TritonBackend::get_num_compute_units() const {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  if (!is_available_) return 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.multiProcessorCount;
#else
  return 0;
#endif
}

bool TritonBackend::set_device(int device_id) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  return cudaSetDevice(device_id) == cudaSuccess;
#else
  return false;
#endif
}

int TritonBackend::get_device() const {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  int device = 0;
  cudaGetDevice(&device);
  return device;
#else
  return 0;
#endif
}

int TritonBackend::get_device_count() const {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  int count = 0;
  cudaGetDeviceCount(&count);
  return count;
#else
  return 0;
#endif
}

// Register Triton backend
REGISTER_BACKEND(TritonBackend);

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_TRITON_ENABLED

