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


#include "yirage/backend/nki_backend.h"

#ifdef YIRAGE_BACKEND_NKI_ENABLED

#include "yirage/backend/backend_registry.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace yirage {
namespace backend {

NKIBackend::NKIBackend() : is_available_(false), num_neuron_cores_(0) {
  is_available_ = check_nki_availability();
}

bool NKIBackend::check_nki_availability() {
  // Check if running on AWS Neuron instance
  // This is a simplified check - actual implementation would use Neuron SDK API
  const char *neuron_rt = getenv("NEURON_RT_ROOT_DIR");
  if (neuron_rt) {
    num_neuron_cores_ = 1; // TODO: Query actual number of NeuronCores
    return true;
  }
  return false;
}

type::BackendType NKIBackend::get_type() const {
  return type::BT_NKI;
}

std::string NKIBackend::get_name() const {
  return "nki";
}

std::string NKIBackend::get_display_name() const {
  return "AWS Neuron (NKI)";
}

bool NKIBackend::is_available() const {
  return is_available_;
}

type::BackendInfo NKIBackend::get_info() const {
  type::BackendInfo info;
  info.type = type::BT_NKI;
  info.name = "nki";
  info.display_name = "AWS Neuron (NKI)";
  info.requires_gpu = true; // Neuron is an accelerator
  info.required_libs = {"neuron-cc", "neuron-runtime"};
  return info;
}

bool NKIBackend::compile(CompileContext const &ctx) {
  // NKI compilation is handled by neuron-cc compiler
  std::cerr << "NKI compilation delegated to neuron-cc" << std::endl;
  return true;
}

std::string NKIBackend::get_compile_flags() const {
  return "-O2";
}

std::vector<std::string> NKIBackend::get_include_dirs() const {
  std::vector<std::string> dirs;
  const char *neuron_root = getenv("NEURON_RT_ROOT_DIR");
  if (neuron_root) {
    dirs.push_back(std::string(neuron_root) + "/include");
  }
  return dirs;
}

std::vector<std::string> NKIBackend::get_library_dirs() const {
  std::vector<std::string> dirs;
  const char *neuron_root = getenv("NEURON_RT_ROOT_DIR");
  if (neuron_root) {
    dirs.push_back(std::string(neuron_root) + "/lib");
  }
  return dirs;
}

std::vector<std::string> NKIBackend::get_link_libraries() const {
  return {"neuron-runtime"};
}

void *NKIBackend::allocate_memory(size_t size) {
  // Use standard malloc for now
  // TODO: Use Neuron runtime memory allocation
  return std::malloc(size);
}

void NKIBackend::free_memory(void *ptr) {
  if (ptr) {
    std::free(ptr);
  }
}

bool NKIBackend::copy_to_device(void *dst, void const *src, size_t size) {
  std::memcpy(dst, src, size);
  return true;
}

bool NKIBackend::copy_to_host(void *dst, void const *src, size_t size) {
  std::memcpy(dst, src, size);
  return true;
}

bool NKIBackend::copy_device_to_device(void *dst, void const *src,
                                       size_t size) {
  std::memcpy(dst, src, size);
  return true;
}

void NKIBackend::synchronize() {
  // TODO: Neuron runtime synchronization
}

size_t NKIBackend::get_max_memory() const {
  // Neuron devices typically have 32GB HBM
  return (size_t)32 * 1024 * 1024 * 1024;
}

size_t NKIBackend::get_max_shared_memory() const {
  // SBUF size
  return 24 * 1024 * 1024; // 24 MB
}

bool NKIBackend::supports_data_type(type::DataType dt) const {
  // Neuron optimized for BF16
  switch (dt) {
  case type::DT_BFLOAT16:
  case type::DT_FLOAT32:
  case type::DT_INT8:
  case type::DT_INT32:
    return true;
  default:
    return false;
  }
}

int NKIBackend::get_compute_capability() const {
  // Return Neuron version as "capability"
  return 1; // Version 1.x
}

int NKIBackend::get_num_compute_units() const {
  return num_neuron_cores_;
}

bool NKIBackend::set_device(int device_id) {
  // TODO: Neuron multi-device support
  return device_id == 0;
}

int NKIBackend::get_device() const {
  return 0;
}

int NKIBackend::get_device_count() const {
  return is_available_ ? num_neuron_cores_ : 0;
}

// Register NKI backend
REGISTER_BACKEND(NKIBackend);

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_NKI_ENABLED

