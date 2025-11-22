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


#include "yirage/backend/cudnn_backend.h"

#ifdef YIRAGE_BACKEND_CUDNN_ENABLED

#include "yirage/backend/backend_registry.h"
#include <iostream>

// Try to include cuDNN if available
#ifdef __has_include
#if __has_include(<cudnn.h>)
#include <cudnn.h>
#define HAS_CUDNN_HEADER 1
#endif
#endif

namespace yirage {
namespace backend {

CUDNNBackend::CUDNNBackend() 
    : CUDABackend(), cudnn_available_(false), cudnn_version_(0) {
  cudnn_available_ = check_cudnn_availability();
}

bool CUDNNBackend::check_cudnn_availability() {
#ifdef HAS_CUDNN_HEADER
  cudnn_version_ = static_cast<int>(cudnnGetVersion());
  return cudnn_version_ >= 8000; // Require cuDNN 8.0+
#else
  return false;
#endif
}

type::BackendType CUDNNBackend::get_type() const {
  return type::BT_CUDNN;
}

std::string CUDNNBackend::get_name() const {
  return "cudnn";
}

std::string CUDNNBackend::get_display_name() const {
  return "cuDNN";
}

bool CUDNNBackend::is_available() const {
  return CUDABackend::is_available() && cudnn_available_;
}

type::BackendInfo CUDNNBackend::get_info() const {
  type::BackendInfo info;
  info.type = type::BT_CUDNN;
  info.name = "cudnn";
  info.display_name = "cuDNN";
  info.requires_gpu = true;
  info.required_libs = {"cudnn", "cudart", "cuda", "cublas"};
  return info;
}

bool CUDNNBackend::compile(CompileContext const &ctx) {
  // cuDNN uses runtime API, no kernel compilation needed
  return true;
}

std::vector<std::string> CUDNNBackend::get_link_libraries() const {
  auto libs = CUDABackend::get_link_libraries();
  libs.push_back("cudnn");
  return libs;
}

// Register cuDNN backend
REGISTER_BACKEND(CUDNNBackend);

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDNN_ENABLED

