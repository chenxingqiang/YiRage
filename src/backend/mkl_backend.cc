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


#include "yirage/backend/mkl_backend.h"

#ifdef YIRAGE_BACKEND_MKL_ENABLED

#include "yirage/backend/backend_registry.h"
#include <iostream>

// Try to include MKL if available
#ifdef __has_include
#if __has_include(<mkl.h>)
#include <mkl.h>
#define HAS_MKL_HEADER 1
#endif
#endif

namespace yirage {
namespace backend {

MKLBackend::MKLBackend() 
    : CPUBackend(), mkl_available_(false), mkl_version_("") {
  mkl_available_ = check_mkl_availability();
}

bool MKLBackend::check_mkl_availability() {
#ifdef HAS_MKL_HEADER
  MKLVersion version;
  mkl_get_version(&version);
  mkl_version_ = std::string(version.ProductVersion);
  return true;
#else
  return false;
#endif
}

type::BackendType MKLBackend::get_type() const {
  return type::BT_MKL;
}

std::string MKLBackend::get_name() const {
  return "mkl";
}

std::string MKLBackend::get_display_name() const {
  return "Intel MKL";
}

bool MKLBackend::is_available() const {
  return CPUBackend::is_available() && mkl_available_;
}

type::BackendInfo MKLBackend::get_info() const {
  type::BackendInfo info;
  info.type = type::BT_MKL;
  info.name = "mkl";
  info.display_name = "Intel MKL";
  info.requires_gpu = false;
  info.required_libs = {"mkl_rt", "mkl_core", "mkl_intel_thread", "iomp5"};
  return info;
}

bool MKLBackend::compile(CompileContext const &ctx) {
  // MKL uses runtime library, no compilation needed
  return true;
}

std::string MKLBackend::get_compile_flags() const {
  return CPUBackend::get_compile_flags() + " -mkl";
}

std::vector<std::string> MKLBackend::get_link_libraries() const {
  auto libs = CPUBackend::get_link_libraries();
  libs.insert(libs.end(), {"mkl_rt", "mkl_core", "mkl_intel_thread", "iomp5"});
  return libs;
}

// Register MKL backend
REGISTER_BACKEND(MKLBackend);

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_MKL_ENABLED

