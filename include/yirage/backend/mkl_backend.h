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


#pragma once

#include "yirage/backend/cpu_backend.h"

#ifdef YIRAGE_BACKEND_MKL_ENABLED

namespace yirage {
namespace backend {

/**
 * @brief MKL backend (extends CPU backend with Intel MKL library)
 */
class MKLBackend : public CPUBackend {
public:
  MKLBackend();
  virtual ~MKLBackend() = default;

  // Override backend info
  type::BackendType get_type() const override;
  std::string get_name() const override;
  std::string get_display_name() const override;
  bool is_available() const override;
  type::BackendInfo get_info() const override;

  // MKL-specific compilation
  bool compile(CompileContext const &ctx) override;
  std::string get_compile_flags() const override;
  std::vector<std::string> get_link_libraries() const override;

private:
  bool check_mkl_availability();
  
  bool mkl_available_;
  std::string mkl_version_;
};

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_MKL_ENABLED

