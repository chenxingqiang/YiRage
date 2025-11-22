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

#include "yirage/backend/backend_interface.h"

#ifdef YIRAGE_BACKEND_MPS_ENABLED

namespace yirage {
namespace backend {

/**
 * @brief Apple Metal Performance Shaders (MPS) backend
 */
class MPSBackend : public BackendInterface {
public:
  MPSBackend();
  virtual ~MPSBackend() = default;

  // Backend Information
  type::BackendType get_type() const override;
  std::string get_name() const override;
  std::string get_display_name() const override;
  bool is_available() const override;
  type::BackendInfo get_info() const override;

  // Compilation
  bool compile(CompileContext const &ctx) override;
  std::string get_compile_flags() const override;
  std::vector<std::string> get_include_dirs() const override;
  std::vector<std::string> get_library_dirs() const override;
  std::vector<std::string> get_link_libraries() const override;

  // Memory Management
  void *allocate_memory(size_t size) override;
  void free_memory(void *ptr) override;
  bool copy_to_device(void *dst, void const *src, size_t size) override;
  bool copy_to_host(void *dst, void const *src, size_t size) override;
  bool copy_device_to_device(void *dst, void const *src, size_t size) override;

  // Synchronization
  void synchronize() override;

  // Capability Query
  size_t get_max_memory() const override;
  size_t get_max_shared_memory() const override;
  bool supports_data_type(type::DataType dt) const override;
  int get_compute_capability() const override;
  int get_num_compute_units() const override;

  // Device Management
  bool set_device(int device_id) override;
  int get_device() const override;
  int get_device_count() const override;

private:
  bool check_mps_availability();

  bool is_available_;
};

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_MPS_ENABLED

