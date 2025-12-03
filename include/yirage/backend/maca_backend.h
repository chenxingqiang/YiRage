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
 */

#pragma once

#include "yirage/backend/backend_interface.h"

#ifdef YIRAGE_BACKEND_MACA_ENABLED

// MACA native runtime API
// Located in $MACA_PATH/include/mcr/
#include <mcr/mc_runtime.h>
#include <mcc/mc_common.h>

namespace yirage {
namespace backend {

/**
 * @brief MetaX MACA GPU Backend
 * 
 * MACA (MetaX Architecture for Compute Acceleration) is MetaX's 
 * CUDA-compatible software stack for their GPU products.
 * 
 * Key features:
 * - CUDA API compatible (can compile standard .cu files)
 * - Supports FP16, FP32, BF16 compute
 * - High-bandwidth memory (HBM) support
 * - Multi-GPU support with MCCL (similar to NCCL)
 * 
 * Environment variables:
 * - MACA_HOME or MACA_PATH: Path to MACA SDK installation
 * 
 * For more info: https://developer.metax-tech.com/
 */
class MACABackend : public BackendInterface {
public:
  MACABackend();
  virtual ~MACABackend() = default;

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

  // MACA-specific methods
  
  /**
   * @brief Get the MACA SDK path
   * @return Path to MACA SDK, or empty string if not found
   */
  std::string get_maca_path() const;
  
  /**
   * @brief Check if MACA SDK is properly installed
   * @return true if SDK is found and functional
   */
  bool is_sdk_available() const;
  
  /**
   * @brief Get MACA driver version
   * @return Driver version string
   */
  std::string get_driver_version() const;
  
  /**
   * @brief Get MACA runtime version
   * @return Runtime version string
   */
  std::string get_runtime_version() const;

private:
  bool check_maca_availability();
  void query_device_properties();
  std::string detect_maca_path();

  bool is_available_;
  int current_device_;
  int device_count_;
  mcDeviceProp device_prop_;
  std::string maca_path_;
};

} // namespace backend
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

