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

#include "yirage/type.h"
#include <memory>
#include <string>
#include <vector>

namespace yirage {
namespace backend {

// Forward declarations
struct CompileContext;

/**
 * @brief Abstract interface for all backend implementations
 * 
 * Each backend (CUDA, CPU, MPS, etc.) must implement this interface
 * to provide hardware-specific functionality.
 */
class BackendInterface {
public:
  virtual ~BackendInterface() = default;

  // ========== Backend Information ==========
  
  /**
   * @brief Get the backend type
   * @return BackendType enum value
   */
  virtual type::BackendType get_type() const = 0;

  /**
   * @brief Get the backend name (e.g., "cuda", "cpu", "mps")
   * @return Backend name string
   */
  virtual std::string get_name() const = 0;

  /**
   * @brief Get the backend display name (e.g., "CUDA", "CPU", "Metal")
   * @return Backend display name
   */
  virtual std::string get_display_name() const = 0;

  /**
   * @brief Check if this backend is available on the current system
   * @return true if available, false otherwise
   */
  virtual bool is_available() const = 0;

  /**
   * @brief Get detailed backend information
   * @return BackendInfo structure
   */
  virtual type::BackendInfo get_info() const = 0;

  // ========== Compilation ==========

  /**
   * @brief Compile kernel code for this backend
   * @param ctx Compilation context with source code and options
   * @return true if compilation succeeded, false otherwise
   */
  virtual bool compile(CompileContext const &ctx) = 0;

  /**
   * @brief Get backend-specific compilation flags
   * @return String containing compilation flags
   */
  virtual std::string get_compile_flags() const = 0;

  /**
   * @brief Get include directories for this backend
   * @return Vector of include directory paths
   */
  virtual std::vector<std::string> get_include_dirs() const = 0;

  /**
   * @brief Get library directories for this backend
   * @return Vector of library directory paths
   */
  virtual std::vector<std::string> get_library_dirs() const = 0;

  /**
   * @brief Get link libraries for this backend
   * @return Vector of library names
   */
  virtual std::vector<std::string> get_link_libraries() const = 0;

  // ========== Memory Management ==========

  /**
   * @brief Allocate device memory
   * @param size Size in bytes
   * @return Pointer to allocated memory, or nullptr on failure
   */
  virtual void *allocate_memory(size_t size) = 0;

  /**
   * @brief Free device memory
   * @param ptr Pointer to memory to free
   */
  virtual void free_memory(void *ptr) = 0;

  /**
   * @brief Copy data from host to device
   * @param dst Device pointer
   * @param src Host pointer
   * @param size Size in bytes
   * @return true if copy succeeded, false otherwise
   */
  virtual bool copy_to_device(void *dst, void const *src, size_t size) = 0;

  /**
   * @brief Copy data from device to host
   * @param dst Host pointer
   * @param src Device pointer
   * @param size Size in bytes
   * @return true if copy succeeded, false otherwise
   */
  virtual bool copy_to_host(void *dst, void const *src, size_t size) = 0;

  /**
   * @brief Copy data from device to device
   * @param dst Device pointer (destination)
   * @param src Device pointer (source)
   * @param size Size in bytes
   * @return true if copy succeeded, false otherwise
   */
  virtual bool copy_device_to_device(void *dst, void const *src, 
                                      size_t size) = 0;

  // ========== Synchronization ==========

  /**
   * @brief Synchronize all operations on this backend
   */
  virtual void synchronize() = 0;

  // ========== Capability Query ==========

  /**
   * @brief Get maximum device memory size
   * @return Maximum memory size in bytes
   */
  virtual size_t get_max_memory() const = 0;

  /**
   * @brief Get maximum shared memory size per block/workgroup
   * @return Maximum shared memory size in bytes
   */
  virtual size_t get_max_shared_memory() const = 0;

  /**
   * @brief Check if backend supports a specific data type
   * @param dt DataType to check
   * @return true if supported, false otherwise
   */
  virtual bool supports_data_type(type::DataType dt) const = 0;

  /**
   * @brief Get compute capability or equivalent metric
   * @return Capability value (meaning depends on backend)
   */
  virtual int get_compute_capability() const = 0;

  /**
   * @brief Get number of compute units (SMs for CUDA, cores for CPU, etc.)
   * @return Number of compute units
   */
  virtual int get_num_compute_units() const = 0;

  // ========== Device Management ==========

  /**
   * @brief Set current device (for multi-device backends)
   * @param device_id Device ID to set
   * @return true if successful, false otherwise
   */
  virtual bool set_device(int device_id) = 0;

  /**
   * @brief Get current device ID
   * @return Current device ID
   */
  virtual int get_device() const = 0;

  /**
   * @brief Get number of available devices
   * @return Number of devices
   */
  virtual int get_device_count() const = 0;
};

/**
 * @brief Compilation context structure
 */
struct CompileContext {
  std::string source_code;
  std::string output_path;
  std::vector<std::string> include_dirs;
  std::vector<std::string> compile_flags;
  bool debug_mode;
  int optimization_level;
  
  CompileContext() 
    : debug_mode(false), optimization_level(2) {}
};

} // namespace backend
} // namespace yirage

