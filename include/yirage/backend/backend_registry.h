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
#include "yirage/type.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace yirage {
namespace backend {

/**
 * @brief Singleton registry for managing backend implementations
 * 
 * This class maintains a registry of all available backends and provides
 * methods to query, retrieve, and manage them.
 */
class BackendRegistry {
public:
  /**
   * @brief Get the singleton instance
   * @return Reference to the BackendRegistry instance
   */
  static BackendRegistry &get_instance();

  // Disable copy and move
  BackendRegistry(BackendRegistry const &) = delete;
  BackendRegistry &operator=(BackendRegistry const &) = delete;
  BackendRegistry(BackendRegistry &&) = delete;
  BackendRegistry &operator=(BackendRegistry &&) = delete;

  /**
   * @brief Register a backend implementation
   * @param backend Unique pointer to backend implementation
   * @return true if registration succeeded, false if backend already exists
   */
  bool register_backend(std::unique_ptr<BackendInterface> backend);

  /**
   * @brief Get a backend by type
   * @param type Backend type
   * @return Pointer to backend, or nullptr if not found
   */
  BackendInterface *get_backend(type::BackendType type);

  /**
   * @brief Get a backend by name
   * @param name Backend name (e.g., "cuda", "cpu")
   * @return Pointer to backend, or nullptr if not found
   */
  BackendInterface *get_backend(std::string const &name);

  /**
   * @brief Get all registered backend types
   * @return Vector of registered backend types
   */
  std::vector<type::BackendType> get_registered_backends() const;

  /**
   * @brief Get all available (registered and functional) backend types
   * @return Vector of available backend types
   */
  std::vector<type::BackendType> get_available_backends() const;

  /**
   * @brief Check if a backend is registered
   * @param type Backend type
   * @return true if registered, false otherwise
   */
  bool is_backend_registered(type::BackendType type) const;

  /**
   * @brief Check if a backend is available (registered and functional)
   * @param type Backend type
   * @return true if available, false otherwise
   */
  bool is_backend_available(type::BackendType type) const;

  /**
   * @brief Get backend information
   * @param type Backend type
   * @return BackendInfo structure, or default-constructed info if not found
   */
  type::BackendInfo get_backend_info(type::BackendType type) const;

  /**
   * @brief Get the default backend type
   * @return Default backend type (CUDA if available, otherwise first available)
   */
  type::BackendType get_default_backend() const;

  /**
   * @brief Set the default backend
   * @param type Backend type to set as default
   * @return true if successful, false if backend not available
   */
  bool set_default_backend(type::BackendType type);

  /**
   * @brief Unregister a backend
   * @param type Backend type to unregister
   * @return true if unregistered, false if not found
   */
  bool unregister_backend(type::BackendType type);

  /**
   * @brief Clear all registered backends
   */
  void clear();

private:
  // Private constructor for singleton
  BackendRegistry();

  // Backend storage
  std::unordered_map<type::BackendType, std::unique_ptr<BackendInterface>>
      backends_;

  // Name to type mapping for quick lookup
  std::unordered_map<std::string, type::BackendType> name_to_type_;

  // Default backend
  type::BackendType default_backend_;

  // Mutex for thread safety
  mutable std::mutex mutex_;
};

/**
 * @brief Helper macro to register a backend at program startup
 * 
 * Usage:
 *   REGISTER_BACKEND(CUDABackend);
 */
#define REGISTER_BACKEND(BackendClass)                                         \
  namespace {                                                                  \
  struct BackendClass##Registrar {                                             \
    BackendClass##Registrar() {                                                \
      yirage::backend::BackendRegistry::get_instance().register_backend(       \
          std::make_unique<BackendClass>());                                   \
    }                                                                          \
  };                                                                           \
  static BackendClass##Registrar g_##BackendClass##_registrar;                \
  }

} // namespace backend
} // namespace yirage

