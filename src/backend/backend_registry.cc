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


#include "yirage/backend/backend_registry.h"
#include <algorithm>
#include <iostream>

namespace yirage {
namespace backend {

BackendRegistry::BackendRegistry() : default_backend_(type::BT_UNKNOWN) {}

BackendRegistry &BackendRegistry::get_instance() {
  static BackendRegistry instance;
  return instance;
}

bool BackendRegistry::register_backend(
    std::unique_ptr<BackendInterface> backend) {
  if (!backend) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  type::BackendType type = backend->get_type();

  // Check if already registered - silently skip
  if (backends_.find(type) != backends_.end()) {
    return false;  // Already registered, not an error
  }

  // Register name to type mapping
  name_to_type_[backend->get_name()] = type;

  // Store backend
  backends_[type] = std::move(backend);

  // Set default backend if not set
  if (default_backend_ == type::BT_UNKNOWN) {
    default_backend_ = type;
  }

  return true;
}

BackendInterface *BackendRegistry::get_backend(type::BackendType type) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = backends_.find(type);
  if (it != backends_.end()) {
    return it->second.get();
  }
  return nullptr;
}

BackendInterface *BackendRegistry::get_backend(std::string const &name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = name_to_type_.find(name);
  if (it != name_to_type_.end()) {
    return get_backend(it->second);
  }
  return nullptr;
}

std::vector<type::BackendType>
BackendRegistry::get_registered_backends() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<type::BackendType> types;
  types.reserve(backends_.size());
  for (auto const &pair : backends_) {
    types.push_back(pair.first);
  }
  return types;
}

std::vector<type::BackendType>
BackendRegistry::get_available_backends() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<type::BackendType> types;
  for (auto const &pair : backends_) {
    if (pair.second->is_available()) {
      types.push_back(pair.first);
    }
  }
  return types;
}

bool BackendRegistry::is_backend_registered(type::BackendType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return backends_.find(type) != backends_.end();
}

bool BackendRegistry::is_backend_available(type::BackendType type) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = backends_.find(type);
  if (it != backends_.end()) {
    return it->second->is_available();
  }
  return false;
}

type::BackendInfo
BackendRegistry::get_backend_info(type::BackendType type) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = backends_.find(type);
  if (it != backends_.end()) {
    return it->second->get_info();
  }
  return type::BackendInfo(); // Return default-constructed info
}

type::BackendType BackendRegistry::get_default_backend() const {
  std::lock_guard<std::mutex> lock(mutex_);

  // Return cached default if available
  if (default_backend_ != type::BT_UNKNOWN &&
      is_backend_available(default_backend_)) {
    return default_backend_;
  }

  // Prefer CUDA if available
  if (is_backend_available(type::BT_CUDA)) {
    return type::BT_CUDA;
  }

  // Otherwise return first available backend
  auto available = get_available_backends();
  if (!available.empty()) {
    return available[0];
  }

  return type::BT_UNKNOWN;
}

bool BackendRegistry::set_default_backend(type::BackendType type) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!is_backend_available(type)) {
    return false;
  }

  default_backend_ = type;
  return true;
}

bool BackendRegistry::unregister_backend(type::BackendType type) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = backends_.find(type);
  if (it == backends_.end()) {
    return false;
  }

  // Remove name mapping
  std::string name = it->second->get_name();
  name_to_type_.erase(name);

  // Remove backend
  backends_.erase(it);

  // Update default if needed
  if (default_backend_ == type) {
    default_backend_ = get_default_backend();
  }

  return true;
}

void BackendRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  backends_.clear();
  name_to_type_.clear();
  default_backend_ = type::BT_UNKNOWN;
}

} // namespace backend
} // namespace yirage

