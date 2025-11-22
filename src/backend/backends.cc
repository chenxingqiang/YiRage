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


#include "yirage/backend/backends.h"
#include <iostream>

namespace yirage {
namespace backend {

// Forward declaration
void register_all_backends();

void initialize_backends() {
  // For static libraries, automatic registration may not work
  // So we manually register backends here
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;
  
  // Manually register all backends
  register_all_backends();
}

std::vector<std::string> get_available_backend_names() {
  auto &registry = BackendRegistry::get_instance();
  auto types = registry.get_available_backends();
  
  std::vector<std::string> names;
  names.reserve(types.size());
  
  for (auto const &type : types) {
    auto *backend = registry.get_backend(type);
    if (backend) {
      names.push_back(backend->get_name());
    }
  }
  
  return names;
}

BackendInterface *get_default_backend() {
  auto &registry = BackendRegistry::get_instance();
  auto default_type = registry.get_default_backend();
  return registry.get_backend(default_type);
}

bool is_backend_available(std::string const &name) {
  auto &registry = BackendRegistry::get_instance();
  auto *backend = registry.get_backend(name);
  return backend != nullptr && backend->is_available();
}

BackendInterface *get_backend_by_name(std::string const &name) {
  auto &registry = BackendRegistry::get_instance();
  return registry.get_backend(name);
}

// Initialize backends at program startup
namespace {
struct BackendInitializer {
  BackendInitializer() {
    initialize_backends();
  }
};

// This will cause initialize_backends() to be called during static
// initialization, after all backends have been registered
static BackendInitializer g_backend_initializer;
} // anonymous namespace

} // namespace backend
} // namespace yirage

