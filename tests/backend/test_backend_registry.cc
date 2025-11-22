/* Copyright 2023-2024 CMU
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
 */

#include "yirage/backend/backends.h"
#include <cassert>
#include <iostream>

using namespace yirage::backend;

int main() {
  std::cout << "Testing YiRage Backend System" << std::endl;
  std::cout << "================================" << std::endl;

  // Test 1: Get singleton instance
  std::cout << "\n[Test 1] Getting BackendRegistry instance..." << std::endl;
  auto &registry = BackendRegistry::get_instance();
  std::cout << "  OK: BackendRegistry instance obtained" << std::endl;

  // Test 2: Get available backends
  std::cout << "\n[Test 2] Querying available backends..." << std::endl;
  auto available = registry.get_available_backends();
  std::cout << "  Found " << available.size() << " available backend(s)"
            << std::endl;

  if (available.empty()) {
    std::cerr << "  ERROR: No backends available!" << std::endl;
    return 1;
  }

  // Test 3: Check each available backend
  std::cout << "\n[Test 3] Checking each available backend..." << std::endl;
  for (auto const &type : available) {
    auto *backend = registry.get_backend(type);
    assert(backend != nullptr);

    std::cout << "\n  Backend: " << backend->get_display_name() << std::endl;
    std::cout << "    Name: " << backend->get_name() << std::endl;
    std::cout << "    Type: " << static_cast<int>(backend->get_type())
              << std::endl;
    std::cout << "    Available: " << (backend->is_available() ? "Yes" : "No")
              << std::endl;

    auto info = backend->get_info();
    std::cout << "    Requires GPU: " << (info.requires_gpu ? "Yes" : "No")
              << std::endl;

    if (backend->is_available()) {
      std::cout << "    Max Memory: "
                << (backend->get_max_memory() / (1024.0 * 1024 * 1024)) << " GB"
                << std::endl;
      std::cout << "    Max Shared Memory: "
                << (backend->get_max_shared_memory() / 1024.0) << " KB"
                << std::endl;
      std::cout << "    Compute Units: " << backend->get_num_compute_units()
                << std::endl;
      std::cout << "    Device Count: " << backend->get_device_count()
                << std::endl;
    }
  }

  // Test 4: Get backend by name
  std::cout << "\n[Test 4] Getting backends by name..." << std::endl;
  std::vector<std::string> backend_names = {"cuda", "cpu", "mps", "nki"};
  for (auto const &name : backend_names) {
    auto *backend = registry.get_backend(name);
    if (backend) {
      std::cout << "  Found: " << name << " -> " << backend->get_display_name()
                << std::endl;
    } else {
      std::cout << "  Not found: " << name << std::endl;
    }
  }

  // Test 5: Get default backend
  std::cout << "\n[Test 5] Getting default backend..." << std::endl;
  auto default_type = registry.get_default_backend();
  auto *default_backend = registry.get_backend(default_type);
  if (default_backend) {
    std::cout << "  Default backend: " << default_backend->get_name()
              << std::endl;
  } else {
    std::cerr << "  ERROR: No default backend!" << std::endl;
    return 1;
  }

  // Test 6: Test backend availability check
  std::cout << "\n[Test 6] Testing backend availability..." << std::endl;
  for (auto const &name : backend_names) {
    bool available = is_backend_available(name);
    std::cout << "  " << name << ": "
              << (available ? "Available" : "Not Available") << std::endl;
  }

  // Test 7: Test data type support
  std::cout << "\n[Test 7] Testing data type support..." << std::endl;
  if (default_backend) {
    std::cout << "  Testing " << default_backend->get_name()
              << " data types:" << std::endl;
    std::cout << "    FP32: "
              << (default_backend->supports_data_type(yirage::type::DT_FLOAT32)
                      ? "Yes"
                      : "No")
              << std::endl;
    std::cout << "    FP16: "
              << (default_backend->supports_data_type(yirage::type::DT_FLOAT16)
                      ? "Yes"
                      : "No")
              << std::endl;
    std::cout << "    BF16: "
              << (default_backend->supports_data_type(yirage::type::DT_BFLOAT16)
                      ? "Yes"
                      : "No")
              << std::endl;
    std::cout << "    INT8: "
              << (default_backend->supports_data_type(yirage::type::DT_INT8)
                      ? "Yes"
                      : "No")
              << std::endl;
  }

  std::cout << "\n================================" << std::endl;
  std::cout << "All tests passed successfully!" << std::endl;
  std::cout << "================================" << std::endl;

  return 0;
}
