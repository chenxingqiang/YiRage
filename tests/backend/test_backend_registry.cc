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

  // Test 8: Test Ascend and MACA backends specifically
  std::cout << "\n[Test 8] Testing Ascend and MACA backends..." << std::endl;
  
  // Test Ascend backend
  auto *ascend_backend = registry.get_backend("ascend");
  if (ascend_backend) {
    std::cout << "  Ascend backend registered" << std::endl;
    std::cout << "    Available: " << (ascend_backend->is_available() ? "Yes" : "No") << std::endl;
    if (ascend_backend->is_available()) {
      std::cout << "    Device Count: " << ascend_backend->get_device_count() << std::endl;
      std::cout << "    Compute Units (AI Cores): " << ascend_backend->get_num_compute_units() << std::endl;
    }
  } else {
    std::cout << "  Ascend backend not registered (requires CANN)" << std::endl;
  }
  
  // Test MACA backend
  auto *maca_backend = registry.get_backend("maca");
  if (maca_backend) {
    std::cout << "  MACA backend registered" << std::endl;
    std::cout << "    Available: " << (maca_backend->is_available() ? "Yes" : "No") << std::endl;
    if (maca_backend->is_available()) {
      std::cout << "    Device Count: " << maca_backend->get_device_count() << std::endl;
      std::cout << "    Compute Units (SMs): " << maca_backend->get_num_compute_units() << std::endl;
      std::cout << "    Warp Size: 64 (vs NVIDIA's 32)" << std::endl;
    }
  } else {
    std::cout << "  MACA backend not registered (requires MetaX MACA SDK)" << std::endl;
  }

  // Test 9: Backend compilation test (if hardware available)
  std::cout << "\n[Test 9] Testing backend compilation infrastructure..." << std::endl;
  for (auto const &type : available) {
    auto *backend = registry.get_backend(type);
    if (backend && backend->is_available()) {
      std::cout << "  " << backend->get_name() << ": ";
      
      // Get compile flags (non-empty indicates compilation support)
      std::string flags = backend->get_compile_flags();
      if (!flags.empty()) {
        std::cout << "Compile flags available" << std::endl;
      } else {
        std::cout << "No compilation needed (JIT or interpreted)" << std::endl;
      }
      
      // Get include directories
      auto include_dirs = backend->get_include_dirs();
      if (!include_dirs.empty()) {
        std::cout << "    Include dirs: " << include_dirs.size() << " path(s)" << std::endl;
      }
      
      // Get library directories
      auto lib_dirs = backend->get_library_dirs();
      if (!lib_dirs.empty()) {
        std::cout << "    Library dirs: " << lib_dirs.size() << " path(s)" << std::endl;
      }
    }
  }

  // Test 10: Multi-backend priority check
  std::cout << "\n[Test 10] Testing multi-backend priority..." << std::endl;
  std::vector<yirage::type::BackendType> priority_order = {
      yirage::type::BT_CUDA,   // Highest priority
      yirage::type::BT_MACA,   // MetaX GPU
      yirage::type::BT_ASCEND, // Huawei NPU
      yirage::type::BT_MPS,    // Apple Silicon
      yirage::type::BT_CPU     // Fallback
  };
  
  std::cout << "  Backend priority order:" << std::endl;
  int priority = 1;
  for (auto const &type : priority_order) {
    auto *backend = registry.get_backend(type);
    if (backend) {
      std::cout << "    " << priority << ". " << backend->get_display_name();
      if (backend->is_available()) {
        std::cout << " [AVAILABLE]";
      }
      std::cout << std::endl;
    }
    priority++;
  }

  std::cout << "\n================================" << std::endl;
  std::cout << "All tests passed successfully!" << std::endl;
  std::cout << "================================" << std::endl;

  return 0;
}
