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

/**
 * @file backends.h
 * @brief Unified header for all backend implementations
 * 
 * This header includes all available backend implementations based on
 * compile-time flags. It also provides convenience functions for
 * initializing and querying backends.
 */

#include "yirage/backend/backend_interface.h"
#include "yirage/backend/backend_registry.h"

// Include backend implementations based on compile flags

// GPU Backends
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include "yirage/backend/cuda_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
#include "yirage/backend/mps_backend.h"
#endif

#ifdef YIRAGE_BACKEND_CUDNN_ENABLED
#include "yirage/backend/cudnn_backend.h"
#endif

// CPU Backends
#ifdef YIRAGE_BACKEND_CPU_ENABLED
#include "yirage/backend/cpu_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MKL_ENABLED
#include "yirage/backend/mkl_backend.h"
#endif

// Specialized Backends
#ifdef YIRAGE_BACKEND_TRITON_ENABLED
#include "yirage/backend/triton_backend.h"
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
#include "yirage/backend/nki_backend.h"
#endif

namespace yirage {
namespace backend {

/**
 * @brief Manually register all compiled backends
 * 
 * For static library builds, this must be called before using backends.
 */
void register_all_backends();

/**
 * @brief Initialize all compiled backends
 * 
 * This function should be called before using backend APIs.
 * It will automatically register all backends.
 */
void initialize_backends();

/**
 * @brief Get list of available backend names
 * @return Vector of backend name strings
 */
std::vector<std::string> get_available_backend_names();

/**
 * @brief Get the default backend
 * @return Pointer to the default backend, or nullptr if none available
 */
BackendInterface *get_default_backend();

/**
 * @brief Check if a specific backend is available
 * @param name Backend name (e.g., "cuda", "cpu", "mps")
 * @return true if available, false otherwise
 */
bool is_backend_available(std::string const &name);

/**
 * @brief Get backend by name
 * @param name Backend name
 * @return Pointer to backend, or nullptr if not found
 */
BackendInterface *get_backend_by_name(std::string const &name);

} // namespace backend
} // namespace yirage

