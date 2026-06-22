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
 */

/**
 * @file pk_backends.h
 * @brief Main header for Persistent Kernel multi-backend support
 * 
 * This header provides the unified interface for persistent kernel backends
 * across different hardware platforms:
 * 
 * Backend Implementation Status and Workload:
 * 
 * - CUDA: NVIDIA GPUs (1 week refactor) - OFFLINE, ONLINE, ONEPASS, GRAPH
 * - CPU: x86/ARM processors (1 week new) - EAGER, GRAPH, OFFLINE
 * - Ascend: Huawei Ascend NPUs (2 weeks new) - OFFLINE, ONLINE, GRAPH
 * - MACA: MetaX MACA GPUs (2 weeks new) - OFFLINE, ONLINE, ONEPASS
 * - MPS: Apple Metal (2 weeks new) - EAGER, GRAPH
 * - Triton: OpenAI Triton (planned)
 * - NKI: AWS Neuron (planned)
 * 
 * Usage:
 * @code
 * #include "persistent_kernel/backends/pk_backends.h"
 * 
 * // Get available backends
 * auto backends = yirage::persistent_kernel::get_available_pk_backends();
 * 
 * // Create a backend
 * auto cuda_backend = yirage::persistent_kernel::create_pk_backend(
 *     yirage::persistent_kernel::PKBackendType::CUDA, 0);
 * 
 * // Check capabilities
 * auto caps = cuda_backend->get_capabilities();
 * if (caps.supports_tma) {
 *     // Use TMA-optimized kernels
 * }
 * 
 * // Initialize and use
 * yirage::persistent_kernel::PKRuntimeConfig config;
 * config.mode = yirage::persistent_kernel::PKMode::ONLINE;
 * config.num_workers = 4;
 * 
 * if (cuda_backend->initialize(config)) {
 *     cuda_backend->launch_worker_kernel(config, 4, 256);
 *     cuda_backend->synchronize();
 *     cuda_backend->finalize();
 * }
 * @endcode
 */

#pragma once

// Base interface
#include "persistent_kernel/pk_backend_interface.h"

// Runtime adapters and utilities
#include "persistent_kernel/pk_runtime_adapter.h"
#include "persistent_kernel/pk_utils.h"

// Backend implementations
#include "persistent_kernel/backends/cuda_pk_backend.h"
#include "persistent_kernel/backends/cpu_pk_backend.h"
#include "persistent_kernel/backends/ascend_pk_backend.h"
#include "persistent_kernel/backends/maca_pk_backend.h"
#include "persistent_kernel/backends/mps_pk_backend.h"
#include "persistent_kernel/backends/rocm_pk_backend.h"

// Persistent kernel runtime implementations (mirroring CUDA persistent_kernel.cuh)
#include "persistent_kernel/pk_runtime_core.h"
#include "persistent_kernel/backends/cpu_pk_runtime.h"
#include "persistent_kernel/backends/ascend_pk_runtime.h"
#include "persistent_kernel/backends/maca_pk_runtime.h"
#include "persistent_kernel/backends/mps_pk_runtime.h"
#include "persistent_kernel/backends/rocm_pk_runtime.h"

namespace yirage {
namespace persistent_kernel {

/**
 * @brief Get the best available backend for the current system
 * 
 * Priority order:
 * 1. CUDA (if NVIDIA GPU available)
 * 2. MACA (if MetaX GPU available)
 * 3. Ascend (if Huawei NPU available)
 * 4. MPS (if Apple Silicon available)
 * 5. CPU (always available as fallback)
 * 
 * @return Best available backend type
 */
inline PKBackendType get_best_available_backend() {
    auto available = get_available_pk_backends();
    if (available.empty()) {
        return PKBackendType::CPU;
    }
    
    // Priority order
    for (auto type : {PKBackendType::CUDA,
                      PKBackendType::ROCM,
                      PKBackendType::MACA,
                      PKBackendType::ASCEND,
                      PKBackendType::MPS,
                      PKBackendType::CPU}) {
        for (auto avail : available) {
            if (avail == type) {
                return type;
            }
        }
    }
    
    return available[0];
}

/**
 * @brief Create the best available backend
 * 
 * @param device_id Device ID to use (for GPU backends)
 * @return Unique pointer to backend instance
 */
inline std::unique_ptr<PKBackendInterface> create_best_backend(int device_id = 0) {
    return create_pk_backend(get_best_available_backend(), device_id);
}

/**
 * @brief Backend capability comparison
 * 
 * @param a First backend type
 * @param b Second backend type
 * @return True if a has higher compute capability than b
 */
inline bool backend_has_higher_capability(PKBackendType a, PKBackendType b) {
    // Simple priority ordering
    auto priority = [](PKBackendType t) -> int {
        switch (t) {
            case PKBackendType::CUDA:   return 100;
            case PKBackendType::ROCM:   return 95;
            case PKBackendType::MACA:   return 90;
            case PKBackendType::ASCEND: return 80;
            case PKBackendType::MPS:    return 70;
            case PKBackendType::TRITON: return 60;
            case PKBackendType::NKI:    return 50;
            case PKBackendType::CPU:    return 10;
            default:                    return 0;
        }
    };
    return priority(a) > priority(b);
}

/**
 * @brief Get default mode for a backend type (aligned with workload plan)
 * 
 * Default modes based on backend strengths:
 * - CUDA: ONLINE (persistent kernel loop for LLM serving)
 * - CPU: EAGER (immediate execution, no GPU scheduling)
 * - Ascend: ONLINE (optimized for continuous inference)
 * - MACA: ONLINE (GPU-like persistent execution)
 * - MPS: EAGER (Metal command buffer model)
 * 
 * @param type Backend type
 * @return Default execution mode for that backend
 */
inline PKMode get_default_mode_for_backend(PKBackendType type) {
    switch (type) {
        case PKBackendType::CUDA:
            return PKMode::ONLINE;  // Best for LLM serving
        case PKBackendType::ROCM:
            return PKMode::ONLINE;  // AMD GPU persistent kernels
        case PKBackendType::CPU:
            return PKMode::EAGER;   // Immediate, no persistent loop
        case PKBackendType::ASCEND:
            return PKMode::ONLINE;  // Continuous NPU inference
        case PKBackendType::MACA:
            return PKMode::ONLINE;  // GPU persistent kernels
        case PKBackendType::MPS:
            return PKMode::EAGER;   // Metal command model
        case PKBackendType::TRITON:
            return PKMode::OFFLINE;
        case PKBackendType::NKI:
            return PKMode::OFFLINE;
        default:
            return PKMode::OFFLINE;
    }
}

} // namespace persistent_kernel
} // namespace yirage
