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

#pragma once
#include <cstddef>
#include <cstdint>

// ============================================================================
// Parallel Search Configuration
// ============================================================================
// Enable parallel search for any GPU-based fingerprint backend
// Only CPU fingerprint requires single-threaded execution
#if defined(YIRAGE_FINGERPRINT_USE_CUDA) || \
    defined(YIRAGE_FINGERPRINT_USE_MACA) || \
    defined(YIRAGE_FINGERPRINT_USE_ASCEND) || \
    defined(YIRAGE_USE_FORMAL_VERIFIER)
#define YIRAGE_ENABLE_PARALLEL_SEARCH
#endif

// ============================================================================

namespace yirage {
namespace config {

uint16_t const FP_P = 167;
uint16_t const FP_Q = 83;
uint32_t const FP_EXP_BASE = 3;
uint16_t const FP_PQ = 13861;
// FP_P_MUL_Q_MOD_1 is a multiplier of P and is 1 module Q
uint16_t const FP_P_MUL_Q_MOD_1 = 167;
// FP_Q_MUL_P_MOD_1 is a multiplier of Q and is 1 module P
uint16_t const FP_Q_MUL_P_MOD_1 = 13695;
size_t const MAX_NUM_THREADBLOCKS_PER_KERNEL = 4096;
int const MAX_NUM_DEVICES = 16;
constexpr int MAX_TENSOR_DIMS = 4;
int const DEFAULT_TB_REDUCTION_DIMX = 64;
int const MAX_NUM_WARP_GROUPS = 4;
int const NUM_THREADS_PER_WARP = 32;
int const NUM_WARPS_PER_GROUP = 4;
int const NUM_THREADS_PER_GROUP = NUM_WARPS_PER_GROUP * NUM_THREADS_PER_WARP;
constexpr int MAX_TMA_DESC_PER_TENSOR = 3;

// Multi-backend configuration
// Each backend has its own memory limits, defined in backend-specific namespaces

#ifdef YIRAGE_BACKEND_CUDA_ENABLED
namespace cuda {
size_t const MAX_DMEM_SIZE = (size_t)2 * 1024 * 1024 * 1024;    // 2 GB
size_t const MAX_SMEM_SIZE = 96 * 1024;                         // 96 KB
}
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
namespace cpu {
size_t const MAX_DMEM_SIZE = (size_t)64 * 1024 * 1024 * 1024;   // 64 GB
size_t const MAX_SMEM_SIZE = (size_t)32 * 1024 * 1024;          // 32 MB (L3 cache)
}
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
namespace mps {
// NOTE: This is a compile-time upper bound. Actual usable memory varies by Mac model:
// M1/M2/M3: 8GB, 16GB, 24GB | Pro: 16GB, 18GB, 32GB, 36GB | Max: 32GB-128GB | Ultra: 64GB-512GB
// Runtime should query system memory (see python/yirage/mps_config.py::get_mps_memory_config)
size_t const MAX_DMEM_SIZE = (size_t)64 * 1024 * 1024 * 1024;   // 64 GB (upper limit for most Macs)
size_t const MAX_SMEM_SIZE = 32 * 1024;                         // 32 KB (threadgroup) - All M-series
}
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
namespace nki {
size_t const MAX_DMEM_SIZE = (size_t)32 * 1024 * 1024 * 1024;   // 32 GB
size_t const MAX_SMEM_SIZE = (size_t)24 * 1024 * 1024;          // 24 MB (SBUF)
}
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
namespace ascend {
// Ascend 910: 32GB HBM, Ascend 910B: 64GB HBM2e
size_t const MAX_DMEM_SIZE = (size_t)64 * 1024 * 1024 * 1024;   // 64 GB (910B)
size_t const MAX_SMEM_SIZE = 512 * 1024;                        // 512 KB (AI Core L1)
}
#endif

// Default limits (fallback if no backend specified)
#if !defined(YIRAGE_BACKEND_CUDA_ENABLED) && !defined(YIRAGE_BACKEND_CPU_ENABLED) && \
    !defined(YIRAGE_BACKEND_MPS_ENABLED) && !defined(YIRAGE_BACKEND_NKI_ENABLED)
#warning "No backend enabled, using default limits"
size_t const MAX_DMEM_SIZE = (size_t)16 * 1024 * 1024 * 1024;   // 16 GB
size_t const MAX_SMEM_SIZE = (size_t)1 * 1024 * 1024;           // 1 MB
#endif

// Backward compatibility: define global constants based on primary backend
#if defined(YIRAGE_BACKEND_USE_CUDA) || defined(YIRAGE_BACKEND_CUDA_ENABLED)
size_t const MAX_DMEM_SIZE = cuda::MAX_DMEM_SIZE;
size_t const MAX_SMEM_SIZE = cuda::MAX_SMEM_SIZE;
#elif defined(YIRAGE_BACKEND_USE_NKI) || defined(YIRAGE_BACKEND_NKI_ENABLED)
size_t const MAX_DMEM_SIZE = nki::MAX_DMEM_SIZE;
size_t const MAX_SMEM_SIZE = nki::MAX_SMEM_SIZE;
#elif defined(YIRAGE_BACKEND_CPU_ENABLED)
size_t const MAX_DMEM_SIZE = cpu::MAX_DMEM_SIZE;
size_t const MAX_SMEM_SIZE = cpu::MAX_SMEM_SIZE;
#elif defined(YIRAGE_BACKEND_MPS_ENABLED)
size_t const MAX_DMEM_SIZE = mps::MAX_DMEM_SIZE;
size_t const MAX_SMEM_SIZE = mps::MAX_SMEM_SIZE;
#endif

// Note that we actually save stensors' fingerprints on GPU device memory
// so MAX_SMEM_FP_SIZE can be larger than MAX_SMEM_SIZE
#if defined(YIRAGE_FINGERPRINT_USE_CUDA) && defined(YIRAGE_FINGERPRINT_USE_CPU)
#error                                                                         \
    "Both YIRAGE_FINGERPRINT_USE_CUDA and YIRAGE_FINGERPRINT_USE_CPU are defined. Please define only one fingerprint type."
#elif defined(YIRAGE_FINGERPRINT_USE_CUDA)
size_t const MAX_DMEM_FP_SIZE = (size_t)2 * 1024 * 1024 * 1024; // 2 GB
size_t const MAX_SMEM_FP_SIZE = (size_t)1024 * 1024;            // 1 MB
#else
size_t const MAX_DMEM_FP_SIZE = (size_t)64 * 1024 * 1024 * 1024; // 64 GB
size_t const MAX_SMEM_FP_SIZE = (size_t)64 * 1024 * 1024;        // 64 MB
#endif

} // namespace config
} // namespace yirage
