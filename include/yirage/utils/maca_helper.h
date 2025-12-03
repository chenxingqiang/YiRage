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
 * 
 * MACA Helper Functions
 * 
 * Helper macros and functions for MetaX MACA backend.
 * Similar to cuda_helper.h but adapted for MACA's mc* API.
 */

#pragma once

#ifdef YIRAGE_BACKEND_MACA_ENABLED

#include <mc_runtime.h>
#include <mc_common.h>
#include <cstdio>
#include <cstdlib>

namespace yirage {
namespace utils {

/**
 * @brief Check MACA API call result and exit on error
 * 
 * Similar to checkCUDA() but for MACA's mc* API
 */
#define checkMACA(expr)                                                        \
  do {                                                                         \
    mcError_t err = (expr);                                                    \
    if (err != mcSuccess) {                                                    \
      fprintf(stderr, "MACA Error: %s at %s:%d\n",                             \
              mcGetErrorString(err), __FILE__, __LINE__);                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * @brief Check MACA API call with custom message
 */
#define checkMACA_MSG(expr, msg)                                               \
  do {                                                                         \
    mcError_t err = (expr);                                                    \
    if (err != mcSuccess) {                                                    \
      fprintf(stderr, "MACA Error [%s]: %s at %s:%d\n",                        \
              (msg), mcGetErrorString(err), __FILE__, __LINE__);               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * @brief Get last MACA error and check
 */
#define checkMACALastError()                                                   \
  do {                                                                         \
    mcError_t err = mcGetLastError();                                          \
    if (err != mcSuccess) {                                                    \
      fprintf(stderr, "MACA Last Error: %s at %s:%d\n",                        \
              mcGetErrorString(err), __FILE__, __LINE__);                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * @brief MACA kernel launch check
 */
#define checkMACAKernelLaunch()                                                \
  do {                                                                         \
    mcError_t err = mcGetLastError();                                          \
    if (err != mcSuccess) {                                                    \
      fprintf(stderr, "MACA Kernel Launch Error: %s at %s:%d\n",               \
              mcGetErrorString(err), __FILE__, __LINE__);                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
    checkMACA(mcDeviceSynchronize());                                          \
  } while (0)

// ============================================================================
// MACA Warp Constants and Utilities
// ============================================================================

// MetaX MACA uses 64-thread warps (not 32 like NVIDIA)
constexpr int MACA_WARP_SIZE = 64;
constexpr unsigned long long MACA_FULL_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @brief Get lane ID within a 64-thread warp
 */
__device__ __forceinline__ int maca_lane_id() {
  return threadIdx.x % MACA_WARP_SIZE;
}

/**
 * @brief Get warp ID within the block
 */
__device__ __forceinline__ int maca_warp_id() {
  return threadIdx.x / MACA_WARP_SIZE;
}

/**
 * @brief Warp-level reduce sum for 64-thread warps
 * 
 * Performs butterfly reduction across all 64 threads.
 * Uses 6 iterations (log2(64) = 6) instead of 5 for NVIDIA.
 */
template <typename T>
__device__ __forceinline__ T maca_warp_reduce_sum(T val) {
  #pragma unroll
  for (int offset = MACA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(MACA_FULL_WARP_MASK, val, offset);
  }
  return val;
}

/**
 * @brief Warp-level reduce max for 64-thread warps
 */
template <typename T>
__device__ __forceinline__ T maca_warp_reduce_max(T val) {
  #pragma unroll
  for (int offset = MACA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other = __shfl_xor_sync(MACA_FULL_WARP_MASK, val, offset);
    val = (val > other) ? val : other;
  }
  return val;
}

/**
 * @brief Block-level reduce sum using 64-thread warps
 */
template <int BLOCK_SIZE, typename T>
__device__ __forceinline__ T maca_block_reduce_sum(T val, T* shared_mem) {
  static_assert(BLOCK_SIZE % MACA_WARP_SIZE == 0, 
                "Block size must be multiple of MACA_WARP_SIZE (64)");
  
  constexpr int NUM_WARPS = BLOCK_SIZE / MACA_WARP_SIZE;
  
  int warp = maca_warp_id();
  int lane = maca_lane_id();
  
  // First reduce within each warp
  val = maca_warp_reduce_sum(val);
  
  // Write warp results to shared memory
  if (lane == 0) {
    shared_mem[warp] = val;
  }
  __syncthreads();
  
  // Final reduction in first warp
  if (warp == 0) {
    val = (lane < NUM_WARPS) ? shared_mem[lane] : T(0);
    
    #pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      val += __shfl_xor_sync(MACA_FULL_WARP_MASK, val, offset);
    }
  }
  
  return val;
}

/**
 * @brief Broadcast value from lane 0 to all threads in warp (64-thread)
 */
template <typename T>
__device__ __forceinline__ T maca_warp_broadcast(T val, int src_lane = 0) {
  return __shfl_sync(MACA_FULL_WARP_MASK, val, src_lane);
}

} // namespace utils
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

