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
 * MACA Warp Utilities
 * 
 * Warp-level primitives adapted for MetaX MACA hardware.
 * Key difference: MACA uses 64-thread warps (vs NVIDIA's 32-thread warps)
 */

#pragma once

#ifdef YIRAGE_BACKEND_MACA_ENABLED

#include <mcr/mc_runtime.h>

namespace yirage {
namespace kernel {
namespace maca {

// MetaX MACA warp size is 64 (not 32 like NVIDIA)
constexpr int MACA_WARP_SIZE = 64;

// Full warp mask for 64 threads (64-bit mask)
constexpr unsigned long long MACA_FULL_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @brief Get lane index within a 64-thread warp
 */
__device__ __forceinline__ int maca_lane_id() {
  return threadIdx.x % MACA_WARP_SIZE;
}

/**
 * @brief Get warp index within the block
 */
__device__ __forceinline__ int maca_warp_id() {
  return threadIdx.x / MACA_WARP_SIZE;
}

/**
 * @brief Warp shuffle for 64-thread warps
 * 
 * MACA provides __shfl_sync compatible with its 64-thread warp model.
 * The mask should be 64-bit for full warp participation.
 */
template <typename T>
__device__ __forceinline__ T maca_shfl_sync(unsigned long long mask, T val, int src_lane) {
  // MACA's __shfl_sync handles 64-thread warps natively
  return __shfl_sync(mask, val, src_lane);
}

/**
 * @brief Warp shuffle XOR for 64-thread warps
 */
template <typename T>
__device__ __forceinline__ T maca_shfl_xor_sync(unsigned long long mask, T val, int lane_mask) {
  return __shfl_xor_sync(mask, val, lane_mask);
}

/**
 * @brief Warp shuffle down for 64-thread warps
 */
template <typename T>
__device__ __forceinline__ T maca_shfl_down_sync(unsigned long long mask, T val, unsigned delta) {
  return __shfl_down_sync(mask, val, delta);
}

/**
 * @brief Warp shuffle up for 64-thread warps
 */
template <typename T>
__device__ __forceinline__ T maca_shfl_up_sync(unsigned long long mask, T val, unsigned delta) {
  return __shfl_up_sync(mask, val, delta);
}

/**
 * @brief Warp-level reduction sum for 64-thread warps
 * 
 * Performs butterfly reduction across all 64 threads in a warp.
 * Requires log2(64) = 6 iterations (vs 5 for 32-thread warps)
 */
template <typename T>
__device__ __forceinline__ T maca_warp_reduce_sum(T val) {
  // 64-thread warp reduction: 6 iterations
  #pragma unroll
  for (int offset = MACA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += maca_shfl_xor_sync(MACA_FULL_WARP_MASK, val, offset);
  }
  return val;
}

/**
 * @brief Warp-level reduction max for 64-thread warps
 */
template <typename T>
__device__ __forceinline__ T maca_warp_reduce_max(T val) {
  #pragma unroll
  for (int offset = MACA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other = maca_shfl_xor_sync(MACA_FULL_WARP_MASK, val, offset);
    val = (val > other) ? val : other;
  }
  return val;
}

/**
 * @brief Warp-level reduction min for 64-thread warps
 */
template <typename T>
__device__ __forceinline__ T maca_warp_reduce_min(T val) {
  #pragma unroll
  for (int offset = MACA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other = maca_shfl_xor_sync(MACA_FULL_WARP_MASK, val, offset);
    val = (val < other) ? val : other;
  }
  return val;
}

/**
 * @brief Block-level reduction sum using 64-thread warps
 * 
 * @tparam BLOCK_SIZE Block size (must be multiple of 64)
 * @param val Value to reduce
 * @param shared_mem Shared memory for inter-warp communication (size: BLOCK_SIZE/64)
 * @return Reduced sum (valid in thread 0)
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
    
    // Reduce across warps (if NUM_WARPS <= 64, this fits in one warp)
    #pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      val += maca_shfl_xor_sync(MACA_FULL_WARP_MASK, val, offset);
    }
  }
  
  return val;
}

/**
 * @brief Warp vote functions for 64-thread warps
 */
__device__ __forceinline__ unsigned long long maca_ballot_sync(unsigned long long mask, int predicate) {
  return __ballot_sync(mask, predicate);
}

__device__ __forceinline__ int maca_any_sync(unsigned long long mask, int predicate) {
  return __any_sync(mask, predicate);
}

__device__ __forceinline__ int maca_all_sync(unsigned long long mask, int predicate) {
  return __all_sync(mask, predicate);
}

/**
 * @brief Population count for 64-bit ballot result
 */
__device__ __forceinline__ int maca_popc(unsigned long long x) {
  return __popcll(x);
}

} // namespace maca
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

