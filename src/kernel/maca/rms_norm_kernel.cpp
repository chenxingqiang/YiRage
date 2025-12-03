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
 * MACA RMS Normalization Kernel
 * 
 * RMS normalization optimized for MetaX MACA's 64-thread warp architecture.
 */

#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/kernel/rms_norm.h"
#include "yirage/utils/maca_helper.h"
#include "yirage/utils/fingerprint_functions.h"
#include "yirage/utils/hash_utils.h"
#include <cassert>

namespace yirage {
namespace kernel {

using namespace yirage::utils;

#ifdef YIRAGE_FINGERPRINT_USE_MACA

// MACA warp constants
constexpr int MACA_WARP_SIZE = 64;
constexpr unsigned long long MACA_FULL_MASK = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @brief Warp-level reduction for 64-thread MACA warps
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_sum_maca(T val) {
  #pragma unroll
  for (int offset = MACA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(MACA_FULL_MASK, val, offset);
  }
  return val;
}

/**
 * @brief Compute RMS norm fingerprint on MACA
 */
__global__ void compute_rms_norm_fingerprint_maca(FPType *input_ptr,
                                                   FPType *output_ptr,
                                                   FPType *div_p_lookup_table,
                                                   FPType *div_q_lookup_table,
                                                   FPType *sqrt_p_lookup_table,
                                                   FPType *sqrt_q_lookup_table,
                                                   int num_samples,
                                                   int norm_size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_samples) {
    FPType square_sum = 0;
    for (int k = 0; k < norm_size; k++) {
      FPType x = input_ptr[i * norm_size + k];
      x = compute_mul_fingerprint(x, x);
      square_sum = compute_add_fingerprint(square_sum, x);
    }
    
    // Compute rooted mean square
    FPType rms = 0;
    {
      FPType x = square_sum;
      FPType n = norm_size % FP_PQ;
      // Compute z = x / n
      FPType z =
          compute_div_fingerprint(x, n, div_p_lookup_table, div_q_lookup_table);
      // Perform sqrt for root-mean-square
      rms =
          compute_sqrt_fingerprint(z, sqrt_p_lookup_table, sqrt_q_lookup_table);
    }
    
    for (int k = 0; k < norm_size; k++) {
      FPType x = input_ptr[i * norm_size + k];
      // Compute x / rms
      FPType z = compute_div_fingerprint(
          x, rms, div_p_lookup_table, div_q_lookup_table);
      output_ptr[i * norm_size + k] = z;
    }
  }
}

/**
 * @brief Optimized RMS norm kernel using MACA warp primitives
 * 
 * This version uses warp shuffle for computing the sum of squares,
 * taking advantage of MACA's 64-thread warp for efficient reduction.
 */
template <int BLOCK_SIZE>
__global__ void rms_norm_optimized_maca(float *input,
                                         float *output,
                                         float *weight,
                                         int hidden_size,
                                         float eps) {
  __shared__ float shared[BLOCK_SIZE / MACA_WARP_SIZE];
  
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  
  float *row_input = input + bid * hidden_size;
  float *row_output = output + bid * hidden_size;
  
  // Compute sum of squares
  float sum_sq = 0.0f;
  for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
    float val = row_input[i];
    sum_sq += val * val;
  }
  
  // Warp-level reduction (6 iterations for 64-thread warp)
  sum_sq = warp_reduce_sum_maca(sum_sq);
  
  int lane = tid % MACA_WARP_SIZE;
  int warp = tid / MACA_WARP_SIZE;
  
  if (lane == 0) {
    shared[warp] = sum_sq;
  }
  __syncthreads();
  
  // Final reduction in first warp
  if (warp == 0) {
    constexpr int NUM_WARPS = BLOCK_SIZE / MACA_WARP_SIZE;
    sum_sq = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    sum_sq = warp_reduce_sum_maca(sum_sq);
  }
  __syncthreads();
  
  // Broadcast result to all threads
  if (tid == 0) {
    shared[0] = sum_sq;
  }
  __syncthreads();
  sum_sq = shared[0];
  
  // Compute RMS normalization
  float rms_inv = rsqrtf(sum_sq / hidden_size + eps);
  
  // Apply normalization and weight
  for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
    float val = row_input[i];
    row_output[i] = val * rms_inv * weight[i];
  }
}

bool KNRMSNormOp::fingerprint(void) {
  // assert a 1-D GPU mesh
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);
  
  int num_samples = output_tensors[0].num_elements() / normalized_size;
  
  // Use block size that's multiple of MACA warpSize (64)
  int const num_threads_per_blk = 128;  // 2 warps
  int num_blocks =
      (num_samples + num_threads_per_blk - 1) / num_threads_per_blk;
  
  yirage::kernel::DeviceMemoryManager *dmm =
      yirage::kernel::DeviceMemoryManager::get_instance();
  
  // Use GPU dmm->gpu_id for computing fingerprint
  checkMACA(mcSetDevice(dmm->gpu_id));

  for (int gpu_id = 0; gpu_id < kgraph->gpu_dim.x; gpu_id++) {
    yirage::type::FPType *input_fp_ptr =
        reinterpret_cast<yirage::type::FPType *>(dmm->fp_base_ptr[gpu_id] +
                                                 input_tensors[0].fp_offset);
    yirage::type::FPType *output_fp_ptr =
        reinterpret_cast<yirage::type::FPType *>(dmm->fp_base_ptr[gpu_id] +
                                                 output_tensors[0].fp_offset);
    
    compute_rms_norm_fingerprint_maca<<<num_blocks, num_threads_per_blk>>>(
        input_fp_ptr,
        output_fp_ptr,
        dmm->div_p_lookup_table,
        dmm->div_q_lookup_table,
        dmm->sqrt_p_lookup_table,
        dmm->sqrt_q_lookup_table,
        num_samples,
        normalized_size);
    checkMACA(mcDeviceSynchronize());
  }
  return true;
}

#endif // YIRAGE_FINGERPRINT_USE_MACA

} // namespace kernel
} // namespace yirage

