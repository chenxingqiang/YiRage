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
 * MACA Reduction Kernel
 * 
 * Reduction operations optimized for MetaX MACA's 64-thread warp architecture.
 */

#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/kernel/reduction.h"
#include "yirage/utils/maca_helper.h"
#include "yirage/utils/fingerprint_functions.h"
#include "yirage/utils/hash_utils.h"
#include <cassert>

namespace yirage {
namespace kernel {

using namespace yirage::type;
using namespace yirage::config;
using namespace yirage::utils;

#ifdef YIRAGE_FINGERPRINT_USE_MACA

// MACA warp size is 64
constexpr int MACA_WARP_SIZE = 64;
constexpr unsigned long long MACA_FULL_MASK = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @brief Warp-level reduction sum optimized for MACA's 64-thread warps
 * 
 * Uses 6 iterations (log2(64)) instead of 5 for NVIDIA's 32-thread warps.
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
 * @brief Block-level reduction using MACA's 64-thread warps
 */
template <int BLOCK_SIZE, typename T>
__device__ T block_reduce_sum_maca(T val) {
  static_assert(BLOCK_SIZE % MACA_WARP_SIZE == 0,
                "Block size must be multiple of 64 for MACA");
  
  constexpr int NUM_WARPS = BLOCK_SIZE / MACA_WARP_SIZE;
  
  __shared__ T shared[NUM_WARPS];
  
  int lane = threadIdx.x % MACA_WARP_SIZE;
  int warp = threadIdx.x / MACA_WARP_SIZE;
  
  // Warp-level reduction
  val = warp_reduce_sum_maca(val);
  
  // Write warp results to shared memory
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();
  
  // Final reduction in first warp
  if (warp == 0) {
    val = (lane < NUM_WARPS) ? shared[lane] : T(0);
    val = warp_reduce_sum_maca(val);
  }
  
  return val;
}

/**
 * @brief Compute reduction fingerprint on MACA
 */
__global__ void compute_reduction_fingerprint_maca(FPType *input_ptr,
                                                    FPType *output_ptr,
                                                    int num_elements,
                                                    int reduction_factor,
                                                    int input_stride,
                                                    int output_stride) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_elements) {
    FPType result = 0;
    int n = i / output_stride;
    int m = i % output_stride;
    for (int k = 0; k < reduction_factor; k++) {
      FPType input = input_ptr[n * input_stride + m + k * output_stride];
      result = compute_add_fingerprint(result, input);
    }
    output_ptr[i] = result;
  }
}

/**
 * @brief Optimized reduction kernel using MACA warp primitives
 * 
 * This kernel performs a parallel reduction using MACA's 64-thread warp
 * shuffle operations for efficient intra-warp communication.
 */
template <int BLOCK_SIZE>
__global__ void parallel_reduce_sum_maca(float *input,
                                          float *output,
                                          int num_elements) {
  __shared__ float shared[BLOCK_SIZE / MACA_WARP_SIZE];
  
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Load and accumulate multiple elements per thread
  float sum = 0.0f;
  for (int i = gid; i < num_elements; i += gridDim.x * blockDim.x) {
    sum += input[i];
  }
  
  // Warp-level reduction (6 iterations for 64-thread warp)
  sum = warp_reduce_sum_maca(sum);
  
  int lane = tid % MACA_WARP_SIZE;
  int warp = tid / MACA_WARP_SIZE;
  
  // First thread in each warp writes to shared memory
  if (lane == 0) {
    shared[warp] = sum;
  }
  __syncthreads();
  
  // Final reduction in first warp
  if (warp == 0) {
    constexpr int NUM_WARPS = BLOCK_SIZE / MACA_WARP_SIZE;
    sum = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    sum = warp_reduce_sum_maca(sum);
    
    if (lane == 0) {
      atomicAdd(output, sum);
    }
  }
}

bool KNReductionOp::fingerprint(void) {
  // assert a 1-D GPU mesh
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);

  int num_elements = output_tensors[0].num_elements();
  
  // Use block size that's multiple of MACA warpSize (64)
  int const num_threads_per_blk = 1024;  // 16 warps
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  
  int output_stride = 1;
  int input_stride = 1;
  for (int i = reduction_dim_idx; i < output_tensors[0].num_dims; i++) {
    output_stride *= output_tensors[0].dim[i];
    input_stride *= input_tensors[0].dim[i];
  }
  
  int reduction_factor = input_tensors[0].dim[reduction_dim_idx] /
                         output_tensors[0].dim[reduction_dim_idx];
  assert(output_stride * reduction_factor == input_stride);
  
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
    
    compute_reduction_fingerprint_maca<<<num_blocks, num_threads_per_blk>>>(
        input_fp_ptr,
        output_fp_ptr,
        num_elements,
        reduction_factor,
        input_stride,
        output_stride);
    checkMACA(mcDeviceSynchronize());
  }
  return true;
}

#endif // YIRAGE_FINGERPRINT_USE_MACA

} // namespace kernel
} // namespace yirage

