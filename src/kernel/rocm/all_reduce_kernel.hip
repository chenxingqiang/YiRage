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
 * ROCm/HIP All-Reduce Kernels
 * 
 * Communication primitives for multi-GPU operations.
 * Uses RCCL (ROCm Collective Communication Library) when available.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Forward declarations for RCCL types (when available)
#ifdef YIRAGE_RCCL_ENABLED
#include <rccl/rccl.h>
#endif

#define WAVEFRONT_SIZE 64

// =============================================================================
// Local All-Reduce (within single GPU)
// =============================================================================

/**
 * @brief Local all-reduce sum across thread blocks
 * 
 * This is a two-pass algorithm:
 * 1. Each block reduces to a single value (stored in workspace)
 * 2. Final block reduces all partial sums
 */
__global__ void all_reduce_sum_local_pass1_hip(
    const float* __restrict__ input,
    float* __restrict__ workspace,
    int size
) {
    __shared__ float shared_sum[4];
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    float local_sum = 0.0f;
    for (int i = idx; i < size; i += stride) {
        local_sum += input[i];
    }
    
    // Wavefront reduction
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor(local_sum, offset);
    }
    
    if (lane_id == 0) {
        shared_sum[wave_id] = local_sum;
    }
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        
        if (lane_id == 0) {
            workspace[blockIdx.x] = local_sum;
        }
    }
}

__global__ void all_reduce_sum_local_pass2_hip(
    float* __restrict__ workspace,
    float* __restrict__ output,
    int num_blocks
) {
    __shared__ float shared_sum[4];
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    float local_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_sum += workspace[i];
    }
    
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor(local_sum, offset);
    }
    
    if (lane_id == 0) {
        shared_sum[wave_id] = local_sum;
    }
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        
        if (lane_id == 0) {
            *output = local_sum;
        }
    }
}

/**
 * @brief Broadcast value to all elements
 */
__global__ void broadcast_hip(
    const float* __restrict__ value,
    float* __restrict__ output,
    int size
) {
    float val = *value;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = val;
    }
}

/**
 * @brief All-reduce with in-place update (add reduced value to each element)
 */
__global__ void all_reduce_add_inplace_hip(
    float* __restrict__ data,
    const float* __restrict__ reduced_value,
    int size
) {
    float val = *reduced_value;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        data[i] += val;
    }
}

/**
 * @brief Ring all-reduce simulation (single GPU)
 * 
 * For multi-GPU, use RCCL directly.
 * This simulates the ring pattern for debugging/testing.
 */
__global__ void ring_all_reduce_sim_hip(
    float* __restrict__ data,
    float* __restrict__ temp,
    int chunk_size,
    int num_chunks,
    int phase  // 0 = scatter-reduce, 1 = all-gather
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (phase == 0) {
        // Scatter-reduce phase
        // Each thread handles one chunk element
        for (int c = 0; c < num_chunks; c++) {
            int idx = c * chunk_size + tid;
            if (tid < chunk_size) {
                temp[idx] = data[idx];  // Simulate receive and reduce
            }
        }
    } else {
        // All-gather phase
        for (int c = 0; c < num_chunks; c++) {
            int idx = c * chunk_size + tid;
            if (tid < chunk_size) {
                data[idx] = temp[idx];  // Simulate scatter
            }
        }
    }
}

// =============================================================================
// Gradient Averaging (for distributed training)
// =============================================================================

/**
 * @brief Average gradients with FP16 gradients, FP32 accumulation
 */
__global__ void gradient_average_f16_hip(
    half* __restrict__ gradients,
    int size,
    int num_workers
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float scale = 1.0f / num_workers;
    
    for (int i = idx; i < size; i += stride) {
        float g = __half2float(gradients[i]);
        gradients[i] = __float2half(g * scale);
    }
}

/**
 * @brief Scale gradients (for gradient clipping)
 */
__global__ void gradient_scale_f16_hip(
    half* __restrict__ gradients,
    int size,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float g = __half2float(gradients[i]);
        gradients[i] = __float2half(g * scale);
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_all_reduce_sum_local_hip(
    const float* input, float* output, float* workspace,
    int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 1024);
    
    // Pass 1: Per-block reduction
    hipLaunchKernelGGL(all_reduce_sum_local_pass1_hip,
                       dim3(num_blocks), dim3(block_size), 0, stream,
                       input, workspace, size);
    
    // Pass 2: Final reduction
    hipLaunchKernelGGL(all_reduce_sum_local_pass2_hip,
                       dim3(1), dim3(256), 0, stream,
                       workspace, output, num_blocks);
}

void launch_broadcast_hip(
    const float* value, float* output,
    int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    
    hipLaunchKernelGGL(broadcast_hip,
                       dim3(num_blocks), dim3(block_size), 0, stream,
                       value, output, size);
}

void launch_gradient_average_f16_hip(
    half* gradients, int size, int num_workers, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    
    hipLaunchKernelGGL(gradient_average_f16_hip,
                       dim3(num_blocks), dim3(block_size), 0, stream,
                       gradients, size, num_workers);
}

void launch_gradient_scale_f16_hip(
    half* gradients, int size, float scale, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    
    hipLaunchKernelGGL(gradient_scale_f16_hip,
                       dim3(num_blocks), dim3(block_size), 0, stream,
                       gradients, size, scale);
}

#ifdef YIRAGE_RCCL_ENABLED

// RCCL-based multi-GPU all-reduce
void launch_all_reduce_rccl_hip(
    float* data, int size,
    ncclComm_t comm, hipStream_t stream
) {
    ncclAllReduce(data, data, size, ncclFloat, ncclSum, comm, stream);
}

void launch_all_reduce_f16_rccl_hip(
    half* data, int size,
    ncclComm_t comm, hipStream_t stream
) {
    ncclAllReduce(data, data, size, ncclHalf, ncclSum, comm, stream);
}

#endif  // YIRAGE_RCCL_ENABLED

}  // extern "C"
