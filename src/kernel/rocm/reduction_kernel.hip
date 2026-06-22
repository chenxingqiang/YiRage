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
 * ROCm/HIP Reduction Kernels
 * 
 * Optimized for 64-thread wavefronts on CDNA architectures.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVEFRONT_SIZE 64

// =============================================================================
// Wavefront Reduction Primitives
// =============================================================================

__device__ __forceinline__ float wave_reduce_sum(float val) {
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

__device__ __forceinline__ float wave_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor(val, 32));
    val = fmaxf(val, __shfl_xor(val, 16));
    val = fmaxf(val, __shfl_xor(val, 8));
    val = fmaxf(val, __shfl_xor(val, 4));
    val = fmaxf(val, __shfl_xor(val, 2));
    val = fmaxf(val, __shfl_xor(val, 1));
    return val;
}

__device__ __forceinline__ float wave_reduce_min(float val) {
    val = fminf(val, __shfl_xor(val, 32));
    val = fminf(val, __shfl_xor(val, 16));
    val = fminf(val, __shfl_xor(val, 8));
    val = fminf(val, __shfl_xor(val, 4));
    val = fminf(val, __shfl_xor(val, 2));
    val = fminf(val, __shfl_xor(val, 1));
    return val;
}

// =============================================================================
// Sum Reduction
// =============================================================================

/**
 * @brief Block-level sum reduction
 * 
 * Uses 256 threads = 4 wavefronts.
 */
__global__ void reduce_sum_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    __shared__ float shared_sum[4];  // One per wavefront
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    // Thread-level accumulation
    float local_sum = 0.0f;
    for (int i = idx; i < size; i += stride) {
        local_sum += input[i];
    }
    
    // Wavefront reduction
    local_sum = wave_reduce_sum(local_sum);
    
    // Store wavefront result
    if (lane_id == 0) {
        shared_sum[wave_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction in first wavefront
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        
        if (lane_id == 0) {
            atomicAdd(output, local_sum);
        }
    }
}

/**
 * @brief Row-wise sum reduction
 */
__global__ void reduce_sum_row_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_rows,
    int row_size
) {
    __shared__ float shared_sum[4];
    
    int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const float* row = input + row_idx * row_size;
    
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_sum += row[i];
    }
    
    local_sum = wave_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        shared_sum[wave_id] = local_sum;
    }
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        
        if (lane_id == 0) {
            output[row_idx] = local_sum;
        }
    }
}

// =============================================================================
// Max Reduction
// =============================================================================

/**
 * @brief Block-level max reduction
 */
__global__ void reduce_max_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    __shared__ float shared_max[4];
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    
    float local_max = -INFINITY;
    for (int i = idx; i < size; i += stride) {
        local_max = fmaxf(local_max, input[i]);
    }
    
    local_max = wave_reduce_max(local_max);
    
    if (lane_id == 0) {
        shared_max[wave_id] = local_max;
    }
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_max = shared_max[lane_id];
        local_max = fmaxf(local_max, __shfl_xor(local_max, 2));
        local_max = fmaxf(local_max, __shfl_xor(local_max, 1));
        
        if (lane_id == 0) {
            // Atomic max for FP32
            int* addr_as_int = reinterpret_cast<int*>(output);
            int old = *addr_as_int;
            int expected;
            do {
                expected = old;
                float old_val = __int_as_float(old);
                float new_val = fmaxf(old_val, local_max);
                old = atomicCAS(addr_as_int, expected, __float_as_int(new_val));
            } while (old != expected);
        }
    }
}

/**
 * @brief Row-wise max reduction
 */
__global__ void reduce_max_row_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_rows,
    int row_size
) {
    __shared__ float shared_max[4];
    
    int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const float* row = input + row_idx * row_size;
    
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    
    local_max = wave_reduce_max(local_max);
    
    if (lane_id == 0) {
        shared_max[wave_id] = local_max;
    }
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_max = shared_max[lane_id];
        local_max = fmaxf(local_max, __shfl_xor(local_max, 2));
        local_max = fmaxf(local_max, __shfl_xor(local_max, 1));
        
        if (lane_id == 0) {
            output[row_idx] = local_max;
        }
    }
}

// =============================================================================
// Softmax
// =============================================================================

/**
 * @brief Row-wise Softmax
 * 
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
 */
__global__ void softmax_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_rows,
    int row_size
) {
    __shared__ float shared_data[4];
    
    int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const float* in_row = input + row_idx * row_size;
    float* out_row = output + row_idx * row_size;
    
    // Phase 1: Find max
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, in_row[i]);
    }
    
    local_max = wave_reduce_max(local_max);
    if (lane_id == 0) shared_data[wave_id] = local_max;
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_max = shared_data[lane_id];
        local_max = fmaxf(local_max, __shfl_xor(local_max, 2));
        local_max = fmaxf(local_max, __shfl_xor(local_max, 1));
        if (lane_id == 0) shared_data[0] = local_max;
    }
    __syncthreads();
    
    float row_max = shared_data[0];
    
    // Phase 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float exp_val = expf(in_row[i] - row_max);
        out_row[i] = exp_val;  // Temporary store
        local_sum += exp_val;
    }
    
    local_sum = wave_reduce_sum(local_sum);
    if (lane_id == 0) shared_data[wave_id] = local_sum;
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_data[lane_id];
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        if (lane_id == 0) shared_data[0] = local_sum;
    }
    __syncthreads();
    
    float inv_sum = 1.0f / shared_data[0];
    
    // Phase 3: Normalize
    for (int i = tid; i < row_size; i += blockDim.x) {
        out_row[i] *= inv_sum;
    }
}

/**
 * @brief Row-wise Softmax FP16
 */
__global__ void softmax_f16_hip(
    const half* __restrict__ input,
    half* __restrict__ output,
    int num_rows,
    int row_size
) {
    __shared__ float shared_data[4];
    
    int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const half* in_row = input + row_idx * row_size;
    half* out_row = output + row_idx * row_size;
    
    // Find max (use FP32 for stability)
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(in_row[i]));
    }
    
    local_max = wave_reduce_max(local_max);
    if (lane_id == 0) shared_data[wave_id] = local_max;
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_max = shared_data[lane_id];
        local_max = fmaxf(local_max, __shfl_xor(local_max, 2));
        local_max = fmaxf(local_max, __shfl_xor(local_max, 1));
        if (lane_id == 0) shared_data[0] = local_max;
    }
    __syncthreads();
    
    float row_max = shared_data[0];
    
    // Compute exp sum
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float exp_val = expf(__half2float(in_row[i]) - row_max);
        out_row[i] = __float2half(exp_val);
        local_sum += exp_val;
    }
    
    local_sum = wave_reduce_sum(local_sum);
    if (lane_id == 0) shared_data[wave_id] = local_sum;
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_data[lane_id];
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        if (lane_id == 0) shared_data[0] = local_sum;
    }
    __syncthreads();
    
    float inv_sum = 1.0f / shared_data[0];
    
    // Normalize
    for (int i = tid; i < row_size; i += blockDim.x) {
        out_row[i] = __float2half(__half2float(out_row[i]) * inv_sum);
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_reduce_sum_f32_hip(
    const float* input, float* output, int size, hipStream_t stream
) {
    // Zero output first
    hipMemsetAsync(output, 0, sizeof(float), stream);
    
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 1024);
    
    hipLaunchKernelGGL(reduce_sum_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, input, output, size);
}

void launch_reduce_sum_row_f32_hip(
    const float* input, float* output,
    int num_rows, int row_size, hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_rows);
    
    hipLaunchKernelGGL(reduce_sum_row_f32_hip, grid, block, 0, stream,
                       input, output, num_rows, row_size);
}

void launch_reduce_max_f32_hip(
    const float* input, float* output, int size, hipStream_t stream
) {
    // Initialize to -infinity
    float neg_inf = -INFINITY;
    hipMemcpyAsync(output, &neg_inf, sizeof(float), hipMemcpyHostToDevice, stream);
    
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 1024);
    
    hipLaunchKernelGGL(reduce_max_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, input, output, size);
}

void launch_softmax_f32_hip(
    const float* input, float* output,
    int num_rows, int row_size, hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_rows);
    
    hipLaunchKernelGGL(softmax_f32_hip, grid, block, 0, stream,
                       input, output, num_rows, row_size);
}

void launch_softmax_f16_hip(
    const half* input, half* output,
    int num_rows, int row_size, hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_rows);
    
    hipLaunchKernelGGL(softmax_f16_hip, grid, block, 0, stream,
                       input, output, num_rows, row_size);
}

}  // extern "C"
