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
 * ROCm/HIP RMSNorm Kernels
 * 
 * RMS Normalization optimized for AMD CDNA architectures.
 * Uses 64-thread wavefront reductions.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVEFRONT_SIZE 64

// =============================================================================
// Wavefront-level Reduction Primitives
// =============================================================================

__device__ __forceinline__ float wavefront_reduce_sum(float val) {
    // 64-thread wavefront: 6 reduction steps
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

// =============================================================================
// RMSNorm Kernels
// =============================================================================

/**
 * @brief RMSNorm for FP32
 * 
 * Each block processes one token.
 * 256 threads = 4 wavefronts.
 */
__global__ void rms_norm_f32_hip(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    __shared__ float shared_sum[4];  // One per wavefront
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const float* in = input + token_idx * hidden_dim;
    float* out = output + token_idx * hidden_dim;
    
    // Phase 1: Compute sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = in[i];
        local_sum += v * v;
    }
    
    // Wavefront reduction
    local_sum = wavefront_reduce_sum(local_sum);
    
    // Store wavefront result
    if (lane_id == 0) {
        shared_sum[wave_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across wavefronts
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        // Reduce 4 values
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        if (lane_id == 0) {
            shared_sum[0] = local_sum;
        }
    }
    __syncthreads();
    
    // Phase 2: Normalize
    float inv_rms = rsqrtf(shared_sum[0] / hidden_dim + eps);
    
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        out[i] = in[i] * inv_rms * weight[i];
    }
}

/**
 * @brief RMSNorm for FP16
 */
__global__ void rms_norm_f16_hip(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    __shared__ float shared_sum[4];
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const half* in = input + token_idx * hidden_dim;
    half* out = output + token_idx * hidden_dim;
    
    // Sum of squares with FP32 accumulation
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = __half2float(in[i]);
        local_sum += v * v;
    }
    
    // Wavefront reduction
    local_sum = wavefront_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        shared_sum[wave_id] = local_sum;
    }
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        if (lane_id == 0) {
            shared_sum[0] = local_sum;
        }
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(shared_sum[0] / hidden_dim + eps);
    
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float v = __half2float(in[i]);
        float w = __half2float(weight[i]);
        out[i] = __float2half(v * inv_rms * w);
    }
}

/**
 * @brief RMSNorm with vectorized loads (FP16)
 * 
 * Uses half2 for 2x throughput on loads/stores.
 */
__global__ void rms_norm_f16_vec2_hip(
    const half2* __restrict__ input,
    const half2* __restrict__ weight,
    half2* __restrict__ output,
    int num_tokens,
    int hidden_dim_half2,
    float eps
) {
    __shared__ float shared_sum[4];
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const half2* in = input + token_idx * hidden_dim_half2;
    half2* out = output + token_idx * hidden_dim_half2;
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim_half2; i += blockDim.x) {
        float2 v = __half22float2(in[i]);
        local_sum += v.x * v.x + v.y * v.y;
    }
    
    local_sum = wavefront_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        shared_sum[wave_id] = local_sum;
    }
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        if (lane_id == 0) {
            shared_sum[0] = local_sum;
        }
    }
    __syncthreads();
    
    int hidden_dim = hidden_dim_half2 * 2;
    float inv_rms = rsqrtf(shared_sum[0] / hidden_dim + eps);
    
    for (int i = tid; i < hidden_dim_half2; i += blockDim.x) {
        float2 v = __half22float2(in[i]);
        float2 w = __half22float2(weight[i]);
        v.x = v.x * inv_rms * w.x;
        v.y = v.y * inv_rms * w.y;
        out[i] = __float22half2_rn(v);
    }
}

/**
 * @brief Batched RMSNorm - multiple tokens per block
 */
__global__ void rms_norm_batched_hip(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    constexpr int TOKENS_PER_BLOCK = 4;
    constexpr int THREADS_PER_TOKEN = 64;  // 1 wavefront per token
    
    __shared__ float shared_sum[TOKENS_PER_BLOCK];
    
    int token_base = blockIdx.x * TOKENS_PER_BLOCK;
    int local_token = threadIdx.x / THREADS_PER_TOKEN;
    int local_tid = threadIdx.x % THREADS_PER_TOKEN;
    
    int token_idx = token_base + local_token;
    if (token_idx >= num_tokens) return;
    
    const half* in = input + token_idx * hidden_dim;
    half* out = output + token_idx * hidden_dim;
    
    // Each thread handles hidden_dim / 64 elements
    float local_sum = 0.0f;
    for (int i = local_tid; i < hidden_dim; i += THREADS_PER_TOKEN) {
        float v = __half2float(in[i]);
        local_sum += v * v;
    }
    
    // Wavefront reduce (single wavefront per token)
    local_sum = wavefront_reduce_sum(local_sum);
    
    if (local_tid == 0) {
        shared_sum[local_token] = local_sum;
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(shared_sum[local_token] / hidden_dim + eps);
    
    for (int i = local_tid; i < hidden_dim; i += THREADS_PER_TOKEN) {
        float v = __half2float(in[i]);
        float w = __half2float(weight[i]);
        out[i] = __float2half(v * inv_rms * w);
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_rms_norm_f32_hip(
    const float* input, const float* weight, float* output,
    int num_tokens, int hidden_dim, float eps,
    hipStream_t stream
) {
    dim3 block(256);  // 4 wavefronts
    dim3 grid(num_tokens);
    
    hipLaunchKernelGGL(rms_norm_f32_hip, grid, block, 0, stream,
                       input, weight, output, num_tokens, hidden_dim, eps);
}

void launch_rms_norm_f16_hip(
    const half* input, const half* weight, half* output,
    int num_tokens, int hidden_dim, float eps,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_tokens);
    
    if (hidden_dim % 2 == 0) {
        // Use vectorized version
        hipLaunchKernelGGL(rms_norm_f16_vec2_hip, grid, block, 0, stream,
                           reinterpret_cast<const half2*>(input),
                           reinterpret_cast<const half2*>(weight),
                           reinterpret_cast<half2*>(output),
                           num_tokens, hidden_dim / 2, eps);
    } else {
        hipLaunchKernelGGL(rms_norm_f16_hip, grid, block, 0, stream,
                           input, weight, output, num_tokens, hidden_dim, eps);
    }
}

void launch_rms_norm_batched_hip(
    const half* input, const half* weight, half* output,
    int num_tokens, int hidden_dim, float eps,
    hipStream_t stream
) {
    constexpr int TOKENS_PER_BLOCK = 4;
    
    dim3 block(256);  // 4 wavefronts = 4 tokens
    dim3 grid((num_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK);
    
    hipLaunchKernelGGL(rms_norm_batched_hip, grid, block, 0, stream,
                       input, weight, output, num_tokens, hidden_dim, eps);
}

}  // extern "C"
