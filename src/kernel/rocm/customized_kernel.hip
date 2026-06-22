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
 * ROCm/HIP Customized Kernels
 * 
 * Template for user-defined custom kernels.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// =============================================================================
// Fused Operations (examples)
// =============================================================================

/**
 * @brief Fused GELU + Linear
 */
__global__ void fused_gelu_linear_f16_hip(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    
    int batch = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || out_idx >= out_features) return;
    
    const half* in = input + batch * in_features;
    half* out = output + batch * out_features;
    const half* w = weight + out_idx * in_features;
    
    // Linear
    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
        sum += __half2float(in[i]) * __half2float(w[i]);
    }
    
    if (bias != nullptr) {
        sum += __half2float(bias[out_idx]);
    }
    
    // GELU
    float x3 = sum * sum * sum;
    float inner = sqrt_2_over_pi * (sum + coeff * x3);
    float gelu = 0.5f * sum * (1.0f + tanhf(inner));
    
    out[out_idx] = __float2half(gelu);
}

/**
 * @brief Fused QKV projection
 */
__global__ void fused_qkv_proj_f16_hip(
    const half* __restrict__ hidden,
    const half* __restrict__ qkv_weight,
    const half* __restrict__ qkv_bias,
    half* __restrict__ query,
    half* __restrict__ key,
    half* __restrict__ value,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int head_dim
) {
    // Each block handles one (batch, seq) position
    int batch = blockIdx.z;
    int seq = blockIdx.y;
    int head = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || seq >= seq_len || head >= num_heads) return;
    
    int pos_offset = (batch * seq_len + seq) * hidden_dim;
    int head_offset = head * head_dim;
    
    const half* h = hidden + pos_offset;
    
    // Q, K, V weights are concatenated: [3 * num_heads * head_dim, hidden_dim]
    int qkv_dim = num_heads * head_dim;
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int out_idx = head_offset + d;
        
        // Q projection
        float q_sum = 0.0f;
        const half* q_w = qkv_weight + out_idx * hidden_dim;
        for (int i = 0; i < hidden_dim; i++) {
            q_sum += __half2float(h[i]) * __half2float(q_w[i]);
        }
        if (qkv_bias != nullptr) {
            q_sum += __half2float(qkv_bias[out_idx]);
        }
        
        // K projection
        float k_sum = 0.0f;
        const half* k_w = qkv_weight + (qkv_dim + out_idx) * hidden_dim;
        for (int i = 0; i < hidden_dim; i++) {
            k_sum += __half2float(h[i]) * __half2float(k_w[i]);
        }
        if (qkv_bias != nullptr) {
            k_sum += __half2float(qkv_bias[qkv_dim + out_idx]);
        }
        
        // V projection
        float v_sum = 0.0f;
        const half* v_w = qkv_weight + (2 * qkv_dim + out_idx) * hidden_dim;
        for (int i = 0; i < hidden_dim; i++) {
            v_sum += __half2float(h[i]) * __half2float(v_w[i]);
        }
        if (qkv_bias != nullptr) {
            v_sum += __half2float(qkv_bias[2 * qkv_dim + out_idx]);
        }
        
        // Output: [batch, num_heads, seq, head_dim]
        int out_pos = ((batch * num_heads + head) * seq_len + seq) * head_dim + d;
        query[out_pos] = __float2half(q_sum);
        key[out_pos] = __float2half(k_sum);
        value[out_pos] = __float2half(v_sum);
    }
}

/**
 * @brief Fused Add + LayerNorm
 */
__global__ void fused_add_layernorm_f16_hip(
    const half* __restrict__ input,
    const half* __restrict__ residual,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    half* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    __shared__ float shared_mean;
    __shared__ float shared_var;
    __shared__ float shared_sum[4];
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / 64;
    int lane_id = tid % 64;
    
    const half* in = input + token_idx * hidden_dim;
    const half* res = residual + token_idx * hidden_dim;
    half* out = output + token_idx * hidden_dim;
    
    // Phase 1: Add and compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(in[i]) + __half2float(res[i]);
        local_sum += val;
    }
    
    // Reduce
    for (int off = 32; off > 0; off /= 2) {
        local_sum += __shfl_xor(local_sum, off);
    }
    if (lane_id == 0) shared_sum[wave_id] = local_sum;
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        local_sum += __shfl_xor(local_sum, 2);
        local_sum += __shfl_xor(local_sum, 1);
        if (lane_id == 0) shared_mean = local_sum / hidden_dim;
    }
    __syncthreads();
    
    float mean = shared_mean;
    
    // Phase 2: Compute variance
    float local_var = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(in[i]) + __half2float(res[i]);
        float diff = val - mean;
        local_var += diff * diff;
    }
    
    for (int off = 32; off > 0; off /= 2) {
        local_var += __shfl_xor(local_var, off);
    }
    if (lane_id == 0) shared_sum[wave_id] = local_var;
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_var = shared_sum[lane_id];
        local_var += __shfl_xor(local_var, 2);
        local_var += __shfl_xor(local_var, 1);
        if (lane_id == 0) shared_var = local_var / hidden_dim;
    }
    __syncthreads();
    
    float inv_std = rsqrtf(shared_var + eps);
    
    // Phase 3: Normalize and scale
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(in[i]) + __half2float(res[i]);
        float normalized = (val - mean) * inv_std;
        float scaled = normalized * __half2float(gamma[i]) + __half2float(beta[i]);
        out[i] = __float2half(scaled);
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_fused_gelu_linear_f16_hip(
    const half* input, const half* weight, const half* bias, half* output,
    int batch_size, int in_features, int out_features,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid((out_features + 255) / 256, batch_size);
    
    hipLaunchKernelGGL(fused_gelu_linear_f16_hip, grid, block, 0, stream,
                       input, weight, bias, output,
                       batch_size, in_features, out_features);
}

void launch_fused_qkv_proj_f16_hip(
    const half* hidden, const half* qkv_weight, const half* qkv_bias,
    half* query, half* key, half* value,
    int batch_size, int seq_len, int hidden_dim, int num_heads, int head_dim,
    hipStream_t stream
) {
    dim3 block(64);  // 1 wavefront per head
    dim3 grid(num_heads, seq_len, batch_size);
    
    hipLaunchKernelGGL(fused_qkv_proj_f16_hip, grid, block, 0, stream,
                       hidden, qkv_weight, qkv_bias, query, key, value,
                       batch_size, seq_len, hidden_dim, num_heads, head_dim);
}

void launch_fused_add_layernorm_f16_hip(
    const half* input, const half* residual, const half* gamma, const half* beta,
    half* output, int num_tokens, int hidden_dim, float eps,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_tokens);
    
    hipLaunchKernelGGL(fused_add_layernorm_f16_hip, grid, block, 0, stream,
                       input, residual, gamma, beta, output,
                       num_tokens, hidden_dim, eps);
}

}  // extern "C"
