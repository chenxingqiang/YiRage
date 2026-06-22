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
 * Metal RMSNorm Kernels for Apple Silicon
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Parameters
// =============================================================================

struct RMSNormParams {
    uint num_tokens;
    uint hidden_dim;
    float eps;
};

// =============================================================================
// SIMD Reduction Helpers
// =============================================================================

inline float simd_sum(float val, uint simd_lane) {
    // 32-thread SIMD reduction
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// =============================================================================
// RMSNorm FP32
// =============================================================================

kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[8];  // Up to 8 SIMD groups
    
    uint token_idx = tg_id;
    if (token_idx >= params.num_tokens) return;
    
    device const float* in = input + token_idx * params.hidden_dim;
    device float* out = output + token_idx * params.hidden_dim;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (uint i = tid; i < params.hidden_dim; i += 256) {
        float v = in[i];
        local_sum += v * v;
    }
    
    // SIMD reduction
    local_sum = simd_sum(local_sum, simd_lane);
    
    // Store SIMD result
    if (simd_lane == 0) {
        shared_sum[simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction across SIMD groups
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = (simd_lane < 8) ? shared_sum[simd_lane] : 0.0f;
        local_sum = simd_sum(local_sum, simd_lane);
        if (simd_lane == 0) {
            shared_sum[0] = local_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize
    float inv_rms = rsqrt(shared_sum[0] / float(params.hidden_dim) + params.eps);
    
    for (uint i = tid; i < params.hidden_dim; i += 256) {
        out[i] = in[i] * inv_rms * weight[i];
    }
}

// =============================================================================
// RMSNorm FP16
// =============================================================================

kernel void rms_norm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[8];
    
    uint token_idx = tg_id;
    if (token_idx >= params.num_tokens) return;
    
    device const half* in = input + token_idx * params.hidden_dim;
    device half* out = output + token_idx * params.hidden_dim;
    
    // Sum of squares with FP32 accumulation
    float local_sum = 0.0f;
    for (uint i = tid; i < params.hidden_dim; i += 256) {
        float v = float(in[i]);
        local_sum += v * v;
    }
    
    local_sum = simd_sum(local_sum, simd_lane);
    
    if (simd_lane == 0) {
        shared_sum[simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum = simd_sum(local_sum, simd_lane);
        if (simd_lane == 0) {
            shared_sum[0] = local_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms = rsqrt(shared_sum[0] / float(params.hidden_dim) + params.eps);
    
    for (uint i = tid; i < params.hidden_dim; i += 256) {
        float v = float(in[i]);
        float w = float(weight[i]);
        out[i] = half(v * inv_rms * w);
    }
}

// =============================================================================
// Vectorized RMSNorm FP16 (using half4)
// =============================================================================

kernel void rms_norm_f16_vec4(
    device const half4* input [[buffer(0)]],
    device const half4* weight [[buffer(1)]],
    device half4* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[8];
    
    uint token_idx = tg_id;
    if (token_idx >= params.num_tokens) return;
    
    uint hidden_dim_vec4 = params.hidden_dim / 4;
    
    device const half4* in = input + token_idx * hidden_dim_vec4;
    device half4* out = output + token_idx * hidden_dim_vec4;
    
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_dim_vec4; i += 256) {
        float4 v = float4(in[i]);
        local_sum += dot(v, v);
    }
    
    local_sum = simd_sum(local_sum, simd_lane);
    
    if (simd_lane == 0) {
        shared_sum[simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum = simd_sum(local_sum, simd_lane);
        if (simd_lane == 0) {
            shared_sum[0] = local_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms = rsqrt(shared_sum[0] / float(params.hidden_dim) + params.eps);
    
    for (uint i = tid; i < hidden_dim_vec4; i += 256) {
        float4 v = float4(in[i]);
        float4 w = float4(weight[i]);
        out[i] = half4(v * inv_rms * w);
    }
}

// =============================================================================
// Batched RMSNorm (multiple tokens per threadgroup)
// =============================================================================

kernel void rms_norm_batched_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // 4 tokens per threadgroup
    constexpr uint TOKENS_PER_TG = 4;
    constexpr uint THREADS_PER_TOKEN = 64;
    
    threadgroup float shared_sum[TOKENS_PER_TG];
    
    uint local_token = tid.y;
    uint local_tid = tid.x;
    uint token_idx = tg_id * TOKENS_PER_TG + local_token;
    
    if (token_idx >= params.num_tokens) return;
    
    device const half* in = input + token_idx * params.hidden_dim;
    device half* out = output + token_idx * params.hidden_dim;
    
    float local_sum = 0.0f;
    for (uint i = local_tid; i < params.hidden_dim; i += THREADS_PER_TOKEN) {
        float v = float(in[i]);
        local_sum += v * v;
    }
    
    // SIMD reduce within token
    local_sum = simd_sum(local_sum, simd_lane);
    
    if (simd_lane == 0) {
        shared_sum[local_token] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms = rsqrt(shared_sum[local_token] / float(params.hidden_dim) + params.eps);
    
    for (uint i = local_tid; i < params.hidden_dim; i += THREADS_PER_TOKEN) {
        float v = float(in[i]);
        float w = float(weight[i]);
        out[i] = half(v * inv_rms * w);
    }
}
