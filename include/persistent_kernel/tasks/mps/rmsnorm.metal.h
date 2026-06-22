/* Copyright 2025 YiRage Team
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

/**
 * @file rmsnorm.metal.h
 * @brief MPS RMS Normalization kernel
 *
 * Root Mean Square Layer Normalization as used in LLaMA/Qwen models.
 * Each threadgroup handles one row (token) of the input.
 */

namespace yirage {
namespace persistent_kernel {
namespace mps {

constexpr const char* RMSNORM_KERNEL_SOURCE = R"(
// RMS Normalization kernel
// Computes: output = input / rms(input) * weight
// where rms(x) = sqrt(mean(x^2) + eps)
kernel void rms_norm_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]]
) {
    // Each threadgroup handles one token
    threadgroup float shared_sum[256];
    
    int token_idx = tgid;
    if (token_idx >= num_tokens) return;
    
    device const float* in_row = input + token_idx * hidden_dim;
    device float* out_row = output + token_idx * hidden_dim;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (int i = tid_in_tg; i < hidden_dim; i += 256) {
        float val = in_row[i];
        local_sum += val * val;
    }
    shared_sum[tid_in_tg] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup (warp-style reduction)
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            shared_sum[tid_in_tg] += shared_sum[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute inverse RMS
    float rms = sqrt(shared_sum[0] / float(hidden_dim) + eps);
    float inv_rms = 1.0f / rms;
    
    // Apply normalization with learned weight
    for (int i = tid_in_tg; i < hidden_dim; i += 256) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}

// Half-precision RMS norm for memory efficiency
kernel void rms_norm_kernel_half(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_sum[256];
    
    int token_idx = tgid;
    if (token_idx >= num_tokens) return;
    
    device const half* in_row = input + token_idx * hidden_dim;
    device half* out_row = output + token_idx * hidden_dim;
    
    // Accumulate in float for precision
    float local_sum = 0.0f;
    for (int i = tid_in_tg; i < hidden_dim; i += 256) {
        float val = float(in_row[i]);
        local_sum += val * val;
    }
    shared_sum[tid_in_tg] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            shared_sum[tid_in_tg] += shared_sum[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rms = sqrt(shared_sum[0] / float(hidden_dim) + eps);
    float inv_rms = 1.0f / rms;
    
    for (int i = tid_in_tg; i < hidden_dim; i += 256) {
        out_row[i] = half(float(in_row[i]) * inv_rms * float(weight[i]));
    }
}
)";

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
