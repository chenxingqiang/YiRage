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
 * Metal Softmax Kernels for Apple Silicon
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Parameters
// =============================================================================

struct SoftmaxParams {
    uint num_rows;
    uint row_size;
};

// =============================================================================
// SIMD Helpers
// =============================================================================

inline float simd_max(float val, uint simd_lane) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum(float val, uint simd_lane) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// =============================================================================
// Row-wise Softmax FP32
// =============================================================================

kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_data[8];
    
    uint row_idx = tg_id;
    if (row_idx >= params.num_rows) return;
    
    device const float* in_row = input + row_idx * params.row_size;
    device float* out_row = output + row_idx * params.row_size;
    
    // Phase 1: Find max
    float local_max = -INFINITY;
    for (uint i = tid; i < params.row_size; i += 256) {
        local_max = max(local_max, in_row[i]);
    }
    
    local_max = simd_max(local_max, simd_lane);
    if (simd_lane == 0) shared_data[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_max = shared_data[simd_lane];
        local_max = simd_max(local_max, simd_lane);
        if (simd_lane == 0) shared_data[0] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float row_max = shared_data[0];
    
    // Phase 2: Compute exp and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < params.row_size; i += 256) {
        float exp_val = exp(in_row[i] - row_max);
        out_row[i] = exp_val;
        local_sum += exp_val;
    }
    
    local_sum = simd_sum(local_sum, simd_lane);
    if (simd_lane == 0) shared_data[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_data[simd_lane];
        local_sum = simd_sum(local_sum, simd_lane);
        if (simd_lane == 0) shared_data[0] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_sum = 1.0f / shared_data[0];
    
    // Phase 3: Normalize
    for (uint i = tid; i < params.row_size; i += 256) {
        out_row[i] *= inv_sum;
    }
}

// =============================================================================
// Row-wise Softmax FP16
// =============================================================================

kernel void softmax_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_data[8];
    
    uint row_idx = tg_id;
    if (row_idx >= params.num_rows) return;
    
    device const half* in_row = input + row_idx * params.row_size;
    device half* out_row = output + row_idx * params.row_size;
    
    // Find max (FP32)
    float local_max = -INFINITY;
    for (uint i = tid; i < params.row_size; i += 256) {
        local_max = max(local_max, float(in_row[i]));
    }
    
    local_max = simd_max(local_max, simd_lane);
    if (simd_lane == 0) shared_data[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_max = shared_data[simd_lane];
        local_max = simd_max(local_max, simd_lane);
        if (simd_lane == 0) shared_data[0] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float row_max = shared_data[0];
    
    // Compute exp sum
    float local_sum = 0.0f;
    for (uint i = tid; i < params.row_size; i += 256) {
        float exp_val = exp(float(in_row[i]) - row_max);
        out_row[i] = half(exp_val);
        local_sum += exp_val;
    }
    
    local_sum = simd_sum(local_sum, simd_lane);
    if (simd_lane == 0) shared_data[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_data[simd_lane];
        local_sum = simd_sum(local_sum, simd_lane);
        if (simd_lane == 0) shared_data[0] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_sum = 1.0f / shared_data[0];
    
    // Normalize
    for (uint i = tid; i < params.row_size; i += 256) {
        out_row[i] = half(float(out_row[i]) * inv_sum);
    }
}

// =============================================================================
// Online Softmax (fused with attention)
// =============================================================================

kernel void online_softmax_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_max[8];
    threadgroup float shared_sum[8];
    
    uint row_idx = tg_id;
    if (row_idx >= params.num_rows) return;
    
    device const half* in_row = input + row_idx * params.row_size;
    device half* out_row = output + row_idx * params.row_size;
    
    // Single-pass online softmax
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    
    for (uint i = tid; i < params.row_size; i += 256) {
        float val = float(in_row[i]);
        float old_max = local_max;
        local_max = max(local_max, val);
        local_sum = local_sum * exp(old_max - local_max) + exp(val - local_max);
    }
    
    // Combine across threads
    // First reduce max
    float global_max = simd_max(local_max, simd_lane);
    if (simd_lane == 0) shared_max[simd_id] = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        global_max = shared_max[simd_lane];
        global_max = simd_max(global_max, simd_lane);
        if (simd_lane == 0) shared_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float row_max = shared_max[0];
    
    // Adjust sum for global max
    local_sum *= exp(local_max - row_max);
    
    float global_sum = simd_sum(local_sum, simd_lane);
    if (simd_lane == 0) shared_sum[simd_id] = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        global_sum = shared_sum[simd_lane];
        global_sum = simd_sum(global_sum, simd_lane);
        if (simd_lane == 0) shared_sum[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_sum = 1.0f / shared_sum[0];
    
    // Compute output
    for (uint i = tid; i < params.row_size; i += 256) {
        float val = float(in_row[i]);
        out_row[i] = half(exp(val - row_max) * inv_sum);
    }
}
