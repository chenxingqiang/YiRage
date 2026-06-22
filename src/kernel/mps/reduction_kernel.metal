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
 * Metal Reduction Kernels for Apple Silicon
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// SIMD Helpers
// =============================================================================

inline float simd_sum(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

inline float simd_max(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_min(float val) {
    val = min(val, simd_shuffle_xor(val, 16));
    val = min(val, simd_shuffle_xor(val, 8));
    val = min(val, simd_shuffle_xor(val, 4));
    val = min(val, simd_shuffle_xor(val, 2));
    val = min(val, simd_shuffle_xor(val, 1));
    return val;
}

// =============================================================================
// Global Sum Reduction
// =============================================================================

kernel void reduce_sum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[8];
    
    float local_sum = 0.0f;
    uint stride = 256 * 1024;  // Assumes max 1024 threadgroups
    
    for (uint i = tg_id * 256 + tid; i < size; i += stride) {
        local_sum += input[i];
    }
    
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            // Atomic add to output
            atomic_fetch_add_explicit((device atomic_float*)output, local_sum,
                                      memory_order_relaxed);
        }
    }
}

// =============================================================================
// Row-wise Sum Reduction
// =============================================================================

kernel void reduce_sum_row_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_rows [[buffer(2)]],
    constant uint& row_size [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[8];
    
    if (row_idx >= num_rows) return;
    
    device const float* row = input + row_idx * row_size;
    
    float local_sum = 0.0f;
    for (uint i = tid; i < row_size; i += 256) {
        local_sum += row[i];
    }
    
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            output[row_idx] = local_sum;
        }
    }
}

// =============================================================================
// Global Max Reduction
// =============================================================================

kernel void reduce_max_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_max[8];
    
    float local_max = -INFINITY;
    uint stride = 256 * 1024;
    
    for (uint i = tg_id * 256 + tid; i < size; i += stride) {
        local_max = max(local_max, input[i]);
    }
    
    local_max = simd_max(local_max);
    if (simd_lane == 0) shared_max[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_max = shared_max[simd_lane];
        local_max = simd_max(local_max);
        if (simd_lane == 0) {
            // Atomic max (using compare-exchange)
            float expected = *output;
            while (local_max > expected) {
                float old = expected;
                expected = atomic_exchange_explicit(
                    (device atomic_float*)output,
                    max(expected, local_max),
                    memory_order_relaxed);
                if (old == expected) break;
            }
        }
    }
}

// =============================================================================
// Row-wise Max Reduction
// =============================================================================

kernel void reduce_max_row_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_rows [[buffer(2)]],
    constant uint& row_size [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_max[8];
    
    if (row_idx >= num_rows) return;
    
    device const float* row = input + row_idx * row_size;
    
    float local_max = -INFINITY;
    for (uint i = tid; i < row_size; i += 256) {
        local_max = max(local_max, row[i]);
    }
    
    local_max = simd_max(local_max);
    if (simd_lane == 0) shared_max[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_max = shared_max[simd_lane];
        local_max = simd_max(local_max);
        if (simd_lane == 0) {
            output[row_idx] = local_max;
        }
    }
}

// =============================================================================
// Mean Reduction
// =============================================================================

kernel void reduce_mean_row_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_rows [[buffer(2)]],
    constant uint& row_size [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[8];
    
    if (row_idx >= num_rows) return;
    
    device const float* row = input + row_idx * row_size;
    
    float local_sum = 0.0f;
    for (uint i = tid; i < row_size; i += 256) {
        local_sum += row[i];
    }
    
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            output[row_idx] = local_sum / float(row_size);
        }
    }
}

// =============================================================================
// Variance Reduction (for LayerNorm)
// =============================================================================

kernel void reduce_variance_row_f32(
    device const float* input [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device float* variance [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& row_size [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[8];
    
    if (row_idx >= num_rows) return;
    
    device const float* row = input + row_idx * row_size;
    float row_mean = mean[row_idx];
    
    float local_sum = 0.0f;
    for (uint i = tid; i < row_size; i += 256) {
        float diff = row[i] - row_mean;
        local_sum += diff * diff;
    }
    
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            variance[row_idx] = local_sum / float(row_size);
        }
    }
}

// =============================================================================
// L2 Norm
// =============================================================================

kernel void reduce_l2_norm_row_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_rows [[buffer(2)]],
    constant uint& row_size [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[8];
    
    if (row_idx >= num_rows) return;
    
    device const float* row = input + row_idx * row_size;
    
    float local_sum = 0.0f;
    for (uint i = tid; i < row_size; i += 256) {
        float val = row[i];
        local_sum += val * val;
    }
    
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            output[row_idx] = sqrt(local_sum);
        }
    }
}
