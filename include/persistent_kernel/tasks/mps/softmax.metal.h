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
 * @file softmax.metal.h
 * @brief MPS Softmax kernel
 *
 * Row-wise softmax implementation with numerical stability.
 * Uses the max-subtraction trick: softmax(x) = softmax(x - max(x))
 */

namespace yirage {
namespace persistent_kernel {
namespace mps {

constexpr const char* SOFTMAX_KERNEL_SOURCE = R"(
// Row-wise softmax with numerical stability
// Each threadgroup handles one row
kernel void softmax_kernel(
    device float* scores [[buffer(0)]],
    constant int& num_rows [[buffer(1)]],
    constant int& row_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    int row = tgid;
    if (row >= num_rows) return;
    
    device float* row_data = scores + row * row_size;
    
    // Phase 1: Find max value in row
    float local_max = -INFINITY;
    for (int i = tid_in_tg; i < row_size; i += 256) {
        local_max = max(local_max, row_data[i]);
    }
    shared_max[tid_in_tg] = local_max;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to find global max
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            shared_max[tid_in_tg] = max(shared_max[tid_in_tg], 
                                         shared_max[tid_in_tg + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];
    
    // Phase 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid_in_tg; i < row_size; i += 256) {
        float val = exp(row_data[i] - row_max);
        row_data[i] = val;  // Store exp value
        local_sum += val;
    }
    shared_sum[tid_in_tg] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to find total sum
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            shared_sum[tid_in_tg] += shared_sum[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_sum[0];
    
    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid_in_tg; i < row_size; i += 256) {
        row_data[i] *= inv_sum;
    }
}

// Online softmax (single-pass, more numerically stable)
kernel void softmax_online_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& num_rows [[buffer(2)]],
    constant int& row_size [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    int row = tgid;
    if (row >= num_rows) return;
    
    device const float* in_row = input + row * row_size;
    device float* out_row = output + row * row_size;
    
    // Online algorithm: track running max and sum
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    
    for (int i = tid_in_tg; i < row_size; i += 256) {
        float x = in_row[i];
        if (x > local_max) {
            local_sum = local_sum * exp(local_max - x) + 1.0f;
            local_max = x;
        } else {
            local_sum += exp(x - local_max);
        }
    }
    
    shared_max[tid_in_tg] = local_max;
    shared_sum[tid_in_tg] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce max and sum together
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            float m1 = shared_max[tid_in_tg];
            float m2 = shared_max[tid_in_tg + stride];
            float s1 = shared_sum[tid_in_tg];
            float s2 = shared_sum[tid_in_tg + stride];
            
            if (m1 > m2) {
                shared_max[tid_in_tg] = m1;
                shared_sum[tid_in_tg] = s1 + s2 * exp(m2 - m1);
            } else {
                shared_max[tid_in_tg] = m2;
                shared_sum[tid_in_tg] = s2 + s1 * exp(m1 - m2);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float row_max = shared_max[0];
    float row_sum = shared_sum[0];
    float log_sum = row_max + log(row_sum);
    
    // Compute final softmax values
    for (int i = tid_in_tg; i < row_size; i += 256) {
        out_row[i] = exp(in_row[i] - log_sum);
    }
}

// Log-softmax for numerical stability in training
kernel void log_softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& num_rows [[buffer(2)]],
    constant int& row_size [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    int row = tgid;
    if (row >= num_rows) return;
    
    device const float* in_row = input + row * row_size;
    device float* out_row = output + row * row_size;
    
    // Find max
    float local_max = -INFINITY;
    for (int i = tid_in_tg; i < row_size; i += 256) {
        local_max = max(local_max, in_row[i]);
    }
    shared_max[tid_in_tg] = local_max;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            shared_max[tid_in_tg] = max(shared_max[tid_in_tg], 
                                         shared_max[tid_in_tg + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];
    
    // Compute sum of exp
    float local_sum = 0.0f;
    for (int i = tid_in_tg; i < row_size; i += 256) {
        local_sum += exp(in_row[i] - row_max);
    }
    shared_sum[tid_in_tg] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            shared_sum[tid_in_tg] += shared_sum[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float log_sum = log(shared_sum[0]) + row_max;
    
    // output = input - log_sum
    for (int i = tid_in_tg; i < row_size; i += 256) {
        out_row[i] = in_row[i] - log_sum;
    }
}
)";

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
