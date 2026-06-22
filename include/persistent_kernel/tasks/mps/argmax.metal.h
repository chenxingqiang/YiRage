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
 * @file argmax.metal.h
 * @brief MPS Argmax kernel for token sampling
 *
 * Finds the index of maximum value in vocabulary logits.
 * Used for greedy decoding in LLM inference.
 */

namespace yirage {
namespace persistent_kernel {
namespace mps {

constexpr const char* ARGMAX_KERNEL_SOURCE = R"(
// Simple argmax kernel
// Each thread handles one token (row of logits)
kernel void argmax_kernel(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant int& num_tokens [[buffer(2)]],
    constant int& vocab_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)num_tokens) return;
    
    device const float* row = input + tid * vocab_size;
    
    int max_idx = 0;
    float max_val = row[0];
    
    for (int i = 1; i < vocab_size; i++) {
        if (row[i] > max_val) {
            max_val = row[i];
            max_idx = i;
        }
    }
    
    output[tid] = max_idx;
}

// Parallel argmax using threadgroup reduction
// More efficient for large vocab sizes
kernel void argmax_parallel_kernel(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant int& num_tokens [[buffer(2)]],
    constant int& vocab_size [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_max[256];
    threadgroup int shared_idx[256];
    
    int token_idx = tgid;
    if (token_idx >= num_tokens) return;
    
    device const float* row = input + token_idx * vocab_size;
    
    // Each thread finds local max in its chunk
    float local_max = -INFINITY;
    int local_idx = 0;
    
    for (int i = tid_in_tg; i < vocab_size; i += tg_size) {
        if (row[i] > local_max) {
            local_max = row[i];
            local_idx = i;
        }
    }
    
    shared_max[tid_in_tg] = local_max;
    shared_idx[tid_in_tg] = local_idx;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce to find global max
    for (int stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            if (shared_max[tid_in_tg + stride] > shared_max[tid_in_tg]) {
                shared_max[tid_in_tg] = shared_max[tid_in_tg + stride];
                shared_idx[tid_in_tg] = shared_idx[tid_in_tg + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid_in_tg == 0) {
        output[token_idx] = shared_idx[0];
    }
}

// Top-k sampling kernel
kernel void topk_kernel(
    device const float* input [[buffer(0)]],
    device int* output_indices [[buffer(1)]],
    device float* output_values [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& vocab_size [[buffer(4)]],
    constant int& k [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    int token_idx = tgid;
    if (token_idx >= num_tokens) return;
    
    device const float* row = input + token_idx * vocab_size;
    device int* out_idx = output_indices + token_idx * k;
    device float* out_val = output_values + token_idx * k;
    
    // Simple O(n*k) top-k for small k
    for (int ki = 0; ki < k; ki++) {
        float max_val = -INFINITY;
        int max_idx = 0;
        
        for (int i = 0; i < vocab_size; i++) {
            bool already_selected = false;
            for (int j = 0; j < ki; j++) {
                if (out_idx[j] == i) {
                    already_selected = true;
                    break;
                }
            }
            if (!already_selected && row[i] > max_val) {
                max_val = row[i];
                max_idx = i;
            }
        }
        
        out_idx[ki] = max_idx;
        out_val[ki] = max_val;
    }
}
)";

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
