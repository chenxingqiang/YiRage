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
 * @file embedding.metal.h
 * @brief MPS Embedding lookup kernel
 *
 * Performs token embedding lookup for LLM inference.
 * Each thread handles one element of the output tensor.
 */

namespace yirage {
namespace persistent_kernel {
namespace mps {

constexpr const char* EMBEDDING_KERNEL_SOURCE = R"(
// Embedding lookup kernel
// Maps token IDs to embedding vectors
kernel void embedding_kernel(
    device const int* input_ids [[buffer(0)]],
    device const float* embedding_table [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& hidden_dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)(num_tokens * hidden_dim)) return;
    
    int token_idx = tid / hidden_dim;
    int dim_idx = tid % hidden_dim;
    int token_id = input_ids[token_idx];
    
    output[tid] = embedding_table[token_id * hidden_dim + dim_idx];
}

// Half-precision embedding kernel for memory efficiency
kernel void embedding_kernel_half(
    device const int* input_ids [[buffer(0)]],
    device const half* embedding_table [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& hidden_dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)(num_tokens * hidden_dim)) return;
    
    int token_idx = tid / hidden_dim;
    int dim_idx = tid % hidden_dim;
    int token_id = input_ids[token_idx];
    
    output[tid] = embedding_table[token_id * hidden_dim + dim_idx];
}
)";

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
