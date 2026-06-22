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
 * @file task_header.h
 * @brief M1 Optimized Metal Kernels (Apple Silicon Gen 1)
 *
 * M1 (2020) characteristics:
 * - 8 GPU cores (Base), 16 (Pro), 32 (Max)
 * - No hardware ray tracing
 * - No simdgroup_matrix operations
 * - 32KB threadgroup memory
 * - Unified memory with 68-400 GB/s bandwidth
 *
 * Optimization strategy:
 * - Focus on memory bandwidth utilization
 * - Use coalesced memory access patterns
 * - Leverage SIMD shuffle operations
 * - Conservative threadgroup sizes (256)
 */

#include "../common/mps_common.h"

namespace yirage {
namespace persistent_kernel {
namespace mps {
namespace m1 {

// M1 specific constants
constexpr int M1_THREADGROUP_SIZE = 256;
constexpr int M1_SIMD_WIDTH = 32;
constexpr int M1_TILE_SIZE = 32;
constexpr int M1_SHARED_MEM_KB = 32;

/**
 * @brief Get M1-optimized kernel source
 */
constexpr const char* M1_KERNEL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// M1 Optimized RMSNorm - Conservative memory access pattern
// =============================================================================
kernel void rms_norm_m1(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // M1: 256 threads, 8 SIMD groups per threadgroup
    constexpr int THREADS_PER_TG = 256;
    constexpr int SIMD_WIDTH = 32;
    
    threadgroup float shared_sum[8];  // One per SIMD group
    
    int token_idx = tgid;
    if (token_idx >= num_tokens) return;
    
    device const float* in_row = input + token_idx * hidden_dim;
    device float* out_row = output + token_idx * hidden_dim;
    
    // Phase 1: Compute sum of squares with SIMD reduction
    float local_sum = 0.0f;
    for (int i = tid_in_tg; i < hidden_dim; i += THREADS_PER_TG) {
        float val = in_row[i];
        local_sum += val * val;
    }
    
    // SIMD shuffle reduction within warp
    local_sum += simd_shuffle_xor(local_sum, 16);
    local_sum += simd_shuffle_xor(local_sum, 8);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 1);
    
    if (simd_lane == 0) {
        shared_sum[simd_group] = local_sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction across SIMD groups
    if (simd_group == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum += simd_shuffle_xor(local_sum, 4);
        local_sum += simd_shuffle_xor(local_sum, 2);
        local_sum += simd_shuffle_xor(local_sum, 1);
        if (simd_lane == 0) {
            shared_sum[0] = local_sum;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float rms = sqrt(shared_sum[0] / float(hidden_dim) + eps);
    float inv_rms = 1.0f / rms;
    
    // Apply normalization
    for (int i = tid_in_tg; i < hidden_dim; i += THREADS_PER_TG) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}

// =============================================================================
// M1 Optimized GEMM - 32x32 tiles
// =============================================================================
kernel void gemm_m1(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr int TILE_SIZE = 32;
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = tgid.y * TILE_SIZE + tid_in_tg.y;
    int col = tgid.x * TILE_SIZE + tid_in_tg.x;
    
    float acc = 0.0f;
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + tid_in_tg.x;
        int b_row = t * TILE_SIZE + tid_in_tg.y;
        
        As[tid_in_tg.y][tid_in_tg.x] = (row < M && a_col < K) ? 
            A[row * K + a_col] : 0.0f;
        Bs[tid_in_tg.y][tid_in_tg.x] = (b_row < K && col < N) ? 
            B[b_row * N + col] : 0.0f;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += As[tid_in_tg.y][k] * Bs[k][tid_in_tg.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// =============================================================================
// M1 Optimized Attention - Memory bandwidth focused
// =============================================================================
kernel void attention_score_m1(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& num_heads [[buffer(4)]],
    constant int& seq_len [[buffer(5)]],
    constant int& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int k_pos = tid.x;
    int q_pos = tid.y;
    int bh = tid.z;
    int b = bh / num_heads;
    int h = bh % num_heads;
    
    if (b >= batch_size || q_pos >= seq_len || k_pos >= seq_len) return;
    
    device const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
    device const float* k = K + ((b * num_heads + h) * seq_len + k_pos) * head_dim;
    
    // Vectorized dot product using float4
    float dot = 0.0f;
    int vec_dim = head_dim / 4;
    
    for (int d = 0; d < vec_dim; d++) {
        float4 qv = *((device const float4*)(q + d * 4));
        float4 kv = *((device const float4*)(k + d * 4));
        dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
    }
    
    // Handle remaining elements
    for (int d = vec_dim * 4; d < head_dim; d++) {
        dot += q[d] * k[d];
    }
    
    scores[((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos] = dot * scale;
}
)";

}  // namespace m1
}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
