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
 * @brief M3 Optimized Metal Kernels (Apple Silicon Gen 3)
 *
 * M3 (2023) NEW FEATURES:
 * - Hardware ray tracing
 * - Mesh shaders
 * - Dynamic caching (better register/shared mem utilization)
 * - simdgroup_matrix operations for faster GEMM
 * - Improved occupancy with more threads per core
 *
 * Optimization strategy:
 * - Use simdgroup_matrix for matrix operations
 * - Leverage dynamic caching for variable workloads
 * - Larger attention tiles (128)
 * - Better instruction-level parallelism
 */

#include "../common/mps_common.h"

namespace yirage {
namespace persistent_kernel {
namespace mps {
namespace m3 {

// M3 specific constants
constexpr int M3_THREADGROUP_SIZE = 256;
constexpr int M3_SIMD_WIDTH = 32;
constexpr int M3_TILE_SIZE = 64;
constexpr int M3_ATTENTION_TILE = 128;
constexpr bool M3_HAS_SIMDGROUP_MATRIX = true;

constexpr const char* M3_KERNEL_SOURCE = R"(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// =============================================================================
// M3 Optimized GEMM using simdgroup_matrix (8x8 tiles)
// =============================================================================
kernel void gemm_m3(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // M3: Use simdgroup_matrix for 8x8 matrix multiply
    // Each SIMD group computes a 8x8 output tile
    constexpr int TILE_SIZE = 64;
    constexpr int SIMD_TILE = 8;
    
    threadgroup float As[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    threadgroup float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int row_base = tgid.y * TILE_SIZE;
    int col_base = tgid.x * TILE_SIZE;
    
    // Each SIMD group handles an 8x8 sub-tile
    int sg_row = (simd_group / 8) * SIMD_TILE;
    int sg_col = (simd_group % 8) * SIMD_TILE;
    
    // Accumulator for 8x8 block (distributed across SIMD lanes)
    simdgroup_float8x8 acc;
    acc = simdgroup_float8x8(0.0f);
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cooperative load of A and B tiles
        for (int load_idx = tid_in_tg.y * 8 + tid_in_tg.x % 8; 
             load_idx < TILE_SIZE * TILE_SIZE; 
             load_idx += 256) {
            int load_row = load_idx / TILE_SIZE;
            int load_col = load_idx % TILE_SIZE;
            
            int a_row = row_base + load_row;
            int a_col = t * TILE_SIZE + load_col;
            As[load_row][load_col] = (a_row < M && a_col < K) ? 
                A[a_row * K + a_col] : 0.0f;
            
            int b_row = t * TILE_SIZE + load_row;
            int b_col = col_base + load_col;
            Bs[load_row][load_col] = (b_row < K && b_col < N) ? 
                B[b_row * N + b_col] : 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute using simdgroup_matrix multiply-accumulate
        for (int k = 0; k < TILE_SIZE; k += SIMD_TILE) {
            simdgroup_float8x8 a_tile;
            simdgroup_float8x8 b_tile;
            
            // Load 8x8 sub-matrices
            simdgroup_load(a_tile, &As[sg_row][k], TILE_SIZE + 1);
            simdgroup_load(b_tile, &Bs[k][sg_col], TILE_SIZE + 1);
            
            // Multiply-accumulate
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result
    int out_row = row_base + sg_row;
    int out_col = col_base + sg_col;
    
    if (out_row < M && out_col < N) {
        simdgroup_store(acc, C + out_row * N + out_col, N);
    }
}

// =============================================================================
// M3 Flash Attention with larger tiles
// =============================================================================
kernel void flash_attention_m3(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& num_heads [[buffer(5)]],
    constant int& seq_len [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // M3: 128 token tiles for better reuse
    constexpr int Q_TILE = 16;    // Queries per block
    constexpr int KV_TILE = 128;  // Keys/Values per tile
    
    threadgroup float tile_k[KV_TILE * 64];
    threadgroup float tile_v[KV_TILE * 64];
    threadgroup float row_max[Q_TILE];
    threadgroup float row_sum[Q_TILE];
    
    int b = tgid.y / num_heads;
    int h = tgid.y % num_heads;
    int q_start = tgid.x * Q_TILE;
    
    if (b >= batch_size) return;
    
    // Initialize per-query state
    if (tid_in_tg < Q_TILE) {
        row_max[tid_in_tg] = -INFINITY;
        row_sum[tid_in_tg] = 0.0f;
    }
    
    // Each thread maintains partial output for one query dimension
    float acc[Q_TILE] = {0.0f};
    
    int num_kv_tiles = (seq_len + KV_TILE - 1) / KV_TILE;
    
    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        int kv_start = kv_t * KV_TILE;
        
        // Cooperative load of K and V tiles
        for (int i = tid_in_tg; i < KV_TILE * head_dim; i += 256) {
            int ki = i / head_dim;
            int d = i % head_dim;
            int k_pos = kv_start + ki;
            
            if (k_pos < seq_len) {
                int k_idx = ((b * num_heads + h) * seq_len + k_pos) * head_dim + d;
                tile_k[i] = K[k_idx];
                tile_v[i] = V[k_idx];
            } else {
                tile_k[i] = 0.0f;
                tile_v[i] = 0.0f;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Process each query in the tile
        for (int qi = 0; qi < Q_TILE && q_start + qi < seq_len; qi++) {
            int q_pos = q_start + qi;
            device const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
            
            // Compute attention scores for this query against all K in tile
            float local_max = row_max[qi];
            float local_sum = row_sum[qi];
            
            for (int ki = tid_in_tg; ki < KV_TILE && kv_start + ki < seq_len; ki += 256) {
                int k_pos = kv_start + ki;
                
                // Causal mask
                if (k_pos > q_pos) continue;
                
                // Compute Q @ K score
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[d] * tile_k[ki * head_dim + d];
                }
                score *= scale;
                
                // Online softmax
                float old_max = local_max;
                local_max = max(local_max, score);
                float exp_diff = exp(old_max - local_max);
                local_sum = local_sum * exp_diff + exp(score - local_max);
                
                // Accumulate weighted V
                float weight = exp(score - local_max);
                for (int d = simd_lane; d < head_dim; d += 32) {
                    acc[qi] += weight * tile_v[ki * head_dim + d];
                }
            }
            
            // Update shared state
            if (simd_lane == 0) {
                row_max[qi] = local_max;
                row_sum[qi] = local_sum;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Normalize and write output
    for (int qi = 0; qi < Q_TILE && q_start + qi < seq_len; qi++) {
        int q_pos = q_start + qi;
        device float* out = output + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
        
        for (int d = tid_in_tg; d < head_dim; d += 256) {
            out[d] = acc[qi] / row_sum[qi];
        }
    }
}

// =============================================================================
// M3 RMSNorm with dynamic caching optimization
// =============================================================================
kernel void rms_norm_m3(
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
    // M3: Dynamic caching allows more flexible shared memory usage
    constexpr int THREADS_PER_TG = 256;
    threadgroup float shared_sum[8];
    
    int token_idx = tgid;
    if (token_idx >= num_tokens) return;
    
    device const float* in_row = input + token_idx * hidden_dim;
    device float* out_row = output + token_idx * hidden_dim;
    
    // Use simd_sum for faster reduction
    float local_sum = 0.0f;
    for (int i = tid_in_tg; i < hidden_dim; i += THREADS_PER_TG) {
        float val = in_row[i];
        local_sum += val * val;
    }
    
    // M3: Use simd_sum (more efficient than shuffle chain)
    local_sum = simd_sum(local_sum);
    
    if (simd_lane == 0) shared_sum[simd_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group == 0) {
        local_sum = (simd_lane < 8) ? shared_sum[simd_lane] : 0.0f;
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) shared_sum[0] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms = rsqrt(shared_sum[0] / float(hidden_dim) + eps);
    
    for (int i = tid_in_tg; i < hidden_dim; i += THREADS_PER_TG) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}
)";

}  // namespace m3
}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
