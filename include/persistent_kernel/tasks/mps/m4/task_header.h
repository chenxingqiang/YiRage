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
 * @brief M4 Optimized Metal Kernels (Apple Silicon Gen 4)
 *
 * M4 (2024) NEW FEATURES:
 * - Enhanced Neural Engine (38 TOPS)
 * - 48KB threadgroup memory (vs 32KB)
 * - Improved memory bandwidth
 * - Better half-precision performance
 * - Optimized transformer workloads
 *
 * Optimization strategy:
 * - Larger tiles using 48KB shared memory
 * - Aggressive use of half precision
 * - Batch processing optimizations
 * - Neural Engine offload for certain operations
 */

#include "../common/mps_common.h"

namespace yirage {
namespace persistent_kernel {
namespace mps {
namespace m4 {

// M4 specific constants
constexpr int M4_THREADGROUP_SIZE = 384;  // Larger threadgroups
constexpr int M4_SIMD_WIDTH = 32;
constexpr int M4_TILE_SIZE = 64;
constexpr int M4_SHARED_MEM_KB = 48;      // Increased!
constexpr int M4_ATTENTION_TILE = 128;

constexpr const char* M4_KERNEL_SOURCE = R"(
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// =============================================================================
// M4 Optimized GEMM - Larger tiles with 48KB shared memory
// =============================================================================
kernel void gemm_m4(
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
    // M4: Can use larger tiles with 48KB shared memory
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 128;  // Wider tiles
    constexpr int TILE_K = 64;
    
    // Total shared: 64*64*4 + 64*128*4 = 16KB + 32KB = 48KB
    threadgroup float As[TILE_M][TILE_K + 1];
    threadgroup float Bs[TILE_K][TILE_N + 1];
    
    int row_base = tgid.y * TILE_M;
    int col_base = tgid.x * TILE_N;
    
    // 384 threads = 12 SIMD groups
    // Each computes 8x8 output using simdgroup_matrix
    int sg_row = (simd_group / 4) * 8;      // 0, 8, 16 for rows
    int sg_col = (simd_group % 4) * 32;     // 0, 32, 64, 96 for cols
    
    simdgroup_float8x8 acc[4];  // 8x32 per SIMD group
    for (int i = 0; i < 4; i++) acc[i] = simdgroup_float8x8(0.0f);
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cooperative load (384 threads loading 64*64 + 64*128 = 12288 elements)
        int tid_linear = tid_in_tg.y * 32 + tid_in_tg.x;
        
        // Load A tile
        for (int i = tid_linear; i < TILE_M * TILE_K; i += 384) {
            int load_row = i / TILE_K;
            int load_col = i % TILE_K;
            int a_row = row_base + load_row;
            int a_col = t * TILE_K + load_col;
            As[load_row][load_col] = (a_row < M && a_col < K) ? 
                A[a_row * K + a_col] : 0.0f;
        }
        
        // Load B tile
        for (int i = tid_linear; i < TILE_K * TILE_N; i += 384) {
            int load_row = i / TILE_N;
            int load_col = i % TILE_N;
            int b_row = t * TILE_K + load_row;
            int b_col = col_base + load_col;
            Bs[load_row][load_col] = (b_row < K && b_col < N) ? 
                B[b_row * N + b_col] : 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute 8x32 block using 4 simdgroup_matrix ops
        for (int k = 0; k < TILE_K; k += 8) {
            simdgroup_float8x8 a_tile;
            simdgroup_load(a_tile, &As[sg_row][k], TILE_K + 1);
            
            for (int n = 0; n < 4; n++) {
                simdgroup_float8x8 b_tile;
                simdgroup_load(b_tile, &Bs[k][sg_col + n * 8], TILE_N + 1);
                simdgroup_multiply_accumulate(acc[n], a_tile, b_tile, acc[n]);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store 8x32 result
    for (int n = 0; n < 4; n++) {
        int out_row = row_base + sg_row;
        int out_col = col_base + sg_col + n * 8;
        if (out_row < M && out_col < N) {
            simdgroup_store(acc[n], C + out_row * N + out_col, N);
        }
    }
}

// =============================================================================
// M4 Batched RMSNorm - Process multiple tokens per threadgroup
// =============================================================================
kernel void rms_norm_batched_m4(
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
    // M4: Process 4 tokens per threadgroup for better efficiency
    constexpr int TOKENS_PER_TG = 4;
    constexpr int THREADS_PER_TOKEN = 96;  // 384 / 4
    
    threadgroup float shared_sum[TOKENS_PER_TG][3];  // 3 SIMD groups per token
    
    int token_base = tgid * TOKENS_PER_TG;
    int local_token = tid_in_tg / THREADS_PER_TOKEN;
    int local_tid = tid_in_tg % THREADS_PER_TOKEN;
    int local_simd = local_tid / 32;
    int local_lane = local_tid % 32;
    
    int token_idx = token_base + local_token;
    if (token_idx >= num_tokens) return;
    
    device const float* in_row = input + token_idx * hidden_dim;
    device float* out_row = output + token_idx * hidden_dim;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (int i = local_tid; i < hidden_dim; i += THREADS_PER_TOKEN) {
        float val = in_row[i];
        local_sum += val * val;
    }
    
    local_sum = simd_sum(local_sum);
    
    if (local_lane == 0) {
        shared_sum[local_token][local_simd] = local_sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (local_simd == 0) {
        local_sum = (local_lane < 3) ? shared_sum[local_token][local_lane] : 0.0f;
        local_sum = simd_sum(local_sum);
        if (local_lane == 0) {
            shared_sum[local_token][0] = local_sum;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms = rsqrt(shared_sum[local_token][0] / float(hidden_dim) + eps);
    
    for (int i = local_tid; i < hidden_dim; i += THREADS_PER_TOKEN) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}

// =============================================================================
// M4 Half-precision GEMM with improved throughput
// =============================================================================
kernel void gemm_m4_half(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // M4 half: Even larger tiles due to 2x smaller data
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 256;  // 4x wider for half precision
    constexpr int TILE_K = 64;
    
    // Half precision uses half the memory
    threadgroup half As[TILE_M][TILE_K + 2];  // +2 for half alignment
    threadgroup half Bs[TILE_K][TILE_N + 2];
    
    int row_base = tgid.y * TILE_M;
    int col_base = tgid.x * TILE_N;
    
    int sg_row = (simd_group / 8) * 8;
    int sg_col = (simd_group % 8) * 32;
    
    simdgroup_half8x8 acc[4];
    for (int i = 0; i < 4; i++) acc[i] = simdgroup_half8x8(0.0h);
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    int tid_linear = tid_in_tg.y * 32 + tid_in_tg.x;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load A tile
        for (int i = tid_linear; i < TILE_M * TILE_K; i += 384) {
            int load_row = i / TILE_K;
            int load_col = i % TILE_K;
            int a_row = row_base + load_row;
            int a_col = t * TILE_K + load_col;
            As[load_row][load_col] = (a_row < M && a_col < K) ? 
                A[a_row * K + a_col] : half(0.0h);
        }
        
        // Load B tile
        for (int i = tid_linear; i < TILE_K * TILE_N; i += 384) {
            int load_row = i / TILE_N;
            int load_col = i % TILE_N;
            int b_row = t * TILE_K + load_row;
            int b_col = col_base + load_col;
            Bs[load_row][load_col] = (b_row < K && b_col < N) ? 
                B[b_row * N + b_col] : half(0.0h);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (int k = 0; k < TILE_K; k += 8) {
            simdgroup_half8x8 a_tile;
            simdgroup_load(a_tile, &As[sg_row][k], TILE_K + 2);
            
            for (int n = 0; n < 4; n++) {
                simdgroup_half8x8 b_tile;
                simdgroup_load(b_tile, &Bs[k][sg_col + n * 8], TILE_N + 2);
                simdgroup_multiply_accumulate(acc[n], a_tile, b_tile, acc[n]);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (int n = 0; n < 4; n++) {
        int out_row = row_base + sg_row;
        int out_col = col_base + sg_col + n * 8;
        if (out_row < M && out_col < N) {
            simdgroup_store(acc[n], C + out_row * N + out_col, N);
        }
    }
}

// =============================================================================
// M4 Fused QKV Projection
// =============================================================================
kernel void qkv_projection_m4(
    device const half* input [[buffer(0)]],
    device const half* qkv_weight [[buffer(1)]],
    device half* q_out [[buffer(2)]],
    device half* k_out [[buffer(3)]],
    device half* v_out [[buffer(4)]],
    constant int& batch_size [[buffer(5)]],
    constant int& seq_len [[buffer(6)]],
    constant int& hidden_dim [[buffer(7)]],
    constant int& num_heads [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]]
) {
    int b = tid.z;
    int s = tid.y;
    int h = tid.x;
    
    if (b >= batch_size || s >= seq_len || h >= num_heads) return;
    
    device const half* in = input + (b * seq_len + s) * hidden_dim;
    
    // Q, K, V weights are stacked: [3 * num_heads * head_dim, hidden_dim]
    int q_offset = h * head_dim;
    int k_offset = num_heads * head_dim + h * head_dim;
    int v_offset = 2 * num_heads * head_dim + h * head_dim;
    
    device half* q = q_out + ((b * num_heads + h) * seq_len + s) * head_dim;
    device half* k = k_out + ((b * num_heads + h) * seq_len + s) * head_dim;
    device half* v = v_out + ((b * num_heads + h) * seq_len + s) * head_dim;
    
    // Compute Q, K, V projections
    for (int d = 0; d < head_dim; d++) {
        float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
        
        device const half* q_w = qkv_weight + (q_offset + d) * hidden_dim;
        device const half* k_w = qkv_weight + (k_offset + d) * hidden_dim;
        device const half* v_w = qkv_weight + (v_offset + d) * hidden_dim;
        
        for (int i = 0; i < hidden_dim; i++) {
            float x = float(in[i]);
            q_val += x * float(q_w[i]);
            k_val += x * float(k_w[i]);
            v_val += x * float(v_w[i]);
        }
        
        q[d] = half(q_val);
        k[d] = half(k_val);
        v[d] = half(v_val);
    }
}
)";

}  // namespace m4
}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
