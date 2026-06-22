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
 * @brief C550 Optimized MACA Kernels (MetaX Enhanced Gen 1)
 *
 * MetaX C550 (2024) improvements over C500:
 * - 112 SMs (Standard), 140 SMs (Pro)
 * - 96KB shared memory (vs 64KB)
 * - Better memory bandwidth
 * - Improved tensor core utilization
 *
 * Optimization strategy:
 * - Larger tiles with 96KB shared memory
 * - 64x128 GEMM tiles
 * - Better prefetching
 */

#include "../common/maca_common.h"

namespace yirage {
namespace persistent_kernel {
namespace maca {
namespace c550 {

constexpr int C550_BLOCK_SIZE = 256;
constexpr int C550_WARP_SIZE = 64;
constexpr int C550_SHARED_MEM_KB = 96;

constexpr const char* C550_KERNEL_SOURCE = R"(
// =============================================================================
// MetaX C550 Optimized Kernels
// Enhanced shared memory (96KB) and improved tensor cores
// =============================================================================

#include <maca_runtime.h>
#include <maca_fp16.h>

#define WARP_SIZE 64
#define WARP_SHUFFLE_DOWN(val, offset) __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, val, offset)
#define WARP_SHUFFLE_XOR(val, mask) __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, val, mask)

// =============================================================================
// C550 GEMM - 64x128 tiles with 96KB shared memory
// =============================================================================
__global__ void gemm_c550(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // C550: 96KB shared memory allows 64x128 tiles
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    
    // 64*32*4 + 32*128*4 = 8KB + 16KB = 24KB per tile pair
    // Can double-buffer: 48KB total, still under 96KB
    __shared__ float As[2][TILE_M][TILE_K + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Each warp handles 16x32 output
    int warp_row = warp_id * 16;
    int warp_col = 0;  // All warps work on full width
    
    float acc[4][8] = {{0.0f}};  // 4x8 = 32 per thread
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    int buffer = 0;
    
    // Load first tile
    for (int i = tid; i < TILE_M * TILE_K; i += 256) {
        int load_row = i / TILE_K;
        int load_col = i % TILE_K;
        int a_row = row_base + load_row;
        int a_col = load_col;
        As[0][load_row][load_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
    }
    
    for (int i = tid; i < TILE_K * TILE_N; i += 256) {
        int load_row = i / TILE_N;
        int load_col = i % TILE_N;
        int b_row = load_row;
        int b_col = col_base + load_col;
        Bs[0][load_row][load_col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
    }
    
    __syncthreads();
    
    for (int t = 0; t < num_tiles; t++) {
        int next_buffer = 1 - buffer;
        
        // Prefetch next tile while computing current
        if (t + 1 < num_tiles) {
            for (int i = tid; i < TILE_M * TILE_K; i += 256) {
                int load_row = i / TILE_K;
                int load_col = i % TILE_K;
                int a_row = row_base + load_row;
                int a_col = (t + 1) * TILE_K + load_col;
                As[next_buffer][load_row][load_col] = (a_row < M && a_col < K) ? 
                    A[a_row * K + a_col] : 0.0f;
            }
            
            for (int i = tid; i < TILE_K * TILE_N; i += 256) {
                int load_row = i / TILE_N;
                int load_col = i % TILE_N;
                int b_row = (t + 1) * TILE_K + load_row;
                int b_col = col_base + load_col;
                Bs[next_buffer][load_row][load_col] = (b_row < K && b_col < N) ? 
                    B[b_row * N + b_col] : 0.0f;
            }
        }
        
        // Compute on current buffer
        for (int k = 0; k < TILE_K; k++) {
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 8; ni++) {
                    int row = warp_row + mi * 4 + lane_id / 16;
                    int col = ni * 16 + (lane_id % 16);
                    if (row < TILE_M && col < TILE_N) {
                        acc[mi][ni] += As[buffer][row][k] * Bs[buffer][k][col];
                    }
                }
            }
        }
        
        buffer = next_buffer;
        __syncthreads();
    }
    
    // Store results
    for (int mi = 0; mi < 4; mi++) {
        for (int ni = 0; ni < 8; ni++) {
            int out_row = row_base + warp_row + mi * 4 + lane_id / 16;
            int out_col = col_base + ni * 16 + (lane_id % 16);
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[mi][ni];
            }
        }
    }
}

// =============================================================================
// C550 Flash Attention - 128 token tiles
// =============================================================================
__global__ void flash_attention_c550(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    constexpr int Q_TILE = 16;
    constexpr int KV_TILE = 128;
    
    extern __shared__ float shared[];
    float* tile_k = shared;
    float* tile_v = shared + KV_TILE * head_dim;
    float* row_max = tile_v + KV_TILE * head_dim;
    float* row_sum = row_max + Q_TILE;
    float* acc = row_sum + Q_TILE;
    
    int b = blockIdx.y / num_heads;
    int h = blockIdx.y % num_heads;
    int q_start = blockIdx.x * Q_TILE;
    
    if (b >= batch_size) return;
    
    int tid = threadIdx.x;
    
    // Initialize
    if (tid < Q_TILE) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    for (int i = tid; i < Q_TILE * head_dim; i += 256) {
        acc[i] = 0.0f;
    }
    __syncthreads();
    
    int num_kv_tiles = (seq_len + KV_TILE - 1) / KV_TILE;
    
    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        int kv_start = kv_t * KV_TILE;
        
        // Load K, V tiles
        for (int i = tid; i < KV_TILE * head_dim; i += 256) {
            int ki = i / head_dim;
            int d = i % head_dim;
            int k_pos = kv_start + ki;
            
            if (k_pos < seq_len) {
                int idx = ((b * num_heads + h) * seq_len + k_pos) * head_dim + d;
                tile_k[i] = K[idx];
                tile_v[i] = V[idx];
            } else {
                tile_k[i] = 0.0f;
                tile_v[i] = 0.0f;
            }
        }
        __syncthreads();
        
        // Process each query
        for (int qi = 0; qi < Q_TILE && q_start + qi < seq_len; qi++) {
            int q_pos = q_start + qi;
            const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
            
            float local_max = row_max[qi];
            float local_sum = row_sum[qi];
            
            for (int ki = tid % 64; ki < KV_TILE && kv_start + ki < seq_len; ki += 64) {
                int k_pos = kv_start + ki;
                if (k_pos > q_pos) continue;
                
                // Dot product
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[d] * tile_k[ki * head_dim + d];
                }
                score *= scale;
                
                // Online softmax
                float old_max = local_max;
                local_max = fmaxf(local_max, score);
                float exp_diff = expf(old_max - local_max);
                local_sum = local_sum * exp_diff + expf(score - local_max);
                
                // Update accumulator
                float weight = expf(score - local_max);
                for (int d = 0; d < head_dim; d++) {
                    atomicAdd(&acc[qi * head_dim + d], 
                             weight * tile_v[ki * head_dim + d] - 
                             (1 - exp_diff) * acc[qi * head_dim + d]);
                }
            }
            
            if (tid % 64 == 0) {
                row_max[qi] = local_max;
                row_sum[qi] = local_sum;
            }
        }
        __syncthreads();
    }
    
    // Normalize and output
    for (int qi = 0; qi < Q_TILE && q_start + qi < seq_len; qi++) {
        int q_pos = q_start + qi;
        float* out = output + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
        
        float inv_sum = 1.0f / row_sum[qi];
        for (int d = tid; d < head_dim; d += 256) {
            out[d] = acc[qi * head_dim + d] * inv_sum;
        }
    }
}

// =============================================================================
// C550 Half-precision RMSNorm
// =============================================================================
__global__ void rms_norm_c550_half(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    constexpr int BLOCK_SIZE = 256;
    __shared__ float shared_sum[4];
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    const __half* in_row = input + token_idx * hidden_dim;
    __half* out_row = output + token_idx * hidden_dim;
    
    // Compute sum of squares (accumulate in float)
    float local_sum = 0.0f;
    for (int i = tid * 2; i < hidden_dim; i += BLOCK_SIZE * 2) {
        if (i + 1 < hidden_dim) {
            __half2 vals = *reinterpret_cast<const __half2*>(in_row + i);
            float2 fvals = __half22float2(vals);
            local_sum += fvals.x * fvals.x + fvals.y * fvals.y;
        }
    }
    
    // Warp reduction
    for (int offset = 32; offset > 0; offset >>= 1) {
        local_sum += WARP_SHUFFLE_XOR(local_sum, offset);
    }
    
    if (lane_id == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    
    if (warp_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        local_sum += WARP_SHUFFLE_XOR(local_sum, 2);
        local_sum += WARP_SHUFFLE_XOR(local_sum, 1);
        if (lane_id == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(shared_sum[0] / float(hidden_dim) + eps);
    
    // Apply normalization (vectorized half2)
    for (int i = tid * 2; i < hidden_dim; i += BLOCK_SIZE * 2) {
        if (i + 1 < hidden_dim) {
            __half2 in_vals = *reinterpret_cast<const __half2*>(in_row + i);
            __half2 w_vals = *reinterpret_cast<const __half2*>(weight + i);
            float2 f_in = __half22float2(in_vals);
            float2 f_w = __half22float2(w_vals);
            float2 f_out;
            f_out.x = f_in.x * inv_rms * f_w.x;
            f_out.y = f_in.y * inv_rms * f_w.y;
            *reinterpret_cast<__half2*>(out_row + i) = __float22half2_rn(f_out);
        }
    }
}
)";

}  // namespace c550
}  // namespace maca
}  // namespace persistent_kernel
}  // namespace yirage
