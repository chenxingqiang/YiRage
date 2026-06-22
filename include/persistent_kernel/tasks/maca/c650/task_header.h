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
 * @brief C650 Optimized MACA Kernels (MetaX Enhanced Gen 2)
 *
 * MetaX C650 (2025) improvements:
 * - 144 SMs (Standard), 180 SMs (Pro)
 * - Better sparsity support
 * - Higher memory bandwidth
 * - 128-192GB HBM
 *
 * Optimization strategy:
 * - 128x256 GEMM tiles
 * - 256 token attention tiles
 * - Improved sparse kernel
 */

#include "../common/maca_common.h"

namespace yirage {
namespace persistent_kernel {
namespace maca {
namespace c650 {

constexpr int C650_BLOCK_SIZE = 512;
constexpr int C650_WARP_SIZE = 64;
constexpr int C650_WARPS_PER_BLOCK = 8;
constexpr int C650_SHARED_MEM_KB = 128;

constexpr const char* C650_KERNEL_SOURCE = R"(
// =============================================================================
// MetaX C650 Optimized Kernels
// Enhanced Gen 2 with improved sparsity and bandwidth
// =============================================================================

#include <maca_runtime.h>
#include <maca_fp16.h>

#define WARP_SIZE 64
#define WARP_SHUFFLE_XOR(val, mask) __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, val, mask)

// =============================================================================
// C650 GEMM - 128x256 tiles
// =============================================================================
__global__ void gemm_c650(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 256;
    constexpr int TILE_K = 64;
    
    // 128*64*4 + 64*256*4 = 32KB + 64KB = 96KB
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // 512 threads = 8 warps
    // Each warp computes 16x32 output
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 128;
    
    float acc[4][8] = {{0.0f}};  // 4x8 per thread
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cooperative load
        for (int i = tid; i < TILE_M * TILE_K; i += 512) {
            int load_row = i / TILE_K;
            int load_col = i % TILE_K;
            int a_row = row_base + load_row;
            int a_col = t * TILE_K + load_col;
            As[load_row][load_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }
        
        for (int i = tid; i < TILE_K * TILE_N; i += 512) {
            int load_row = i / TILE_N;
            int load_col = i % TILE_N;
            int b_row = t * TILE_K + load_row;
            int b_col = col_base + load_col;
            Bs[load_row][load_col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute
        for (int k = 0; k < TILE_K; k++) {
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 8; ni++) {
                    int row = warp_row + mi * 8 + lane_id / 8;
                    int col = warp_col + ni * 16 + (lane_id % 8) * 2;
                    if (row < TILE_M && col < TILE_N) {
                        acc[mi][ni] += As[row][k] * Bs[k][col];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store
    for (int mi = 0; mi < 4; mi++) {
        for (int ni = 0; ni < 8; ni++) {
            int out_row = row_base + warp_row + mi * 8 + lane_id / 8;
            int out_col = col_base + warp_col + ni * 16 + (lane_id % 8) * 2;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[mi][ni];
            }
        }
    }
}

// =============================================================================
// C650 Flash Attention - 256 token tiles
// =============================================================================
__global__ void flash_attention_c650(
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
    constexpr int Q_TILE = 32;
    constexpr int KV_TILE = 256;
    
    extern __shared__ float shared[];
    float* tile_k = shared;
    float* tile_v = shared + KV_TILE * 128;  // Max 128 head_dim
    float* row_max = tile_v + KV_TILE * 128;
    float* row_sum = row_max + Q_TILE;
    float* acc = row_sum + Q_TILE;
    
    int b = blockIdx.y / num_heads;
    int h = blockIdx.y % num_heads;
    int q_start = blockIdx.x * Q_TILE;
    
    if (b >= batch_size) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Initialize
    if (tid < Q_TILE) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    for (int i = tid; i < Q_TILE * head_dim; i += 512) {
        acc[i] = 0.0f;
    }
    __syncthreads();
    
    int num_kv_tiles = (seq_len + KV_TILE - 1) / KV_TILE;
    
    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        int kv_start = kv_t * KV_TILE;
        
        // Load K, V
        for (int i = tid; i < KV_TILE * head_dim; i += 512) {
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
        
        // Each warp handles 4 queries
        int qi_per_warp = (Q_TILE + 7) / 8;
        int qi_start = warp_id * qi_per_warp;
        int qi_end = min(qi_start + qi_per_warp, Q_TILE);
        
        for (int qi = qi_start; qi < qi_end && q_start + qi < seq_len; qi++) {
            int q_pos = q_start + qi;
            const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
            
            float local_max = -INFINITY;
            float local_sum = 0.0f;
            
            // Compute attention for KV tokens
            for (int ki = lane_id; ki < KV_TILE && kv_start + ki < seq_len; ki += WARP_SIZE) {
                int k_pos = kv_start + ki;
                if (k_pos > q_pos) continue;
                
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
            }
            
            // Warp reduce for max and sum
            for (int offset = 32; offset > 0; offset >>= 1) {
                float other_max = WARP_SHUFFLE_XOR(local_max, offset);
                float other_sum = WARP_SHUFFLE_XOR(local_sum, offset);
                if (other_max > local_max) {
                    local_sum = local_sum * expf(local_max - other_max) + other_sum;
                    local_max = other_max;
                } else {
                    local_sum = local_sum + other_sum * expf(other_max - local_max);
                }
            }
            
            if (lane_id == 0) {
                // Update global state
                float old_max = row_max[qi];
                float new_max = fmaxf(old_max, local_max);
                float old_sum = row_sum[qi];
                row_sum[qi] = old_sum * expf(old_max - new_max) + local_sum * expf(local_max - new_max);
                row_max[qi] = new_max;
            }
        }
        
        __syncthreads();
        
        // Compute weighted V accumulation (separate pass for clarity)
        for (int qi = 0; qi < Q_TILE && q_start + qi < seq_len; qi++) {
            int q_pos = q_start + qi;
            const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
            float qmax = row_max[qi];
            
            for (int ki = tid % KV_TILE; ki < KV_TILE && kv_start + ki < seq_len; ki += 512) {
                int k_pos = kv_start + ki;
                if (k_pos > q_pos) continue;
                
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[d] * tile_k[ki * head_dim + d];
                }
                score *= scale;
                
                float weight = expf(score - qmax);
                
                for (int d = 0; d < head_dim; d++) {
                    atomicAdd(&acc[qi * head_dim + d], weight * tile_v[ki * head_dim + d]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Normalize and output
    for (int qi = 0; qi < Q_TILE && q_start + qi < seq_len; qi++) {
        int q_pos = q_start + qi;
        float* out = output + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
        
        float inv_sum = 1.0f / row_sum[qi];
        for (int d = tid; d < head_dim; d += 512) {
            out[d] = acc[qi * head_dim + d] * inv_sum;
        }
    }
}

// =============================================================================
// C650 Fused MLP (Gate + Up + SiLU + Down)
// =============================================================================
__global__ void fused_mlp_c650(
    const __half* __restrict__ input,
    const __half* __restrict__ gate_weight,
    const __half* __restrict__ up_weight,
    const __half* __restrict__ down_weight,
    __half* __restrict__ output,
    int batch_tokens,
    int hidden_dim,
    int intermediate_dim
) {
    extern __shared__ __half shared_intermediate[];
    
    int token_idx = blockIdx.x;
    if (token_idx >= batch_tokens) return;
    
    int tid = threadIdx.x;
    
    const __half* in = input + token_idx * hidden_dim;
    __half* out = output + token_idx * hidden_dim;
    
    // Step 1: Gate and Up projections + SiLU
    for (int i = tid; i < intermediate_dim; i += 512) {
        float gate_val = 0.0f;
        float up_val = 0.0f;
        
        for (int d = 0; d < hidden_dim; d++) {
            float x = __half2float(in[d]);
            gate_val += x * __half2float(gate_weight[i * hidden_dim + d]);
            up_val += x * __half2float(up_weight[i * hidden_dim + d]);
        }
        
        // SiLU(gate) * up
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
        shared_intermediate[i] = __float2half((gate_val * sigmoid_gate) * up_val);
    }
    
    __syncthreads();
    
    // Step 2: Down projection
    for (int d = tid; d < hidden_dim; d += 512) {
        float val = 0.0f;
        for (int i = 0; i < intermediate_dim; i++) {
            val += __half2float(shared_intermediate[i]) * 
                   __half2float(down_weight[d * intermediate_dim + i]);
        }
        out[d] = __float2half(val);
    }
}
)";

}  // namespace c650
}  // namespace maca
}  // namespace persistent_kernel
}  // namespace yirage
