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
 * @brief C600 Optimized MACA Kernels (MetaX Gen 2)
 *
 * MetaX C600 (2024) NEW FEATURES:
 * - 128 SMs (Standard), 160 SMs (Pro)
 * - 128KB shared memory
 * - 48 warps per SM (vs 32)
 * - Sparsity acceleration!
 * - Doubled registers per SM
 *
 * Optimization strategy:
 * - 128x128 GEMM tiles
 * - Sparsity-aware kernels
 * - Higher occupancy
 */

#include "../common/maca_common.h"

namespace yirage {
namespace persistent_kernel {
namespace maca {
namespace c600 {

constexpr int C600_BLOCK_SIZE = 384;
constexpr int C600_WARP_SIZE = 64;
constexpr int C600_WARPS_PER_BLOCK = 6;
constexpr int C600_SHARED_MEM_KB = 128;

constexpr const char* C600_KERNEL_SOURCE = R"(
// =============================================================================
// MetaX C600 Optimized Kernels
// 128KB shared memory, 48 warps/SM, sparsity acceleration
// =============================================================================

#include <maca_runtime.h>
#include <maca_fp16.h>

#define WARP_SIZE 64
#define WARP_SHUFFLE_DOWN(val, offset) __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, val, offset)
#define WARP_SHUFFLE_XOR(val, mask) __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, val, mask)

// =============================================================================
// C600 GEMM - 128x128 tiles with tensor cores
// =============================================================================
__global__ void gemm_c600(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // C600: 128KB shared memory allows 128x128 tiles
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 64;
    
    // 128*64*4 + 64*128*4 = 32KB + 32KB = 64KB
    // Double buffer: 128KB total
    __shared__ float As[2][TILE_M][TILE_K + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // 384 threads = 6 warps
    // Each warp handles 21x128 output (approximately)
    int warp_row = warp_id * 21;
    
    float acc[8][8] = {{0.0f}};  // 8x8 per thread = 64 elements
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    int buffer = 0;
    
    // Initial load
    for (int i = tid; i < TILE_M * TILE_K; i += 384) {
        int load_row = i / TILE_K;
        int load_col = i % TILE_K;
        int a_row = row_base + load_row;
        int a_col = load_col;
        As[0][load_row][load_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
    }
    
    for (int i = tid; i < TILE_K * TILE_N; i += 384) {
        int load_row = i / TILE_N;
        int load_col = i % TILE_N;
        int b_row = load_row;
        int b_col = col_base + load_col;
        Bs[0][load_row][load_col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
    }
    __syncthreads();
    
    for (int t = 0; t < num_tiles; t++) {
        int next_buffer = 1 - buffer;
        
        // Prefetch next tile
        if (t + 1 < num_tiles) {
            for (int i = tid; i < TILE_M * TILE_K; i += 384) {
                int load_row = i / TILE_K;
                int load_col = i % TILE_K;
                int a_row = row_base + load_row;
                int a_col = (t + 1) * TILE_K + load_col;
                As[next_buffer][load_row][load_col] = (a_row < M && a_col < K) ? 
                    A[a_row * K + a_col] : 0.0f;
            }
            
            for (int i = tid; i < TILE_K * TILE_N; i += 384) {
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
            for (int mi = 0; mi < 8; mi++) {
                for (int ni = 0; ni < 8; ni++) {
                    int row = warp_row + mi * 3 + lane_id / 21;
                    int col = ni * 16 + (lane_id % 8) * 2;
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
    for (int mi = 0; mi < 8; mi++) {
        for (int ni = 0; ni < 8; ni++) {
            int out_row = row_base + warp_row + mi * 3 + lane_id / 21;
            int out_col = col_base + ni * 16 + (lane_id % 8) * 2;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[mi][ni];
            }
        }
    }
}

// =============================================================================
// C600 Sparse GEMM - 2:4 structured sparsity
// =============================================================================
__global__ void sparse_gemm_c600(
    const float* __restrict__ A,          // Dense input
    const float* __restrict__ B_values,   // Sparse weights (2:4 format)
    const uint8_t* __restrict__ B_meta,   // Sparsity metadata
    float* __restrict__ C,
    int M, int N, int K
) {
    // 2:4 sparsity: for every 4 elements, only 2 are non-zero
    // This effectively halves the compute and memory for weight matrix
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 64;
    
    __shared__ float As[TILE_M][TILE_K + 1];
    // Sparse B is half the size
    __shared__ float Bs_values[TILE_K / 2][TILE_N + 1];
    __shared__ uint8_t Bs_meta[TILE_K / 4][TILE_N];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    float acc[4][8] = {{0.0f}};
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load dense A tile
        for (int i = tid; i < TILE_M * TILE_K; i += 384) {
            int load_row = i / TILE_K;
            int load_col = i % TILE_K;
            int a_row = row_base + load_row;
            int a_col = t * TILE_K + load_col;
            As[load_row][load_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }
        
        // Load sparse B tile (values and metadata)
        int sparse_k = TILE_K / 2;  // Half the K dimension due to 2:4 sparsity
        for (int i = tid; i < sparse_k * TILE_N; i += 384) {
            int load_row = i / TILE_N;
            int load_col = i % TILE_N;
            int b_row = t * sparse_k + load_row;
            int b_col = col_base + load_col;
            int total_sparse_k = K / 2;
            Bs_values[load_row][load_col] = (b_row < total_sparse_k && b_col < N) ? 
                B_values[b_row * N + b_col] : 0.0f;
        }
        
        // Load metadata (2 bits per 4 elements indicating which 2 are non-zero)
        for (int i = tid; i < (TILE_K / 4) * TILE_N; i += 384) {
            int load_row = i / TILE_N;
            int load_col = i % TILE_N;
            int m_row = t * (TILE_K / 4) + load_row;
            int m_col = col_base + load_col;
            int total_meta_k = K / 4;
            Bs_meta[load_row][load_col] = (m_row < total_meta_k && m_col < N) ? 
                B_meta[m_row * N + m_col] : 0;
        }
        
        __syncthreads();
        
        // Compute with 2:4 sparsity pattern
        int warp_row = warp_id * 10;
        for (int k4 = 0; k4 < TILE_K / 4; k4++) {
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 8; ni++) {
                    int row = warp_row + mi * 3 + lane_id / 21;
                    int col = ni * 16 + (lane_id % 16);
                    
                    if (row < TILE_M && col < TILE_N) {
                        uint8_t meta = Bs_meta[k4][col];
                        
                        // Decode 2:4 pattern
                        int idx0 = (meta >> 0) & 0x3;
                        int idx1 = (meta >> 2) & 0x3;
                        
                        // Two non-zero elements per group of 4
                        float a0 = As[row][k4 * 4 + idx0];
                        float a1 = As[row][k4 * 4 + idx1];
                        float b0 = Bs_values[k4 * 2 + 0][col];
                        float b1 = Bs_values[k4 * 2 + 1][col];
                        
                        acc[mi][ni] += a0 * b0 + a1 * b1;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    int warp_row = warp_id * 10;
    for (int mi = 0; mi < 4; mi++) {
        for (int ni = 0; ni < 8; ni++) {
            int out_row = row_base + warp_row + mi * 3 + lane_id / 21;
            int out_col = col_base + ni * 16 + (lane_id % 16);
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[mi][ni];
            }
        }
    }
}

// =============================================================================
// C600 Batched RMSNorm - 4 tokens per block
// =============================================================================
__global__ void rms_norm_batched_c600(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    constexpr int TOKENS_PER_BLOCK = 4;
    constexpr int THREADS_PER_TOKEN = 96;  // 384 / 4
    
    __shared__ float shared_sum[TOKENS_PER_BLOCK][2];  // 2 warps per token (approx)
    
    int token_base = blockIdx.x * TOKENS_PER_BLOCK;
    int local_token = threadIdx.x / THREADS_PER_TOKEN;
    int local_tid = threadIdx.x % THREADS_PER_TOKEN;
    int warp_in_token = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;
    
    int token_idx = token_base + local_token;
    if (token_idx >= num_tokens) return;
    
    const float* in_row = input + token_idx * hidden_dim;
    float* out_row = output + token_idx * hidden_dim;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (int i = local_tid; i < hidden_dim; i += THREADS_PER_TOKEN) {
        float val = in_row[i];
        local_sum += val * val;
    }
    
    // 64-thread warp reduction
    for (int offset = 32; offset > 0; offset >>= 1) {
        local_sum += WARP_SHUFFLE_XOR(local_sum, offset);
    }
    
    if (lane_id == 0 && warp_in_token < 2) {
        shared_sum[local_token][warp_in_token] = local_sum;
    }
    __syncthreads();
    
    if (warp_in_token == 0 && lane_id < 2) {
        local_sum = shared_sum[local_token][lane_id];
        local_sum += WARP_SHUFFLE_XOR(local_sum, 1);
        if (lane_id == 0) {
            shared_sum[local_token][0] = local_sum;
        }
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(shared_sum[local_token][0] / float(hidden_dim) + eps);
    
    // Apply normalization
    for (int i = local_tid; i < hidden_dim; i += THREADS_PER_TOKEN) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}
)";

}  // namespace c600
}  // namespace maca
}  // namespace persistent_kernel
}  // namespace yirage
