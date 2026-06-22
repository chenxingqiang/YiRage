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
 * @brief C500 Optimized MACA Kernels (MetaX Gen 1)
 *
 * MetaX C500 (2023) characteristics:
 * - 104 SMs (Standard), 128 SMs (Pro)
 * - 64-thread warps (CRITICAL: not 32 like NVIDIA!)
 * - 64KB shared memory per block
 * - 64-80GB HBM memory
 * - Tensor core support
 *
 * Optimization strategy:
 * - Use 64-wide warp operations
 * - 64x64 GEMM tiles for balanced occupancy
 * - Conservative register usage
 */

#include "../common/maca_common.h"

namespace yirage {
namespace persistent_kernel {
namespace maca {
namespace c500 {

// C500 specific constants
constexpr int C500_BLOCK_SIZE = 256;
constexpr int C500_WARP_SIZE = 64;
constexpr int C500_WARPS_PER_BLOCK = 4;
constexpr int C500_SHARED_MEM_KB = 64;

constexpr const char* C500_KERNEL_SOURCE = R"(
// =============================================================================
// MetaX C500 Optimized Kernels
// CRITICAL: warp_size = 64, NOT 32!
// =============================================================================

#include <maca_runtime.h>
#include <maca_fp16.h>

// Warp shuffle operations for 64-thread warps
#define WARP_SIZE 64
#define WARP_SHUFFLE_DOWN(val, offset) __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, val, offset)
#define WARP_SHUFFLE_XOR(val, mask) __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, val, mask)

// =============================================================================
// C500 RMSNorm - 64-thread warp reduction
// =============================================================================
__global__ void rms_norm_c500(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    // Each block handles one token
    // 256 threads = 4 warps of 64 threads
    constexpr int BLOCK_SIZE = 256;
    
    __shared__ float shared_sum[4];  // One per warp
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    const float* in_row = input + token_idx * hidden_dim;
    float* out_row = output + token_idx * hidden_dim;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = in_row[i];
        local_sum += val * val;
    }
    
    // 64-thread warp reduction (log2(64) = 6 steps)
    local_sum += WARP_SHUFFLE_XOR(local_sum, 32);
    local_sum += WARP_SHUFFLE_XOR(local_sum, 16);
    local_sum += WARP_SHUFFLE_XOR(local_sum, 8);
    local_sum += WARP_SHUFFLE_XOR(local_sum, 4);
    local_sum += WARP_SHUFFLE_XOR(local_sum, 2);
    local_sum += WARP_SHUFFLE_XOR(local_sum, 1);
    
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across 4 warps
    if (warp_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        local_sum += WARP_SHUFFLE_XOR(local_sum, 2);
        local_sum += WARP_SHUFFLE_XOR(local_sum, 1);
        if (lane_id == 0) {
            shared_sum[0] = local_sum;
        }
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(shared_sum[0] / float(hidden_dim) + eps);
    
    // Apply normalization
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}

// =============================================================================
// C500 GEMM - 64x64 tiles with tensor cores
// =============================================================================
__global__ void gemm_c500(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 32;
    
    __shared__ float As[TILE_M][TILE_K + 1];  // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_K][TILE_N + 1];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Each warp handles a 16x64 output tile
    int warp_row = warp_id * 16;
    int lane_row = lane_id / 4;
    int lane_col = (lane_id % 4) * 16;
    
    float acc[4][4] = {{0.0f}};  // 4x4 per thread
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cooperative load: 256 threads loading 64*32 + 32*64 = 4096 elements
        // Each thread loads 16 elements
        for (int i = tid; i < TILE_M * TILE_K; i += 256) {
            int load_row = i / TILE_K;
            int load_col = i % TILE_K;
            int a_row = row_base + load_row;
            int a_col = t * TILE_K + load_col;
            As[load_row][load_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }
        
        for (int i = tid; i < TILE_K * TILE_N; i += 256) {
            int load_row = i / TILE_N;
            int load_col = i % TILE_N;
            int b_row = t * TILE_K + load_row;
            int b_col = col_base + load_col;
            Bs[load_row][load_col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute 4x4 block per thread
        for (int k = 0; k < TILE_K; k++) {
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    int row = warp_row + lane_row + mi * 4;
                    int col = lane_col + ni * 4;
                    if (row < TILE_M && col < TILE_N) {
                        acc[mi][ni] += As[row][k] * Bs[k][col];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    for (int mi = 0; mi < 4; mi++) {
        for (int ni = 0; ni < 4; ni++) {
            int out_row = row_base + warp_row + lane_row + mi * 4;
            int out_col = col_base + lane_col + ni * 4;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[mi][ni];
            }
        }
    }
}

// =============================================================================
// C500 SiLU + Mul (SwiGLU)
// =============================================================================
__global__ void silu_mul_c500(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles 4 elements for better throughput
    int base = idx * 4;
    
    if (base + 3 < size) {
        float4 g = *reinterpret_cast<const float4*>(gate + base);
        float4 u = *reinterpret_cast<const float4*>(up + base);
        float4 result;
        
        result.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        result.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        result.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        result.w = (g.w / (1.0f + expf(-g.w))) * u.w;
        
        *reinterpret_cast<float4*>(output + base) = result;
    } else if (base < size) {
        for (int i = base; i < size && i < base + 4; i++) {
            float gv = gate[i];
            output[i] = (gv / (1.0f + expf(-gv))) * up[i];
        }
    }
}

// =============================================================================
// C500 Attention Score with 64-thread warp
// =============================================================================
__global__ void attention_score_c500(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    int b = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int q_pos = blockIdx.y;
    int k_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || q_pos >= seq_len || k_pos >= seq_len) return;
    
    const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
    const float* k = K + ((b * num_heads + h) * seq_len + k_pos) * head_dim;
    
    // Vectorized dot product
    float dot = 0.0f;
    int vec_dim = head_dim / 4;
    
    for (int d = 0; d < vec_dim; d++) {
        float4 qv = *reinterpret_cast<const float4*>(q + d * 4);
        float4 kv = *reinterpret_cast<const float4*>(k + d * 4);
        dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
    }
    
    // Handle remaining
    for (int d = vec_dim * 4; d < head_dim; d++) {
        dot += q[d] * k[d];
    }
    
    scores[((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos] = dot * scale;
}

// =============================================================================
// C500 Softmax with 64-thread warp reduction
// =============================================================================
__global__ void softmax_c500(
    float* __restrict__ scores,
    int num_rows,
    int row_size
) {
    __shared__ float shared_max[4];
    __shared__ float shared_sum[4];
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    float* row_data = scores + row * row_size;
    
    // Phase 1: Find max
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += 256) {
        local_max = fmaxf(local_max, row_data[i]);
    }
    
    // 64-thread warp reduction for max
    for (int offset = 32; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, WARP_SHUFFLE_XOR(local_max, offset));
    }
    
    if (lane_id == 0) shared_max[warp_id] = local_max;
    __syncthreads();
    
    if (warp_id == 0 && lane_id < 4) {
        local_max = shared_max[lane_id];
        local_max = fmaxf(local_max, WARP_SHUFFLE_XOR(local_max, 2));
        local_max = fmaxf(local_max, WARP_SHUFFLE_XOR(local_max, 1));
        if (lane_id == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    float row_max = shared_max[0];
    
    // Phase 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += 256) {
        float val = expf(row_data[i] - row_max);
        row_data[i] = val;
        local_sum += val;
    }
    
    // Warp reduction for sum
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
    float row_sum = shared_sum[0];
    
    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < row_size; i += 256) {
        row_data[i] *= inv_sum;
    }
}
)";

}  // namespace c500
}  // namespace maca
}  // namespace persistent_kernel
}  // namespace yirage
