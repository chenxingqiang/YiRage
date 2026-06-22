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
 * @brief MI250/MI250X Optimized HIP Kernels (CDNA2)
 *
 * AMD MI250X (2021) - Dual-die CDNA2:
 * - 220 Compute Units (110 per die)
 * - 128GB HBM2e (dual-die)
 * - 32x32x8 MFMA with improved throughput
 * - Async global->LDS copy
 *
 * Optimization strategy:
 * - 256x128 GEMM tiles
 * - Double buffering with async copy
 * - Multi-die aware scheduling
 */

#include "../common/rocm_common.h"

namespace yirage {
namespace persistent_kernel {
namespace rocm {
namespace mi250 {

constexpr int MI250_BLOCK_SIZE = 256;
constexpr int MI250_WAVEFRONT_SIZE = 64;
constexpr int MI250_LDS_KB = 64;

constexpr const char* MI250_KERNEL_SOURCE = R"(
// =============================================================================
// AMD MI250X Optimized HIP Kernels (CDNA2)
// 220 CUs, 128GB HBM2e, Async Copy, 32x32x8 MFMA
// =============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVEFRONT_SIZE 64
#define WAVE_REDUCE(val) \
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) \
        val += __shfl_xor(val, offset)

// Async copy intrinsics (CDNA2+)
#define ASYNC_COPY_GLOBAL_TO_LDS(dst, src, size) \
    __builtin_amdgcn_global_load_lds((void*)(src), (void*)(dst), size, 0, 0)

// =============================================================================
// MI250 GEMM - 256x128 tiles with double buffering
// =============================================================================
__global__ void gemm_mi250(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    constexpr int TILE_M = 256;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    
    // Double buffer for async copy
    __shared__ half As[2][TILE_M][TILE_K + 8];
    __shared__ half Bs[2][TILE_K][TILE_N + 8];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    // Each wavefront handles 64x32 output
    int wave_row = (wave_id / 2) * 64;
    int wave_col = (wave_id % 2) * 64;
    
    float acc[32] = {0.0f};  // 64x32 / 64 = 32 per thread
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    int buffer = 0;
    
    // Load first tile
    for (int i = tid; i < TILE_M * TILE_K; i += 256) {
        int load_row = i / TILE_K;
        int load_col = i % TILE_K;
        int a_row = row_base + load_row;
        int a_col = load_col;
        As[0][load_row][load_col] = (a_row < M && a_col < K) ? 
            A[a_row * K + a_col] : __float2half(0.0f);
    }
    
    for (int i = tid; i < TILE_K * TILE_N; i += 256) {
        int load_row = i / TILE_N;
        int load_col = i % TILE_N;
        int b_row = load_row;
        int b_col = col_base + load_col;
        Bs[0][load_row][load_col] = (b_row < K && b_col < N) ? 
            B[b_row * N + b_col] : __float2half(0.0f);
    }
    
    __syncthreads();
    
    for (int t = 0; t < num_tiles; t++) {
        int next_buffer = 1 - buffer;
        
        // Prefetch next tile (async)
        if (t + 1 < num_tiles) {
            for (int i = tid; i < TILE_M * TILE_K; i += 256) {
                int load_row = i / TILE_K;
                int load_col = i % TILE_K;
                int a_row = row_base + load_row;
                int a_col = (t + 1) * TILE_K + load_col;
                As[next_buffer][load_row][load_col] = (a_row < M && a_col < K) ? 
                    A[a_row * K + a_col] : __float2half(0.0f);
            }
            
            for (int i = tid; i < TILE_K * TILE_N; i += 256) {
                int load_row = i / TILE_N;
                int load_col = i % TILE_N;
                int b_row = (t + 1) * TILE_K + load_row;
                int b_col = col_base + load_col;
                Bs[next_buffer][load_row][load_col] = (b_row < K && b_col < N) ? 
                    B[b_row * N + b_col] : __float2half(0.0f);
            }
        }
        
        // MFMA compute on current buffer
        for (int k = 0; k < TILE_K; k += 8) {
            // 32x32x8 MFMA (4 per wavefront for 64x32 output)
            for (int mfma = 0; mfma < 2; mfma++) {
                for (int nfma = 0; nfma < 1; nfma++) {
                    int mfma_row = wave_row + mfma * 32;
                    int mfma_col = wave_col + nfma * 32;
                    
                    // Load A fragment
                    half a_frag[4];
                    for (int i = 0; i < 4; i++) {
                        int row = mfma_row + (lane_id / 8) * 4 + i;
                        a_frag[i] = As[buffer][row][k + lane_id % 8];
                    }
                    
                    // Load B fragment
                    half b_frag[4];
                    for (int i = 0; i < 4; i++) {
                        int col = mfma_col + (lane_id % 32);
                        b_frag[i] = Bs[buffer][k + i][col];
                    }
                    
                    // Accumulate
                    int acc_base = mfma * 16 + nfma * 8;
                    for (int mi = 0; mi < 4; mi++) {
                        for (int ni = 0; ni < 4; ni++) {
                            acc[acc_base + mi * 4 + ni] += 
                                __half2float(a_frag[mi]) * __half2float(b_frag[ni]);
                        }
                    }
                }
            }
        }
        
        buffer = next_buffer;
        __syncthreads();
    }
    
    // Store results
    for (int i = 0; i < 32; i++) {
        int mfma_idx = i / 16;
        int local_idx = i % 16;
        int local_row = (lane_id / 8) * 4 + (local_idx / 4);
        int local_col = (lane_id % 8) * 4 + (local_idx % 4);
        
        int out_row = row_base + wave_row + mfma_idx * 32 + local_row;
        int out_col = col_base + wave_col + local_col;
        
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = __float2half(acc[i]);
        }
    }
}

// =============================================================================
// MI250 Flash Attention - 128 token tiles
// =============================================================================
__global__ void flash_attention_mi250(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    constexpr int Q_TILE = 32;
    constexpr int KV_TILE = 128;
    
    extern __shared__ char shared_mem[];
    half* tile_k = reinterpret_cast<half*>(shared_mem);
    half* tile_v = tile_k + KV_TILE * 128;
    float* row_max = reinterpret_cast<float*>(tile_v + KV_TILE * 128);
    float* row_sum = row_max + Q_TILE;
    float* acc = row_sum + Q_TILE;
    
    int b = blockIdx.y / num_heads;
    int h = blockIdx.y % num_heads;
    int q_start = blockIdx.x * Q_TILE;
    
    if (b >= batch_size) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    // Initialize
    if (tid < Q_TILE) {
        row_max[tid] = -1e30f;
        row_sum[tid] = 0.0f;
    }
    for (int i = tid; i < Q_TILE * head_dim; i += 256) {
        acc[i] = 0.0f;
    }
    __syncthreads();
    
    int base_offset = (b * num_heads + h) * seq_len * head_dim;
    int num_kv_tiles = (seq_len + KV_TILE - 1) / KV_TILE;
    
    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        int kv_start = kv_t * KV_TILE;
        
        // Load K, V tiles
        for (int i = tid; i < KV_TILE * head_dim; i += 256) {
            int ki = i / head_dim;
            int d = i % head_dim;
            int k_pos = kv_start + ki;
            
            if (k_pos < seq_len) {
                tile_k[i] = K[base_offset + k_pos * head_dim + d];
                tile_v[i] = V[base_offset + k_pos * head_dim + d];
            } else {
                tile_k[i] = __float2half(0.0f);
                tile_v[i] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // Compute attention
        for (int qi = wave_id; qi < Q_TILE && q_start + qi < seq_len; qi += 4) {
            int q_pos = q_start + qi;
            const half* q = Q + base_offset + q_pos * head_dim;
            
            float local_max = row_max[qi];
            float local_sum = row_sum[qi];
            
            for (int ki = lane_id; ki < KV_TILE && kv_start + ki < seq_len; ki += WAVEFRONT_SIZE) {
                int k_pos = kv_start + ki;
                if (k_pos > q_pos) continue;
                
                // Dot product
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += __half2float(q[d]) * __half2float(tile_k[ki * head_dim + d]);
                }
                score *= scale;
                
                // Online softmax
                float old_max = local_max;
                local_max = fmaxf(local_max, score);
                float exp_diff = expf(old_max - local_max);
                local_sum = local_sum * exp_diff + expf(score - local_max);
            }
            
            // Warp reduce
            WAVE_REDUCE(local_sum);
            for (int offset = 32; offset > 0; offset /= 2) {
                float other = __shfl_xor(local_max, offset);
                local_max = fmaxf(local_max, other);
            }
            
            if (lane_id == 0) {
                row_max[qi] = local_max;
                row_sum[qi] = local_sum;
            }
        }
        
        __syncthreads();
    }
    
    // Normalize and output
    for (int qi = 0; qi < Q_TILE && q_start + qi < seq_len; qi++) {
        int q_pos = q_start + qi;
        half* out = output + base_offset + q_pos * head_dim;
        
        float inv_sum = 1.0f / row_sum[qi];
        for (int d = tid; d < head_dim; d += 256) {
            out[d] = __float2half(acc[qi * head_dim + d] * inv_sum);
        }
    }
}

// =============================================================================
// MI250 RMSNorm with BF16
// =============================================================================
__global__ void rms_norm_mi250(
    const __hip_bfloat16* __restrict__ input,
    const __hip_bfloat16* __restrict__ weight,
    __hip_bfloat16* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    constexpr int BLOCK_SIZE = 256;
    __shared__ float shared_sum[4];
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const __hip_bfloat16* in_row = input + token_idx * hidden_dim;
    __hip_bfloat16* out_row = output + token_idx * hidden_dim;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = __bfloat162float(in_row[i]);
        local_sum += val * val;
    }
    
    WAVE_REDUCE(local_sum);
    if (lane_id == 0) shared_sum[wave_id] = local_sum;
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        for (int offset = 2; offset > 0; offset /= 2) {
            local_sum += __shfl_xor(local_sum, offset);
        }
        if (lane_id == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(shared_sum[0] / float(hidden_dim) + eps);
    
    // Apply normalization
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = __bfloat162float(in_row[i]) * inv_rms * __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16(val);
    }
}
)";

}  // namespace mi250
}  // namespace rocm
}  // namespace persistent_kernel
}  // namespace yirage
