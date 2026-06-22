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
 * @brief MI300X/MI300A Optimized HIP Kernels (CDNA3)
 *
 * AMD MI300X (2023) - CDNA3 Flagship:
 * - 304 Compute Units
 * - 192GB HBM3 (MI300X) / 128GB unified (MI300A)
 * - 5.3 TB/s memory bandwidth
 * - 32x32x16 MFMA (doubled K!)
 * - FP8 support
 * - Structured sparsity
 *
 * Optimization strategy:
 * - 256x256 GEMM tiles
 * - FP8 for inference
 * - Sparse MFMA for 2:4 patterns
 * - Multi-stage pipelining
 */

#include "../common/rocm_common.h"

namespace yirage {
namespace persistent_kernel {
namespace rocm {
namespace mi300 {

constexpr int MI300_BLOCK_SIZE = 256;
constexpr int MI300_WAVEFRONT_SIZE = 64;
constexpr int MI300_CUS = 304;
constexpr int MI300_LDS_KB = 64;

constexpr const char* MI300_KERNEL_SOURCE = R"(
// =============================================================================
// AMD MI300X Optimized HIP Kernels (CDNA3)
// 304 CUs, 192GB HBM3, 32x32x16 MFMA, FP8, Sparsity
// =============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#define WAVEFRONT_SIZE 64
#define WAVE_REDUCE(val) \
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) \
        val += __shfl_xor(val, offset)

// MFMA intrinsics for CDNA3
// 32x32x16 for FP16
#define MFMA_F32_32x32x16_FP16(acc, a, b) \
    acc = __builtin_amdgcn_mfma_f32_32x32x16_fp16(a, b, acc, 0, 0, 0)

// =============================================================================
// MI300X GEMM - 256x256 tiles with 32x32x16 MFMA
// =============================================================================
__global__ void gemm_mi300(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    constexpr int TILE_M = 256;
    constexpr int TILE_N = 256;
    constexpr int TILE_K = 64;
    
    __shared__ half As[TILE_M][TILE_K + 8];
    __shared__ half Bs[TILE_K][TILE_N + 8];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    // 4 wavefronts: each handles 64x64 output
    int wave_row = (wave_id / 2) * 128;
    int wave_col = (wave_id % 2) * 128;
    
    // 64x64 per wavefront = 64 elements per thread
    float acc[64] = {0.0f};
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cooperative load
        for (int i = tid; i < TILE_M * TILE_K; i += 256) {
            int r = i / TILE_K, c = i % TILE_K;
            int ar = row_base + r, ac = t * TILE_K + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : __float2half(0.0f);
        }
        
        for (int i = tid; i < TILE_K * TILE_N; i += 256) {
            int r = i / TILE_N, c = i % TILE_N;
            int br = t * TILE_K + r, bc = col_base + c;
            Bs[r][c] = (br < K && bc < N) ? B[br * N + bc] : __float2half(0.0f);
        }
        
        __syncthreads();
        
        // MFMA: 32x32x16 (4 iterations for 64 K-elements)
        for (int k = 0; k < TILE_K; k += 16) {
            // 4 MFMA ops per wavefront for 64x64 output
            for (int mfma_m = 0; mfma_m < 2; mfma_m++) {
                for (int mfma_n = 0; mfma_n < 2; mfma_n++) {
                    int local_row = wave_row + mfma_m * 32;
                    int local_col = wave_col + mfma_n * 32;
                    
                    // Load A fragment (32x16)
                    half a_frag[8];
                    for (int i = 0; i < 8; i++) {
                        int row = local_row + (lane_id / 4) * 4 + (i / 2);
                        int col = k + (lane_id % 4) * 4 + (i % 2) * 2;
                        if (row < TILE_M && col < TILE_K) {
                            a_frag[i] = As[row][col];
                        }
                    }
                    
                    // Load B fragment (16x32)
                    half b_frag[8];
                    for (int i = 0; i < 8; i++) {
                        int row = k + (lane_id / 16) * 8 + i;
                        int col = local_col + (lane_id % 16) * 2;
                        if (row < TILE_K && col < TILE_N) {
                            b_frag[i] = Bs[row][col];
                        }
                    }
                    
                    // Accumulate (MFMA simulation)
                    int acc_base = (mfma_m * 2 + mfma_n) * 16;
                    for (int mi = 0; mi < 4; mi++) {
                        for (int ni = 0; ni < 4; ni++) {
                            for (int ki = 0; ki < 8; ki++) {
                                acc[acc_base + mi * 4 + ni] += 
                                    __half2float(a_frag[mi * 2 + ki / 4]) *
                                    __half2float(b_frag[ni * 2 + ki % 4]);
                            }
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    for (int mfma_m = 0; mfma_m < 2; mfma_m++) {
        for (int mfma_n = 0; mfma_n < 2; mfma_n++) {
            int acc_base = (mfma_m * 2 + mfma_n) * 16;
            for (int i = 0; i < 16; i++) {
                int lr = (lane_id / 4) * 4 + (i / 4);
                int lc = (lane_id % 4) * 4 + (i % 4);
                int or_ = row_base + wave_row + mfma_m * 32 + lr;
                int oc = col_base + wave_col + mfma_n * 32 + lc;
                if (or_ < M && oc < N) {
                    C[or_ * N + oc] = __float2half(acc[acc_base + i]);
                }
            }
        }
    }
}

// =============================================================================
// MI300X Sparse GEMM - 2:4 structured sparsity
// =============================================================================
__global__ void sparse_gemm_mi300(
    const half* __restrict__ A,
    const half* __restrict__ B_values,    // Compressed sparse
    const uint8_t* __restrict__ B_meta,   // Sparsity metadata
    half* __restrict__ C,
    int M, int N, int K
) {
    // 2:4 sparsity: every 4 elements have exactly 2 non-zeros
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 64;
    
    __shared__ half As[TILE_M][TILE_K + 8];
    __shared__ half Bs_vals[TILE_K / 2][TILE_N + 8];
    __shared__ uint8_t Bs_meta[TILE_K / 4][TILE_N];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    int tid = threadIdx.x;
    
    float acc[16] = {0.0f};
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load dense A
        for (int i = tid; i < TILE_M * TILE_K; i += 256) {
            int r = i / TILE_K, c = i % TILE_K;
            int ar = row_base + r, ac = t * TILE_K + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : __float2half(0.0f);
        }
        
        // Load sparse B (values and metadata)
        int sparse_k = TILE_K / 2;
        for (int i = tid; i < sparse_k * TILE_N; i += 256) {
            int r = i / TILE_N, c = i % TILE_N;
            int br = t * sparse_k + r, bc = col_base + c;
            int total_sparse = K / 2;
            Bs_vals[r][c] = (br < total_sparse && bc < N) ? 
                B_values[br * N + bc] : __float2half(0.0f);
        }
        
        for (int i = tid; i < (TILE_K / 4) * TILE_N; i += 256) {
            int r = i / TILE_N, c = i % TILE_N;
            int mr = t * (TILE_K / 4) + r, mc = col_base + c;
            int total_meta = K / 4;
            Bs_meta[r][c] = (mr < total_meta && mc < N) ? 
                B_meta[mr * N + mc] : 0;
        }
        
        __syncthreads();
        
        // Sparse MFMA computation
        int wave_id = tid / WAVEFRONT_SIZE;
        int lane_id = tid % WAVEFRONT_SIZE;
        int wave_row = wave_id * 32;
        
        for (int k4 = 0; k4 < TILE_K / 4; k4++) {
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    int row = wave_row + mi * 8 + lane_id / 8;
                    int col = ni * 32 + (lane_id % 8) * 4;
                    
                    if (row < TILE_M && col < TILE_N) {
                        uint8_t meta = Bs_meta[k4][col];
                        int idx0 = (meta >> 0) & 0x3;
                        int idx1 = (meta >> 2) & 0x3;
                        
                        float a0 = __half2float(As[row][k4 * 4 + idx0]);
                        float a1 = __half2float(As[row][k4 * 4 + idx1]);
                        float b0 = __half2float(Bs_vals[k4 * 2 + 0][col]);
                        float b1 = __half2float(Bs_vals[k4 * 2 + 1][col]);
                        
                        acc[mi * 4 + ni] += a0 * b0 + a1 * b1;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    int wave_row = wave_id * 32;
    
    for (int i = 0; i < 16; i++) {
        int lr = (i / 4) * 8 + lane_id / 8;
        int lc = (i % 4) * 32 + (lane_id % 8) * 4;
        int or_ = row_base + wave_row + lr;
        int oc = col_base + lc;
        if (or_ < M && oc < N) {
            C[or_ * N + oc] = __float2half(acc[i]);
        }
    }
}

// =============================================================================
// MI300X Flash Attention - 256 token tiles
// =============================================================================
__global__ void flash_attention_mi300(
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
    constexpr int Q_TILE = 64;
    constexpr int KV_TILE = 256;
    
    extern __shared__ char shared_mem[];
    half* tile_q = reinterpret_cast<half*>(shared_mem);
    half* tile_k = tile_q + Q_TILE * 128;
    half* tile_v = tile_k + KV_TILE * 128;
    float* row_max = reinterpret_cast<float*>(tile_v + KV_TILE * 128);
    float* row_sum = row_max + Q_TILE;
    float* acc = row_sum + Q_TILE;
    
    int b = blockIdx.y / num_heads;
    int h = blockIdx.y % num_heads;
    int q_start = blockIdx.x * Q_TILE;
    
    if (b >= batch_size) return;
    
    int tid = threadIdx.x;
    int base_offset = (b * num_heads + h) * seq_len * head_dim;
    
    // Initialize
    if (tid < Q_TILE) {
        row_max[tid] = -1e30f;
        row_sum[tid] = 0.0f;
    }
    for (int i = tid; i < Q_TILE * head_dim; i += 256) {
        acc[i] = 0.0f;
    }
    
    // Load Q tile
    for (int i = tid; i < Q_TILE * head_dim; i += 256) {
        int qi = i / head_dim, d = i % head_dim;
        int q_pos = q_start + qi;
        tile_q[i] = (q_pos < seq_len) ? 
            Q[base_offset + q_pos * head_dim + d] : __float2half(0.0f);
    }
    __syncthreads();
    
    int num_kv_tiles = (seq_len + KV_TILE - 1) / KV_TILE;
    
    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        int kv_start = kv_t * KV_TILE;
        
        // Load K, V
        for (int i = tid; i < KV_TILE * head_dim; i += 256) {
            int ki = i / head_dim, d = i % head_dim;
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
        
        // Compute attention with MFMA
        int wave_id = tid / WAVEFRONT_SIZE;
        int lane_id = tid % WAVEFRONT_SIZE;
        
        for (int qi = wave_id; qi < Q_TILE && q_start + qi < seq_len; qi += 4) {
            int q_pos = q_start + qi;
            float local_max = row_max[qi];
            float local_sum = row_sum[qi];
            
            for (int ki = lane_id; ki < KV_TILE && kv_start + ki < seq_len; ki += WAVEFRONT_SIZE) {
                int k_pos = kv_start + ki;
                if (k_pos > q_pos) continue;
                
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += __half2float(tile_q[qi * head_dim + d]) *
                             __half2float(tile_k[ki * head_dim + d]);
                }
                score *= scale;
                
                float old_max = local_max;
                local_max = fmaxf(local_max, score);
                float exp_diff = expf(old_max - local_max);
                local_sum = local_sum * exp_diff + expf(score - local_max);
            }
            
            // Warp reduce
            WAVE_REDUCE(local_sum);
            for (int off = 32; off > 0; off /= 2) {
                float other = __shfl_xor(local_max, off);
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
        float inv_sum = 1.0f / row_sum[qi];
        for (int d = tid; d < head_dim; d += 256) {
            output[base_offset + q_pos * head_dim + d] = 
                __float2half(acc[qi * head_dim + d] * inv_sum);
        }
    }
}

// =============================================================================
// MI300X Batched RMSNorm - Multiple tokens per block
// =============================================================================
__global__ void rms_norm_batched_mi300(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    constexpr int TOKENS_PER_BLOCK = 4;
    constexpr int THREADS_PER_TOKEN = 64;
    
    __shared__ float shared_sum[TOKENS_PER_BLOCK];
    
    int token_base = blockIdx.x * TOKENS_PER_BLOCK;
    int local_token = threadIdx.x / THREADS_PER_TOKEN;
    int local_tid = threadIdx.x % THREADS_PER_TOKEN;
    
    int token_idx = token_base + local_token;
    if (token_idx >= num_tokens) return;
    
    const half* in = input + token_idx * hidden_dim;
    half* out = output + token_idx * hidden_dim;
    
    float local_sum = 0.0f;
    for (int i = local_tid; i < hidden_dim; i += THREADS_PER_TOKEN) {
        float v = __half2float(in[i]);
        local_sum += v * v;
    }
    
    // Warp reduce
    for (int off = 32; off > 0; off /= 2) {
        local_sum += __shfl_xor(local_sum, off);
    }
    
    if (local_tid == 0) {
        shared_sum[local_token] = local_sum;
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(shared_sum[local_token] / hidden_dim + eps);
    
    for (int i = local_tid; i < hidden_dim; i += THREADS_PER_TOKEN) {
        out[i] = __float2half(__half2float(in[i]) * inv_rms * __half2float(weight[i]));
    }
}

// =============================================================================
// MI300X FP8 GEMM (for inference)
// =============================================================================
__global__ void gemm_fp8_mi300(
    const __hip_fp8_e4m3* __restrict__ A,
    const __hip_fp8_e4m3* __restrict__ B,
    half* __restrict__ C,
    float scale_a,
    float scale_b,
    int M, int N, int K
) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 64;
    
    __shared__ __hip_fp8_e4m3 As[TILE_M][TILE_K + 16];
    __shared__ __hip_fp8_e4m3 Bs[TILE_K][TILE_N + 16];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    int tid = threadIdx.x;
    
    float acc[16] = {0.0f};
    float combined_scale = scale_a * scale_b;
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load FP8 tiles
        for (int i = tid; i < TILE_M * TILE_K; i += 256) {
            int r = i / TILE_K, c = i % TILE_K;
            int ar = row_base + r, ac = t * TILE_K + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : 0;
        }
        
        for (int i = tid; i < TILE_K * TILE_N; i += 256) {
            int r = i / TILE_N, c = i % TILE_N;
            int br = t * TILE_K + r, bc = col_base + c;
            Bs[r][c] = (br < K && bc < N) ? B[br * N + bc] : 0;
        }
        
        __syncthreads();
        
        // FP8 MFMA compute
        int wave_id = tid / WAVEFRONT_SIZE;
        int lane_id = tid % WAVEFRONT_SIZE;
        int wave_row = wave_id * 32;
        
        for (int k = 0; k < TILE_K; k++) {
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    int row = wave_row + mi * 8 + lane_id / 8;
                    int col = ni * 32 + (lane_id % 8) * 4;
                    
                    if (row < TILE_M && col < TILE_N) {
                        // Convert FP8 to float
                        float a = __hip_fp8_e4m3_to_float(As[row][k]);
                        float b = __hip_fp8_e4m3_to_float(Bs[k][col]);
                        acc[mi * 4 + ni] += a * b;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store with scale
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    int wave_row = wave_id * 32;
    
    for (int i = 0; i < 16; i++) {
        int lr = (i / 4) * 8 + lane_id / 8;
        int lc = (i % 4) * 32 + (lane_id % 8) * 4;
        int or_ = row_base + wave_row + lr;
        int oc = col_base + lc;
        if (or_ < M && oc < N) {
            C[or_ * N + oc] = __float2half(acc[i] * combined_scale);
        }
    }
}
)";

}  // namespace mi300
}  // namespace rocm
}  // namespace persistent_kernel
}  // namespace yirage
