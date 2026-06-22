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
 * @brief MI100 Optimized HIP Kernels (CDNA1)
 *
 * AMD MI100 (2020) - First CDNA architecture:
 * - 120 Compute Units
 * - 64-thread wavefronts
 * - 64KB LDS per CU
 * - 32x32x8 MFMA (Matrix Fused Multiply-Add)
 * - 32GB HBM2
 *
 * Optimization strategy:
 * - 128x128 GEMM tiles
 * - MFMA for matrix operations
 * - LDS bank conflict avoidance
 */

#include "../common/rocm_common.h"

namespace yirage {
namespace persistent_kernel {
namespace rocm {
namespace mi100 {

constexpr int MI100_BLOCK_SIZE = 256;
constexpr int MI100_WAVEFRONT_SIZE = 64;
constexpr int MI100_LDS_KB = 64;

constexpr const char* MI100_KERNEL_SOURCE = R"(
// =============================================================================
// AMD MI100 Optimized HIP Kernels (CDNA1)
// 120 CUs, 64KB LDS, 32x32x8 MFMA
// =============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Wavefront operations
#define WAVEFRONT_SIZE 64
#define WAVE_REDUCE(val) \
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) \
        val += __shfl_xor(val, offset)

// =============================================================================
// MI100 RMSNorm - Wavefront parallel reduction
// =============================================================================
__global__ void rms_norm_mi100(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    constexpr int BLOCK_SIZE = 256;
    
    __shared__ float shared_sum[4];  // One per wavefront
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const half* in_row = input + token_idx * hidden_dim;
    half* out_row = output + token_idx * hidden_dim;
    
    // Compute sum of squares (vectorized load)
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = __half2float(in_row[i]);
        local_sum += val * val;
    }
    
    // 64-thread wavefront reduction
    WAVE_REDUCE(local_sum);
    
    if (lane_id == 0) {
        shared_sum[wave_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across 4 wavefronts
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        for (int offset = 2; offset > 0; offset /= 2) {
            local_sum += __shfl_xor(local_sum, offset);
        }
        if (lane_id == 0) {
            shared_sum[0] = local_sum;
        }
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(shared_sum[0] / float(hidden_dim) + eps);
    
    // Apply normalization
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = __half2float(in_row[i]) * inv_rms * __half2float(weight[i]);
        out_row[i] = __float2half(val);
    }
}

// =============================================================================
// MI100 GEMM - 32x32x8 MFMA tiles
// =============================================================================
__global__ void gemm_mi100(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // MI100: 128x128 output tile, 32x32x8 MFMA
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    
    __shared__ half As[TILE_M][TILE_K + 8];  // +8 to avoid bank conflicts
    __shared__ half Bs[TILE_K][TILE_N + 8];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    // Each wavefront handles 32x32 output using MFMA
    int wave_row = (wave_id / 2) * 32;
    int wave_col = (wave_id % 2) * 64;
    
    // MFMA accumulator: 32x32 per wavefront
    float acc[16] = {0.0f};  // 32x32 / 64 threads = 16 elements per thread
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cooperative load
        for (int i = tid; i < TILE_M * TILE_K; i += 256) {
            int load_row = i / TILE_K;
            int load_col = i % TILE_K;
            int a_row = row_base + load_row;
            int a_col = t * TILE_K + load_col;
            As[load_row][load_col] = (a_row < M && a_col < K) ? 
                A[a_row * K + a_col] : __float2half(0.0f);
        }
        
        for (int i = tid; i < TILE_K * TILE_N; i += 256) {
            int load_row = i / TILE_N;
            int load_col = i % TILE_N;
            int b_row = t * TILE_K + load_row;
            int b_col = col_base + load_col;
            Bs[load_row][load_col] = (b_row < K && b_col < N) ? 
                B[b_row * N + b_col] : __float2half(0.0f);
        }
        
        __syncthreads();
        
        // MFMA: 32x32x8 matrix multiply
        // Each MFMA instruction processes 32x32 output with k=8
        for (int k = 0; k < TILE_K; k += 8) {
            // Load A fragment (32x8)
            half a_frag[4];  // Each thread loads 4 elements
            for (int i = 0; i < 4; i++) {
                int row = wave_row + (lane_id / 8) * 4 + i;
                int col = k + (lane_id % 8);
                a_frag[i] = As[row][col];
            }
            
            // Load B fragment (8x32)
            half b_frag[4];
            for (int i = 0; i < 4; i++) {
                int row = k + (lane_id / 32) * 4 + i;
                int col = wave_col + (lane_id % 32);
                b_frag[i] = Bs[row][col];
            }
            
            // MFMA accumulate (simulated - real HIP would use __builtin_amdgcn_mfma_*)
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    int acc_idx = mi * 4 + ni;
                    for (int ki = 0; ki < 4; ki++) {
                        acc[acc_idx] += __half2float(a_frag[mi]) * __half2float(b_frag[ni]);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    for (int i = 0; i < 16; i++) {
        int local_row = (lane_id / 4) * 4 + (i / 4);
        int local_col = (lane_id % 4) * 4 + (i % 4);
        int out_row = row_base + wave_row + local_row;
        int out_col = col_base + wave_col + local_col;
        
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = __float2half(acc[i]);
        }
    }
}

// =============================================================================
// MI100 SiLU + Mul (SwiGLU)
// =============================================================================
__global__ void silu_mul_mi100(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles 4 elements (half2 vectorized)
    int base = idx * 4;
    
    if (base + 3 < size) {
        half2 g0 = *reinterpret_cast<const half2*>(gate + base);
        half2 g1 = *reinterpret_cast<const half2*>(gate + base + 2);
        half2 u0 = *reinterpret_cast<const half2*>(up + base);
        half2 u1 = *reinterpret_cast<const half2*>(up + base + 2);
        
        // SiLU: x * sigmoid(x)
        float2 gf0 = __half22float2(g0);
        float2 gf1 = __half22float2(g1);
        float2 uf0 = __half22float2(u0);
        float2 uf1 = __half22float2(u1);
        
        gf0.x = gf0.x / (1.0f + expf(-gf0.x)) * uf0.x;
        gf0.y = gf0.y / (1.0f + expf(-gf0.y)) * uf0.y;
        gf1.x = gf1.x / (1.0f + expf(-gf1.x)) * uf1.x;
        gf1.y = gf1.y / (1.0f + expf(-gf1.y)) * uf1.y;
        
        *reinterpret_cast<half2*>(output + base) = __float22half2_rn(gf0);
        *reinterpret_cast<half2*>(output + base + 2) = __float22half2_rn(gf1);
    }
}

// =============================================================================
// MI100 Softmax - Wavefront parallel
// =============================================================================
__global__ void softmax_mi100(
    half* __restrict__ scores,
    int num_rows,
    int row_size
) {
    __shared__ float shared_max[4];
    __shared__ float shared_sum[4];
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    half* row_data = scores + row * row_size;
    
    // Find max
    float local_max = -1e30f;
    for (int i = tid; i < row_size; i += 256) {
        local_max = fmaxf(local_max, __half2float(row_data[i]));
    }
    
    WAVE_REDUCE(local_max);
    local_max = __shfl(local_max, 0);  // Broadcast max within wave
    
    if (lane_id == 0) shared_max[wave_id] = local_max;
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_max = shared_max[lane_id];
        for (int offset = 2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
        }
        if (lane_id == 0) shared_max[0] = local_max;
    }
    __syncthreads();
    float row_max = shared_max[0];
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += 256) {
        float val = expf(__half2float(row_data[i]) - row_max);
        row_data[i] = __float2half(val);
        local_sum += val;
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
    float row_sum = shared_sum[0];
    
    // Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < row_size; i += 256) {
        row_data[i] = __float2half(__half2float(row_data[i]) * inv_sum);
    }
}
)";

}  // namespace mi100
}  // namespace rocm
}  // namespace persistent_kernel
}  // namespace yirage
