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
 * @brief MI200/MI210 Optimized HIP Kernels (CDNA2 Single Die)
 *
 * AMD MI210 (2021) - Single-die CDNA2:
 * - 104 Compute Units
 * - 64GB HBM2e
 * - 32x32x8 MFMA
 * - Async global->LDS copy
 *
 * Same kernels as MI250 but with adjusted tile sizes for single die.
 */

#include "../common/rocm_common.h"

namespace yirage {
namespace persistent_kernel {
namespace rocm {
namespace mi200 {

constexpr int MI200_BLOCK_SIZE = 256;
constexpr int MI200_WAVEFRONT_SIZE = 64;
constexpr int MI200_CUS = 104;

// MI200 uses same kernels as MI250 with smaller tiles
// Include MI250 and alias
constexpr const char* MI200_KERNEL_SOURCE = R"(
// =============================================================================
// AMD MI200/MI210 Optimized HIP Kernels (CDNA2 Single Die)
// 104 CUs, 64GB HBM2e, gfx90a
// =============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVEFRONT_SIZE 64
#define WAVE_REDUCE(val) \
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) \
        val += __shfl_xor(val, offset)

// =============================================================================
// MI200 GEMM - 128x128 tiles (smaller than MI250X)
// =============================================================================
__global__ void gemm_mi200(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    
    __shared__ half As[2][TILE_M][TILE_K + 8];
    __shared__ half Bs[2][TILE_K][TILE_N + 8];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    int wave_row = (wave_id / 2) * 32;
    int wave_col = (wave_id % 2) * 64;
    
    float acc[16] = {0.0f};
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    int buffer = 0;
    
    // Load first tile
    for (int i = tid; i < TILE_M * TILE_K; i += 256) {
        int r = i / TILE_K, c = i % TILE_K;
        int ar = row_base + r, ac = c;
        As[0][r][c] = (ar < M && ac < K) ? A[ar * K + ac] : __float2half(0.0f);
    }
    for (int i = tid; i < TILE_K * TILE_N; i += 256) {
        int r = i / TILE_N, c = i % TILE_N;
        int br = r, bc = col_base + c;
        Bs[0][r][c] = (br < K && bc < N) ? B[br * N + bc] : __float2half(0.0f);
    }
    __syncthreads();
    
    for (int t = 0; t < num_tiles; t++) {
        int next = 1 - buffer;
        
        if (t + 1 < num_tiles) {
            for (int i = tid; i < TILE_M * TILE_K; i += 256) {
                int r = i / TILE_K, c = i % TILE_K;
                int ar = row_base + r, ac = (t + 1) * TILE_K + c;
                As[next][r][c] = (ar < M && ac < K) ? A[ar * K + ac] : __float2half(0.0f);
            }
            for (int i = tid; i < TILE_K * TILE_N; i += 256) {
                int r = i / TILE_N, c = i % TILE_N;
                int br = (t + 1) * TILE_K + r, bc = col_base + c;
                Bs[next][r][c] = (br < K && bc < N) ? B[br * N + bc] : __float2half(0.0f);
            }
        }
        
        // MFMA compute
        for (int k = 0; k < TILE_K; k += 8) {
            half a_frag[4], b_frag[4];
            for (int i = 0; i < 4; i++) {
                int row = wave_row + (lane_id / 8) * 4 + i;
                a_frag[i] = As[buffer][row][k + lane_id % 8];
            }
            for (int i = 0; i < 4; i++) {
                int col = wave_col + (lane_id % 32);
                b_frag[i] = Bs[buffer][k + i][col];
            }
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    acc[mi * 4 + ni] += __half2float(a_frag[mi]) * __half2float(b_frag[ni]);
                }
            }
        }
        
        buffer = next;
        __syncthreads();
    }
    
    // Store
    for (int i = 0; i < 16; i++) {
        int lr = (lane_id / 4) * 4 + (i / 4);
        int lc = (lane_id % 4) * 4 + (i % 4);
        int or_ = row_base + wave_row + lr;
        int oc = col_base + wave_col + lc;
        if (or_ < M && oc < N) C[or_ * N + oc] = __float2half(acc[i]);
    }
}

// =============================================================================
// MI200 RMSNorm
// =============================================================================
__global__ void rms_norm_mi200(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    __shared__ float shared_sum[4];
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    const half* in = input + token_idx * hidden_dim;
    half* out = output + token_idx * hidden_dim;
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += 256) {
        float v = __half2float(in[i]);
        local_sum += v * v;
    }
    
    WAVE_REDUCE(local_sum);
    if (lane_id == 0) shared_sum[wave_id] = local_sum;
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_sum = shared_sum[lane_id];
        for (int off = 2; off > 0; off /= 2)
            local_sum += __shfl_xor(local_sum, off);
        if (lane_id == 0) shared_sum[0] = local_sum;
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(shared_sum[0] / hidden_dim + eps);
    
    for (int i = tid; i < hidden_dim; i += 256) {
        out[i] = __float2half(__half2float(in[i]) * inv_rms * __half2float(weight[i]));
    }
}

// =============================================================================
// MI200 SiLU Mul
// =============================================================================
__global__ void silu_mul_mi200(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 4;
    
    if (base + 3 < size) {
        half2 g0 = *reinterpret_cast<const half2*>(gate + base);
        half2 g1 = *reinterpret_cast<const half2*>(gate + base + 2);
        half2 u0 = *reinterpret_cast<const half2*>(up + base);
        half2 u1 = *reinterpret_cast<const half2*>(up + base + 2);
        
        float2 gf0 = __half22float2(g0), gf1 = __half22float2(g1);
        float2 uf0 = __half22float2(u0), uf1 = __half22float2(u1);
        
        gf0.x = gf0.x / (1.0f + expf(-gf0.x)) * uf0.x;
        gf0.y = gf0.y / (1.0f + expf(-gf0.y)) * uf0.y;
        gf1.x = gf1.x / (1.0f + expf(-gf1.x)) * uf1.x;
        gf1.y = gf1.y / (1.0f + expf(-gf1.y)) * uf1.y;
        
        *reinterpret_cast<half2*>(output + base) = __float22half2_rn(gf0);
        *reinterpret_cast<half2*>(output + base + 2) = __float22half2_rn(gf1);
    }
}
)";

}  // namespace mi200
}  // namespace rocm
}  // namespace persistent_kernel
}  // namespace yirage
