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
 * @brief M2 Optimized Metal Kernels (Apple Silicon Gen 2)
 *
 * M2 (2022) characteristics:
 * - 10 GPU cores (Base), 19 (Pro), 38 (Max)
 * - Improved memory bandwidth (100-400 GB/s)
 * - Better power efficiency
 * - Enhanced media engine
 * - Still no hardware ray tracing
 *
 * Optimization strategy:
 * - Larger tiles for better memory utilization
 * - Improved instruction scheduling
 * - Better half-precision support
 */

#include "../common/mps_common.h"

namespace yirage {
namespace persistent_kernel {
namespace mps {
namespace m2 {

// M2 specific constants
constexpr int M2_THREADGROUP_SIZE = 256;
constexpr int M2_SIMD_WIDTH = 32;
constexpr int M2_TILE_SIZE_M = 32;
constexpr int M2_TILE_SIZE_N = 64;  // Wider tiles for better BW util
constexpr int M2_SHARED_MEM_KB = 32;

constexpr const char* M2_KERNEL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// M2 Optimized RMSNorm - Better memory prefetching
// =============================================================================
kernel void rms_norm_m2(
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
    constexpr int THREADS_PER_TG = 256;
    threadgroup float shared_sum[8];
    
    int token_idx = tgid;
    if (token_idx >= num_tokens) return;
    
    device const float* in_row = input + token_idx * hidden_dim;
    device float* out_row = output + token_idx * hidden_dim;
    
    // M2: Use float4 for better memory throughput
    float local_sum = 0.0f;
    int vec_elements = hidden_dim / 4;
    int thread_vec_start = tid_in_tg * (vec_elements / THREADS_PER_TG);
    int thread_vec_count = (vec_elements + THREADS_PER_TG - 1) / THREADS_PER_TG;
    
    for (int vi = 0; vi < thread_vec_count && thread_vec_start + vi < vec_elements; vi++) {
        int idx = (thread_vec_start + vi) * 4;
        if (idx + 3 < hidden_dim) {
            float4 vals = *((device const float4*)(in_row + idx));
            local_sum += vals.x * vals.x + vals.y * vals.y + 
                        vals.z * vals.z + vals.w * vals.w;
        }
    }
    
    // Handle remaining elements
    for (int i = vec_elements * 4 + tid_in_tg; i < hidden_dim; i += THREADS_PER_TG) {
        float val = in_row[i];
        local_sum += val * val;
    }
    
    // SIMD reduction
    local_sum += simd_shuffle_xor(local_sum, 16);
    local_sum += simd_shuffle_xor(local_sum, 8);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 1);
    
    if (simd_lane == 0) shared_sum[simd_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum += simd_shuffle_xor(local_sum, 4);
        local_sum += simd_shuffle_xor(local_sum, 2);
        local_sum += simd_shuffle_xor(local_sum, 1);
        if (simd_lane == 0) shared_sum[0] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms = rsqrt(shared_sum[0] / float(hidden_dim) + eps);
    
    // Vectorized output with float4
    for (int vi = 0; vi < thread_vec_count && thread_vec_start + vi < vec_elements; vi++) {
        int idx = (thread_vec_start + vi) * 4;
        if (idx + 3 < hidden_dim) {
            float4 in_vals = *((device const float4*)(in_row + idx));
            float4 w_vals = *((device const float4*)(weight + idx));
            float4 out_vals = in_vals * inv_rms * w_vals;
            *((device float4*)(out_row + idx)) = out_vals;
        }
    }
}

// =============================================================================
// M2 Optimized GEMM - 32x64 tiles for better bandwidth
// =============================================================================
kernel void gemm_m2(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 32;
    
    threadgroup float As[TILE_M][TILE_K];
    threadgroup float Bs[TILE_K][TILE_N];
    
    int row_base = tgid.y * TILE_M;
    int col_base = tgid.x * TILE_N;
    
    // Each thread computes a 2x2 block
    int local_row = (tid_in_tg.y / 2) * 2;
    int local_col = (tid_in_tg.x / 2) * 4 + (tid_in_tg.y % 2) * 2;
    
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cooperative load
        int a_row = row_base + tid_in_tg.y;
        int a_col = t * TILE_K + tid_in_tg.x;
        As[tid_in_tg.y][tid_in_tg.x] = (a_row < M && a_col < K) ? 
            A[a_row * K + a_col] : 0.0f;
        
        int b_row = t * TILE_K + tid_in_tg.y;
        int b_col = col_base + tid_in_tg.x;
        Bs[tid_in_tg.y][tid_in_tg.x] = (b_row < K && b_col < N) ? 
            B[b_row * N + b_col] : 0.0f;
        Bs[tid_in_tg.y][tid_in_tg.x + 32] = (b_row < K && b_col + 32 < N) ? 
            B[b_row * N + b_col + 32] : 0.0f;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute 2x2 block
        for (int k = 0; k < TILE_K; k++) {
            float a0 = As[local_row][k];
            float a1 = As[local_row + 1][k];
            float b0 = Bs[k][local_col];
            float b1 = Bs[k][local_col + 1];
            
            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store results
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int out_row = row_base + local_row + i;
            int out_col = col_base + local_col + j;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[i][j];
            }
        }
    }
}

// =============================================================================
// M2 Half-precision support
// =============================================================================
kernel void rms_norm_m2_half(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr int THREADS_PER_TG = 256;
    threadgroup float shared_sum[8];
    
    int token_idx = tgid;
    if (token_idx >= num_tokens) return;
    
    device const half* in_row = input + token_idx * hidden_dim;
    device half* out_row = output + token_idx * hidden_dim;
    
    // Use half8 for maximum throughput
    float local_sum = 0.0f;
    for (int i = tid_in_tg * 8; i < hidden_dim; i += THREADS_PER_TG * 8) {
        if (i + 7 < hidden_dim) {
            half8 vals = *((device const half8*)(in_row + i));
            local_sum += float(vals[0]) * float(vals[0]) + 
                        float(vals[1]) * float(vals[1]) +
                        float(vals[2]) * float(vals[2]) + 
                        float(vals[3]) * float(vals[3]) +
                        float(vals[4]) * float(vals[4]) + 
                        float(vals[5]) * float(vals[5]) +
                        float(vals[6]) * float(vals[6]) + 
                        float(vals[7]) * float(vals[7]);
        }
    }
    
    // Reduction (same as float version)
    local_sum += simd_shuffle_xor(local_sum, 16);
    local_sum += simd_shuffle_xor(local_sum, 8);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 1);
    
    if (simd_lane == 0) shared_sum[simd_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum += simd_shuffle_xor(local_sum, 4);
        local_sum += simd_shuffle_xor(local_sum, 2);
        local_sum += simd_shuffle_xor(local_sum, 1);
        if (simd_lane == 0) shared_sum[0] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_rms = rsqrt(shared_sum[0] / float(hidden_dim) + eps);
    
    // Output with half8
    for (int i = tid_in_tg * 8; i < hidden_dim; i += THREADS_PER_TG * 8) {
        if (i + 7 < hidden_dim) {
            half8 in_vals = *((device const half8*)(in_row + i));
            half8 w_vals = *((device const half8*)(weight + i));
            half8 out_vals;
            for (int j = 0; j < 8; j++) {
                out_vals[j] = half(float(in_vals[j]) * inv_rms * float(w_vals[j]));
            }
            *((device half8*)(out_row + i)) = out_vals;
        }
    }
}
)";

}  // namespace m2
}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
