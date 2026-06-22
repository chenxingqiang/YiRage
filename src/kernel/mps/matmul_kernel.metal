/* Copyright 2025 Chen Xingqiang (YiRage Project)
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
 *
 * Metal Matrix Multiplication Kernels for Apple Silicon
 * 
 * Optimized for M1/M2/M3/M4/M5 GPU architectures.
 * Uses threadgroup memory (shared memory) and SIMD operations.
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Constants
// =============================================================================

constant int TILE_SIZE = 32;
constant int SIMDGROUP_SIZE = 32;

// =============================================================================
// Helper Structures
// =============================================================================

struct MatmulParams {
    uint M;
    uint N;
    uint K;
    float alpha;
    float beta;
};

// =============================================================================
// Basic GEMM (for small matrices or warmup)
// =============================================================================

kernel void gemm_basic(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= params.M || col >= params.N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        sum += A[row * params.K + k] * B[k * params.N + col];
    }
    
    C[row * params.N + col] = params.alpha * sum + params.beta * C[row * params.N + col];
}

// =============================================================================
// Tiled GEMM with Threadgroup Memory
// =============================================================================

kernel void gemm_tiled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_id [[threadgroup_position_in_grid]]
) {
    // Threadgroup memory for tiles
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    uint row = tg_id.y * TILE_SIZE + tid.y;
    uint col = tg_id.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;
    uint num_tiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < num_tiles; t++) {
        // Load A tile
        uint a_col = t * TILE_SIZE + tid.x;
        if (row < params.M && a_col < params.K) {
            As[tid.y][tid.x] = A[row * params.K + a_col];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }
        
        // Load B tile
        uint b_row = t * TILE_SIZE + tid.y;
        if (b_row < params.K && col < params.N) {
            Bs[tid.y][tid.x] = B[b_row * params.N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < params.M && col < params.N) {
        C[row * params.N + col] = params.alpha * sum + params.beta * C[row * params.N + col];
    }
}

// =============================================================================
// FP16 Tiled GEMM (optimized for Apple Silicon)
// =============================================================================

kernel void gemm_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup half As[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    threadgroup half Bs[TILE_SIZE][TILE_SIZE + 1];
    
    uint row = tg_id.y * TILE_SIZE + tid.y;
    uint col = tg_id.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;  // Accumulate in FP32
    uint num_tiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + tid.x;
        if (row < params.M && a_col < params.K) {
            As[tid.y][tid.x] = A[row * params.K + a_col];
        } else {
            As[tid.y][tid.x] = half(0.0f);
        }
        
        uint b_row = t * TILE_SIZE + tid.y;
        if (b_row < params.K && col < params.N) {
            Bs[tid.y][tid.x] = B[b_row * params.N + col];
        } else {
            Bs[tid.y][tid.x] = half(0.0f);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(As[tid.y][k]) * float(Bs[k][tid.x]);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < params.M && col < params.N) {
        float existing = float(C[row * params.N + col]);
        C[row * params.N + col] = half(params.alpha * sum + params.beta * existing);
    }
}

// =============================================================================
// SIMD-optimized GEMM (uses Apple's SIMD operations)
// =============================================================================

kernel void gemm_simd_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // 64x64 tile per threadgroup, 8x8 per thread
    constexpr uint TG_M = 64;
    constexpr uint TG_N = 64;
    constexpr uint TG_K = 32;
    constexpr uint THREAD_M = 8;
    constexpr uint THREAD_N = 8;
    
    threadgroup half As[TG_M][TG_K + 1];
    threadgroup half Bs[TG_K][TG_N + 1];
    
    uint tg_row = tg_id.y * TG_M;
    uint tg_col = tg_id.x * TG_N;
    
    // Thread position within 8x8 grid
    uint thread_row = (tid.y * 8 + tid.x / 8) * THREAD_M;
    uint thread_col = (tid.x % 8) * THREAD_N;
    
    float acc[THREAD_M][THREAD_N] = {{0.0f}};
    
    uint num_tiles = (params.K + TG_K - 1) / TG_K;
    
    for (uint t = 0; t < num_tiles; t++) {
        // Cooperative load
        for (uint i = tid.y * 8 + tid.x; i < TG_M * TG_K; i += 64) {
            uint r = i / TG_K;
            uint c = i % TG_K;
            uint global_r = tg_row + r;
            uint global_c = t * TG_K + c;
            As[r][c] = (global_r < params.M && global_c < params.K) ? 
                       A[global_r * params.K + global_c] : half(0.0f);
        }
        
        for (uint i = tid.y * 8 + tid.x; i < TG_K * TG_N; i += 64) {
            uint r = i / TG_N;
            uint c = i % TG_N;
            uint global_r = t * TG_K + r;
            uint global_c = tg_col + c;
            Bs[r][c] = (global_r < params.K && global_c < params.N) ?
                       B[global_r * params.N + global_c] : half(0.0f);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute 8x8 output per thread
        for (uint k = 0; k < TG_K; k++) {
            for (uint mi = 0; mi < THREAD_M; mi++) {
                for (uint ni = 0; ni < THREAD_N; ni++) {
                    uint a_row = thread_row + mi;
                    uint b_col = thread_col + ni;
                    if (a_row < TG_M && b_col < TG_N) {
                        acc[mi][ni] += float(As[a_row][k]) * float(Bs[k][b_col]);
                    }
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store results
    for (uint mi = 0; mi < THREAD_M; mi++) {
        for (uint ni = 0; ni < THREAD_N; ni++) {
            uint global_r = tg_row + thread_row + mi;
            uint global_c = tg_col + thread_col + ni;
            if (global_r < params.M && global_c < params.N) {
                C[global_r * params.N + global_c] = half(acc[mi][ni]);
            }
        }
    }
}

// =============================================================================
// Batched GEMM
// =============================================================================

kernel void gemm_batched_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    constant uint& batch_count [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint3 tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup half As[TILE_SIZE][TILE_SIZE + 1];
    threadgroup half Bs[TILE_SIZE][TILE_SIZE + 1];
    
    uint batch = tg_id.z;
    if (batch >= batch_count) return;
    
    uint stride_a = params.M * params.K;
    uint stride_b = params.K * params.N;
    uint stride_c = params.M * params.N;
    
    device const half* A_batch = A + batch * stride_a;
    device const half* B_batch = B + batch * stride_b;
    device half* C_batch = C + batch * stride_c;
    
    uint row = tg_id.y * TILE_SIZE + tid.y;
    uint col = tg_id.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;
    uint num_tiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + tid.x;
        As[tid.y][tid.x] = (row < params.M && a_col < params.K) ?
                           A_batch[row * params.K + a_col] : half(0.0f);
        
        uint b_row = t * TILE_SIZE + tid.y;
        Bs[tid.y][tid.x] = (b_row < params.K && col < params.N) ?
                           B_batch[b_row * params.N + col] : half(0.0f);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += float(As[tid.y][k]) * float(Bs[k][tid.x]);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < params.M && col < params.N) {
        C_batch[row * params.N + col] = half(sum);
    }
}
