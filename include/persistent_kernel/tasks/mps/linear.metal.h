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
 * @file linear.metal.h
 * @brief MPS Linear (GEMM) kernels
 *
 * Matrix multiplication kernels for linear layers.
 * For production, use MPSMatrixMultiplication for better performance.
 */

namespace yirage {
namespace persistent_kernel {
namespace mps {

constexpr const char* LINEAR_KERNEL_SOURCE = R"(
// Generic GEMM kernel: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
kernel void gemm_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int row = tid.y;
    int col = tid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Tiled GEMM for better cache utilization
// Uses 16x16 tiles in shared memory
kernel void gemm_tiled_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    constexpr int TILE_SIZE = 16;
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = tgid.y * TILE_SIZE + tid_in_tg.y;
    int col = tgid.x * TILE_SIZE + tid_in_tg.x;
    
    float sum = 0.0f;
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load tile of A into shared memory
        int a_row = row;
        int a_col = t * TILE_SIZE + tid_in_tg.x;
        if (a_row < M && a_col < K) {
            As[tid_in_tg.y][tid_in_tg.x] = A[a_row * K + a_col];
        } else {
            As[tid_in_tg.y][tid_in_tg.x] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + tid_in_tg.y;
        int b_col = col;
        if (b_row < K && b_col < N) {
            Bs[tid_in_tg.y][tid_in_tg.x] = B[b_row * N + b_col];
        } else {
            Bs[tid_in_tg.y][tid_in_tg.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[tid_in_tg.y][k] * Bs[k][tid_in_tg.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Linear layer with bias: Y = X @ W^T + b
kernel void linear_bias_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& in_features [[buffer(5)]],
    constant int& out_features [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int batch = tid.y;
    int out_idx = tid.x;
    
    if (batch >= batch_size || out_idx >= out_features) return;
    
    device const float* in_row = input + batch * in_features;
    device const float* w_row = weight + out_idx * in_features;
    
    float sum = bias[out_idx];
    for (int i = 0; i < in_features; i++) {
        sum += in_row[i] * w_row[i];
    }
    
    output[batch * out_features + out_idx] = sum;
}
)";

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
