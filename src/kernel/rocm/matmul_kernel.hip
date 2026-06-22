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
 * ROCm/HIP Matrix Multiplication Kernels
 * 
 * Optimized for CDNA architectures (MI100, MI200, MI300 series)
 * Key differences from CUDA:
 * - 64-thread wavefronts instead of 32-thread warps
 * - MFMA instructions instead of Tensor Cores
 * - LDS instead of shared memory (same hardware, different terminology)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVEFRONT_SIZE 64
#define WAVE_REDUCE_SUM(val) \
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) \
        val += __shfl_xor(val, offset)

// =============================================================================
// GEMM Kernels
// =============================================================================

/**
 * @brief Basic GEMM kernel for warm-up and small matrices
 */
__global__ void gemm_basic_hip(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/**
 * @brief Tiled GEMM with LDS for MI100+ (128x128 tiles)
 */
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_tiled_hip(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half As[TILE_M][TILE_K + 8];  // Padding to avoid bank conflicts
    __shared__ half Bs[TILE_K][TILE_N + 8];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    // Each wavefront handles 32x32 output tile
    int wave_row = wave_id / 2 * 32;
    int wave_col = wave_id % 2 * 64;
    
    // Accumulator registers
    float acc[4][4] = {{0.0f}};
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cooperative load A tile
        for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int ar = row_base + r;
            int ac = t * TILE_K + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : __float2half(0.0f);
        }
        
        // Cooperative load B tile
        for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int br = t * TILE_K + r;
            int bc = col_base + c;
            Bs[r][c] = (br < K && bc < N) ? B[br * N + bc] : __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute using MFMA-like pattern
        for (int k = 0; k < TILE_K; k++) {
            // Each thread computes 4x4 output elements
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    int row = wave_row + (lane_id / 8) * 4 + mi;
                    int col = wave_col + (lane_id % 8) * 8 + ni;
                    
                    if (row < TILE_M && col < TILE_N) {
                        acc[mi][ni] += __half2float(As[row][k]) * 
                                       __half2float(Bs[k][col]);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    for (int mi = 0; mi < 4; mi++) {
        for (int ni = 0; ni < 4; ni++) {
            int row = row_base + wave_row + (lane_id / 8) * 4 + mi;
            int col = col_base + wave_col + (lane_id % 8) * 8 + ni;
            
            if (row < M && col < N) {
                C[row * N + col] = __float2half(acc[mi][ni]);
            }
        }
    }
}

/**
 * @brief GEMM with double buffering for MI200+
 */
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_double_buffer_hip(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half As[2][TILE_M][TILE_K + 8];
    __shared__ half Bs[2][TILE_K][TILE_N + 8];
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    int tid = threadIdx.x;
    
    float acc[16] = {0.0f};
    
    int num_tiles = (K + TILE_K - 1) / TILE_K;
    int buffer = 0;
    
    // Load first tile
    for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
        int r = i / TILE_K, c = i % TILE_K;
        int ar = row_base + r, ac = c;
        As[0][r][c] = (ar < M && ac < K) ? A[ar * K + ac] : __float2half(0.0f);
    }
    for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
        int r = i / TILE_N, c = i % TILE_N;
        int br = r, bc = col_base + c;
        Bs[0][r][c] = (br < K && bc < N) ? B[br * N + bc] : __float2half(0.0f);
    }
    __syncthreads();
    
    for (int t = 0; t < num_tiles; t++) {
        int next = 1 - buffer;
        
        // Prefetch next tile
        if (t + 1 < num_tiles) {
            for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
                int r = i / TILE_K, c = i % TILE_K;
                int ar = row_base + r, ac = (t + 1) * TILE_K + c;
                As[next][r][c] = (ar < M && ac < K) ? 
                    A[ar * K + ac] : __float2half(0.0f);
            }
            for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
                int r = i / TILE_N, c = i % TILE_N;
                int br = (t + 1) * TILE_K + r, bc = col_base + c;
                Bs[next][r][c] = (br < K && bc < N) ? 
                    B[br * N + bc] : __float2half(0.0f);
            }
        }
        
        // Compute on current buffer
        int wave_id = tid / WAVEFRONT_SIZE;
        int lane_id = tid % WAVEFRONT_SIZE;
        int wave_row = wave_id * 32;
        
        for (int k = 0; k < TILE_K; k++) {
            for (int mi = 0; mi < 4; mi++) {
                for (int ni = 0; ni < 4; ni++) {
                    int row = wave_row + (lane_id / 8) * 4 + mi;
                    int col = (lane_id % 8) * 4 + ni;
                    
                    if (row < TILE_M && col < TILE_N) {
                        acc[mi * 4 + ni] += __half2float(As[buffer][row][k]) *
                                            __half2float(Bs[buffer][k][col]);
                    }
                }
            }
        }
        
        buffer = next;
        __syncthreads();
    }
    
    // Store results
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    int wave_row = wave_id * 32;
    
    for (int i = 0; i < 16; i++) {
        int mi = i / 4, ni = i % 4;
        int row = row_base + wave_row + (lane_id / 8) * 4 + mi;
        int col = col_base + (lane_id % 8) * 4 + ni;
        
        if (row < M && col < N) {
            C[row * N + col] = __float2half(acc[i]);
        }
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_gemm_basic_hip(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    hipStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    hipLaunchKernelGGL(gemm_basic_hip, grid, block, 0, stream,
                       A, B, C, M, N, K, alpha, beta);
}

void launch_gemm_tiled_hip(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    hipStream_t stream
) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    
    dim3 block(256);  // 4 wavefronts
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    hipLaunchKernelGGL((gemm_tiled_hip<TILE_M, TILE_N, TILE_K>),
                       grid, block, 0, stream, A, B, C, M, N, K);
}

void launch_gemm_double_buffer_hip(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    hipStream_t stream
) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    
    dim3 block(256);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    hipLaunchKernelGGL((gemm_double_buffer_hip<TILE_M, TILE_N, TILE_K>),
                       grid, block, 0, stream, A, B, C, M, N, K);
}

}  // extern "C"
