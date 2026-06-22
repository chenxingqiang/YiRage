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
 * ROCm/HIP Device Tensor Operations
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// =============================================================================
// Tensor Copy Operations
// =============================================================================

/**
 * @brief Copy tensor with format conversion
 */
__global__ void copy_f32_to_f16_hip(
    const float* __restrict__ src,
    half* __restrict__ dst,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void copy_f16_to_f32_hip(
    const half* __restrict__ src,
    float* __restrict__ dst,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        dst[i] = __half2float(src[i]);
    }
}

/**
 * @brief Copy with stride (for non-contiguous tensors)
 */
__global__ void copy_strided_f32_hip(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int size,
    int src_stride,
    int dst_stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        dst[idx * dst_stride] = src[idx * src_stride];
    }
}

// =============================================================================
// Tensor Fill Operations
// =============================================================================

/**
 * @brief Fill tensor with constant value
 */
__global__ void fill_f32_hip(
    float* __restrict__ data,
    float value,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        data[i] = value;
    }
}

__global__ void fill_f16_hip(
    half* __restrict__ data,
    half value,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        data[i] = value;
    }
}

/**
 * @brief Fill with zeros (optimized)
 */
__global__ void zero_f32_hip(
    float4* __restrict__ data,
    int size_float4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int i = idx; i < size_float4; i += stride) {
        data[i] = zero;
    }
}

// =============================================================================
// Tensor Transpose Operations
// =============================================================================

/**
 * @brief 2D Transpose using LDS
 */
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_2d_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load tile into LDS (coalesced)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }
    
    __syncthreads();
    
    // Transposed position
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Store transposed tile (coalesced)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

/**
 * @brief Batch transpose for attention
 */
__global__ void batch_transpose_f16_hip(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size,
    int rows,
    int cols
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int in_idx = batch * rows * cols + row * cols + col;
        int out_idx = batch * cols * rows + col * rows + row;
        output[out_idx] = input[in_idx];
    }
}

// =============================================================================
// Tensor Reshape/View Operations
// =============================================================================

/**
 * @brief Permute dimensions (generalized transpose)
 * 
 * @param perm Permutation indices (e.g., [0, 2, 1, 3] for transposing middle dims)
 */
__global__ void permute_4d_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int d0, int d1, int d2, int d3,
    int s0, int s1, int s2, int s3,  // Strides for output
    int perm0, int perm1, int perm2, int perm3
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    
    if (idx < total) {
        // Compute input indices
        int i0 = idx / (d1 * d2 * d3);
        int rem = idx % (d1 * d2 * d3);
        int i1 = rem / (d2 * d3);
        rem = rem % (d2 * d3);
        int i2 = rem / d3;
        int i3 = rem % d3;
        
        // Map to output indices based on permutation
        int indices[4] = {i0, i1, i2, i3};
        int out_idx = indices[perm0] * s0 + indices[perm1] * s1 + 
                      indices[perm2] * s2 + indices[perm3] * s3;
        
        output[out_idx] = input[idx];
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_copy_f32_to_f16_hip(
    const float* src, half* dst, int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(copy_f32_to_f16_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, src, dst, size);
}

void launch_copy_f16_to_f32_hip(
    const half* src, float* dst, int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(copy_f16_to_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, src, dst, size);
}

void launch_fill_f32_hip(
    float* data, float value, int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(fill_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, data, value, size);
}

void launch_fill_f16_hip(
    half* data, float value, int size, hipStream_t stream
) {
    half h_value = __float2half(value);
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(fill_f16_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, data, h_value, size);
}

void launch_transpose_2d_f32_hip(
    const float* input, float* output,
    int rows, int cols, hipStream_t stream
) {
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    
    hipLaunchKernelGGL((transpose_2d_f32_hip<TILE_DIM, BLOCK_ROWS>),
                       grid, block, 0, stream, input, output, rows, cols);
}

void launch_batch_transpose_f16_hip(
    const half* input, half* output,
    int batch_size, int rows, int cols, hipStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16, batch_size);
    
    hipLaunchKernelGGL(batch_transpose_f16_hip, grid, block, 0, stream,
                       input, output, batch_size, rows, cols);
}

}  // extern "C"
