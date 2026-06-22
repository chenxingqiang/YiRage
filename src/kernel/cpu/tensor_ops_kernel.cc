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
 * CPU Tensor Operations
 */

#include "kernel/cpu/cpu_kernel_config.h"

#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace yirage {
namespace kernel {
namespace cpu {

// =============================================================================
// Fill Operations
// =============================================================================

void fill_f32(float* data, float value, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = value;
    }
}

#ifdef __AVX2__
void fill_avx2_f32(float* data, float value, int size) {
    __m256 val_vec = _mm256_set1_ps(value);
    int i = 0;
    for (; i <= size - 8; i += 8) {
        _mm256_storeu_ps(&data[i], val_vec);
    }
    for (; i < size; i++) {
        data[i] = value;
    }
}
#endif

void fill_zero_f32(float* data, int size) {
    std::memset(data, 0, size * sizeof(float));
}

// =============================================================================
// Copy Operations
// =============================================================================

void copy_f32(const float* src, float* dst, int size) {
    std::memcpy(dst, src, size * sizeof(float));
}

void copy_strided_f32(
    const float* src, float* dst,
    int size,
    int src_stride, int dst_stride
) {
    for (int i = 0; i < size; i++) {
        dst[i * dst_stride] = src[i * src_stride];
    }
}

// =============================================================================
// Transpose Operations
// =============================================================================

void transpose_2d_f32(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    // Cache-friendly tiled transpose
    constexpr int TILE = 32;
    
    for (int i0 = 0; i0 < rows; i0 += TILE) {
        for (int j0 = 0; j0 < cols; j0 += TILE) {
            int i_end = std::min(i0 + TILE, rows);
            int j_end = std::min(j0 + TILE, cols);
            
            for (int i = i0; i < i_end; i++) {
                for (int j = j0; j < j_end; j++) {
                    output[j * rows + i] = input[i * cols + j];
                }
            }
        }
    }
}

#ifdef __AVX2__
void transpose_2d_avx2_f32(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    // 8x8 transpose using AVX2
    constexpr int TILE = 8;
    
    for (int i0 = 0; i0 < rows; i0 += TILE) {
        for (int j0 = 0; j0 < cols; j0 += TILE) {
            if (i0 + TILE <= rows && j0 + TILE <= cols) {
                // Full 8x8 tile
                __m256 r0 = _mm256_loadu_ps(&input[(i0 + 0) * cols + j0]);
                __m256 r1 = _mm256_loadu_ps(&input[(i0 + 1) * cols + j0]);
                __m256 r2 = _mm256_loadu_ps(&input[(i0 + 2) * cols + j0]);
                __m256 r3 = _mm256_loadu_ps(&input[(i0 + 3) * cols + j0]);
                __m256 r4 = _mm256_loadu_ps(&input[(i0 + 4) * cols + j0]);
                __m256 r5 = _mm256_loadu_ps(&input[(i0 + 5) * cols + j0]);
                __m256 r6 = _mm256_loadu_ps(&input[(i0 + 6) * cols + j0]);
                __m256 r7 = _mm256_loadu_ps(&input[(i0 + 7) * cols + j0]);
                
                // Transpose within 4x4 blocks
                __m256 t0 = _mm256_unpacklo_ps(r0, r1);
                __m256 t1 = _mm256_unpackhi_ps(r0, r1);
                __m256 t2 = _mm256_unpacklo_ps(r2, r3);
                __m256 t3 = _mm256_unpackhi_ps(r2, r3);
                __m256 t4 = _mm256_unpacklo_ps(r4, r5);
                __m256 t5 = _mm256_unpackhi_ps(r4, r5);
                __m256 t6 = _mm256_unpacklo_ps(r6, r7);
                __m256 t7 = _mm256_unpackhi_ps(r6, r7);
                
                r0 = _mm256_shuffle_ps(t0, t2, 0x44);
                r1 = _mm256_shuffle_ps(t0, t2, 0xEE);
                r2 = _mm256_shuffle_ps(t1, t3, 0x44);
                r3 = _mm256_shuffle_ps(t1, t3, 0xEE);
                r4 = _mm256_shuffle_ps(t4, t6, 0x44);
                r5 = _mm256_shuffle_ps(t4, t6, 0xEE);
                r6 = _mm256_shuffle_ps(t5, t7, 0x44);
                r7 = _mm256_shuffle_ps(t5, t7, 0xEE);
                
                // Permute 128-bit halves
                t0 = _mm256_permute2f128_ps(r0, r4, 0x20);
                t1 = _mm256_permute2f128_ps(r1, r5, 0x20);
                t2 = _mm256_permute2f128_ps(r2, r6, 0x20);
                t3 = _mm256_permute2f128_ps(r3, r7, 0x20);
                t4 = _mm256_permute2f128_ps(r0, r4, 0x31);
                t5 = _mm256_permute2f128_ps(r1, r5, 0x31);
                t6 = _mm256_permute2f128_ps(r2, r6, 0x31);
                t7 = _mm256_permute2f128_ps(r3, r7, 0x31);
                
                _mm256_storeu_ps(&output[(j0 + 0) * rows + i0], t0);
                _mm256_storeu_ps(&output[(j0 + 1) * rows + i0], t1);
                _mm256_storeu_ps(&output[(j0 + 2) * rows + i0], t2);
                _mm256_storeu_ps(&output[(j0 + 3) * rows + i0], t3);
                _mm256_storeu_ps(&output[(j0 + 4) * rows + i0], t4);
                _mm256_storeu_ps(&output[(j0 + 5) * rows + i0], t5);
                _mm256_storeu_ps(&output[(j0 + 6) * rows + i0], t6);
                _mm256_storeu_ps(&output[(j0 + 7) * rows + i0], t7);
            } else {
                // Partial tile - scalar
                int i_end = std::min(i0 + TILE, rows);
                int j_end = std::min(j0 + TILE, cols);
                for (int i = i0; i < i_end; i++) {
                    for (int j = j0; j < j_end; j++) {
                        output[j * rows + i] = input[i * cols + j];
                    }
                }
            }
        }
    }
}
#endif

// =============================================================================
// Batch Transpose
// =============================================================================

void batch_transpose_f32(
    const float* input,
    float* output,
    int batch_size,
    int rows,
    int cols
) {
    int stride = rows * cols;
    for (int b = 0; b < batch_size; b++) {
        transpose_2d_f32(input + b * stride, output + b * stride, rows, cols);
    }
}

// =============================================================================
// Permute (Generalized Transpose)
// =============================================================================

void permute_4d_f32(
    const float* input,
    float* output,
    int d0, int d1, int d2, int d3,
    int perm0, int perm1, int perm2, int perm3
) {
    int dims[4] = {d0, d1, d2, d3};
    int out_dims[4] = {dims[perm0], dims[perm1], dims[perm2], dims[perm3]};
    
    // Compute output strides
    int out_strides[4];
    out_strides[3] = 1;
    out_strides[2] = out_dims[3];
    out_strides[1] = out_strides[2] * out_dims[2];
    out_strides[0] = out_strides[1] * out_dims[1];
    
    // Map output strides to input dimensions
    int strides_for_input[4];
    strides_for_input[perm0] = out_strides[0];
    strides_for_input[perm1] = out_strides[1];
    strides_for_input[perm2] = out_strides[2];
    strides_for_input[perm3] = out_strides[3];
    
    int total = d0 * d1 * d2 * d3;
    for (int idx = 0; idx < total; idx++) {
        int i0 = idx / (d1 * d2 * d3);
        int rem = idx % (d1 * d2 * d3);
        int i1 = rem / (d2 * d3);
        rem = rem % (d2 * d3);
        int i2 = rem / d3;
        int i3 = rem % d3;
        
        int out_idx = i0 * strides_for_input[0] + 
                      i1 * strides_for_input[1] +
                      i2 * strides_for_input[2] + 
                      i3 * strides_for_input[3];
        
        output[out_idx] = input[idx];
    }
}

// =============================================================================
// Concatenate
// =============================================================================

void concat_f32(
    const float* const* inputs,
    float* output,
    const int* input_sizes,
    int num_inputs
) {
    int offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        std::memcpy(output + offset, inputs[i], input_sizes[i] * sizeof(float));
        offset += input_sizes[i];
    }
}

// =============================================================================
// Slice
// =============================================================================

void slice_f32(
    const float* input,
    float* output,
    int start,
    int end,
    int stride
) {
    int out_idx = 0;
    for (int i = start; i < end; i += stride) {
        output[out_idx++] = input[i];
    }
}

void slice_2d_f32(
    const float* input,
    float* output,
    int in_rows, int in_cols,
    int start_row, int end_row,
    int start_col, int end_col
) {
    int out_cols = end_col - start_col;
    
    for (int i = start_row; i < end_row; i++) {
        const float* src = input + i * in_cols + start_col;
        float* dst = output + (i - start_row) * out_cols;
        std::memcpy(dst, src, out_cols * sizeof(float));
    }
}

// =============================================================================
// Gather
// =============================================================================

void gather_f32(
    const float* input,
    const int* indices,
    float* output,
    int num_indices,
    int dim
) {
    for (int i = 0; i < num_indices; i++) {
        int idx = indices[i];
        if (idx >= 0) {
            std::memcpy(output + i * dim, input + idx * dim, dim * sizeof(float));
        }
    }
}

// =============================================================================
// Parallel Operations
// =============================================================================

void transpose_2d_parallel_f32(
    const float* input,
    float* output,
    int rows,
    int cols,
    int num_threads
) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    std::vector<std::thread> threads;
    int rows_per_thread = (rows + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        int start_row = t * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, rows);
        
        if (start_row >= rows) break;
        
        threads.emplace_back([=]() {
            for (int i = start_row; i < end_row; i++) {
                for (int j = 0; j < cols; j++) {
                    output[j * rows + i] = input[i * cols + j];
                }
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
}

}  // namespace cpu
}  // namespace kernel
}  // namespace yirage
