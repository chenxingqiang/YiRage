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
 * CPU Kernel Interface
 * 
 * Main header for CPU kernel implementations with SIMD optimization.
 * Supports x86 (SSE, AVX, AVX2, AVX-512) and ARM (NEON).
 */

#pragma once

#include "kernel/cpu/cpu_kernel_config.h"

namespace yirage {
namespace kernel {
namespace cpu {

// =============================================================================
// GEMM Operations
// =============================================================================

/**
 * @brief SIMD-optimized GEMM
 * @param A Input matrix A [M x K]
 * @param B Input matrix B [K x N]
 * @param C Output matrix C [M x N]
 * @param M, N, K Matrix dimensions
 * @param alpha, beta Scaling factors: C = alpha * A * B + beta * C
 * @param simd_type SIMD instruction set to use
 */
void gemm_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    SIMDType simd_type = SIMDType::AUTO);

/**
 * @brief Multi-threaded GEMM
 */
void gemm_parallel_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    int num_threads = -1);

/**
 * @brief Fused RMS norm + GEMM (fp32): Y = rms_norm(X) @ W
 */
void rms_matmul_f32(
    const float* X, const float* W, float* Y,
    int M, int N, int K,
    float epsilon,
    int num_threads = -1);

// =============================================================================
// RMSNorm Operations
// =============================================================================

/**
 * @brief SIMD-optimized RMSNorm
 */
void rms_norm_f32(
    const float* input,
    const float* weight,
    float* output,
    int num_tokens,
    int hidden_dim,
    float eps,
    SIMDType simd_type = SIMDType::AUTO);

/**
 * @brief Multi-threaded RMSNorm
 */
void rms_norm_parallel_f32(
    const float* input,
    const float* weight,
    float* output,
    int num_tokens,
    int hidden_dim,
    float eps,
    int num_threads = -1);

// =============================================================================
// Softmax Operations
// =============================================================================

/**
 * @brief SIMD-optimized row-wise Softmax
 */
void softmax_f32(
    const float* input,
    float* output,
    int num_rows,
    int row_size,
    SIMDType simd_type = SIMDType::AUTO);

// =============================================================================
// Element-wise Binary Operations
// =============================================================================

void add_f32(const float* a, const float* b, float* c, int size,
             SIMDType simd_type = SIMDType::AUTO);

void mul_f32(const float* a, const float* b, float* c, int size,
             SIMDType simd_type = SIMDType::AUTO);

void silu_mul_f32(const float* gate, const float* up, float* out, int size,
                  SIMDType simd_type = SIMDType::AUTO);

// =============================================================================
// Element-wise Unary Operations
// =============================================================================

void relu_f32(const float* input, float* output, int size,
              SIMDType simd_type = SIMDType::AUTO);

void gelu_f32(const float* input, float* output, int size,
              SIMDType simd_type = SIMDType::AUTO);

void silu_f32(const float* input, float* output, int size,
              SIMDType simd_type = SIMDType::AUTO);

// =============================================================================
// Reduction Operations
// =============================================================================

float reduce_sum_f32(const float* input, int size,
                     SIMDType simd_type = SIMDType::AUTO);

float reduce_max_f32(const float* input, int size,
                     SIMDType simd_type = SIMDType::AUTO);

void reduce_sum_row_f32(
    const float* input, float* output,
    int num_rows, int row_size,
    SIMDType simd_type = SIMDType::AUTO);

void reduce_max_row_f32(
    const float* input, float* output,
    int num_rows, int row_size,
    SIMDType simd_type = SIMDType::AUTO);

void reduce_mean_row_f32(
    const float* input, float* output,
    int num_rows, int row_size,
    SIMDType simd_type = SIMDType::AUTO);

void argmax_row_f32(
    const float* input, int* output,
    int num_rows, int row_size);

// =============================================================================
// Embedding Operations
// =============================================================================

void embedding_lookup_f32(
    const int* token_ids,
    const float* embedding_table,
    float* output,
    int num_tokens,
    int embedding_dim,
    int vocab_size);

void embedding_lookup_parallel_f32(
    const int* token_ids,
    const float* embedding_table,
    float* output,
    int num_tokens,
    int embedding_dim,
    int vocab_size,
    int num_threads = -1);

void add_position_embedding_f32(
    float* embeddings,
    const float* position_table,
    const int* positions,
    int num_tokens,
    int embedding_dim,
    int max_positions,
    SIMDType simd_type = SIMDType::AUTO);

void apply_rope_f32(
    float* query,
    float* key,
    const float* cos_cache,
    const float* sin_cache,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int rotary_dim);

void precompute_rope_cache_f32(
    float* cos_cache,
    float* sin_cache,
    int max_seq_len,
    int head_dim,
    float base = 10000.0f);

void lm_head_f32(
    const float* hidden,
    const float* weight,
    float* logits,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    SIMDType simd_type = SIMDType::AUTO);

// =============================================================================
// Tensor Operations
// =============================================================================

void fill_f32(float* data, float value, int size);
void fill_zero_f32(float* data, int size);
void copy_f32(const float* src, float* dst, int size);

void transpose_2d_f32(
    const float* input, float* output,
    int rows, int cols);

void batch_transpose_f32(
    const float* input, float* output,
    int batch_size, int rows, int cols);

void permute_4d_f32(
    const float* input, float* output,
    int d0, int d1, int d2, int d3,
    int perm0, int perm1, int perm2, int perm3);

void concat_f32(
    const float* const* inputs, float* output,
    const int* input_sizes, int num_inputs);

void slice_2d_f32(
    const float* input, float* output,
    int in_rows, int in_cols,
    int start_row, int end_row,
    int start_col, int end_col);

void gather_f32(
    const float* input, const int* indices, float* output,
    int num_indices, int dim);

// =============================================================================
// High-Level Executor
// =============================================================================

/**
 * @brief CPU Kernel Executor with automatic SIMD detection
 */
class CPUKernelExecutor {
public:
    CPUKernelExecutor();
    ~CPUKernelExecutor();
    
    /**
     * @brief Initialize with automatic SIMD detection
     */
    bool initialize();
    
    /**
     * @brief Get detected SIMD type
     */
    SIMDType get_simd_type() const { return simd_type_; }
    
    /**
     * @brief Get number of available threads
     */
    int get_num_threads() const { return num_threads_; }
    
    /**
     * @brief Set number of threads for parallel operations
     */
    void set_num_threads(int num_threads);
    
    // GEMM
    void gemm(const float* A, const float* B, float* C,
              int M, int N, int K,
              float alpha = 1.0f, float beta = 0.0f);
    
    // RMSNorm
    void rms_norm(const float* input, const float* weight, float* output,
                  int num_tokens, int hidden_dim, float eps = 1e-5f);
    
    // Softmax
    void softmax(const float* input, float* output,
                 int num_rows, int row_size);
    
    // Element-wise
    void add(const float* a, const float* b, float* c, int size);
    void mul(const float* a, const float* b, float* c, int size);
    void silu_mul(const float* gate, const float* up, float* out, int size);
    void relu(const float* input, float* output, int size);
    void gelu(const float* input, float* output, int size);

private:
    SIMDType simd_type_;
    int num_threads_;
    bool initialized_;
};

}  // namespace cpu
}  // namespace kernel
}  // namespace yirage
