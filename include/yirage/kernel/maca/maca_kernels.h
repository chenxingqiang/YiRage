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
 * This file is part of YiRage (Yi Revolutionary AGile Engine)
 * 
 * MACA Fingerprint Kernels
 * 
 * Fingerprint computation kernels for MetaX MACA backend.
 * Used for verification of kernel optimizations.
 * 
 * Note: All kernels use MACA's mc* API and 64-thread warp operations.
 */

#pragma once

#include "yirage/kernel/graph.h"
#include "yirage/type.h"

#ifdef YIRAGE_BACKEND_MACA_ENABLED

namespace yirage {
namespace kernel {
namespace maca {

/**
 * @brief MACA fingerprint kernels for verification
 * 
 * These kernels compute fingerprints for MACA GPU operations
 * to verify correctness of optimizations.
 * 
 * All kernels are optimized for MACA's 64-thread warp architecture:
 * - Block sizes are multiples of 64
 * - Warp reductions use 6 iterations (log2(64))
 * - Full warp mask is 64-bit (0xFFFFFFFFFFFFFFFF)
 */

// ============================================================
// Matmul Fingerprint
// ============================================================

/**
 * @brief Compute matrix multiplication fingerprint
 * @param A_ptr Pointer to matrix A fingerprint data
 * @param B_ptr Pointer to matrix B fingerprint data
 * @param C_ptr Pointer to output matrix C fingerprint data
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param num_batches Number of batches
 * @return true if computation succeeded
 */
bool compute_matmul_fingerprint(
    void *A_ptr, void *B_ptr, void *C_ptr,
    int m, int n, int k, int num_batches);

// ============================================================
// Element-wise Operation Fingerprints
// ============================================================

/**
 * @brief Compute element-wise unary operation fingerprint
 * @param op_type Operation type (EXP, LOG, RELU, etc.)
 * @param input_ptr Input tensor fingerprint
 * @param output_ptr Output tensor fingerprint
 * @param num_elements Number of elements
 * @return true if computation succeeded
 */
bool compute_element_unary_fingerprint(
    type::KNOperatorType op_type,
    void *input_ptr, void *output_ptr,
    int num_elements);

/**
 * @brief Compute element-wise binary operation fingerprint
 * @param op_type Operation type (ADD, MUL, DIV, etc.)
 * @param input1_ptr First input tensor fingerprint
 * @param input2_ptr Second input tensor fingerprint
 * @param output_ptr Output tensor fingerprint
 * @param num_elements Number of elements
 * @return true if computation succeeded
 */
bool compute_element_binary_fingerprint(
    type::KNOperatorType op_type,
    void *input1_ptr, void *input2_ptr, void *output_ptr,
    int num_elements);

// ============================================================
// Normalization Fingerprints
// ============================================================

/**
 * @brief Compute RMS normalization fingerprint
 * Uses MACA 64-thread warp reduce for computing square sum
 * @param input_ptr Input tensor fingerprint
 * @param output_ptr Output tensor fingerprint
 * @param num_samples Number of samples
 * @param norm_size Normalization size
 * @return true if computation succeeded
 */
bool compute_rms_norm_fingerprint(
    void *input_ptr, void *output_ptr,
    int num_samples, int norm_size);

/**
 * @brief Compute layer normalization fingerprint
 * @param input_ptr Input tensor fingerprint
 * @param output_ptr Output tensor fingerprint
 * @param num_samples Number of samples
 * @param norm_size Normalization size
 * @return true if computation succeeded
 */
bool compute_layer_norm_fingerprint(
    void *input_ptr, void *output_ptr,
    int num_samples, int norm_size);

// ============================================================
// Reduction Fingerprints
// ============================================================

/**
 * @brief Compute reduction fingerprint
 * @param input_ptr Input tensor fingerprint
 * @param output_ptr Output tensor fingerprint
 * @param reduction_dim Dimension to reduce
 * @param reduction_size Size of reduction dimension
 * @return true if computation succeeded
 */
bool compute_reduction_fingerprint(
    void *input_ptr, void *output_ptr,
    int reduction_dim, int reduction_size);

/**
 * @brief Compute all-reduce fingerprint
 * Uses MACA's mccl (NCCL equivalent) for multi-GPU
 * @param input_ptr Input tensor fingerprint
 * @param output_ptr Output tensor fingerprint
 * @param num_elements Number of elements
 * @param num_devices Number of GPU devices
 * @return true if computation succeeded
 */
bool compute_all_reduce_fingerprint(
    void *input_ptr, void *output_ptr,
    int num_elements, int num_devices);

// ============================================================
// Attention Fingerprints
// ============================================================

/**
 * @brief Compute attention fingerprint
 * @param Q_ptr Query tensor fingerprint
 * @param K_ptr Key tensor fingerprint
 * @param V_ptr Value tensor fingerprint
 * @param output_ptr Output tensor fingerprint
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim Head dimension
 * @param causal Whether to use causal masking
 * @return true if computation succeeded
 */
bool compute_attention_fingerprint(
    void *Q_ptr, void *K_ptr, void *V_ptr, void *output_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim,
    bool causal);

// ============================================================
// Device Tensor Operations
// ============================================================

/**
 * @brief Compute device tensor copy fingerprint
 * @param src_ptr Source tensor fingerprint
 * @param dst_ptr Destination tensor fingerprint
 * @param num_elements Number of elements
 * @return true if computation succeeded
 */
bool compute_tensor_copy_fingerprint(
    void *src_ptr, void *dst_ptr,
    int num_elements);

/**
 * @brief Compute transpose fingerprint
 * @param input_ptr Input tensor fingerprint
 * @param output_ptr Output tensor fingerprint
 * @param dims Tensor dimensions
 * @param perm Permutation indices
 * @return true if computation succeeded
 */
bool compute_transpose_fingerprint(
    void *input_ptr, void *output_ptr,
    std::vector<int> const &dims,
    std::vector<int> const &perm);

// ============================================================
// Utility Functions
// ============================================================

/**
 * @brief Initialize MACA fingerprint computation
 * Allocates lookup tables on GPU
 * @param device_id GPU device ID
 * @return true if initialization succeeded
 */
bool init_maca_fingerprint(int device_id = 0);

/**
 * @brief Cleanup MACA fingerprint resources
 */
void cleanup_maca_fingerprint();

/**
 * @brief Check if MACA fingerprint is initialized
 * @return true if initialized
 */
bool is_maca_fingerprint_initialized();

} // namespace maca
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

