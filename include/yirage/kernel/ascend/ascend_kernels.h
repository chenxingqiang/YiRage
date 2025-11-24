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
 */

#pragma once

#include "yirage/kernel/graph.h"

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED

namespace yirage {
namespace kernel {
namespace ascend {

/**
 * @brief Ascend fingerprint kernels for verification
 * 
 * These kernels compute fingerprints for Ascend NPU operations
 * to verify correctness of optimizations.
 */

// Matmul fingerprint
bool compute_matmul_fingerprint(
    void *A_ptr, void *B_ptr, void *C_ptr,
    int m, int n, int k, int num_batches);

// Element-wise unary operation fingerprint
bool compute_element_unary_fingerprint(
    type::KNOperatorType op_type,
    void *input_ptr, void *output_ptr,
    int num_elements);

// Element-wise binary operation fingerprint  
bool compute_element_binary_fingerprint(
    type::KNOperatorType op_type,
    void *input1_ptr, void *input2_ptr, void *output_ptr,
    int num_elements);

// RMS Norm fingerprint
bool compute_rms_norm_fingerprint(
    void *input_ptr, void *output_ptr,
    int num_samples, int norm_size);

// Reduction fingerprint
bool compute_reduction_fingerprint(
    void *input_ptr, void *output_ptr,
    int reduction_dim, int reduction_size);

} // namespace ascend
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

