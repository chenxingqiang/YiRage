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
 * ROCm/HIP Kernel Interface
 * 
 * Main header for AMD ROCm kernel implementations.
 */

#pragma once

#include "kernel/rocm/rocm_kernel_config.h"

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

namespace yirage {
namespace kernel {
namespace rocm {

// =============================================================================
// Kernel Launch Interface
// =============================================================================

#ifdef YIRAGE_BACKEND_ROCM_ENABLED

// Forward declarations for HIP kernels
extern "C" {

// GEMM
void launch_gemm_basic_hip(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    hipStream_t stream);

void launch_gemm_tiled_hip(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    hipStream_t stream);

void launch_gemm_double_buffer_hip(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    hipStream_t stream);

// RMSNorm
void launch_rms_norm_f32_hip(
    const float* input, const float* weight, float* output,
    int num_tokens, int hidden_dim, float eps,
    hipStream_t stream);

void launch_rms_norm_f16_hip(
    const half* input, const half* weight, half* output,
    int num_tokens, int hidden_dim, float eps,
    hipStream_t stream);

void launch_rms_norm_batched_hip(
    const half* input, const half* weight, half* output,
    int num_tokens, int hidden_dim, float eps,
    hipStream_t stream);

// Element-wise Binary
void launch_add_f32_hip(
    const float* a, const float* b, float* c,
    int size, hipStream_t stream);

void launch_add_f16_hip(
    const half* a, const half* b, half* c,
    int size, hipStream_t stream);

void launch_mul_f32_hip(
    const float* a, const float* b, float* c,
    int size, hipStream_t stream);

void launch_silu_mul_f16_hip(
    const half* gate, const half* up, half* output,
    int size, hipStream_t stream);

// Element-wise Unary
void launch_relu_f32_hip(
    const float* input, float* output, int size, hipStream_t stream);

void launch_gelu_f32_hip(
    const float* input, float* output, int size, hipStream_t stream);

void launch_gelu_f16_hip(
    const half* input, half* output, int size, hipStream_t stream);

void launch_silu_f32_hip(
    const float* input, float* output, int size, hipStream_t stream);

void launch_silu_f16_hip(
    const half* input, half* output, int size, hipStream_t stream);

void launch_sigmoid_f32_hip(
    const float* input, float* output, int size, hipStream_t stream);

void launch_tanh_f32_hip(
    const float* input, float* output, int size, hipStream_t stream);

// Reduction
void launch_reduce_sum_f32_hip(
    const float* input, float* output, int size, hipStream_t stream);

void launch_reduce_sum_row_f32_hip(
    const float* input, float* output,
    int num_rows, int row_size, hipStream_t stream);

void launch_reduce_max_f32_hip(
    const float* input, float* output, int size, hipStream_t stream);

void launch_softmax_f32_hip(
    const float* input, float* output,
    int num_rows, int row_size, hipStream_t stream);

void launch_softmax_f16_hip(
    const half* input, half* output,
    int num_rows, int row_size, hipStream_t stream);

// All-Reduce
void launch_all_reduce_sum_local_hip(
    const float* input, float* output, float* workspace,
    int size, hipStream_t stream);

void launch_broadcast_hip(
    const float* value, float* output,
    int size, hipStream_t stream);

void launch_gradient_average_f16_hip(
    half* gradients, int size, int num_workers, hipStream_t stream);

void launch_gradient_scale_f16_hip(
    half* gradients, int size, float scale, hipStream_t stream);

// Tensor Operations
void launch_copy_f32_to_f16_hip(
    const float* src, half* dst, int size, hipStream_t stream);

void launch_copy_f16_to_f32_hip(
    const half* src, float* dst, int size, hipStream_t stream);

void launch_fill_f32_hip(
    float* data, float value, int size, hipStream_t stream);

void launch_fill_f16_hip(
    half* data, float value, int size, hipStream_t stream);

void launch_transpose_2d_f32_hip(
    const float* input, float* output,
    int rows, int cols, hipStream_t stream);

void launch_batch_transpose_f16_hip(
    const half* input, half* output,
    int batch_size, int rows, int cols, hipStream_t stream);

}  // extern "C"

#endif  // YIRAGE_BACKEND_ROCM_ENABLED

// =============================================================================
// High-Level Kernel Interface
// =============================================================================

/**
 * @brief ROCm Kernel Executor
 * 
 * Provides a unified interface for executing kernels on AMD GPUs.
 */
class ROCmKernelExecutor {
public:
    ROCmKernelExecutor();
    ~ROCmKernelExecutor();
    
    // Initialize for specific architecture
    bool initialize(ROCmArch arch);
    
    // GEMM
    void gemm_f32(const float* A, const float* B, float* C,
                  int M, int N, int K,
                  float alpha = 1.0f, float beta = 0.0f);
    
    void gemm_f16(const void* A, const void* B, void* C,
                  int M, int N, int K);
    
    // RMSNorm
    void rms_norm_f32(const float* input, const float* weight, float* output,
                      int num_tokens, int hidden_dim, float eps = 1e-5f);
    
    void rms_norm_f16(const void* input, const void* weight, void* output,
                      int num_tokens, int hidden_dim, float eps = 1e-5f);
    
    // Softmax
    void softmax_f32(const float* input, float* output,
                     int num_rows, int row_size);
    
    void softmax_f16(const void* input, void* output,
                     int num_rows, int row_size);
    
    // Synchronize
    void synchronize();
    
private:
    ROCmArch arch_;
    bool initialized_;
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipStream_t stream_;
#else
    void* stream_;
#endif
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if ROCm is available
 */
bool is_rocm_available();

/**
 * @brief Get number of ROCm devices
 */
int get_rocm_device_count();

/**
 * @brief Get current device
 */
int get_current_rocm_device();

/**
 * @brief Set current device
 */
void set_rocm_device(int device_id);

/**
 * @brief Get device name
 */
std::string get_rocm_device_name(int device_id = -1);

/**
 * @brief Get device architecture
 */
ROCmArch get_rocm_device_arch(int device_id = -1);

/**
 * @brief Get device compute units
 */
int get_rocm_compute_units(int device_id = -1);

/**
 * @brief Get device memory (bytes)
 */
size_t get_rocm_device_memory(int device_id = -1);

}  // namespace rocm
}  // namespace kernel
}  // namespace yirage
