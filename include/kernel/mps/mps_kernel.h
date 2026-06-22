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
 * MPS Kernel Interface for Apple Silicon
 * 
 * Main header for Metal Performance Shaders kernel implementations.
 */

#pragma once

#include "kernel/mps/mps_kernel_config.h"

#include <string>
#include <memory>

#ifdef __APPLE__
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#else
// Forward declarations for non-Objective-C++ code
typedef void* MTLDevice;
typedef void* MTLCommandQueue;
typedef void* MTLLibrary;
typedef void* MTLComputePipelineState;
typedef void* MTLBuffer;
typedef void* MTLCommandBuffer;
#endif
#endif

namespace yirage {
namespace kernel {
namespace mps {

// =============================================================================
// MPS Kernel Manager
// =============================================================================

/**
 * @brief Manages Metal compute pipelines and kernel execution
 */
class MPSKernelManager {
public:
    MPSKernelManager();
    ~MPSKernelManager();
    
    /**
     * @brief Initialize with default device
     */
    bool initialize();
    
    /**
     * @brief Check if MPS is available
     */
    static bool is_available();
    
    /**
     * @brief Get device name
     */
    std::string get_device_name() const;
    
    /**
     * @brief Get GPU family
     */
    int get_gpu_family() const;
    
    /**
     * @brief Compile Metal shader library from source
     */
    bool compile_library(const std::string& source);
    
    /**
     * @brief Load pre-compiled Metal library
     */
    bool load_library(const std::string& path);
    
    /**
     * @brief Get compute pipeline for kernel
     */
    void* get_pipeline(const std::string& kernel_name);
    
    /**
     * @brief Synchronize all pending operations
     */
    void synchronize();
    
    // Accessors
#ifdef __APPLE__
    void* get_device() const { return device_; }
    void* get_command_queue() const { return command_queue_; }
#endif

private:
    bool initialized_;
    int gpu_family_;
    
#ifdef __APPLE__
    void* device_;           // id<MTLDevice>
    void* command_queue_;    // id<MTLCommandQueue>
    void* library_;          // id<MTLLibrary>
    std::unordered_map<std::string, void*> pipelines_;  // Cached pipelines
#endif
};

// =============================================================================
// High-Level Kernel Executor
// =============================================================================

/**
 * @brief High-level interface for executing MPS kernels
 */
class MPSKernelExecutor {
public:
    MPSKernelExecutor();
    ~MPSKernelExecutor();
    
    /**
     * @brief Initialize executor
     */
    bool initialize();
    
    // GEMM operations
    void gemm_f32(const float* A, const float* B, float* C,
                  int M, int N, int K,
                  float alpha = 1.0f, float beta = 0.0f);
    
    void gemm_f16(const void* A, const void* B, void* C,
                  int M, int N, int K);
    
    // RMSNorm
    void rms_norm_f16(const void* input, const void* weight, void* output,
                      int num_tokens, int hidden_dim, float eps = 1e-5f);
    
    // Softmax
    void softmax_f16(const void* input, void* output,
                     int num_rows, int row_size);
    
    // Element-wise operations
    void add_f16(const void* a, const void* b, void* c, int size);
    void mul_f16(const void* a, const void* b, void* c, int size);
    void silu_mul_f16(const void* gate, const void* up, void* output, int size);
    
    // Attention
    void attention_f16(const void* Q, const void* K, const void* V, void* output,
                       int batch_size, int num_heads, int seq_len, int head_dim,
                       float scale, bool causal = true);
    
    // Embedding
    void embedding_lookup_f16(const int* token_ids, const void* table, void* output,
                              int num_tokens, int embedding_dim, int vocab_size);
    
    // Reduction operations (corresponds to reduction_kernel.metal)
    void reduce_sum_f16(const void* input, void* output, int num_rows, int row_size);
    void reduce_max_f16(const void* input, void* output, int num_rows, int row_size);
    void reduce_mean_f16(const void* input, void* output, int num_rows, int row_size);
    
    // Tensor operations (corresponds to tensor_ops_kernel.metal)
    void transpose_f16(const void* input, void* output, int rows, int cols);
    void concat_f16(const void* const* inputs, void* output,
                    const int* input_sizes, int num_inputs);
    void slice_f16(const void* input, void* output,
                   int start_row, int end_row, int start_col, int end_col);
    void copy_f16(const void* src, void* dst, int size);
    void fill_f16(void* data, float value, int size);
    
    // Synchronize
    void synchronize();

private:
    std::unique_ptr<MPSKernelManager> manager_;
    bool initialized_;
};

// =============================================================================
// Buffer Management
// =============================================================================

/**
 * @brief Allocate Metal buffer
 * @param size Size in bytes
 * @return Pointer to buffer (MTLBuffer*)
 */
void* mps_malloc(size_t size);

/**
 * @brief Free Metal buffer
 */
void mps_free(void* ptr);

/**
 * @brief Copy host to device
 */
void mps_memcpy_h2d(void* dst, const void* src, size_t size);

/**
 * @brief Copy device to host
 */
void mps_memcpy_d2h(void* dst, const void* src, size_t size);

/**
 * @brief Copy device to device
 */
void mps_memcpy_d2d(void* dst, const void* src, size_t size);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if MPS is available
 */
bool is_mps_available();

/**
 * @brief Get Metal device name
 */
std::string get_mps_device_name();

/**
 * @brief Get Apple GPU family (7=M1, 8=M2, 9=M3, 10=M4)
 */
int get_apple_gpu_family();

/**
 * @brief Get unified memory size
 */
size_t get_mps_unified_memory();

/**
 * @brief Get recommended max working set size
 */
size_t get_mps_max_buffer_length();

}  // namespace mps
}  // namespace kernel
}  // namespace yirage
