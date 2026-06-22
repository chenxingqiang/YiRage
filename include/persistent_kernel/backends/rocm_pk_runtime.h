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
 * @file rocm_pk_runtime.h
 * @brief ROCm/HIP Persistent Kernel Runtime Implementation
 *
 * This file contains the device-side implementation of the persistent kernel
 * runtime for AMD GPUs using HIP. It mirrors the CUDA implementation but
 * with HIP-specific optimizations for CDNA architectures.
 *
 * Key differences from CUDA:
 * - 64-thread wavefronts instead of 32-thread warps
 * - LDS (Local Data Share) instead of shared memory
 * - MFMA instead of Tensor Cores
 * - Different memory hierarchy
 */

#include "persistent_kernel/pk_runtime_core.h"
#include "persistent_kernel/tasks/rocm/task_header.h"

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include <hip/hip_runtime.h>
#endif

namespace yirage {
namespace persistent_kernel {
namespace rocm {

// =============================================================================
// HIP Macros (matching CUDA equivalents)
// =============================================================================

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#define HIP_CHECK(call)                                                       \
    do {                                                                       \
        hipError_t err = call;                                                \
        if (err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, hipGetErrorString(err));              \
            abort();                                                           \
        }                                                                      \
    } while (0)
#else
#define HIP_CHECK(call) (void)(call)
#endif

// =============================================================================
// Device-side Atomic Operations (64-thread wavefront)
// =============================================================================

#ifdef __HIP_DEVICE_COMPILE__

/**
 * @brief Atomically increment and get previous value
 */
__device__ __forceinline__ int atomicInc_rocm(int* addr) {
    return atomicAdd(addr, 1);
}

/**
 * @brief Atomically decrement and get previous value
 */
__device__ __forceinline__ int atomicDec_rocm(int* addr) {
    return atomicSub(addr, 1);
}

/**
 * @brief Memory fence for LDS
 */
__device__ __forceinline__ void lds_fence() {
    __threadfence_block();
}

/**
 * @brief Memory fence for global memory
 */
__device__ __forceinline__ void global_fence() {
    __threadfence();
}

/**
 * @brief Wavefront barrier (all 64 threads)
 */
__device__ __forceinline__ void wavefront_barrier() {
    __syncthreads();
}

/**
 * @brief Wavefront vote all
 */
__device__ __forceinline__ bool wavefront_all(bool predicate) {
    return __all(predicate);
}

/**
 * @brief Wavefront vote any
 */
__device__ __forceinline__ bool wavefront_any(bool predicate) {
    return __any(predicate);
}

/**
 * @brief Wavefront ballot (64-bit for 64-thread wavefront)
 */
__device__ __forceinline__ uint64_t wavefront_ballot(bool predicate) {
    return __ballot(predicate);
}

/**
 * @brief Wavefront shuffle (broadcast from lane)
 */
template<typename T>
__device__ __forceinline__ T wavefront_shuffle(T val, int src_lane) {
    return __shfl(val, src_lane);
}

/**
 * @brief Wavefront shuffle XOR
 */
template<typename T>
__device__ __forceinline__ T wavefront_shuffle_xor(T val, int lane_mask) {
    return __shfl_xor(val, lane_mask);
}

/**
 * @brief Wavefront shuffle down
 */
template<typename T>
__device__ __forceinline__ T wavefront_shuffle_down(T val, int delta) {
    return __shfl_down(val, delta);
}

/**
 * @brief Wavefront reduce sum (64 threads)
 */
template<typename T>
__device__ __forceinline__ T wavefront_reduce_sum(T val) {
    // 64-thread wavefront: log2(64) = 6 steps
    val += wavefront_shuffle_xor(val, 32);
    val += wavefront_shuffle_xor(val, 16);
    val += wavefront_shuffle_xor(val, 8);
    val += wavefront_shuffle_xor(val, 4);
    val += wavefront_shuffle_xor(val, 2);
    val += wavefront_shuffle_xor(val, 1);
    return val;
}

/**
 * @brief Wavefront reduce max (64 threads)
 */
template<typename T>
__device__ __forceinline__ T wavefront_reduce_max(T val) {
    val = max(val, wavefront_shuffle_xor(val, 32));
    val = max(val, wavefront_shuffle_xor(val, 16));
    val = max(val, wavefront_shuffle_xor(val, 8));
    val = max(val, wavefront_shuffle_xor(val, 4));
    val = max(val, wavefront_shuffle_xor(val, 2));
    val = max(val, wavefront_shuffle_xor(val, 1));
    return val;
}

#endif  // __HIP_DEVICE_COMPILE__

// =============================================================================
// Task ID / Event ID Utilities (matching CUDA implementation)
// =============================================================================

/**
 * @brief Get device ID from event ID
 */
inline int get_event_device_id(int event_id) {
    return event_id >> 24;
}

/**
 * @brief Get local event ID
 */
inline int get_local_event_id(int event_id) {
    return event_id & 0x00FFFFFF;
}

/**
 * @brief Make global event ID
 */
inline int make_event_id(int device_id, int local_id) {
    return (device_id << 24) | (local_id & 0x00FFFFFF);
}

/**
 * @brief Get device ID from task ID
 */
inline int get_task_device_id(int task_id) {
    return task_id >> 24;
}

/**
 * @brief Get local task ID
 */
inline int get_local_task_id(int task_id) {
    return task_id & 0x00FFFFFF;
}

/**
 * @brief Make global task ID
 */
inline int make_task_id(int device_id, int local_id) {
    return (device_id << 24) | (local_id & 0x00FFFFFF);
}

// =============================================================================
// ROCm Runtime Configuration
// =============================================================================

/**
 * @brief ROCm-specific runtime configuration
 */
struct ROCmRuntimeConfig {
    // Device info
    int device_id;
    AMDArch architecture;
    int compute_units;
    int wavefront_size;
    size_t lds_per_cu;
    
    // Kernel configuration
    int num_worker_blocks;
    int threads_per_block;
    int waves_per_block;
    
    // Memory
    size_t lds_usage;
    bool use_managed_memory;
    
    // Features
    bool use_mfma;
    bool use_fp8;
    bool use_sparsity;
    bool use_async_copy;
    
    // Streams
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipStream_t worker_stream;
    hipStream_t scheduler_stream;
#else
    void* worker_stream;
    void* scheduler_stream;
#endif
};

// =============================================================================
// Device-side Worker Kernel
// =============================================================================

#ifdef __HIP_DEVICE_COMPILE__

/**
 * @brief Execute a single task on ROCm device
 * @param task Task descriptor
 * @param config Runtime configuration
 */
__device__ void execute_task_rocm(const PKTaskDesc& task,
                                   PKRuntimeConfig* config) {
    // Get task parameters
    void* params = task.params;
    PKTaskType type = task.type;
    
    // Dispatch based on task type
    // Note: Actual implementation would call architecture-specific kernels
    switch (type) {
        case PKTaskType::GEMM:
            // Call appropriate GEMM kernel based on architecture
            break;
        case PKTaskType::RMSNORM:
            // Call RMSNorm kernel
            break;
        case PKTaskType::SILU_MUL:
            // Call SiLU+Mul kernel
            break;
        case PKTaskType::SOFTMAX:
            // Call Softmax kernel
            break;
        case PKTaskType::ATTENTION:
            // Call Attention kernel
            break;
        default:
            break;
    }
}

/**
 * @brief ROCm Worker kernel - processes tasks from queue
 */
__global__ void rocm_worker_kernel(PKRuntimeConfig* config) {
    __shared__ int shared_task_id;
    __shared__ PKTaskDesc shared_task;
    
    int tid = threadIdx.x;
    int wave_id = tid / ROCM_WAVEFRONT_SIZE;
    int lane_id = tid % ROCM_WAVEFRONT_SIZE;
    
    while (true) {
        // Leader thread fetches next task
        if (tid == 0) {
            // Atomically get next task from queue
            shared_task_id = atomicInc_rocm(config->worker_queue_next_task_id);
        }
        __syncthreads();
        
        int task_id = shared_task_id;
        
        // Check for termination
        if (task_id >= config->total_num_tasks) {
            break;
        }
        
        // Wait for task to be ready
        if (tid == 0) {
            while (atomicAdd(config->task_ready_flags + task_id, 0) == 0) {
                // Spin wait
            }
            shared_task = config->all_tasks[task_id];
        }
        __syncthreads();
        
        // Execute task
        execute_task_rocm(shared_task, config);
        
        // Signal completion
        if (tid == 0) {
            atomicAdd(config->task_complete_flags + task_id, 1);
        }
        __syncthreads();
    }
}

#endif  // __HIP_DEVICE_COMPILE__

// =============================================================================
// Host-side Runtime Functions
// =============================================================================

/**
 * @brief Initialize ROCm runtime
 */
void init_rocm_runtime(ROCmRuntimeConfig& config);

/**
 * @brief Finalize ROCm runtime
 */
void finalize_rocm_runtime(ROCmRuntimeConfig& config);

/**
 * @brief Launch ROCm worker kernel
 */
void launch_rocm_worker(const ROCmRuntimeConfig& config,
                        PKRuntimeConfig* device_config);

/**
 * @brief Launch ROCm scheduler kernel
 */
void launch_rocm_scheduler(const ROCmRuntimeConfig& config,
                           PKRuntimeConfig* device_config);

/**
 * @brief Synchronize ROCm streams
 */
void sync_rocm_runtime(const ROCmRuntimeConfig& config);

/**
 * @brief Get optimal launch configuration
 */
void get_rocm_launch_config(AMDArch arch,
                            int& num_blocks,
                            int& threads_per_block,
                            size_t& lds_size);

// =============================================================================
// Memory Management
// =============================================================================

/**
 * @brief Allocate device memory
 */
void* rocm_malloc(size_t size);

/**
 * @brief Allocate managed memory (accessible from host and device)
 */
void* rocm_malloc_managed(size_t size);

/**
 * @brief Free device memory
 */
void rocm_free(void* ptr);

/**
 * @brief Copy memory
 */
void rocm_memcpy(void* dst, const void* src, size_t size, int direction);

/**
 * @brief Set memory
 */
void rocm_memset(void* ptr, int value, size_t size);

}  // namespace rocm
}  // namespace persistent_kernel
}  // namespace yirage
