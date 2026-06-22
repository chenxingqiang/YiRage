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

#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Supported persistent kernel backends
 */
enum class PKBackendType {
    CUDA = 0,      // NVIDIA CUDA
    CPU = 1,       // CPU (x86/ARM)
    MPS = 2,       // Apple Metal
    ASCEND = 3,    // Huawei Ascend NPU
    MACA = 4,      // MetaX MACA GPU
    TRITON = 5,    // OpenAI Triton
    NKI = 6,       // AWS Neuron
    ROCM = 7,      // AMD ROCm/HIP
    NUM_BACKENDS
};

/**
 * @brief Persistent kernel execution modes
 */
enum class PKMode {
    OFFLINE = 0,    // Pre-compile all kernels, batch processing
    ONLINE = 1,     // JIT compile as needed, single request
    ONEPASS = 2,    // Single-pass execution
    EAGER = 3,      // Immediate execution (no compilation)
    GRAPH = 4,      // Graph-based execution
    STREAMING = 5,  // Streaming/pipelined execution
    NUM_MODES
};

/**
 * @brief Data types supported by persistent kernel
 */
enum class PKDataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    INT4 = 4,
    FP8_E4M3 = 5,
    FP8_E5M2 = 6,
    NUM_TYPES
};

// =============================================================================
// Capability Structures
// =============================================================================

/**
 * @brief Backend capability description
 */
struct PKCapabilities {
    // Hardware features
    bool supports_tma;              // Tensor Memory Accelerator (Hopper+)
    bool supports_tensor_cores;     // Tensor cores / AI cores
    bool supports_async_copy;       // Async memory copy
    bool supports_nvshmem;          // NVSHMEM for multi-GPU
    bool supports_fp8;              // FP8 data type
    
    // Memory limits
    size_t max_shared_memory;       // Max shared memory per block
    size_t max_global_memory;       // Max global memory
    size_t max_threads_per_block;   // Max threads per block
    size_t max_blocks_per_sm;       // Max blocks per SM
    
    // Compute capability
    int compute_major;              // Major version (e.g., 9 for Hopper)
    int compute_minor;              // Minor version (e.g., 0)
    
    // Supported modes
    std::vector<PKMode> supported_modes;
    
    // Supported data types
    std::vector<PKDataType> supported_dtypes;
};

// =============================================================================
// Runtime Configuration (Backend-Agnostic)
// =============================================================================

/**
 * @brief Backend-agnostic runtime configuration
 */
struct PKRuntimeConfig {
    // Mode configuration
    PKMode mode;
    
    // Worker configuration
    int num_workers;
    int num_local_schedulers;
    int num_remote_schedulers;
    int threads_per_worker;
    
    // Memory configuration
    size_t max_seq_length;
    size_t max_num_batched_requests;
    size_t max_num_batched_tokens;
    size_t max_num_pages;
    size_t page_size;
    
    // Token configuration
    int64_t eos_token_id;
    
    // Multi-GPU configuration
    int num_gpus;
    int my_gpu_id;
    
    // Backend-specific context (opaque pointer)
    void* backend_context;
    
    // Stream/queue handle (opaque pointer)
    void* stream_handle;
    
    // Profiling
    bool profiling_enabled;
    void* profiler_buffer;
};

// =============================================================================
// Task Description (Backend-Agnostic)
// =============================================================================

/**
 * @brief Task types for persistent kernel
 */
enum class PKTaskType {
    TERMINATE = 0,
    BEGIN_TASK_GRAPH = 10,
    
    // Compute tasks
    EMBEDDING = 101,
    RMS_NORM = 102,
    RMS_NORM_LINEAR = 103,
    LINEAR = 104,
    LINEAR_RESIDUAL = 105,
    ATTENTION = 106,
    PAGED_ATTENTION = 107,
    SILU_MUL = 108,
    SILU_MUL_LINEAR = 109,
    ARGMAX = 110,
    ROTARY_EMBEDDING = 111,
    
    // MOE tasks
    MOE_GATE = 120,
    MOE_LINEAR = 121,
    MOE_SILU_LINEAR = 122,
    
    // Communication tasks
    ALLREDUCE = 200,
    REDUCE = 201,
    NVSHMEM_COPY = 202,
    
    // Custom task
    CUSTOM = 999,
};

/**
 * @brief Tensor descriptor for task inputs/outputs
 */
struct PKTensorDesc {
    void* data;                     // Pointer to data
    PKDataType dtype;               // Data type
    int num_dims;                   // Number of dimensions
    int64_t dims[8];               // Dimension sizes
    int64_t strides[8];            // Strides
};

/**
 * @brief Task descriptor
 */
struct PKTaskDesc {
    PKTaskType type;
    int task_id;
    
    // Inputs and outputs
    int num_inputs;
    int num_outputs;
    PKTensorDesc inputs[8];
    PKTensorDesc outputs[4];
    
    // Task-specific configuration
    void* config;
    size_t config_size;
    
    // Task parameters (for task kernels)
    void* params;
    size_t params_size;
};

// =============================================================================
// Memory Interface
// =============================================================================

/**
 * @brief Abstract memory allocator interface
 */
class PKMemoryAllocator {
public:
    virtual ~PKMemoryAllocator() = default;
    
    /**
     * @brief Allocate device memory
     * @param size Size in bytes
     * @return Pointer to allocated memory
     */
    virtual void* allocate(size_t size) = 0;
    
    /**
     * @brief Free device memory
     * @param ptr Pointer to free
     */
    virtual void free(void* ptr) = 0;
    
    /**
     * @brief Copy from host to device
     */
    virtual void copy_h2d(void* dst, const void* src, size_t size) = 0;
    
    /**
     * @brief Copy from device to host
     */
    virtual void copy_d2h(void* dst, const void* src, size_t size) = 0;
    
    /**
     * @brief Copy from device to device
     */
    virtual void copy_d2d(void* dst, const void* src, size_t size) = 0;
    
    /**
     * @brief Async copy from host to device
     */
    virtual void copy_h2d_async(void* dst, const void* src, size_t size, 
                                 void* stream) = 0;
    
    /**
     * @brief Set memory to value
     */
    virtual void memset(void* ptr, int value, size_t size) = 0;
    
    /**
     * @brief Get total device memory
     */
    virtual size_t get_total_memory() const = 0;
    
    /**
     * @brief Get free device memory
     */
    virtual size_t get_free_memory() const = 0;
};

// =============================================================================
// Atomic Operations Interface
// =============================================================================

/**
 * @brief Abstract atomic operations interface
 */
class PKAtomicOps {
public:
    virtual ~PKAtomicOps() = default;
    
    // 64-bit atomic operations
    virtual uint64_t fetch_add_u64(uint64_t* addr, uint64_t val) = 0;
    virtual uint64_t fetch_sub_u64(uint64_t* addr, uint64_t val) = 0;
    virtual uint64_t compare_exchange_u64(uint64_t* addr, uint64_t expected, 
                                           uint64_t desired) = 0;
    virtual void store_release_u64(uint64_t* addr, uint64_t val) = 0;
    virtual uint64_t load_acquire_u64(uint64_t* addr) = 0;
    
    // 32-bit atomic operations
    virtual uint32_t fetch_add_u32(uint32_t* addr, uint32_t val) = 0;
    virtual uint32_t fetch_sub_u32(uint32_t* addr, uint32_t val) = 0;
    virtual uint32_t compare_exchange_u32(uint32_t* addr, uint32_t expected,
                                           uint32_t desired) = 0;
    
    // Memory fence
    virtual void memory_fence() = 0;
    virtual void thread_fence() = 0;
};

// =============================================================================
// Task Executor Interface
// =============================================================================

/**
 * @brief Abstract task executor interface
 */
class PKTaskExecutor {
public:
    virtual ~PKTaskExecutor() = default;
    
    /**
     * @brief Check if this executor supports a task type
     */
    virtual bool supports_task(PKTaskType type) const = 0;
    
    /**
     * @brief Execute a task
     * @param desc Task descriptor
     * @param config Runtime configuration
     * @param shared_memory Shared memory pointer
     * @param shared_memory_size Size of shared memory
     */
    virtual void execute(const PKTaskDesc& desc,
                         const PKRuntimeConfig& config,
                         void* shared_memory,
                         size_t shared_memory_size) = 0;
    
    /**
     * @brief Get required shared memory size for a task
     */
    virtual size_t get_shared_memory_size(PKTaskType type) const = 0;
    
    /**
     * @brief Get task name
     */
    virtual const char* get_task_name(PKTaskType type) const = 0;
};

// =============================================================================
// Persistent Kernel Backend Interface
// =============================================================================

/**
 * @brief Abstract persistent kernel backend interface
 * 
 * All backend implementations (CUDA, CPU, Ascend, etc.) must implement
 * this interface to provide persistent kernel functionality.
 */
class PKBackendInterface {
public:
    virtual ~PKBackendInterface() = default;
    
    // ========== Backend Information ==========
    
    /**
     * @brief Get backend type
     */
    virtual PKBackendType get_type() const = 0;
    
    /**
     * @brief Get backend name (e.g., "cuda", "cpu", "ascend")
     */
    virtual std::string get_name() const = 0;
    
    /**
     * @brief Get display name (e.g., "NVIDIA CUDA", "CPU", "Huawei Ascend")
     */
    virtual std::string get_display_name() const = 0;
    
    /**
     * @brief Check if backend is available on this system
     */
    virtual bool is_available() const = 0;
    
    /**
     * @brief Get backend capabilities
     */
    virtual PKCapabilities get_capabilities() const = 0;
    
    // ========== Mode Support ==========
    
    /**
     * @brief Check if a mode is supported
     */
    virtual bool supports_mode(PKMode mode) const = 0;
    
    /**
     * @brief Get default mode for this backend
     */
    virtual PKMode get_default_mode() const = 0;
    
    /**
     * @brief Get all supported modes
     */
    virtual std::vector<PKMode> get_supported_modes() const = 0;
    
    // ========== Component Access ==========
    
    /**
     * @brief Get memory allocator
     */
    virtual PKMemoryAllocator* get_allocator() = 0;
    
    /**
     * @brief Get atomic operations
     */
    virtual PKAtomicOps* get_atomic_ops() = 0;
    
    /**
     * @brief Get task executor
     */
    virtual PKTaskExecutor* get_executor() = 0;
    
    // ========== Initialization ==========
    
    /**
     * @brief Initialize backend with configuration
     */
    virtual bool initialize(const PKRuntimeConfig& config) = 0;
    
    /**
     * @brief Finalize backend and release resources
     */
    virtual void finalize() = 0;
    
    /**
     * @brief Reset backend for new session
     */
    virtual void reset() = 0;
    
    // ========== Stream/Queue Management ==========
    
    /**
     * @brief Create a new stream/queue
     */
    virtual void* create_stream() = 0;
    
    /**
     * @brief Destroy a stream/queue
     */
    virtual void destroy_stream(void* stream) = 0;
    
    /**
     * @brief Synchronize a stream/queue
     */
    virtual void synchronize_stream(void* stream) = 0;
    
    /**
     * @brief Synchronize all streams
     */
    virtual void synchronize() = 0;
    
    // ========== Kernel Launch ==========
    
    /**
     * @brief Launch worker kernel
     */
    virtual void launch_worker_kernel(const PKRuntimeConfig& config,
                                       int num_workers,
                                       int threads_per_worker) = 0;
    
    /**
     * @brief Launch scheduler kernel
     */
    virtual void launch_scheduler_kernel(const PKRuntimeConfig& config) = 0;
    
    // ========== Mode-Specific Operations ==========
    
    /**
     * @brief Prepare next batch (mode-specific)
     * @return true if more batches to process, false if done
     */
    virtual bool prepare_next_batch(PKRuntimeConfig& config) = 0;
    
    /**
     * @brief Process batch results (mode-specific)
     */
    virtual void process_batch_results(PKRuntimeConfig& config) = 0;
    
    // ========== Device Management ==========
    
    /**
     * @brief Set current device
     */
    virtual bool set_device(int device_id) = 0;
    
    /**
     * @brief Get current device
     */
    virtual int get_device() const = 0;
    
    /**
     * @brief Get number of available devices
     */
    virtual int get_device_count() const = 0;
    
    // ========== Compilation ==========
    
    /**
     * @brief Get compile flags for this backend and mode
     */
    virtual std::vector<std::string> get_compile_flags(PKMode mode) const = 0;
    
    /**
     * @brief Get include directories
     */
    virtual std::vector<std::string> get_include_dirs() const = 0;
};

// =============================================================================
// Backend Factory
// =============================================================================

/**
 * @brief Create a persistent kernel backend
 * @param type Backend type
 * @param device_id Device ID (for GPU backends)
 * @return Unique pointer to backend instance
 */
std::unique_ptr<PKBackendInterface> create_pk_backend(
    PKBackendType type, 
    int device_id = 0
);

/**
 * @brief Get available backends on this system
 */
std::vector<PKBackendType> get_available_pk_backends();

/**
 * @brief Get backend name from type
 */
const char* pk_backend_type_to_name(PKBackendType type);

/**
 * @brief Get mode name from type
 */
const char* pk_mode_to_name(PKMode mode);

/**
 * @brief Check if mode is supported by backend
 */
bool pk_is_mode_supported(PKBackendType backend, PKMode mode);

} // namespace persistent_kernel
} // namespace yirage
