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

#include "persistent_kernel/backends/cuda_pk_backend.h"
#include <iostream>
#include <sstream>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// CudaMemoryAllocator Implementation
// =============================================================================

CudaMemoryAllocator::CudaMemoryAllocator()
    : device_id_(0), use_nvshmem_(false) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaGetDevice(&device_id_);
#endif
}

CudaMemoryAllocator::~CudaMemoryAllocator() = default;

void* CudaMemoryAllocator::allocate(size_t size) {
    void* ptr = nullptr;
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#ifdef USE_NVSHMEM
    if (use_nvshmem_) {
        ptr = nvshmem_malloc(size);
    } else {
        cudaMalloc(&ptr, size);
    }
#else
    cudaMalloc(&ptr, size);
#endif
#endif
    return ptr;
}

void CudaMemoryAllocator::free(void* ptr) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    if (ptr) {
#ifdef USE_NVSHMEM
        if (use_nvshmem_) {
            nvshmem_free(ptr);
        } else {
            cudaFree(ptr);
        }
#else
        cudaFree(ptr);
#endif
    }
#endif
}

void CudaMemoryAllocator::copy_h2d(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
#endif
}

void CudaMemoryAllocator::copy_d2h(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
#endif
}

void CudaMemoryAllocator::copy_d2d(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
#endif
}

void CudaMemoryAllocator::copy_h2d_async(void* dst, const void* src, 
                                          size_t size, void* stream) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream));
#endif
}

void CudaMemoryAllocator::memset(void* ptr, int value, size_t size) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaMemset(ptr, value, size);
#endif
}

size_t CudaMemoryAllocator::get_total_memory() const {
    size_t total = 0;
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    size_t free;
    cudaMemGetInfo(&free, &total);
#endif
    return total;
}

size_t CudaMemoryAllocator::get_free_memory() const {
    size_t free = 0;
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    size_t total;
    cudaMemGetInfo(&free, &total);
#endif
    return free;
}

// =============================================================================
// CudaAtomicOps Implementation
// =============================================================================

CudaAtomicOps::CudaAtomicOps() = default;
CudaAtomicOps::~CudaAtomicOps() = default;

// Note: These are host-side stubs. Actual atomics are in device code.
uint64_t CudaAtomicOps::fetch_add_u64(uint64_t* addr, uint64_t val) {
    // Host-side stub - actual implementation is in mpk_atoms.cuh
    return 0;
}

uint64_t CudaAtomicOps::fetch_sub_u64(uint64_t* addr, uint64_t val) {
    return 0;
}

uint64_t CudaAtomicOps::compare_exchange_u64(uint64_t* addr, uint64_t expected,
                                              uint64_t desired) {
    return 0;
}

void CudaAtomicOps::store_release_u64(uint64_t* addr, uint64_t val) {
    // Host-side stub
}

uint64_t CudaAtomicOps::load_acquire_u64(uint64_t* addr) {
    return 0;
}

uint32_t CudaAtomicOps::fetch_add_u32(uint32_t* addr, uint32_t val) {
    return 0;
}

uint32_t CudaAtomicOps::fetch_sub_u32(uint32_t* addr, uint32_t val) {
    return 0;
}

uint32_t CudaAtomicOps::compare_exchange_u32(uint32_t* addr, uint32_t expected,
                                              uint32_t desired) {
    return 0;
}

void CudaAtomicOps::memory_fence() {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaDeviceSynchronize();
#endif
}

void CudaAtomicOps::thread_fence() {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaDeviceSynchronize();
#endif
}

// =============================================================================
// CudaTaskExecutor Implementation
// =============================================================================

CudaTaskExecutor::CudaTaskExecutor() : compute_capability_(0) {}
CudaTaskExecutor::~CudaTaskExecutor() = default;

bool CudaTaskExecutor::supports_task(PKTaskType type) const {
    switch (type) {
        case PKTaskType::TERMINATE:
        case PKTaskType::BEGIN_TASK_GRAPH:
        case PKTaskType::EMBEDDING:
        case PKTaskType::RMS_NORM:
        case PKTaskType::RMS_NORM_LINEAR:
        case PKTaskType::LINEAR:
        case PKTaskType::LINEAR_RESIDUAL:
        case PKTaskType::ATTENTION:
        case PKTaskType::PAGED_ATTENTION:
        case PKTaskType::SILU_MUL:
        case PKTaskType::SILU_MUL_LINEAR:
        case PKTaskType::ARGMAX:
        case PKTaskType::ROTARY_EMBEDDING:
        case PKTaskType::MOE_GATE:
        case PKTaskType::MOE_LINEAR:
        case PKTaskType::ALLREDUCE:
        case PKTaskType::REDUCE:
        case PKTaskType::NVSHMEM_COPY:
            return true;
        default:
            return false;
    }
}

void CudaTaskExecutor::execute(const PKTaskDesc& desc,
                                const PKRuntimeConfig& config,
                                void* shared_memory,
                                size_t shared_memory_size) {
    // Task execution is handled by device kernels
    // This is a placeholder for kernel dispatch logic
}

size_t CudaTaskExecutor::get_shared_memory_size(PKTaskType type) const {
    // Return required shared memory based on compute capability
    if (compute_capability_ >= 90) {
        return 227 * 1024 - 6 * 1024;  // Hopper
    } else if (compute_capability_ >= 86) {
        return 99 * 1024 - 3 * 1024;
    } else if (compute_capability_ >= 80) {
        return 163 * 1024 - 3 * 1024;  // Ampere
    }
    return 48 * 1024;  // Default
}

const char* CudaTaskExecutor::get_task_name(PKTaskType type) const {
    switch (type) {
        case PKTaskType::TERMINATE: return "TERMINATE";
        case PKTaskType::BEGIN_TASK_GRAPH: return "BEGIN_TASK_GRAPH";
        case PKTaskType::EMBEDDING: return "EMBEDDING";
        case PKTaskType::RMS_NORM: return "RMS_NORM";
        case PKTaskType::RMS_NORM_LINEAR: return "RMS_NORM_LINEAR";
        case PKTaskType::LINEAR: return "LINEAR";
        case PKTaskType::LINEAR_RESIDUAL: return "LINEAR_RESIDUAL";
        case PKTaskType::ATTENTION: return "ATTENTION";
        case PKTaskType::PAGED_ATTENTION: return "PAGED_ATTENTION";
        case PKTaskType::SILU_MUL: return "SILU_MUL";
        case PKTaskType::SILU_MUL_LINEAR: return "SILU_MUL_LINEAR";
        case PKTaskType::ARGMAX: return "ARGMAX";
        case PKTaskType::ROTARY_EMBEDDING: return "ROTARY_EMBEDDING";
        case PKTaskType::MOE_GATE: return "MOE_GATE";
        case PKTaskType::MOE_LINEAR: return "MOE_LINEAR";
        case PKTaskType::ALLREDUCE: return "ALLREDUCE";
        case PKTaskType::REDUCE: return "REDUCE";
        case PKTaskType::NVSHMEM_COPY: return "NVSHMEM_COPY";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// CudaPKBackend Implementation
// =============================================================================

CudaPKBackend::CudaPKBackend(int device_id)
    : device_id_(device_id),
      initialized_(false),
      compute_capability_major_(0),
      compute_capability_minor_(0),
      worker_stream_(nullptr),
      scheduler_stream_(nullptr) {
    detect_capabilities();
    allocator_ = std::make_unique<CudaMemoryAllocator>();
    atomic_ops_ = std::make_unique<CudaAtomicOps>();
    executor_ = std::make_unique<CudaTaskExecutor>();
}

CudaPKBackend::~CudaPKBackend() {
    if (initialized_) {
        finalize();
    }
}

void CudaPKBackend::detect_capabilities() {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return;
    }
    
    if (device_id_ >= device_count) {
        device_id_ = 0;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    compute_capability_major_ = prop.major;
    compute_capability_minor_ = prop.minor;
#endif
}

PKBackendType CudaPKBackend::get_type() const {
    return PKBackendType::CUDA;
}

std::string CudaPKBackend::get_name() const {
    return "cuda";
}

std::string CudaPKBackend::get_display_name() const {
    std::stringstream ss;
    ss << "NVIDIA CUDA (SM " << compute_capability_major_ << "." 
       << compute_capability_minor_ << ")";
    return ss.str();
}

bool CudaPKBackend::is_available() const {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

PKCapabilities CudaPKBackend::get_capabilities() const {
    PKCapabilities caps;
    caps.supports_tma = (compute_capability_major_ >= 9);  // Hopper+
    caps.supports_tensor_cores = (compute_capability_major_ >= 7);  // Volta+
    caps.supports_async_copy = (compute_capability_major_ >= 8);  // Ampere+
    caps.supports_nvshmem = true;
    caps.supports_fp8 = (compute_capability_major_ >= 9);  // Hopper+
    
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    caps.max_shared_memory = prop.sharedMemPerBlockOptin;
    caps.max_global_memory = prop.totalGlobalMem;
    caps.max_threads_per_block = prop.maxThreadsPerBlock;
    caps.max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
#else
    caps.max_shared_memory = 0;
    caps.max_global_memory = 0;
    caps.max_threads_per_block = 0;
    caps.max_blocks_per_sm = 0;
#endif
    
    caps.compute_major = compute_capability_major_;
    caps.compute_minor = compute_capability_minor_;
    
    caps.supported_modes = get_supported_modes();
    caps.supported_dtypes = {
        PKDataType::FP32, PKDataType::FP16, PKDataType::BF16,
        PKDataType::INT8
    };
    if (caps.supports_fp8) {
        caps.supported_dtypes.push_back(PKDataType::FP8_E4M3);
        caps.supported_dtypes.push_back(PKDataType::FP8_E5M2);
    }
    
    return caps;
}

bool CudaPKBackend::supports_mode(PKMode mode) const {
    // CUDA: OFFLINE, ONLINE, ONEPASS, GRAPH
    switch (mode) {
        case PKMode::OFFLINE:
        case PKMode::ONLINE:
        case PKMode::ONEPASS:
        case PKMode::GRAPH:
            return true;
        case PKMode::EAGER:
        case PKMode::STREAMING:
            return false;  // Not in workload plan
        default:
            return false;
    }
}

PKMode CudaPKBackend::get_default_mode() const {
    return PKMode::ONLINE;
}

std::vector<PKMode> CudaPKBackend::get_supported_modes() const {
    // CUDA: OFFLINE, ONLINE, ONEPASS, GRAPH (1 week refactor)
    return {
        PKMode::OFFLINE,
        PKMode::ONLINE,
        PKMode::ONEPASS,
        PKMode::GRAPH,
        PKMode::STREAMING
    };
}

PKMemoryAllocator* CudaPKBackend::get_allocator() {
    return allocator_.get();
}

PKAtomicOps* CudaPKBackend::get_atomic_ops() {
    return atomic_ops_.get();
}

PKTaskExecutor* CudaPKBackend::get_executor() {
    return executor_.get();
}

bool CudaPKBackend::initialize(const PKRuntimeConfig& config) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        return false;
    }
    
    cudaStreamCreate(&worker_stream_);
    cudaStreamCreate(&scheduler_stream_);
    
    initialized_ = true;
    return true;
#else
    return false;
#endif
}

void CudaPKBackend::finalize() {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    if (initialized_) {
        if (worker_stream_) {
            cudaStreamDestroy(worker_stream_);
            worker_stream_ = nullptr;
        }
        if (scheduler_stream_) {
            cudaStreamDestroy(scheduler_stream_);
            scheduler_stream_ = nullptr;
        }
        initialized_ = false;
    }
#endif
}

void CudaPKBackend::reset() {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    if (worker_stream_) cudaStreamSynchronize(worker_stream_);
    if (scheduler_stream_) cudaStreamSynchronize(scheduler_stream_);
#endif
}

void* CudaPKBackend::create_stream() {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return static_cast<void*>(stream);
#else
    return nullptr;
#endif
}

void CudaPKBackend::destroy_stream(void* stream) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    if (stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }
#endif
}

void CudaPKBackend::synchronize_stream(void* stream) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    if (stream) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    }
#endif
}

void CudaPKBackend::synchronize() {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaDeviceSynchronize();
#endif
}

void CudaPKBackend::launch_worker_kernel(const PKRuntimeConfig& config,
                                          int num_workers,
                                          int threads_per_worker) {
    // Kernel launch is handled in persistent_kernel.cuh
}

void CudaPKBackend::launch_scheduler_kernel(const PKRuntimeConfig& config) {
    // Kernel launch is handled in persistent_kernel.cuh
}

bool CudaPKBackend::prepare_next_batch(PKRuntimeConfig& config) {
    // Mode-specific batch preparation handled in device code
    return true;
}

void CudaPKBackend::process_batch_results(PKRuntimeConfig& config) {
    // Mode-specific result processing
}

bool CudaPKBackend::set_device(int device_id) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaError_t err = cudaSetDevice(device_id);
    if (err == cudaSuccess) {
        device_id_ = device_id;
        detect_capabilities();
        return true;
    }
#endif
    return false;
}

int CudaPKBackend::get_device() const {
    return device_id_;
}

int CudaPKBackend::get_device_count() const {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
#else
    return 0;
#endif
}

std::vector<std::string> CudaPKBackend::get_compile_flags(PKMode mode) const {
    std::vector<std::string> flags;
    
    int cc = compute_capability_major_ * 10 + compute_capability_minor_;
    flags.push_back("-DYPK_TARGET_CC=" + std::to_string(cc));
    
    switch (mode) {
        case PKMode::OFFLINE:
            flags.push_back("-DMODE_OFFLINE");
            break;
        case PKMode::ONLINE:
            flags.push_back("-DMODE_ONLINE");
            break;
        case PKMode::ONEPASS:
            flags.push_back("-DMODE_ONEPASS");
            break;
        case PKMode::EAGER:
            flags.push_back("-DMODE_EAGER");
            break;
        case PKMode::GRAPH:
            flags.push_back("-DMODE_GRAPH");
            break;
        case PKMode::STREAMING:
            flags.push_back("-DMODE_STREAMING");
            break;
        default:
            break;
    }
    
    // Architecture-specific flags
    if (compute_capability_major_ >= 9) {
        flags.push_back("-DYIRAGE_GRACE_HOPPER");
        flags.push_back("-DYPK_ENABLE_TMA");
    } else if (compute_capability_major_ >= 10) {
        flags.push_back("-DYIRAGE_GRACE_BLACKWELL");
        flags.push_back("-DYPK_ENABLE_TMA");
    }
    
    // Common CUDA flags
    flags.push_back("--expt-relaxed-constexpr");
    flags.push_back("-std=c++17");
    flags.push_back("-arch=sm_" + std::to_string(cc));
    
    return flags;
}

std::vector<std::string> CudaPKBackend::get_include_dirs() const {
    return {
        "${CUDA_HOME}/include",
        "${CUTLASS_HOME}/include"
    };
}

int CudaPKBackend::get_compute_capability() const {
    return compute_capability_major_ * 10 + compute_capability_minor_;
}

size_t CudaPKBackend::get_max_shared_memory() const {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    return prop.sharedMemPerBlockOptin;
#else
    return 0;
#endif
}

bool CudaPKBackend::supports_tma() const {
    return compute_capability_major_ >= 9;
}

bool CudaPKBackend::supports_nvshmem() const {
#ifdef USE_NVSHMEM
    return true;
#else
    return false;
#endif
}

} // namespace persistent_kernel
} // namespace yirage
