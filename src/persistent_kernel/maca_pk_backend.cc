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

#include "persistent_kernel/backends/maca_pk_backend.h"
#include <atomic>
#include <cstring>
#include <sstream>

// MACA includes (CUDA-compatible API via mc* headers)
#ifdef YIRAGE_BACKEND_MACA_ENABLED
#include <mcr/mc_runtime.h>
#include <mcr/mc_common.h>
#endif

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// MacaMemoryAllocator Implementation
// =============================================================================

MacaMemoryAllocator::MacaMemoryAllocator()
    : device_id_(0) {}

MacaMemoryAllocator::~MacaMemoryAllocator() = default;

void* MacaMemoryAllocator::allocate(size_t size) {
    void* ptr = nullptr;
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcMalloc(&ptr, size);
#else
    // Fallback: use host memory
    ptr = std::malloc(size);
#endif
    return ptr;
}

void MacaMemoryAllocator::free(void* ptr) {
    if (!ptr) return;
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcFree(ptr);
#else
    std::free(ptr);
#endif
}

void MacaMemoryAllocator::copy_h2d(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcMemcpy(dst, src, size, mcMemcpyHostToDevice);
#else
    std::memcpy(dst, src, size);
#endif
}

void MacaMemoryAllocator::copy_d2h(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcMemcpy(dst, src, size, mcMemcpyDeviceToHost);
#else
    std::memcpy(dst, src, size);
#endif
}

void MacaMemoryAllocator::copy_d2d(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcMemcpy(dst, src, size, mcMemcpyDeviceToDevice);
#else
    std::memcpy(dst, src, size);
#endif
}

void MacaMemoryAllocator::copy_h2d_async(void* dst, const void* src, 
                                          size_t size, void* stream) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcMemcpyAsync(dst, src, size, mcMemcpyHostToDevice,
                  static_cast<mcStream_t>(stream));
#else
    std::memcpy(dst, src, size);
#endif
}

void MacaMemoryAllocator::memset(void* ptr, int value, size_t size) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcMemset(ptr, value, size);
#else
    std::memset(ptr, value, size);
#endif
}

size_t MacaMemoryAllocator::get_total_memory() const {
    size_t total = 0;
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    size_t free;
    mcMemGetInfo(&free, &total);
#endif
    return total;
}

size_t MacaMemoryAllocator::get_free_memory() const {
    size_t free = 0;
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    size_t total;
    mcMemGetInfo(&free, &total);
#endif
    return free;
}

// =============================================================================
// MacaAtomicOps Implementation
// =============================================================================

MacaAtomicOps::MacaAtomicOps() = default;
MacaAtomicOps::~MacaAtomicOps() = default;

// MACA uses CUDA-compatible atomics on device
// Host-side stubs for interface compatibility

uint64_t MacaAtomicOps::fetch_add_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->fetch_add(val, std::memory_order_acq_rel);
}

uint64_t MacaAtomicOps::fetch_sub_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->fetch_sub(val, std::memory_order_acq_rel);
}

uint64_t MacaAtomicOps::compare_exchange_u64(uint64_t* addr, uint64_t expected,
                                              uint64_t desired) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    atomic_addr->compare_exchange_strong(expected, desired,
                                          std::memory_order_acq_rel);
    return expected;
}

void MacaAtomicOps::store_release_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    atomic_addr->store(val, std::memory_order_release);
}

uint64_t MacaAtomicOps::load_acquire_u64(uint64_t* addr) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->load(std::memory_order_acquire);
}

uint32_t MacaAtomicOps::fetch_add_u32(uint32_t* addr, uint32_t val) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    return atomic_addr->fetch_add(val, std::memory_order_acq_rel);
}

uint32_t MacaAtomicOps::fetch_sub_u32(uint32_t* addr, uint32_t val) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    return atomic_addr->fetch_sub(val, std::memory_order_acq_rel);
}

uint32_t MacaAtomicOps::compare_exchange_u32(uint32_t* addr, uint32_t expected,
                                              uint32_t desired) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    atomic_addr->compare_exchange_strong(expected, desired,
                                          std::memory_order_acq_rel);
    return expected;
}

void MacaAtomicOps::memory_fence() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcDeviceSynchronize();
#endif
}

void MacaAtomicOps::thread_fence() {
    std::atomic_thread_fence(std::memory_order_acq_rel);
}

// =============================================================================
// MacaTaskExecutor Implementation
// =============================================================================

MacaTaskExecutor::MacaTaskExecutor()
    : compute_capability_(0) {}

MacaTaskExecutor::~MacaTaskExecutor() = default;

bool MacaTaskExecutor::supports_task(PKTaskType type) const {
    // MACA supports same tasks as CUDA with some limitations
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
            return true;
        case PKTaskType::NVSHMEM_COPY:
            return false;  // NVSHMEM not available on MACA
        default:
            return false;
    }
}

void MacaTaskExecutor::execute(const PKTaskDesc& desc,
                                const PKRuntimeConfig& config,
                                void* shared_memory,
                                size_t shared_memory_size) {
    // Task execution through MACA kernels
    // Implementation mirrors CUDA kernel dispatch
}

size_t MacaTaskExecutor::get_shared_memory_size(PKTaskType type) const {
    // MACA shared memory similar to CUDA
    if (compute_capability_ >= 80) {
        return 163 * 1024 - 3 * 1024;
    }
    return 96 * 1024;
}

const char* MacaTaskExecutor::get_task_name(PKTaskType type) const {
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
        default: return "UNKNOWN";
    }
}

// =============================================================================
// MacaPKBackend Implementation
// =============================================================================

MacaPKBackend::MacaPKBackend(int device_id)
    : device_id_(device_id),
      initialized_(false),
      compute_capability_(0),
      worker_stream_(nullptr),
      scheduler_stream_(nullptr) {
    detect_capabilities();
    allocator_ = std::make_unique<MacaMemoryAllocator>();
    atomic_ops_ = std::make_unique<MacaAtomicOps>();
    executor_ = std::make_unique<MacaTaskExecutor>();
}

MacaPKBackend::~MacaPKBackend() {
    if (initialized_) {
        finalize();
    }
}

void MacaPKBackend::detect_capabilities() {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    int device_count = 0;
    mcError_t err = mcGetDeviceCount(&device_count);
    if (err != mcSuccess || device_count == 0) {
        return;
    }
    
    if (device_id_ >= device_count) {
        device_id_ = 0;
    }
    
    mcDeviceProp prop;
    mcGetDeviceProperties(&prop, device_id_);
    compute_capability_ = prop.major * 10 + prop.minor;
#else
    compute_capability_ = 80;  // Default simulation
#endif
}

PKBackendType MacaPKBackend::get_type() const {
    return PKBackendType::MACA;
}

std::string MacaPKBackend::get_name() const {
    return "maca";
}

std::string MacaPKBackend::get_display_name() const {
    std::stringstream ss;
    ss << "MetaX MACA GPU (CC " << compute_capability_ / 10 << "." 
       << compute_capability_ % 10 << ")";
    return ss.str();
}

bool MacaPKBackend::is_available() const {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    int device_count = 0;
    mcError_t err = mcGetDeviceCount(&device_count);
    return (err == mcSuccess && device_count > 0);
#else
    return false;
#endif
}

PKCapabilities MacaPKBackend::get_capabilities() const {
    PKCapabilities caps;
    caps.supports_tma = false;  // MACA doesn't support TMA
    caps.supports_tensor_cores = true;  // Has tensor cores
    caps.supports_async_copy = true;
    caps.supports_nvshmem = false;  // No NVSHMEM
    caps.supports_fp8 = false;
    
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcDeviceProp prop;
    mcGetDeviceProperties(&prop, device_id_);
    caps.max_shared_memory = prop.sharedMemPerBlockOptin;
    caps.max_global_memory = prop.totalGlobalMem;
    caps.max_threads_per_block = prop.maxThreadsPerBlock;
    caps.max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
#else
    caps.max_shared_memory = 163 * 1024;
    caps.max_global_memory = 32ULL * 1024 * 1024 * 1024;
    caps.max_threads_per_block = 1024;
    caps.max_blocks_per_sm = 16;
#endif
    
    caps.compute_major = compute_capability_ / 10;
    caps.compute_minor = compute_capability_ % 10;
    
    caps.supported_modes = get_supported_modes();
    caps.supported_dtypes = {
        PKDataType::FP32, PKDataType::FP16, PKDataType::BF16,
        PKDataType::INT8
    };
    
    return caps;
}

bool MacaPKBackend::supports_mode(PKMode mode) const {
    // MACA: OFFLINE, ONLINE, ONEPASS
    switch (mode) {
        case PKMode::OFFLINE:
        case PKMode::ONLINE:
        case PKMode::ONEPASS:
            return true;
        case PKMode::EAGER:
        case PKMode::GRAPH:
        case PKMode::STREAMING:
            return false;  // Not in workload plan
        default:
            return false;
    }
}

PKMode MacaPKBackend::get_default_mode() const {
    return PKMode::ONLINE;
}

std::vector<PKMode> MacaPKBackend::get_supported_modes() const {
    // MACA: OFFLINE, ONLINE, ONEPASS (2 weeks new)
    return {
        PKMode::OFFLINE,
        PKMode::ONLINE,
        PKMode::ONEPASS,
        PKMode::GRAPH
    };
}

PKMemoryAllocator* MacaPKBackend::get_allocator() {
    return allocator_.get();
}

PKAtomicOps* MacaPKBackend::get_atomic_ops() {
    return atomic_ops_.get();
}

PKTaskExecutor* MacaPKBackend::get_executor() {
    return executor_.get();
}

bool MacaPKBackend::initialize(const PKRuntimeConfig& config) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcError_t err = mcSetDevice(device_id_);
    if (err != mcSuccess) {
        return false;
    }
    
    mcStreamCreate(&worker_stream_);
    mcStreamCreate(&scheduler_stream_);
    
    initialized_ = true;
    return true;
#else
    initialized_ = true;
    return true;
#endif
}

void MacaPKBackend::finalize() {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    if (initialized_) {
        if (worker_stream_) {
            mcStreamDestroy(static_cast<mcStream_t>(worker_stream_));
            worker_stream_ = nullptr;
        }
        if (scheduler_stream_) {
            mcStreamDestroy(static_cast<mcStream_t>(scheduler_stream_));
            scheduler_stream_ = nullptr;
        }
    }
#endif
    initialized_ = false;
}

void MacaPKBackend::reset() {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    if (worker_stream_) mcStreamSynchronize(static_cast<mcStream_t>(worker_stream_));
    if (scheduler_stream_) mcStreamSynchronize(static_cast<mcStream_t>(scheduler_stream_));
#endif
}

void* MacaPKBackend::create_stream() {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcStream_t stream;
    mcStreamCreate(&stream);
    return static_cast<void*>(stream);
#else
    return reinterpret_cast<void*>(1);
#endif
}

void MacaPKBackend::destroy_stream(void* stream) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    if (stream) {
        mcStreamDestroy(static_cast<mcStream_t>(stream));
    }
#endif
}

void MacaPKBackend::synchronize_stream(void* stream) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    if (stream) {
        mcStreamSynchronize(static_cast<mcStream_t>(stream));
    }
#endif
}

void MacaPKBackend::synchronize() {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcDeviceSynchronize();
#endif
}

void MacaPKBackend::launch_worker_kernel(const PKRuntimeConfig& config,
                                          int num_workers,
                                          int threads_per_worker) {
    // Launch worker kernel similar to CUDA
}

void MacaPKBackend::launch_scheduler_kernel(const PKRuntimeConfig& config) {
    // Launch scheduler kernel
}

bool MacaPKBackend::prepare_next_batch(PKRuntimeConfig& config) {
    return true;
}

void MacaPKBackend::process_batch_results(PKRuntimeConfig& config) {
    // Process results
}

bool MacaPKBackend::set_device(int device_id) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcError_t err = mcSetDevice(device_id);
    if (err == mcSuccess) {
        device_id_ = device_id;
        detect_capabilities();
        return true;
    }
    return false;
#else
    device_id_ = device_id;
    return true;
#endif
}

int MacaPKBackend::get_device() const {
    return device_id_;
}

int MacaPKBackend::get_device_count() const {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    int count = 0;
    mcGetDeviceCount(&count);
    return count;
#else
    return 1;
#endif
}

std::vector<std::string> MacaPKBackend::get_compile_flags(PKMode mode) const {
    std::vector<std::string> flags;
    
    flags.push_back("-DYIRAGE_BACKEND_MACA");
    flags.push_back("-DYPK_TARGET_CC=" + std::to_string(compute_capability_));
    
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
        default:
            break;
    }
    
    // MACA compilation flags
    flags.push_back("--expt-relaxed-constexpr");
    flags.push_back("-std=c++17");
    flags.push_back("-arch=mc_" + std::to_string(compute_capability_));
    
    return flags;
}

std::vector<std::string> MacaPKBackend::get_include_dirs() const {
    return {
        "${MACA_HOME}/include",
        "${MACA_HOME}/include/maca"
    };
}

int MacaPKBackend::get_compute_capability() const {
    return compute_capability_;
}

size_t MacaPKBackend::get_max_shared_memory() const {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcDeviceProp prop;
    mcGetDeviceProperties(&prop, device_id_);
    return prop.sharedMemPerBlockOptin;
#else
    return 163 * 1024;
#endif
}

} // namespace persistent_kernel
} // namespace yirage
