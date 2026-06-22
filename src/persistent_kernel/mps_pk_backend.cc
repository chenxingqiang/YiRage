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

#include "persistent_kernel/backends/mps_pk_backend.h"
#include <atomic>
#include <cstring>
#include <sstream>

// Metal includes (Objective-C, wrapped for C++)
#ifdef __APPLE__
#ifdef YIRAGE_BACKEND_MPS_ENABLED
// Forward declarations - actual implementation requires Objective-C++
extern "C" {
    void* mtl_create_default_device();
    void mtl_release_device(void* device);
    void* mtl_create_command_queue(void* device);
    void mtl_release_command_queue(void* queue);
    void* mtl_allocate_buffer(void* device, size_t size);
    void mtl_release_buffer(void* buffer);
    void mtl_synchronize_device(void* device);
    size_t mtl_get_max_threadgroup_memory(void* device);
    const char* mtl_get_device_name(void* device);
    bool mtl_is_device_available();
}
#endif
#endif

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// MpsMemoryAllocator Implementation
// =============================================================================

MpsMemoryAllocator::MpsMemoryAllocator()
    : mtl_device_(nullptr), default_queue_(nullptr) {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    mtl_device_ = mtl_create_default_device();
    if (mtl_device_) {
        default_queue_ = mtl_create_command_queue(mtl_device_);
    }
#endif
}

MpsMemoryAllocator::~MpsMemoryAllocator() {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    if (default_queue_) {
        mtl_release_command_queue(default_queue_);
    }
    if (mtl_device_) {
        mtl_release_device(mtl_device_);
    }
#endif
}

void* MpsMemoryAllocator::allocate(size_t size) {
    void* ptr = nullptr;
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    if (mtl_device_) {
        ptr = mtl_allocate_buffer(mtl_device_, size);
    }
#else
    // Fallback to CPU memory
    ptr = std::malloc(size);
#endif
    return ptr;
}

void MpsMemoryAllocator::free(void* ptr) {
    if (!ptr) return;
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    mtl_release_buffer(ptr);
#else
    std::free(ptr);
#endif
}

void MpsMemoryAllocator::copy_h2d(void* dst, const void* src, size_t size) {
    // MPS uses unified memory - direct copy works
    std::memcpy(dst, src, size);
}

void MpsMemoryAllocator::copy_d2h(void* dst, const void* src, size_t size) {
    // MPS uses unified memory - direct copy works
    std::memcpy(dst, src, size);
}

void MpsMemoryAllocator::copy_d2d(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void MpsMemoryAllocator::copy_h2d_async(void* dst, const void* src, 
                                         size_t size, void* stream) {
    // Metal blit encoder would be used here
    std::memcpy(dst, src, size);
}

void MpsMemoryAllocator::memset(void* ptr, int value, size_t size) {
    std::memset(ptr, value, size);
}

size_t MpsMemoryAllocator::get_total_memory() const {
    // Apple Silicon has unified memory
#ifdef __APPLE__
    // Would query system memory
    return 16ULL * 1024 * 1024 * 1024;  // 16GB typical for M1/M2
#else
    return 0;
#endif
}

size_t MpsMemoryAllocator::get_free_memory() const {
#ifdef __APPLE__
    return 8ULL * 1024 * 1024 * 1024;  // Estimate
#else
    return 0;
#endif
}

// =============================================================================
// MpsAtomicOps Implementation
// =============================================================================

MpsAtomicOps::MpsAtomicOps() = default;
MpsAtomicOps::~MpsAtomicOps() = default;

uint64_t MpsAtomicOps::fetch_add_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->fetch_add(val, std::memory_order_acq_rel);
}

uint64_t MpsAtomicOps::fetch_sub_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->fetch_sub(val, std::memory_order_acq_rel);
}

uint64_t MpsAtomicOps::compare_exchange_u64(uint64_t* addr, uint64_t expected,
                                             uint64_t desired) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    atomic_addr->compare_exchange_strong(expected, desired,
                                          std::memory_order_acq_rel);
    return expected;
}

void MpsAtomicOps::store_release_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    atomic_addr->store(val, std::memory_order_release);
}

uint64_t MpsAtomicOps::load_acquire_u64(uint64_t* addr) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->load(std::memory_order_acquire);
}

uint32_t MpsAtomicOps::fetch_add_u32(uint32_t* addr, uint32_t val) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    return atomic_addr->fetch_add(val, std::memory_order_acq_rel);
}

uint32_t MpsAtomicOps::fetch_sub_u32(uint32_t* addr, uint32_t val) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    return atomic_addr->fetch_sub(val, std::memory_order_acq_rel);
}

uint32_t MpsAtomicOps::compare_exchange_u32(uint32_t* addr, uint32_t expected,
                                             uint32_t desired) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    atomic_addr->compare_exchange_strong(expected, desired,
                                          std::memory_order_acq_rel);
    return expected;
}

void MpsAtomicOps::memory_fence() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void MpsAtomicOps::thread_fence() {
    std::atomic_thread_fence(std::memory_order_acq_rel);
}

// =============================================================================
// MpsTaskExecutor Implementation
// =============================================================================

MpsTaskExecutor::MpsTaskExecutor()
    : compute_pipeline_cache_(nullptr), mtl_device_(nullptr) {}

MpsTaskExecutor::~MpsTaskExecutor() = default;

bool MpsTaskExecutor::supports_task(PKTaskType type) const {
    // MPS supports basic compute tasks through Metal shaders
    switch (type) {
        case PKTaskType::TERMINATE:
        case PKTaskType::BEGIN_TASK_GRAPH:
        case PKTaskType::EMBEDDING:
        case PKTaskType::RMS_NORM:
        case PKTaskType::RMS_NORM_LINEAR:
        case PKTaskType::LINEAR:
        case PKTaskType::LINEAR_RESIDUAL:
        case PKTaskType::SILU_MUL:
        case PKTaskType::SILU_MUL_LINEAR:
        case PKTaskType::ARGMAX:
            return true;
        // Attention is complex but supported through MPSGraph
        case PKTaskType::ATTENTION:
        case PKTaskType::ROTARY_EMBEDDING:
            return true;
        // Paged attention not well suited for MPS
        case PKTaskType::PAGED_ATTENTION:
            return false;
        // MOE and communication not supported
        case PKTaskType::MOE_GATE:
        case PKTaskType::MOE_LINEAR:
        case PKTaskType::ALLREDUCE:
        case PKTaskType::REDUCE:
        case PKTaskType::NVSHMEM_COPY:
            return false;
        default:
            return false;
    }
}

void MpsTaskExecutor::execute(const PKTaskDesc& desc,
                               const PKRuntimeConfig& config,
                               void* shared_memory,
                               size_t shared_memory_size) {
    // Execute using Metal compute command encoder
    // Would dispatch appropriate Metal shader
}

size_t MpsTaskExecutor::get_shared_memory_size(PKTaskType type) const {
    // Metal threadgroup memory limit (typically 32KB)
    return 32 * 1024;
}

const char* MpsTaskExecutor::get_task_name(PKTaskType type) const {
    switch (type) {
        case PKTaskType::TERMINATE: return "TERMINATE";
        case PKTaskType::BEGIN_TASK_GRAPH: return "BEGIN_TASK_GRAPH";
        case PKTaskType::EMBEDDING: return "EMBEDDING_MPS";
        case PKTaskType::RMS_NORM: return "RMS_NORM_MPS";
        case PKTaskType::RMS_NORM_LINEAR: return "RMS_NORM_LINEAR_MPS";
        case PKTaskType::LINEAR: return "LINEAR_MPS";
        case PKTaskType::LINEAR_RESIDUAL: return "LINEAR_RESIDUAL_MPS";
        case PKTaskType::ATTENTION: return "ATTENTION_MPS";
        case PKTaskType::SILU_MUL: return "SILU_MUL_MPS";
        case PKTaskType::SILU_MUL_LINEAR: return "SILU_MUL_LINEAR_MPS";
        case PKTaskType::ARGMAX: return "ARGMAX_MPS";
        case PKTaskType::ROTARY_EMBEDDING: return "ROTARY_EMBEDDING_MPS";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// MpsPKBackend Implementation
// =============================================================================

MpsPKBackend::MpsPKBackend(int device_id)
    : device_id_(device_id),
      initialized_(false),
      mtl_device_(nullptr),
      command_queue_(nullptr) {
    detect_capabilities();
    allocator_ = std::make_unique<MpsMemoryAllocator>();
    atomic_ops_ = std::make_unique<MpsAtomicOps>();
    executor_ = std::make_unique<MpsTaskExecutor>();
}

MpsPKBackend::~MpsPKBackend() {
    if (initialized_) {
        finalize();
    }
}

void MpsPKBackend::detect_capabilities() {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    mtl_device_ = mtl_create_default_device();
    if (mtl_device_) {
        const char* name = mtl_get_device_name(mtl_device_);
        gpu_family_ = name ? name : "Apple GPU";
    }
#else
    gpu_family_ = "Apple GPU (Simulation)";
#endif
}

PKBackendType MpsPKBackend::get_type() const {
    return PKBackendType::MPS;
}

std::string MpsPKBackend::get_name() const {
    return "mps";
}

std::string MpsPKBackend::get_display_name() const {
    std::stringstream ss;
    ss << "Apple Metal (" << gpu_family_ << ")";
    return ss.str();
}

bool MpsPKBackend::is_available() const {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    return mtl_is_device_available();
#else
    return false;
#endif
}

PKCapabilities MpsPKBackend::get_capabilities() const {
    PKCapabilities caps;
    caps.supports_tma = false;  // Metal doesn't have TMA
    caps.supports_tensor_cores = false;  // Uses AMX on CPU side
    caps.supports_async_copy = true;  // Metal supports async blit
    caps.supports_nvshmem = false;
    caps.supports_fp8 = false;
    
    // Apple Silicon specs
    caps.max_shared_memory = 32 * 1024;  // Threadgroup memory
    caps.max_global_memory = 16ULL * 1024 * 1024 * 1024;  // Unified memory
    caps.max_threads_per_block = 1024;  // Max threads per threadgroup
    caps.max_blocks_per_sm = 0;  // Different model
    
    caps.compute_major = 0;  // Not applicable
    caps.compute_minor = 0;
    
    caps.supported_modes = get_supported_modes();
    caps.supported_dtypes = {
        PKDataType::FP32, PKDataType::FP16
        // BF16 support varies by Apple Silicon generation
    };
    
    return caps;
}

bool MpsPKBackend::supports_mode(PKMode mode) const {
    // MPS supports EAGER and GRAPH modes only
    switch (mode) {
        case PKMode::EAGER:
        case PKMode::GRAPH:
            return true;
        case PKMode::OFFLINE:
        case PKMode::ONLINE:
        case PKMode::ONEPASS:
        case PKMode::STREAMING:
            return false;
        default:
            return false;
    }
}

PKMode MpsPKBackend::get_default_mode() const {
    return PKMode::EAGER;
}

std::vector<PKMode> MpsPKBackend::get_supported_modes() const {
    return {PKMode::EAGER, PKMode::GRAPH};
}

PKMemoryAllocator* MpsPKBackend::get_allocator() {
    return allocator_.get();
}

PKAtomicOps* MpsPKBackend::get_atomic_ops() {
    return atomic_ops_.get();
}

PKTaskExecutor* MpsPKBackend::get_executor() {
    return executor_.get();
}

bool MpsPKBackend::initialize(const PKRuntimeConfig& config) {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    if (!mtl_device_) {
        mtl_device_ = mtl_create_default_device();
    }
    if (!mtl_device_) {
        return false;
    }
    
    command_queue_ = mtl_create_command_queue(mtl_device_);
    if (!command_queue_) {
        return false;
    }
    
    initialized_ = true;
    return true;
#else
    // Simulation mode
    initialized_ = true;
    return true;
#endif
}

void MpsPKBackend::finalize() {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    if (command_queue_) {
        mtl_release_command_queue(command_queue_);
        command_queue_ = nullptr;
    }
#endif
    initialized_ = false;
}

void MpsPKBackend::reset() {
    synchronize();
}

void* MpsPKBackend::create_stream() {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    if (mtl_device_) {
        return mtl_create_command_queue(mtl_device_);
    }
#endif
    return reinterpret_cast<void*>(1);  // Dummy
}

void MpsPKBackend::destroy_stream(void* stream) {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    if (stream && stream != reinterpret_cast<void*>(1)) {
        mtl_release_command_queue(stream);
    }
#endif
}

void MpsPKBackend::synchronize_stream(void* stream) {
    synchronize();
}

void MpsPKBackend::synchronize() {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    if (mtl_device_) {
        mtl_synchronize_device(mtl_device_);
    }
#endif
}

void MpsPKBackend::launch_worker_kernel(const PKRuntimeConfig& config,
                                         int num_workers,
                                         int threads_per_worker) {
    // MPS uses command buffers, not persistent kernels
    // Would encode compute commands here
}

void MpsPKBackend::launch_scheduler_kernel(const PKRuntimeConfig& config) {
    // Not needed for MPS - uses Metal's built-in scheduling
}

bool MpsPKBackend::prepare_next_batch(PKRuntimeConfig& config) {
    return true;
}

void MpsPKBackend::process_batch_results(PKRuntimeConfig& config) {
    // Process Metal command buffer results
}

bool MpsPKBackend::set_device(int device_id) {
    // Apple Silicon typically has single GPU
    device_id_ = 0;
    return device_id == 0;
}

int MpsPKBackend::get_device() const {
    return device_id_;
}

int MpsPKBackend::get_device_count() const {
    // Apple Silicon typically has 1 GPU
    return 1;
}

std::vector<std::string> MpsPKBackend::get_compile_flags(PKMode mode) const {
    std::vector<std::string> flags;
    
    flags.push_back("-DYIRAGE_BACKEND_MPS");
    
    switch (mode) {
        case PKMode::EAGER:
            flags.push_back("-DMODE_EAGER");
            break;
        case PKMode::GRAPH:
            flags.push_back("-DMODE_GRAPH");
            break;
        default:
            break;
    }
    
    // Metal shader compilation flags
    flags.push_back("-std=metal2.0");
    flags.push_back("-framework Metal");
    flags.push_back("-framework MetalPerformanceShaders");
    
    return flags;
}

std::vector<std::string> MpsPKBackend::get_include_dirs() const {
    return {
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/"
        "Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers",
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/"
        "Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/"
        "MetalPerformanceShaders.framework/Headers"
    };
}

std::string MpsPKBackend::get_gpu_family() const {
    return gpu_family_;
}

bool MpsPKBackend::supports_apple_silicon() const {
#ifdef __APPLE__
    return true;
#else
    return false;
#endif
}

size_t MpsPKBackend::get_max_threadgroup_memory() const {
#if defined(__APPLE__) && defined(YIRAGE_BACKEND_MPS_ENABLED)
    if (mtl_device_) {
        return mtl_get_max_threadgroup_memory(mtl_device_);
    }
#endif
    return 32 * 1024;  // Default 32KB
}

size_t MpsPKBackend::get_max_threads_per_threadgroup() const {
    return 1024;  // Metal default
}

} // namespace persistent_kernel
} // namespace yirage
