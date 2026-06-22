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

#include "persistent_kernel/backends/ascend_pk_backend.h"
#include <atomic>
#include <cstring>
#include <sstream>

// Ascend ACL includes (conditional compilation)
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
#include <acl/acl.h>
#include <acl/acl_rt.h>
#endif

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// AscendMemoryAllocator Implementation
// =============================================================================

AscendMemoryAllocator::AscendMemoryAllocator()
    : device_id_(0), acl_context_(nullptr) {}

AscendMemoryAllocator::~AscendMemoryAllocator() = default;

void* AscendMemoryAllocator::allocate(size_t size) {
    void* ptr = nullptr;
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return nullptr;
    }
#else
    // Fallback: use host memory for simulation
    ptr = std::malloc(size);
#endif
    return ptr;
}

void AscendMemoryAllocator::free(void* ptr) {
    if (!ptr) return;
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtFree(ptr);
#else
    std::free(ptr);
#endif
}

void AscendMemoryAllocator::copy_h2d(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
#else
    std::memcpy(dst, src, size);
#endif
}

void AscendMemoryAllocator::copy_d2h(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
#else
    std::memcpy(dst, src, size);
#endif
}

void AscendMemoryAllocator::copy_d2d(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
#else
    std::memcpy(dst, src, size);
#endif
}

void AscendMemoryAllocator::copy_h2d_async(void* dst, const void* src, 
                                            size_t size, void* stream) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE,
                     static_cast<aclrtStream>(stream));
#else
    std::memcpy(dst, src, size);
#endif
}

void AscendMemoryAllocator::memset(void* ptr, int value, size_t size) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtMemset(ptr, size, value, size);
#else
    std::memset(ptr, value, size);
#endif
}

size_t AscendMemoryAllocator::get_total_memory() const {
    size_t total = 0;
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    size_t free;
    aclrtGetMemInfo(ACL_HBM_MEM, &free, &total);
#endif
    return total;
}

size_t AscendMemoryAllocator::get_free_memory() const {
    size_t free = 0;
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    size_t total;
    aclrtGetMemInfo(ACL_HBM_MEM, &free, &total);
#endif
    return free;
}

// =============================================================================
// AscendAtomicOps Implementation
// =============================================================================

AscendAtomicOps::AscendAtomicOps() = default;
AscendAtomicOps::~AscendAtomicOps() = default;

// Ascend NPU atomics are typically handled through CANN operators
// These are host-side stubs for interface compatibility

uint64_t AscendAtomicOps::fetch_add_u64(uint64_t* addr, uint64_t val) {
    // Use C++ atomic as fallback for host-side operations
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->fetch_add(val, std::memory_order_acq_rel);
}

uint64_t AscendAtomicOps::fetch_sub_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->fetch_sub(val, std::memory_order_acq_rel);
}

uint64_t AscendAtomicOps::compare_exchange_u64(uint64_t* addr, uint64_t expected,
                                                uint64_t desired) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    atomic_addr->compare_exchange_strong(expected, desired,
                                          std::memory_order_acq_rel);
    return expected;
}

void AscendAtomicOps::store_release_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    atomic_addr->store(val, std::memory_order_release);
}

uint64_t AscendAtomicOps::load_acquire_u64(uint64_t* addr) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->load(std::memory_order_acquire);
}

uint32_t AscendAtomicOps::fetch_add_u32(uint32_t* addr, uint32_t val) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    return atomic_addr->fetch_add(val, std::memory_order_acq_rel);
}

uint32_t AscendAtomicOps::fetch_sub_u32(uint32_t* addr, uint32_t val) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    return atomic_addr->fetch_sub(val, std::memory_order_acq_rel);
}

uint32_t AscendAtomicOps::compare_exchange_u32(uint32_t* addr, uint32_t expected,
                                                uint32_t desired) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    atomic_addr->compare_exchange_strong(expected, desired,
                                          std::memory_order_acq_rel);
    return expected;
}

void AscendAtomicOps::memory_fence() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtSynchronizeDevice();
#endif
}

void AscendAtomicOps::thread_fence() {
    std::atomic_thread_fence(std::memory_order_acq_rel);
}

// =============================================================================
// AscendTaskExecutor Implementation
// =============================================================================

AscendTaskExecutor::AscendTaskExecutor()
    : acl_stream_(nullptr) {}

AscendTaskExecutor::~AscendTaskExecutor() = default;

bool AscendTaskExecutor::supports_task(PKTaskType type) const {
    switch (type) {
        case PKTaskType::TERMINATE:
        case PKTaskType::BEGIN_TASK_GRAPH:
        case PKTaskType::EMBEDDING:
        case PKTaskType::RMS_NORM:
        case PKTaskType::RMS_NORM_LINEAR:
        case PKTaskType::LINEAR:
        case PKTaskType::LINEAR_RESIDUAL:
        case PKTaskType::ATTENTION:
        case PKTaskType::SILU_MUL:
        case PKTaskType::SILU_MUL_LINEAR:
        case PKTaskType::ARGMAX:
        case PKTaskType::ROTARY_EMBEDDING:
            return true;
        // Paged attention requires specific Ascend kernel implementation
        case PKTaskType::PAGED_ATTENTION:
            return true;  // Supported through CANN custom ops
        // MOE tasks
        case PKTaskType::MOE_GATE:
        case PKTaskType::MOE_LINEAR:
            return true;
        // Communication - HCCL instead of NVSHMEM
        case PKTaskType::ALLREDUCE:
        case PKTaskType::REDUCE:
            return true;  // Via HCCL
        case PKTaskType::NVSHMEM_COPY:
            return false;  // NVSHMEM not supported
        default:
            return false;
    }
}

void AscendTaskExecutor::execute(const PKTaskDesc& desc,
                                  const PKRuntimeConfig& config,
                                  void* shared_memory,
                                  size_t shared_memory_size) {
    // Task execution is handled through CANN operators
    // This dispatches to the appropriate Ascend kernel
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    // Implementation would call aclnn* operators based on task type
    // Example: aclnnMatmul, aclnnLayerNorm, aclnnSoftmax, etc.
#endif
}

size_t AscendTaskExecutor::get_shared_memory_size(PKTaskType type) const {
    // Ascend NPU uses L0/L1 buffer instead of shared memory
    // Return typical L1 buffer size for computation
    return 512 * 1024;  // 512KB L1 buffer typical for Ascend 910
}

const char* AscendTaskExecutor::get_task_name(PKTaskType type) const {
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
        case PKTaskType::ALLREDUCE: return "ALLREDUCE_HCCL";
        case PKTaskType::REDUCE: return "REDUCE_HCCL";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// AscendPKBackend Implementation
// =============================================================================

AscendPKBackend::AscendPKBackend(int device_id)
    : device_id_(device_id),
      initialized_(false),
      acl_context_(nullptr) {
    detect_capabilities();
    allocator_ = std::make_unique<AscendMemoryAllocator>();
    atomic_ops_ = std::make_unique<AscendAtomicOps>();
    executor_ = std::make_unique<AscendTaskExecutor>();
}

AscendPKBackend::~AscendPKBackend() {
    if (initialized_) {
        finalize();
    }
}

void AscendPKBackend::detect_capabilities() {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    // Query SoC version
    const char* soc = aclrtGetSocName();
    if (soc) {
        soc_version_ = soc;
    } else {
        soc_version_ = "Ascend910";  // Default
    }
#else
    soc_version_ = "Ascend910-Simulation";
#endif
}

PKBackendType AscendPKBackend::get_type() const {
    return PKBackendType::ASCEND;
}

std::string AscendPKBackend::get_name() const {
    return "ascend";
}

std::string AscendPKBackend::get_display_name() const {
    std::stringstream ss;
    ss << "Huawei Ascend NPU (" << soc_version_ << ")";
    return ss.str();
}

bool AscendPKBackend::is_available() const {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    uint32_t count = 0;
    aclError ret = aclrtGetDeviceCount(&count);
    return (ret == ACL_SUCCESS && count > 0);
#else
    return false;
#endif
}

PKCapabilities AscendPKBackend::get_capabilities() const {
    PKCapabilities caps;
    caps.supports_tma = false;  // Ascend has different memory architecture
    caps.supports_tensor_cores = true;  // Cube cores
    caps.supports_async_copy = true;  // DVPP/AIPP support
    caps.supports_nvshmem = false;  // Use HCCL instead
    caps.supports_fp8 = false;  // Currently not in mainstream Ascend
    
    // Ascend 910 specifications
    caps.max_shared_memory = 512 * 1024;  // L1 buffer
    caps.max_global_memory = 32ULL * 1024 * 1024 * 1024;  // 32GB HBM
    caps.max_threads_per_block = 0;  // Different compute model
    caps.max_blocks_per_sm = 0;  // Uses AI cores, not SM
    
    caps.compute_major = 0;  // Not applicable
    caps.compute_minor = 0;
    
    caps.supported_modes = get_supported_modes();
    caps.supported_dtypes = {
        PKDataType::FP32, PKDataType::FP16, PKDataType::BF16,
        PKDataType::INT8
    };
    
    return caps;
}

bool AscendPKBackend::supports_mode(PKMode mode) const {
    // Ascend: OFFLINE, ONLINE, GRAPH
    switch (mode) {
        case PKMode::OFFLINE:
        case PKMode::ONLINE:
        case PKMode::GRAPH:
            return true;
        case PKMode::ONEPASS:
        case PKMode::EAGER:
        case PKMode::STREAMING:
            return false;  // Not in workload plan
        default:
            return false;
    }
}

PKMode AscendPKBackend::get_default_mode() const {
    return PKMode::ONLINE;
}

std::vector<PKMode> AscendPKBackend::get_supported_modes() const {
    // Ascend: OFFLINE, ONLINE, GRAPH (2 weeks new)
    return {
        PKMode::OFFLINE,
        PKMode::ONLINE,
        PKMode::GRAPH
    };
}

PKMemoryAllocator* AscendPKBackend::get_allocator() {
    return allocator_.get();
}

PKAtomicOps* AscendPKBackend::get_atomic_ops() {
    return atomic_ops_.get();
}

PKTaskExecutor* AscendPKBackend::get_executor() {
    return executor_.get();
}

bool AscendPKBackend::initialize(const PKRuntimeConfig& config) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    // Initialize ACL
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        return false;
    }
    
    // Set device
    ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
        return false;
    }
    
    // Create context
    ret = aclrtCreateContext(&acl_context_, device_id_);
    if (ret != ACL_SUCCESS) {
        return false;
    }
    
    initialized_ = true;
    return true;
#else
    initialized_ = true;
    return true;  // Simulation mode
#endif
}

void AscendPKBackend::finalize() {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    if (initialized_) {
        if (acl_context_) {
            aclrtDestroyContext(acl_context_);
            acl_context_ = nullptr;
        }
        aclrtResetDevice(device_id_);
        aclFinalize();
    }
#endif
    initialized_ = false;
}

void AscendPKBackend::reset() {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtSynchronizeDevice();
#endif
}

void* AscendPKBackend::create_stream() {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtStream stream;
    aclrtCreateStream(&stream);
    return static_cast<void*>(stream);
#else
    return reinterpret_cast<void*>(1);  // Dummy
#endif
}

void AscendPKBackend::destroy_stream(void* stream) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    if (stream) {
        aclrtDestroyStream(static_cast<aclrtStream>(stream));
    }
#endif
}

void AscendPKBackend::synchronize_stream(void* stream) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    if (stream) {
        aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
    }
#endif
}

void AscendPKBackend::synchronize() {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtSynchronizeDevice();
#endif
}

void AscendPKBackend::launch_worker_kernel(const PKRuntimeConfig& config,
                                            int num_workers,
                                            int threads_per_worker) {
    // Ascend uses operator-based execution, not persistent kernels
    // Task graphs are built and executed through CANN
}

void AscendPKBackend::launch_scheduler_kernel(const PKRuntimeConfig& config) {
    // Scheduler is handled by CANN runtime
}

bool AscendPKBackend::prepare_next_batch(PKRuntimeConfig& config) {
    // Batch preparation for Ascend
    return true;
}

void AscendPKBackend::process_batch_results(PKRuntimeConfig& config) {
    // Process results
}

bool AscendPKBackend::set_device(int device_id) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclError ret = aclrtSetDevice(device_id);
    if (ret == ACL_SUCCESS) {
        device_id_ = device_id;
        return true;
    }
    return false;
#else
    device_id_ = device_id;
    return true;
#endif
}

int AscendPKBackend::get_device() const {
    return device_id_;
}

int AscendPKBackend::get_device_count() const {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    uint32_t count = 0;
    aclrtGetDeviceCount(&count);
    return static_cast<int>(count);
#else
    return 1;
#endif
}

std::vector<std::string> AscendPKBackend::get_compile_flags(PKMode mode) const {
    std::vector<std::string> flags;
    
    flags.push_back("-DYIRAGE_BACKEND_ASCEND");
    
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
        case PKMode::GRAPH:
            flags.push_back("-DMODE_GRAPH");
            break;
        default:
            break;
    }
    
    // CANN compilation flags
    flags.push_back("-std=c++17");
    flags.push_back("-DASCEND_910");
    
    return flags;
}

std::vector<std::string> AscendPKBackend::get_include_dirs() const {
    return {
        "${ASCEND_HOME}/include",
        "${ASCEND_HOME}/include/aclnn",
        "${CANN_HOME}/include"
    };
}

std::string AscendPKBackend::get_soc_version() const {
    return soc_version_;
}

bool AscendPKBackend::supports_vector_core() const {
    return true;  // Ascend 910 has vector cores
}

bool AscendPKBackend::supports_cube_core() const {
    return true;  // Ascend 910 has cube cores for matrix ops
}

} // namespace persistent_kernel
} // namespace yirage
