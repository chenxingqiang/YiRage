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

#include "persistent_kernel/backends/cpu_pk_backend.h"
#include <atomic>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#ifdef __linux__
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif
#endif

#ifdef __x86_64__
#include <cpuid.h>
#endif

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// CpuMemoryAllocator Implementation
// =============================================================================

CpuMemoryAllocator::CpuMemoryAllocator()
    : total_allocated_(0), use_aligned_alloc_(true), alignment_(64) {}

CpuMemoryAllocator::~CpuMemoryAllocator() = default;

void* CpuMemoryAllocator::allocate(size_t size) {
    void* ptr = nullptr;
    if (use_aligned_alloc_) {
#ifdef _WIN32
        ptr = _aligned_malloc(size, alignment_);
#else
        if (posix_memalign(&ptr, alignment_, size) != 0) {
            ptr = nullptr;
        }
#endif
    } else {
        ptr = std::malloc(size);
    }
    if (ptr) {
        total_allocated_ += size;
    }
    return ptr;
}

void CpuMemoryAllocator::free(void* ptr) {
    if (ptr) {
#ifdef _WIN32
        if (use_aligned_alloc_) {
            _aligned_free(ptr);
        } else {
            std::free(ptr);
        }
#else
        std::free(ptr);
#endif
    }
}

void CpuMemoryAllocator::copy_h2d(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void CpuMemoryAllocator::copy_d2h(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void CpuMemoryAllocator::copy_d2d(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
}

void CpuMemoryAllocator::copy_h2d_async(void* dst, const void* src, 
                                         size_t size, void* stream) {
    // CPU doesn't have async copy; just do sync copy
    std::memcpy(dst, src, size);
}

void CpuMemoryAllocator::memset(void* ptr, int value, size_t size) {
    std::memset(ptr, value, size);
}

size_t CpuMemoryAllocator::get_total_memory() const {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<size_t>(status.ullTotalPhys);
#elif defined(__APPLE__)
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
        return static_cast<size_t>(memsize);
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
    return 0;
#else
    return 0;
#endif
}

size_t CpuMemoryAllocator::get_free_memory() const {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<size_t>(status.ullAvailPhys);
#elif defined(__APPLE__)
    mach_port_t host_port = mach_host_self();
    mach_msg_type_number_t host_size = sizeof(vm_statistics64_data_t) / sizeof(integer_t);
    vm_size_t page_size;
    vm_statistics64_data_t vm_stat;
    
    host_page_size(host_port, &page_size);
    if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) == KERN_SUCCESS) {
        return static_cast<size_t>(vm_stat.free_count) * page_size;
    }
    return 0;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
    return 0;
#else
    return 0;
#endif
}

// =============================================================================
// CpuAtomicOps Implementation
// =============================================================================

CpuAtomicOps::CpuAtomicOps() = default;
CpuAtomicOps::~CpuAtomicOps() = default;

uint64_t CpuAtomicOps::fetch_add_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->fetch_add(val, std::memory_order_acq_rel);
}

uint64_t CpuAtomicOps::fetch_sub_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->fetch_sub(val, std::memory_order_acq_rel);
}

uint64_t CpuAtomicOps::compare_exchange_u64(uint64_t* addr, uint64_t expected,
                                             uint64_t desired) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    atomic_addr->compare_exchange_strong(expected, desired, 
                                          std::memory_order_acq_rel);
    return expected;
}

void CpuAtomicOps::store_release_u64(uint64_t* addr, uint64_t val) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    atomic_addr->store(val, std::memory_order_release);
}

uint64_t CpuAtomicOps::load_acquire_u64(uint64_t* addr) {
    std::atomic<uint64_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint64_t>*>(addr);
    return atomic_addr->load(std::memory_order_acquire);
}

uint32_t CpuAtomicOps::fetch_add_u32(uint32_t* addr, uint32_t val) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    return atomic_addr->fetch_add(val, std::memory_order_acq_rel);
}

uint32_t CpuAtomicOps::fetch_sub_u32(uint32_t* addr, uint32_t val) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    return atomic_addr->fetch_sub(val, std::memory_order_acq_rel);
}

uint32_t CpuAtomicOps::compare_exchange_u32(uint32_t* addr, uint32_t expected,
                                             uint32_t desired) {
    std::atomic<uint32_t>* atomic_addr = 
        reinterpret_cast<std::atomic<uint32_t>*>(addr);
    atomic_addr->compare_exchange_strong(expected, desired,
                                          std::memory_order_acq_rel);
    return expected;
}

void CpuAtomicOps::memory_fence() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void CpuAtomicOps::thread_fence() {
    std::atomic_thread_fence(std::memory_order_acq_rel);
}

// =============================================================================
// CpuTaskExecutor Implementation
// =============================================================================

CpuTaskExecutor::CpuTaskExecutor()
    : num_threads_(std::thread::hardware_concurrency()),
      use_avx_(false),
      use_avx512_(false) {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        use_avx_ = (ecx & (1 << 28)) != 0;
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        use_avx512_ = (ebx & (1 << 16)) != 0;  // AVX-512F
    }
#endif
}

CpuTaskExecutor::~CpuTaskExecutor() = default;

bool CpuTaskExecutor::supports_task(PKTaskType type) const {
    // CPU supports basic compute tasks
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
        // Attention and paged attention require more complex implementation
        case PKTaskType::ATTENTION:
        case PKTaskType::PAGED_ATTENTION:
        case PKTaskType::ROTARY_EMBEDDING:
            return true;  // Basic support
        // MOE and communication not supported on CPU
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

void CpuTaskExecutor::execute(const PKTaskDesc& desc,
                               const PKRuntimeConfig& config,
                               void* shared_memory,
                               size_t shared_memory_size) {
    // CPU task execution - placeholder for actual implementation
    // Each task type would have its own CPU-optimized implementation
}

size_t CpuTaskExecutor::get_shared_memory_size(PKTaskType type) const {
    // CPU doesn't have shared memory; return 0
    return 0;
}

const char* CpuTaskExecutor::get_task_name(PKTaskType type) const {
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
        default: return "UNKNOWN";
    }
}

// =============================================================================
// CpuPKBackend Implementation
// =============================================================================

CpuPKBackend::CpuPKBackend(int num_threads)
    : num_threads_(num_threads > 0 ? num_threads : 
                   static_cast<int>(std::thread::hardware_concurrency())),
      initialized_(false),
      supports_avx_(false),
      supports_avx512_(false),
      running_(false) {
    detect_capabilities();
    allocator_ = std::make_unique<CpuMemoryAllocator>();
    atomic_ops_ = std::make_unique<CpuAtomicOps>();
    executor_ = std::make_unique<CpuTaskExecutor>();
}

CpuPKBackend::~CpuPKBackend() {
    if (initialized_) {
        finalize();
    }
}

void CpuPKBackend::detect_capabilities() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    // Check AVX support
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        supports_avx_ = (ecx & (1 << 28)) != 0;
    }
    
    // Check AVX-512 support
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        supports_avx512_ = (ebx & (1 << 16)) != 0;  // AVX-512F
    }
    
    // Get CPU name
    char brand[48];
    unsigned int* brand_ptr = reinterpret_cast<unsigned int*>(brand);
    __get_cpuid(0x80000002, &brand_ptr[0], &brand_ptr[1], 
                &brand_ptr[2], &brand_ptr[3]);
    __get_cpuid(0x80000003, &brand_ptr[4], &brand_ptr[5], 
                &brand_ptr[6], &brand_ptr[7]);
    __get_cpuid(0x80000004, &brand_ptr[8], &brand_ptr[9], 
                &brand_ptr[10], &brand_ptr[11]);
    brand[47] = '\0';
    cpu_name_ = brand;
    
    // Trim whitespace
    size_t start = cpu_name_.find_first_not_of(" ");
    size_t end = cpu_name_.find_last_not_of(" ");
    if (start != std::string::npos) {
        cpu_name_ = cpu_name_.substr(start, end - start + 1);
    }
#else
    cpu_name_ = "Unknown CPU";
#endif
}

PKBackendType CpuPKBackend::get_type() const {
    return PKBackendType::CPU;
}

std::string CpuPKBackend::get_name() const {
    return "cpu";
}

std::string CpuPKBackend::get_display_name() const {
    std::stringstream ss;
    ss << "CPU (" << cpu_name_ << ", " << num_threads_ << " threads)";
    return ss.str();
}

bool CpuPKBackend::is_available() const {
    return true;  // CPU is always available
}

PKCapabilities CpuPKBackend::get_capabilities() const {
    PKCapabilities caps;
    caps.supports_tma = false;
    caps.supports_tensor_cores = false;
    caps.supports_async_copy = false;
    caps.supports_nvshmem = false;
    caps.supports_fp8 = false;
    
    caps.max_shared_memory = 0;  // CPU doesn't have shared memory
    caps.max_global_memory = allocator_->get_total_memory();
    caps.max_threads_per_block = num_threads_;
    caps.max_blocks_per_sm = 1;
    
    caps.compute_major = 0;
    caps.compute_minor = 0;
    
    caps.supported_modes = get_supported_modes();
    caps.supported_dtypes = {
        PKDataType::FP32, PKDataType::FP16, PKDataType::BF16
    };
    
    return caps;
}

bool CpuPKBackend::supports_mode(PKMode mode) const {
    // CPU: EAGER, GRAPH, OFFLINE
    switch (mode) {
        case PKMode::EAGER:
        case PKMode::GRAPH:
        case PKMode::OFFLINE:
            return true;
        case PKMode::ONLINE:
        case PKMode::ONEPASS:
        case PKMode::STREAMING:
            return false;  // Not in workload plan
        default:
            return false;
    }
}

PKMode CpuPKBackend::get_default_mode() const {
    return PKMode::EAGER;  // Immediate execution, best for CPU
}

std::vector<PKMode> CpuPKBackend::get_supported_modes() const {
    // CPU: EAGER, GRAPH, OFFLINE (1 week new)
    return {PKMode::EAGER, PKMode::GRAPH, PKMode::OFFLINE};
}

PKMemoryAllocator* CpuPKBackend::get_allocator() {
    return allocator_.get();
}

PKAtomicOps* CpuPKBackend::get_atomic_ops() {
    return atomic_ops_.get();
}

PKTaskExecutor* CpuPKBackend::get_executor() {
    return executor_.get();
}

bool CpuPKBackend::initialize(const PKRuntimeConfig& config) {
    if (initialized_) return true;
    
    num_threads_ = config.num_workers > 0 ? 
                   config.num_workers : num_threads_;
    
    initialized_ = true;
    return true;
}

void CpuPKBackend::finalize() {
    stop_worker_threads();
    initialized_ = false;
}

void CpuPKBackend::reset() {
    stop_worker_threads();
}

void* CpuPKBackend::create_stream() {
    // CPU uses thread pool instead of streams
    // Return a dummy handle
    return reinterpret_cast<void*>(1);
}

void CpuPKBackend::destroy_stream(void* stream) {
    // No-op for CPU
}

void CpuPKBackend::synchronize_stream(void* stream) {
    // Wait for all worker threads to complete
    synchronize();
}

void CpuPKBackend::synchronize() {
    // Wait for all pending work to complete
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void CpuPKBackend::start_worker_threads(int num_workers) {
    running_ = true;
    for (int i = 0; i < num_workers; ++i) {
        worker_threads_.emplace_back([this, i]() {
            // Worker thread main loop
            while (running_) {
                // Execute tasks from queue
                std::this_thread::yield();
            }
        });
    }
}

void CpuPKBackend::stop_worker_threads() {
    running_ = false;
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void CpuPKBackend::launch_worker_kernel(const PKRuntimeConfig& config,
                                         int num_workers,
                                         int threads_per_worker) {
    start_worker_threads(num_workers);
}

void CpuPKBackend::launch_scheduler_kernel(const PKRuntimeConfig& config) {
    // CPU doesn't need separate scheduler kernel
}

bool CpuPKBackend::prepare_next_batch(PKRuntimeConfig& config) {
    // Basic batch preparation for CPU
    return true;
}

void CpuPKBackend::process_batch_results(PKRuntimeConfig& config) {
    // Process results on CPU
}

bool CpuPKBackend::set_device(int device_id) {
    // CPU only has device 0
    return device_id == 0;
}

int CpuPKBackend::get_device() const {
    return 0;
}

int CpuPKBackend::get_device_count() const {
    return 1;
}

std::vector<std::string> CpuPKBackend::get_compile_flags(PKMode mode) const {
    std::vector<std::string> flags;
    
    flags.push_back("-DYIRAGE_BACKEND_CPU");
    
    switch (mode) {
        case PKMode::OFFLINE:
            flags.push_back("-DMODE_OFFLINE");
            break;
        case PKMode::ONEPASS:
            flags.push_back("-DMODE_ONEPASS");
            break;
        default:
            break;
    }
    
    // CPU optimization flags
    flags.push_back("-O3");
    flags.push_back("-std=c++17");
    
    if (supports_avx512_) {
        flags.push_back("-mavx512f");
        flags.push_back("-mavx512dq");
        flags.push_back("-mavx512vl");
    } else if (supports_avx_) {
        flags.push_back("-mavx");
        flags.push_back("-mavx2");
        flags.push_back("-mfma");
    }
    
    // OpenMP for parallel execution
    flags.push_back("-fopenmp");
    
    return flags;
}

std::vector<std::string> CpuPKBackend::get_include_dirs() const {
    return {};
}

int CpuPKBackend::get_num_cores() const {
    return static_cast<int>(std::thread::hardware_concurrency());
}

int CpuPKBackend::get_num_threads() const {
    return num_threads_;
}

bool CpuPKBackend::supports_avx() const {
    return supports_avx_;
}

bool CpuPKBackend::supports_avx512() const {
    return supports_avx512_;
}

std::string CpuPKBackend::get_cpu_name() const {
    return cpu_name_;
}

} // namespace persistent_kernel
} // namespace yirage
