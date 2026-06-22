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

#include "persistent_kernel/pk_backend_interface.h"
#include "persistent_kernel/pk_runtime_adapter.h"
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cstdio>
#include <cctype>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// Backend Selection Utilities
// =============================================================================

/**
 * @brief Backend priority for auto-selection
 */
struct PKBackendPriority {
    PKBackendType type;
    int priority;  // Higher = preferred
    
    static constexpr int CUDA_PRIORITY = 100;
    static constexpr int ROCM_PRIORITY = 95;   // AMD ROCm/HIP
    static constexpr int MACA_PRIORITY = 90;
    static constexpr int ASCEND_PRIORITY = 85;
    static constexpr int MPS_PRIORITY = 70;
    static constexpr int TRITON_PRIORITY = 60;
    static constexpr int NKI_PRIORITY = 50;
    static constexpr int CPU_PRIORITY = 10;
};

/**
 * @brief Get priority for a backend type
 */
inline int get_backend_priority(PKBackendType type) {
    switch (type) {
        case PKBackendType::CUDA:   return PKBackendPriority::CUDA_PRIORITY;
        case PKBackendType::ROCM:   return PKBackendPriority::ROCM_PRIORITY;
        case PKBackendType::MACA:   return PKBackendPriority::MACA_PRIORITY;
        case PKBackendType::ASCEND: return PKBackendPriority::ASCEND_PRIORITY;
        case PKBackendType::MPS:    return PKBackendPriority::MPS_PRIORITY;
        case PKBackendType::TRITON: return PKBackendPriority::TRITON_PRIORITY;
        case PKBackendType::NKI:    return PKBackendPriority::NKI_PRIORITY;
        case PKBackendType::CPU:    return PKBackendPriority::CPU_PRIORITY;
        default:                    return 0;
    }
}

/**
 * @brief Select best backend from available options
 */
inline PKBackendType select_best_backend() {
    auto available = get_available_pk_backends();
    
    if (available.empty()) {
        return PKBackendType::CPU;
    }
    
    PKBackendType best = available[0];
    int best_priority = get_backend_priority(best);
    
    for (auto type : available) {
        int priority = get_backend_priority(type);
        if (priority > best_priority) {
            best = type;
            best_priority = priority;
        }
    }
    
    return best;
}

/**
 * @brief Select backend by name (case-insensitive)
 * @return Backend type, or NUM_BACKENDS if not found
 */
inline PKBackendType select_backend_by_name(const std::string& name) {
    std::string lower_name = name;
    for (auto& c : lower_name) {
        c = std::tolower(c);
    }
    
    if (lower_name == "cuda" || lower_name == "nvidia") {
        return PKBackendType::CUDA;
    } else if (lower_name == "rocm" || lower_name == "hip" || lower_name == "amd") {
        return PKBackendType::ROCM;
    } else if (lower_name == "cpu" || lower_name == "host") {
        return PKBackendType::CPU;
    } else if (lower_name == "mps" || lower_name == "metal" || lower_name == "apple") {
        return PKBackendType::MPS;
    } else if (lower_name == "ascend" || lower_name == "huawei" || lower_name == "npu") {
        return PKBackendType::ASCEND;
    } else if (lower_name == "maca" || lower_name == "metax") {
        return PKBackendType::MACA;
    } else if (lower_name == "triton" || lower_name == "openai") {
        return PKBackendType::TRITON;
    } else if (lower_name == "nki" || lower_name == "neuron" || lower_name == "aws") {
        return PKBackendType::NKI;
    } else if (lower_name == "auto" || lower_name == "best") {
        return select_best_backend();
    }
    
    return PKBackendType::NUM_BACKENDS;  // Not found
}

// =============================================================================
// Mode Selection Utilities
// =============================================================================

/**
 * @brief Select best mode for a backend and use case
 */
inline PKMode select_best_mode(PKBackendType backend, bool is_batch = false,
                               bool is_streaming = false) {
    auto backend_ptr = create_pk_backend(backend, 0);
    if (!backend_ptr) {
        return PKMode::OFFLINE;
    }
    
    auto supported = backend_ptr->get_supported_modes();
    
    // Priority based on use case
    if (is_streaming && backend_ptr->supports_mode(PKMode::STREAMING)) {
        return PKMode::STREAMING;
    }
    
    if (is_batch) {
        if (backend_ptr->supports_mode(PKMode::OFFLINE)) {
            return PKMode::OFFLINE;
        }
    } else {
        if (backend_ptr->supports_mode(PKMode::ONLINE)) {
            return PKMode::ONLINE;
        }
    }
    
    // Fallback to default
    return backend_ptr->get_default_mode();
}

/**
 * @brief Parse mode from string
 */
inline PKMode parse_mode(const std::string& mode_str) {
    std::string lower = mode_str;
    for (auto& c : lower) {
        c = std::tolower(c);
    }
    
    if (lower == "offline" || lower == "batch") {
        return PKMode::OFFLINE;
    } else if (lower == "online" || lower == "single") {
        return PKMode::ONLINE;
    } else if (lower == "onepass" || lower == "forward") {
        return PKMode::ONEPASS;
    } else if (lower == "eager" || lower == "immediate") {
        return PKMode::EAGER;
    } else if (lower == "graph" || lower == "compiled") {
        return PKMode::GRAPH;
    } else if (lower == "streaming" || lower == "pipeline") {
        return PKMode::STREAMING;
    }
    
    return PKMode::ONLINE;  // Default
}

// =============================================================================
// Configuration Builders
// =============================================================================

/**
 * @brief Builder for PKRuntimeConfig
 */
class PKRuntimeConfigBuilder {
public:
    PKRuntimeConfigBuilder() {
        config_.mode = PKMode::ONLINE;
        config_.num_workers = 4;
        config_.num_local_schedulers = 1;
        config_.num_remote_schedulers = 0;
        config_.threads_per_worker = 256;
        config_.max_seq_length = 2048;
        config_.max_num_batched_requests = 16;
        config_.max_num_batched_tokens = 64;
        config_.max_num_pages = 1024;
        config_.page_size = 64;
        config_.eos_token_id = -1;
        config_.num_gpus = 1;
        config_.my_gpu_id = 0;
        config_.backend_context = nullptr;
        config_.stream_handle = nullptr;
        config_.profiling_enabled = false;
        config_.profiler_buffer = nullptr;
    }
    
    PKRuntimeConfigBuilder& mode(PKMode m) {
        config_.mode = m;
        return *this;
    }
    
    PKRuntimeConfigBuilder& workers(int n) {
        config_.num_workers = n;
        return *this;
    }
    
    PKRuntimeConfigBuilder& schedulers(int local, int remote = 0) {
        config_.num_local_schedulers = local;
        config_.num_remote_schedulers = remote;
        return *this;
    }
    
    PKRuntimeConfigBuilder& threads_per_worker(int n) {
        config_.threads_per_worker = n;
        return *this;
    }
    
    PKRuntimeConfigBuilder& max_seq_length(size_t n) {
        config_.max_seq_length = n;
        return *this;
    }
    
    PKRuntimeConfigBuilder& batch_config(size_t max_requests, size_t max_tokens) {
        config_.max_num_batched_requests = max_requests;
        config_.max_num_batched_tokens = max_tokens;
        return *this;
    }
    
    PKRuntimeConfigBuilder& paging(size_t max_pages, size_t page_size) {
        config_.max_num_pages = max_pages;
        config_.page_size = page_size;
        return *this;
    }
    
    PKRuntimeConfigBuilder& eos_token(int64_t id) {
        config_.eos_token_id = id;
        return *this;
    }
    
    PKRuntimeConfigBuilder& multi_gpu(int num_gpus, int my_id) {
        config_.num_gpus = num_gpus;
        config_.my_gpu_id = my_id;
        return *this;
    }
    
    PKRuntimeConfigBuilder& profiling(bool enable, void* buffer = nullptr) {
        config_.profiling_enabled = enable;
        config_.profiler_buffer = buffer;
        return *this;
    }
    
    PKRuntimeConfig build() const {
        return config_;
    }
    
private:
    PKRuntimeConfig config_;
};

// =============================================================================
// Profiling Utilities
// =============================================================================

/**
 * @brief Simple profiling timer
 */
class PKProfiler {
public:
    PKProfiler() : enabled_(false), start_time_(0), total_time_(0), count_(0) {}
    
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }
    
    void start() {
        if (!enabled_) return;
        start_time_ = get_time_ns();
    }
    
    void stop() {
        if (!enabled_) return;
        total_time_ += get_time_ns() - start_time_;
        ++count_;
    }
    
    double get_total_ms() const {
        return total_time_ / 1e6;
    }
    
    double get_avg_ms() const {
        return count_ > 0 ? (total_time_ / count_) / 1e6 : 0.0;
    }
    
    size_t get_count() const {
        return count_;
    }
    
    void reset() {
        total_time_ = 0;
        count_ = 0;
    }
    
private:
    bool enabled_;
    uint64_t start_time_;
    uint64_t total_time_;
    size_t count_;
    
    static uint64_t get_time_ns() {
        auto now = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch());
        return ns.count();
    }
};

// =============================================================================
// Error Handling
// =============================================================================

/**
 * @brief PK error codes
 */
enum class PKError {
    SUCCESS = 0,
    BACKEND_NOT_AVAILABLE,
    MODE_NOT_SUPPORTED,
    INITIALIZATION_FAILED,
    EXECUTION_FAILED,
    OUT_OF_MEMORY,
    INVALID_ARGUMENT,
    INTERNAL_ERROR
};

/**
 * @brief Convert error code to string
 */
inline const char* pk_error_to_string(PKError error) {
    switch (error) {
        case PKError::SUCCESS: return "Success";
        case PKError::BACKEND_NOT_AVAILABLE: return "Backend not available";
        case PKError::MODE_NOT_SUPPORTED: return "Mode not supported";
        case PKError::INITIALIZATION_FAILED: return "Initialization failed";
        case PKError::EXECUTION_FAILED: return "Execution failed";
        case PKError::OUT_OF_MEMORY: return "Out of memory";
        case PKError::INVALID_ARGUMENT: return "Invalid argument";
        case PKError::INTERNAL_ERROR: return "Internal error";
        default: return "Unknown error";
    }
}

// =============================================================================
// Logging Utilities
// =============================================================================

/**
 * @brief Simple logging levels
 */
enum class PKLogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    NONE = 4
};

/**
 * @brief Global log level (can be set at runtime)
 */
inline PKLogLevel& get_pk_log_level() {
    static PKLogLevel level = PKLogLevel::INFO;
    return level;
}

inline void set_pk_log_level(PKLogLevel level) {
    get_pk_log_level() = level;
}

/**
 * @brief Simple logging macro
 */
#define PK_LOG(level, ...) \
    do { \
        if (static_cast<int>(level) >= \
            static_cast<int>(::yirage::persistent_kernel::get_pk_log_level())) { \
            fprintf(stderr, "[PK %s] ", \
                level == ::yirage::persistent_kernel::PKLogLevel::DEBUG ? "DEBUG" : \
                level == ::yirage::persistent_kernel::PKLogLevel::INFO ? "INFO" : \
                level == ::yirage::persistent_kernel::PKLogLevel::WARNING ? "WARN" : "ERROR"); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
        } \
    } while (0)

#define PK_DEBUG(...) PK_LOG(::yirage::persistent_kernel::PKLogLevel::DEBUG, __VA_ARGS__)
#define PK_INFO(...)  PK_LOG(::yirage::persistent_kernel::PKLogLevel::INFO, __VA_ARGS__)
#define PK_WARN(...)  PK_LOG(::yirage::persistent_kernel::PKLogLevel::WARNING, __VA_ARGS__)
#define PK_ERROR(...) PK_LOG(::yirage::persistent_kernel::PKLogLevel::ERROR, __VA_ARGS__)

} // namespace persistent_kernel
} // namespace yirage
