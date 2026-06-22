// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_pk_utils_gtest.cc
 * @brief Persistent Kernel Utilities Unit Tests
 *
 * Tests for utility components:
 *   - PKBackendPriority and backend selection
 *   - Mode parsing and selection
 *   - PKRuntimeConfigBuilder
 *   - PKProfiler
 *   - PKError and error handling
 *   - PKLogLevel and logging
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cctype>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// Enums
// =============================================================================

enum class PKBackendType {
    CUDA = 0,
    CPU = 1,
    MPS = 2,
    ASCEND = 3,
    MACA = 4,
    TRITON = 5,
    NKI = 6,
    ROCM = 7,
    NUM_BACKENDS
};

enum class PKMode {
    OFFLINE = 0,
    ONLINE = 1,
    ONEPASS = 2,
    EAGER = 3,
    GRAPH = 4,
    STREAMING = 5,
    NUM_MODES
};

// =============================================================================
// Backend Priority
// =============================================================================

struct PKBackendPriority {
    PKBackendType type;
    int priority;
    
    static constexpr int CUDA_PRIORITY = 100;
    static constexpr int ROCM_PRIORITY = 95;
    static constexpr int MACA_PRIORITY = 90;
    static constexpr int ASCEND_PRIORITY = 85;
    static constexpr int MPS_PRIORITY = 70;
    static constexpr int TRITON_PRIORITY = 60;
    static constexpr int NKI_PRIORITY = 50;
    static constexpr int CPU_PRIORITY = 10;
};

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

inline PKBackendType select_best_backend(std::vector<PKBackendType> const& available) {
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

inline PKBackendType select_backend_by_name(std::string const& name) {
    std::string lower = name;
    for (auto& c : lower) {
        c = std::tolower(c);
    }
    
    if (lower == "cuda" || lower == "nvidia") return PKBackendType::CUDA;
    if (lower == "rocm" || lower == "hip" || lower == "amd") return PKBackendType::ROCM;
    if (lower == "cpu" || lower == "host") return PKBackendType::CPU;
    if (lower == "mps" || lower == "metal" || lower == "apple") return PKBackendType::MPS;
    if (lower == "ascend" || lower == "huawei" || lower == "npu") return PKBackendType::ASCEND;
    if (lower == "maca" || lower == "metax") return PKBackendType::MACA;
    if (lower == "triton" || lower == "openai") return PKBackendType::TRITON;
    if (lower == "nki" || lower == "neuron" || lower == "aws") return PKBackendType::NKI;
    
    return PKBackendType::NUM_BACKENDS;  // Not found
}

inline bool backend_has_higher_capability(PKBackendType a, PKBackendType b) {
    return get_backend_priority(a) > get_backend_priority(b);
}

// =============================================================================
// Mode Selection
// =============================================================================

inline PKMode parse_mode(std::string const& mode_str) {
    std::string lower = mode_str;
    for (auto& c : lower) {
        c = std::tolower(c);
    }
    
    if (lower == "offline" || lower == "batch") return PKMode::OFFLINE;
    if (lower == "online" || lower == "single") return PKMode::ONLINE;
    if (lower == "onepass" || lower == "forward") return PKMode::ONEPASS;
    if (lower == "eager" || lower == "immediate") return PKMode::EAGER;
    if (lower == "graph" || lower == "compiled") return PKMode::GRAPH;
    if (lower == "streaming" || lower == "pipeline") return PKMode::STREAMING;
    
    return PKMode::ONLINE;  // Default
}

inline PKMode get_default_mode_for_backend(PKBackendType type) {
    switch (type) {
        case PKBackendType::CUDA:
        case PKBackendType::ROCM:
        case PKBackendType::MACA:
        case PKBackendType::ASCEND:
            return PKMode::ONLINE;
        case PKBackendType::MPS:
        case PKBackendType::CPU:
            return PKMode::EAGER;
        default:
            return PKMode::OFFLINE;
    }
}

// =============================================================================
// PKRuntimeConfig
// =============================================================================

struct PKRuntimeConfig {
    PKMode mode = PKMode::ONLINE;
    
    int num_workers = 4;
    int num_local_schedulers = 1;
    int num_remote_schedulers = 0;
    int threads_per_worker = 256;
    
    size_t max_seq_length = 2048;
    size_t max_num_batched_requests = 16;
    size_t max_num_batched_tokens = 64;
    size_t max_num_pages = 1024;
    size_t page_size = 64;
    
    int64_t eos_token_id = -1;
    
    int num_gpus = 1;
    int my_gpu_id = 0;
    
    void* backend_context = nullptr;
    void* stream_handle = nullptr;
    
    bool profiling_enabled = false;
    void* profiler_buffer = nullptr;
};

// =============================================================================
// PKRuntimeConfigBuilder
// =============================================================================

class PKRuntimeConfigBuilder {
public:
    PKRuntimeConfigBuilder() = default;
    
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
// PKProfiler
// =============================================================================

class PKProfiler {
public:
    PKProfiler() = default;
    
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
    bool enabled_ = false;
    uint64_t start_time_ = 0;
    uint64_t total_time_ = 0;
    size_t count_ = 0;
    
    static uint64_t get_time_ns() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
};

// =============================================================================
// PKError
// =============================================================================

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
// PKLogLevel
// =============================================================================

enum class PKLogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    NONE = 4
};

inline PKLogLevel& get_pk_log_level() {
    static PKLogLevel level = PKLogLevel::INFO;
    return level;
}

inline void set_pk_log_level(PKLogLevel level) {
    get_pk_log_level() = level;
}

inline const char* pk_log_level_to_string(PKLogLevel level) {
    switch (level) {
        case PKLogLevel::DEBUG: return "DEBUG";
        case PKLogLevel::INFO: return "INFO";
        case PKLogLevel::WARNING: return "WARNING";
        case PKLogLevel::ERROR: return "ERROR";
        case PKLogLevel::NONE: return "NONE";
        default: return "UNKNOWN";
    }
}

}  // namespace persistent_kernel
}  // namespace yirage

using namespace yirage::persistent_kernel;

// =============================================================================
// Backend Priority Tests
// =============================================================================

class PKBackendPriorityTest : public ::testing::Test {};

TEST_F(PKBackendPriorityTest, GetBackendPriority) {
    EXPECT_EQ(get_backend_priority(PKBackendType::CUDA), 100);
    EXPECT_EQ(get_backend_priority(PKBackendType::ROCM), 95);
    EXPECT_EQ(get_backend_priority(PKBackendType::MACA), 90);
    EXPECT_EQ(get_backend_priority(PKBackendType::CPU), 10);
}

TEST_F(PKBackendPriorityTest, CUDAHighestPriority) {
    EXPECT_TRUE(backend_has_higher_capability(PKBackendType::CUDA, PKBackendType::CPU));
    EXPECT_TRUE(backend_has_higher_capability(PKBackendType::CUDA, PKBackendType::ROCM));
    EXPECT_TRUE(backend_has_higher_capability(PKBackendType::CUDA, PKBackendType::MPS));
}

TEST_F(PKBackendPriorityTest, SelectBestBackend) {
    std::vector<PKBackendType> available = {PKBackendType::CPU, PKBackendType::CUDA};
    EXPECT_EQ(select_best_backend(available), PKBackendType::CUDA);
    
    available = {PKBackendType::CPU, PKBackendType::MPS};
    EXPECT_EQ(select_best_backend(available), PKBackendType::MPS);
    
    available = {PKBackendType::CPU};
    EXPECT_EQ(select_best_backend(available), PKBackendType::CPU);
}

TEST_F(PKBackendPriorityTest, SelectBestBackendEmpty) {
    std::vector<PKBackendType> empty;
    EXPECT_EQ(select_best_backend(empty), PKBackendType::CPU);  // Default
}

TEST_F(PKBackendPriorityTest, SelectBackendByName) {
    EXPECT_EQ(select_backend_by_name("cuda"), PKBackendType::CUDA);
    EXPECT_EQ(select_backend_by_name("CUDA"), PKBackendType::CUDA);
    EXPECT_EQ(select_backend_by_name("nvidia"), PKBackendType::CUDA);
    EXPECT_EQ(select_backend_by_name("cpu"), PKBackendType::CPU);
    EXPECT_EQ(select_backend_by_name("mps"), PKBackendType::MPS);
    EXPECT_EQ(select_backend_by_name("metal"), PKBackendType::MPS);
    EXPECT_EQ(select_backend_by_name("ascend"), PKBackendType::ASCEND);
    EXPECT_EQ(select_backend_by_name("rocm"), PKBackendType::ROCM);
    EXPECT_EQ(select_backend_by_name("hip"), PKBackendType::ROCM);
}

TEST_F(PKBackendPriorityTest, SelectBackendByNameInvalid) {
    EXPECT_EQ(select_backend_by_name("invalid"), PKBackendType::NUM_BACKENDS);
    EXPECT_EQ(select_backend_by_name("xyz"), PKBackendType::NUM_BACKENDS);
}

// =============================================================================
// Mode Selection Tests
// =============================================================================

class PKModeSelectionTest : public ::testing::Test {};

TEST_F(PKModeSelectionTest, ParseMode) {
    EXPECT_EQ(parse_mode("offline"), PKMode::OFFLINE);
    EXPECT_EQ(parse_mode("OFFLINE"), PKMode::OFFLINE);
    EXPECT_EQ(parse_mode("batch"), PKMode::OFFLINE);
    
    EXPECT_EQ(parse_mode("online"), PKMode::ONLINE);
    EXPECT_EQ(parse_mode("single"), PKMode::ONLINE);
    
    EXPECT_EQ(parse_mode("eager"), PKMode::EAGER);
    EXPECT_EQ(parse_mode("immediate"), PKMode::EAGER);
    
    EXPECT_EQ(parse_mode("graph"), PKMode::GRAPH);
    EXPECT_EQ(parse_mode("compiled"), PKMode::GRAPH);
    
    EXPECT_EQ(parse_mode("streaming"), PKMode::STREAMING);
    EXPECT_EQ(parse_mode("pipeline"), PKMode::STREAMING);
}

TEST_F(PKModeSelectionTest, ParseModeDefault) {
    EXPECT_EQ(parse_mode("unknown"), PKMode::ONLINE);  // Default
    EXPECT_EQ(parse_mode(""), PKMode::ONLINE);
}

TEST_F(PKModeSelectionTest, GetDefaultModeForBackend) {
    EXPECT_EQ(get_default_mode_for_backend(PKBackendType::CUDA), PKMode::ONLINE);
    EXPECT_EQ(get_default_mode_for_backend(PKBackendType::ROCM), PKMode::ONLINE);
    EXPECT_EQ(get_default_mode_for_backend(PKBackendType::CPU), PKMode::EAGER);
    EXPECT_EQ(get_default_mode_for_backend(PKBackendType::MPS), PKMode::EAGER);
    EXPECT_EQ(get_default_mode_for_backend(PKBackendType::TRITON), PKMode::OFFLINE);
}

// =============================================================================
// PKRuntimeConfigBuilder Tests
// =============================================================================

class PKRuntimeConfigBuilderTest : public ::testing::Test {};

TEST_F(PKRuntimeConfigBuilderTest, DefaultValues) {
    auto config = PKRuntimeConfigBuilder().build();
    
    EXPECT_EQ(config.mode, PKMode::ONLINE);
    EXPECT_EQ(config.num_workers, 4);
    EXPECT_EQ(config.threads_per_worker, 256);
    EXPECT_EQ(config.max_seq_length, 2048u);
}

TEST_F(PKRuntimeConfigBuilderTest, SetMode) {
    auto config = PKRuntimeConfigBuilder()
        .mode(PKMode::OFFLINE)
        .build();
    
    EXPECT_EQ(config.mode, PKMode::OFFLINE);
}

TEST_F(PKRuntimeConfigBuilderTest, SetWorkers) {
    auto config = PKRuntimeConfigBuilder()
        .workers(8)
        .threads_per_worker(512)
        .build();
    
    EXPECT_EQ(config.num_workers, 8);
    EXPECT_EQ(config.threads_per_worker, 512);
}

TEST_F(PKRuntimeConfigBuilderTest, SetSchedulers) {
    auto config = PKRuntimeConfigBuilder()
        .schedulers(2, 4)
        .build();
    
    EXPECT_EQ(config.num_local_schedulers, 2);
    EXPECT_EQ(config.num_remote_schedulers, 4);
}

TEST_F(PKRuntimeConfigBuilderTest, SetBatchConfig) {
    auto config = PKRuntimeConfigBuilder()
        .batch_config(32, 128)
        .build();
    
    EXPECT_EQ(config.max_num_batched_requests, 32u);
    EXPECT_EQ(config.max_num_batched_tokens, 128u);
}

TEST_F(PKRuntimeConfigBuilderTest, SetPaging) {
    auto config = PKRuntimeConfigBuilder()
        .paging(2048, 128)
        .build();
    
    EXPECT_EQ(config.max_num_pages, 2048u);
    EXPECT_EQ(config.page_size, 128u);
}

TEST_F(PKRuntimeConfigBuilderTest, SetMultiGPU) {
    auto config = PKRuntimeConfigBuilder()
        .multi_gpu(8, 3)
        .build();
    
    EXPECT_EQ(config.num_gpus, 8);
    EXPECT_EQ(config.my_gpu_id, 3);
}

TEST_F(PKRuntimeConfigBuilderTest, SetProfiling) {
    auto config = PKRuntimeConfigBuilder()
        .profiling(true, reinterpret_cast<void*>(0x1234))
        .build();
    
    EXPECT_TRUE(config.profiling_enabled);
    EXPECT_EQ(config.profiler_buffer, reinterpret_cast<void*>(0x1234));
}

TEST_F(PKRuntimeConfigBuilderTest, ChainedCalls) {
    auto config = PKRuntimeConfigBuilder()
        .mode(PKMode::ONLINE)
        .workers(16)
        .max_seq_length(4096)
        .multi_gpu(4, 0)
        .profiling(true)
        .build();
    
    EXPECT_EQ(config.mode, PKMode::ONLINE);
    EXPECT_EQ(config.num_workers, 16);
    EXPECT_EQ(config.max_seq_length, 4096u);
    EXPECT_EQ(config.num_gpus, 4);
    EXPECT_TRUE(config.profiling_enabled);
}

// =============================================================================
// PKProfiler Tests
// =============================================================================

class PKProfilerTest : public ::testing::Test {};

TEST_F(PKProfilerTest, DefaultDisabled) {
    PKProfiler profiler;
    EXPECT_FALSE(profiler.is_enabled());
}

TEST_F(PKProfilerTest, EnableDisable) {
    PKProfiler profiler;
    
    profiler.enable();
    EXPECT_TRUE(profiler.is_enabled());
    
    profiler.disable();
    EXPECT_FALSE(profiler.is_enabled());
}

TEST_F(PKProfilerTest, StartStopDisabled) {
    PKProfiler profiler;
    
    profiler.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.stop();
    
    EXPECT_EQ(profiler.get_count(), 0u);  // Not counted when disabled
    EXPECT_DOUBLE_EQ(profiler.get_total_ms(), 0.0);
}

TEST_F(PKProfilerTest, StartStopEnabled) {
    PKProfiler profiler;
    profiler.enable();
    
    profiler.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.stop();
    
    EXPECT_EQ(profiler.get_count(), 1u);
    EXPECT_GT(profiler.get_total_ms(), 0.0);
}

TEST_F(PKProfilerTest, MultipleMeasurements) {
    PKProfiler profiler;
    profiler.enable();
    
    for (int i = 0; i < 5; ++i) {
        profiler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        profiler.stop();
    }
    
    EXPECT_EQ(profiler.get_count(), 5u);
    EXPECT_GT(profiler.get_total_ms(), 0.0);
    EXPECT_GT(profiler.get_avg_ms(), 0.0);
}

TEST_F(PKProfilerTest, Reset) {
    PKProfiler profiler;
    profiler.enable();
    
    profiler.start();
    profiler.stop();
    
    EXPECT_EQ(profiler.get_count(), 1u);
    
    profiler.reset();
    
    EXPECT_EQ(profiler.get_count(), 0u);
    EXPECT_DOUBLE_EQ(profiler.get_total_ms(), 0.0);
}

TEST_F(PKProfilerTest, AverageWithNoMeasurements) {
    PKProfiler profiler;
    EXPECT_DOUBLE_EQ(profiler.get_avg_ms(), 0.0);
}

// =============================================================================
// PKError Tests
// =============================================================================

class PKErrorTest : public ::testing::Test {};

TEST_F(PKErrorTest, SuccessValue) {
    EXPECT_EQ(static_cast<int>(PKError::SUCCESS), 0);
}

TEST_F(PKErrorTest, ErrorToString) {
    EXPECT_STREQ(pk_error_to_string(PKError::SUCCESS), "Success");
    EXPECT_STREQ(pk_error_to_string(PKError::BACKEND_NOT_AVAILABLE), "Backend not available");
    EXPECT_STREQ(pk_error_to_string(PKError::MODE_NOT_SUPPORTED), "Mode not supported");
    EXPECT_STREQ(pk_error_to_string(PKError::INITIALIZATION_FAILED), "Initialization failed");
    EXPECT_STREQ(pk_error_to_string(PKError::EXECUTION_FAILED), "Execution failed");
    EXPECT_STREQ(pk_error_to_string(PKError::OUT_OF_MEMORY), "Out of memory");
    EXPECT_STREQ(pk_error_to_string(PKError::INVALID_ARGUMENT), "Invalid argument");
    EXPECT_STREQ(pk_error_to_string(PKError::INTERNAL_ERROR), "Internal error");
}

// =============================================================================
// PKLogLevel Tests
// =============================================================================

class PKLogLevelTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Reset to default
        set_pk_log_level(PKLogLevel::INFO);
    }
};

TEST_F(PKLogLevelTest, DefaultLevel) {
    EXPECT_EQ(get_pk_log_level(), PKLogLevel::INFO);
}

TEST_F(PKLogLevelTest, SetLogLevel) {
    set_pk_log_level(PKLogLevel::DEBUG);
    EXPECT_EQ(get_pk_log_level(), PKLogLevel::DEBUG);
    
    set_pk_log_level(PKLogLevel::ERROR);
    EXPECT_EQ(get_pk_log_level(), PKLogLevel::ERROR);
    
    set_pk_log_level(PKLogLevel::NONE);
    EXPECT_EQ(get_pk_log_level(), PKLogLevel::NONE);
}

TEST_F(PKLogLevelTest, LogLevelToString) {
    EXPECT_STREQ(pk_log_level_to_string(PKLogLevel::DEBUG), "DEBUG");
    EXPECT_STREQ(pk_log_level_to_string(PKLogLevel::INFO), "INFO");
    EXPECT_STREQ(pk_log_level_to_string(PKLogLevel::WARNING), "WARNING");
    EXPECT_STREQ(pk_log_level_to_string(PKLogLevel::ERROR), "ERROR");
    EXPECT_STREQ(pk_log_level_to_string(PKLogLevel::NONE), "NONE");
}

TEST_F(PKLogLevelTest, LogLevelOrdering) {
    EXPECT_LT(static_cast<int>(PKLogLevel::DEBUG), static_cast<int>(PKLogLevel::INFO));
    EXPECT_LT(static_cast<int>(PKLogLevel::INFO), static_cast<int>(PKLogLevel::WARNING));
    EXPECT_LT(static_cast<int>(PKLogLevel::WARNING), static_cast<int>(PKLogLevel::ERROR));
    EXPECT_LT(static_cast<int>(PKLogLevel::ERROR), static_cast<int>(PKLogLevel::NONE));
}

// =============================================================================
// Parameterized Backend Priority Tests
// =============================================================================

struct BackendPriorityTestParam {
    PKBackendType type;
    int expected_priority;
};

class BackendPriorityParameterizedTest
    : public ::testing::TestWithParam<BackendPriorityTestParam> {};

TEST_P(BackendPriorityParameterizedTest, Priority) {
    auto param = GetParam();
    EXPECT_EQ(get_backend_priority(param.type), param.expected_priority);
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendPriorities,
    BackendPriorityParameterizedTest,
    ::testing::Values(
        BackendPriorityTestParam{PKBackendType::CUDA, 100},
        BackendPriorityTestParam{PKBackendType::ROCM, 95},
        BackendPriorityTestParam{PKBackendType::MACA, 90},
        BackendPriorityTestParam{PKBackendType::ASCEND, 85},
        BackendPriorityTestParam{PKBackendType::MPS, 70},
        BackendPriorityTestParam{PKBackendType::TRITON, 60},
        BackendPriorityTestParam{PKBackendType::NKI, 50},
        BackendPriorityTestParam{PKBackendType::CPU, 10}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
