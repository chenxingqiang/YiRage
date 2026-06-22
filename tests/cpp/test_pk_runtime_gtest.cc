// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_pk_runtime_gtest.cc
 * @brief Persistent Kernel Runtime Unit Tests (Google Test version)
 *
 * Tests for PK (Persistent Kernel) runtime across all backends.
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>

namespace yirage {
namespace pk {

// =============================================================================
// Mock PK Runtime Types for Testing
// =============================================================================

enum class PKBackendType {
    CUDA,
    ROCM,
    CPU,
    MPS,
    ASCEND,
    MACA,
};

struct PKConfig {
    PKBackendType backend = PKBackendType::CPU;
    int32_t num_streams = 4;
    size_t work_buffer_size = 1024 * 1024;  // 1MB
    bool enable_profiling = false;
    int32_t timeout_ms = 1000;
};

struct PKStatus {
    bool initialized = false;
    bool running = false;
    int32_t active_streams = 0;
    size_t allocated_memory = 0;
    int32_t kernels_launched = 0;
};

// Mock PK runtime for testing
class MockPKRuntime {
public:
    explicit MockPKRuntime(const PKConfig& config) 
        : config_(config), initialized_(false), running_(false) {}
    
    ~MockPKRuntime() {
        if (running_) {
            stop();
        }
    }
    
    bool initialize() {
        if (initialized_) return false;
        
        // Simulate initialization
        allocated_memory_ = config_.work_buffer_size;
        initialized_ = true;
        return true;
    }
    
    bool start() {
        if (!initialized_ || running_) return false;
        
        active_streams_ = config_.num_streams;
        running_ = true;
        return true;
    }
    
    bool stop() {
        if (!running_) return false;
        
        running_ = false;
        active_streams_ = 0;
        return true;
    }
    
    bool launch_kernel(const std::string& name) {
        if (!running_) return false;
        
        kernels_launched_++;
        return true;
    }
    
    bool synchronize() {
        if (!running_) return false;
        
        // Simulate sync
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        return true;
    }
    
    void* allocate(size_t size) {
        if (!initialized_) return nullptr;
        
        allocated_memory_ += size;
        // Return dummy pointer (don't actually allocate)
        return reinterpret_cast<void*>(allocated_memory_);
    }
    
    void deallocate(void* ptr) {
        // No-op for mock
        (void)ptr;
    }
    
    PKStatus get_status() const {
        PKStatus status;
        status.initialized = initialized_;
        status.running = running_;
        status.active_streams = active_streams_;
        status.allocated_memory = allocated_memory_;
        status.kernels_launched = kernels_launched_;
        return status;
    }
    
    PKConfig get_config() const { return config_; }
    
private:
    PKConfig config_;
    bool initialized_;
    bool running_;
    int32_t active_streams_ = 0;
    size_t allocated_memory_ = 0;
    int32_t kernels_launched_ = 0;
};

// Thread pool for CPU backend
class MockThreadPool {
public:
    explicit MockThreadPool(int num_threads) 
        : num_threads_(num_threads), active_(false) {}
    
    bool start() {
        if (active_) return false;
        active_ = true;
        return true;
    }
    
    bool stop() {
        if (!active_) return false;
        active_ = false;
        return true;
    }
    
    int get_num_threads() const { return num_threads_; }
    bool is_active() const { return active_; }
    
private:
    int num_threads_;
    bool active_;
};

}  // namespace pk
}  // namespace yirage

using namespace yirage::pk;

// =============================================================================
// PKConfig Tests
// =============================================================================

class PKConfigTest : public ::testing::Test {
protected:
    PKConfig config;
};

TEST_F(PKConfigTest, DefaultValues) {
    EXPECT_EQ(config.backend, PKBackendType::CPU);
    EXPECT_EQ(config.num_streams, 4);
    EXPECT_EQ(config.work_buffer_size, 1024u * 1024u);
    EXPECT_FALSE(config.enable_profiling);
    EXPECT_EQ(config.timeout_ms, 1000);
}

TEST_F(PKConfigTest, CustomConfiguration) {
    config.backend = PKBackendType::CUDA;
    config.num_streams = 8;
    config.work_buffer_size = 4 * 1024 * 1024;
    config.enable_profiling = true;
    
    EXPECT_EQ(config.backend, PKBackendType::CUDA);
    EXPECT_EQ(config.num_streams, 8);
    EXPECT_EQ(config.work_buffer_size, 4u * 1024u * 1024u);
    EXPECT_TRUE(config.enable_profiling);
}

// =============================================================================
// CUDA PK Backend Tests
// =============================================================================

class CUDAPKBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.backend = PKBackendType::CUDA;
        config.num_streams = 4;
        runtime = std::make_unique<MockPKRuntime>(config);
    }
    
    PKConfig config;
    std::unique_ptr<MockPKRuntime> runtime;
};

TEST_F(CUDAPKBackendTest, Initialization) {
    EXPECT_TRUE(runtime->initialize());
    
    auto status = runtime->get_status();
    EXPECT_TRUE(status.initialized);
    EXPECT_FALSE(status.running);
}

TEST_F(CUDAPKBackendTest, DoubleInitializationFails) {
    EXPECT_TRUE(runtime->initialize());
    EXPECT_FALSE(runtime->initialize());  // Should fail
}

TEST_F(CUDAPKBackendTest, MemoryAllocation) {
    EXPECT_TRUE(runtime->initialize());
    
    void* ptr = runtime->allocate(4096);
    EXPECT_NE(ptr, nullptr);
    
    auto status = runtime->get_status();
    EXPECT_GT(status.allocated_memory, 4096u);
}

TEST_F(CUDAPKBackendTest, KernelLaunch) {
    EXPECT_TRUE(runtime->initialize());
    EXPECT_TRUE(runtime->start());
    
    EXPECT_TRUE(runtime->launch_kernel("matmul"));
    EXPECT_TRUE(runtime->launch_kernel("silu"));
    
    auto status = runtime->get_status();
    EXPECT_EQ(status.kernels_launched, 2);
}

TEST_F(CUDAPKBackendTest, Synchronization) {
    EXPECT_TRUE(runtime->initialize());
    EXPECT_TRUE(runtime->start());
    
    EXPECT_TRUE(runtime->synchronize());
}

// =============================================================================
// CPU PK Backend Tests
// =============================================================================

class CPUPKBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.backend = PKBackendType::CPU;
        config.num_streams = 8;  // Thread count for CPU
        runtime = std::make_unique<MockPKRuntime>(config);
    }
    
    PKConfig config;
    std::unique_ptr<MockPKRuntime> runtime;
};

TEST_F(CPUPKBackendTest, ThreadPoolCreation) {
    MockThreadPool pool(8);
    EXPECT_EQ(pool.get_num_threads(), 8);
    EXPECT_FALSE(pool.is_active());
}

TEST_F(CPUPKBackendTest, ThreadPoolStartStop) {
    MockThreadPool pool(4);
    
    EXPECT_TRUE(pool.start());
    EXPECT_TRUE(pool.is_active());
    
    EXPECT_TRUE(pool.stop());
    EXPECT_FALSE(pool.is_active());
}

TEST_F(CPUPKBackendTest, RuntimeLifecycle) {
    EXPECT_TRUE(runtime->initialize());
    EXPECT_TRUE(runtime->start());
    
    auto status = runtime->get_status();
    EXPECT_TRUE(status.initialized);
    EXPECT_TRUE(status.running);
    EXPECT_EQ(status.active_streams, 8);
    
    EXPECT_TRUE(runtime->stop());
    
    status = runtime->get_status();
    EXPECT_FALSE(status.running);
    EXPECT_EQ(status.active_streams, 0);
}

// =============================================================================
// MPS PK Backend Tests
// =============================================================================

class MPSPKBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.backend = PKBackendType::MPS;
        config.num_streams = 2;  // MPS typically uses fewer streams
        runtime = std::make_unique<MockPKRuntime>(config);
    }
    
    PKConfig config;
    std::unique_ptr<MockPKRuntime> runtime;
};

TEST_F(MPSPKBackendTest, Initialization) {
    EXPECT_TRUE(runtime->initialize());
    
    auto cfg = runtime->get_config();
    EXPECT_EQ(cfg.backend, PKBackendType::MPS);
    EXPECT_EQ(cfg.num_streams, 2);
}

// =============================================================================
// Ascend PK Backend Tests
// =============================================================================

class AscendPKBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.backend = PKBackendType::ASCEND;
        config.num_streams = 4;
        runtime = std::make_unique<MockPKRuntime>(config);
    }
    
    PKConfig config;
    std::unique_ptr<MockPKRuntime> runtime;
};

TEST_F(AscendPKBackendTest, Initialization) {
    EXPECT_TRUE(runtime->initialize());
    
    auto cfg = runtime->get_config();
    EXPECT_EQ(cfg.backend, PKBackendType::ASCEND);
}

// =============================================================================
// MACA PK Backend Tests
// =============================================================================

class MACAPKBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.backend = PKBackendType::MACA;
        config.num_streams = 4;
        runtime = std::make_unique<MockPKRuntime>(config);
    }
    
    PKConfig config;
    std::unique_ptr<MockPKRuntime> runtime;
};

TEST_F(MACAPKBackendTest, Initialization) {
    EXPECT_TRUE(runtime->initialize());
    
    auto cfg = runtime->get_config();
    EXPECT_EQ(cfg.backend, PKBackendType::MACA);
}

// =============================================================================
// Cross-Backend Tests
// =============================================================================

class PKCrossBackendTest : public ::testing::TestWithParam<PKBackendType> {};

TEST_P(PKCrossBackendTest, BasicLifecycle) {
    PKBackendType backend = GetParam();
    
    PKConfig config;
    config.backend = backend;
    config.num_streams = 4;
    
    MockPKRuntime runtime(config);
    
    EXPECT_TRUE(runtime.initialize());
    EXPECT_TRUE(runtime.start());
    
    auto status = runtime.get_status();
    EXPECT_TRUE(status.initialized);
    EXPECT_TRUE(status.running);
    
    EXPECT_TRUE(runtime.stop());
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    PKCrossBackendTest,
    ::testing::Values(
        PKBackendType::CUDA,
        PKBackendType::ROCM,
        PKBackendType::CPU,
        PKBackendType::MPS,
        PKBackendType::ASCEND,
        PKBackendType::MACA
    )
);

// =============================================================================
// Stress Tests
// =============================================================================

class PKStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.backend = PKBackendType::CPU;
        config.num_streams = 8;
        runtime = std::make_unique<MockPKRuntime>(config);
        runtime->initialize();
        runtime->start();
    }
    
    void TearDown() override {
        runtime->stop();
    }
    
    PKConfig config;
    std::unique_ptr<MockPKRuntime> runtime;
};

TEST_F(PKStressTest, MultipleKernelLaunches) {
    const int num_launches = 1000;
    
    for (int i = 0; i < num_launches; ++i) {
        EXPECT_TRUE(runtime->launch_kernel("kernel_" + std::to_string(i)));
    }
    
    auto status = runtime->get_status();
    EXPECT_EQ(status.kernels_launched, num_launches);
}

TEST_F(PKStressTest, RepeatedStartStop) {
    runtime->stop();  // Stop first
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(runtime->start());
        EXPECT_TRUE(runtime->stop());
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
