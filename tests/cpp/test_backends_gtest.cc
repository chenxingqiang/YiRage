// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_backends_gtest.cc
 * @brief Backend API Unit Tests (Google Test version)
 *
 * Tests for backend registration, discovery, and hardware capabilities.
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace yirage {
namespace backend {

// =============================================================================
// Mock Backend Types for Testing
// =============================================================================

enum class BackendType {
    CUDA = 0,
    ROCM = 1,
    CPU = 2,
    MPS = 3,
    ASCEND = 4,
    MACA = 5,
    TPU = 6,
    XPU = 7,
    FPGA = 8,
    TRITON = 9,
    NKI = 10,
    MLIR = 11,
};

enum class DataType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT32,
    INT8,
};

struct BackendInfo {
    std::string name;
    std::string display_name;
    bool requires_gpu;
    bool is_available;
    int32_t warp_size;
    int32_t max_threads_per_block;
    size_t max_shared_memory;
    std::vector<DataType> supported_dtypes;
};

// Mock backend interface
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual const char* get_name() const = 0;
    virtual const char* get_display_name() const = 0;
    virtual BackendType get_type() const = 0;
    virtual bool is_available() const = 0;
    virtual BackendInfo get_info() const = 0;
    virtual int get_device_count() const = 0;
    virtual size_t get_max_memory() const = 0;
    virtual int get_num_compute_units() const = 0;
    virtual bool supports_data_type(DataType dtype) const = 0;
};

// Mock CPU backend
class CPUBackend : public IBackend {
public:
    const char* get_name() const override { return "cpu"; }
    const char* get_display_name() const override { return "CPU"; }
    BackendType get_type() const override { return BackendType::CPU; }
    bool is_available() const override { return true; }
    
    BackendInfo get_info() const override {
        return BackendInfo{
            "cpu", "CPU", false, true, 1, 1, 0,
            {DataType::FLOAT32, DataType::FLOAT16, DataType::INT32}
        };
    }
    
    int get_device_count() const override { return 1; }
    size_t get_max_memory() const override { return 16ULL * 1024 * 1024 * 1024; }  // 16GB
    int get_num_compute_units() const override { return 8; }
    
    bool supports_data_type(DataType dtype) const override {
        return dtype == DataType::FLOAT32 || 
               dtype == DataType::FLOAT16 || 
               dtype == DataType::INT32;
    }
};

// Mock CUDA backend
class CUDABackend : public IBackend {
public:
    const char* get_name() const override { return "cuda"; }
    const char* get_display_name() const override { return "NVIDIA CUDA"; }
    BackendType get_type() const override { return BackendType::CUDA; }
    bool is_available() const override { return cuda_available_; }
    
    BackendInfo get_info() const override {
        return BackendInfo{
            "cuda", "NVIDIA CUDA", true, cuda_available_, 32, 1024, 49152,
            {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16, DataType::INT32, DataType::INT8}
        };
    }
    
    int get_device_count() const override { return cuda_available_ ? 1 : 0; }
    size_t get_max_memory() const override { return 24ULL * 1024 * 1024 * 1024; }  // 24GB
    int get_num_compute_units() const override { return 128; }
    
    bool supports_data_type(DataType dtype) const override { return true; }
    
    void set_available(bool available) { cuda_available_ = available; }
    
private:
    bool cuda_available_ = false;
};

// Mock backend registry
class BackendRegistry {
public:
    static BackendRegistry& get_instance() {
        static BackendRegistry instance;
        return instance;
    }
    
    void register_backend(std::unique_ptr<IBackend> backend) {
        backends_[backend->get_type()] = std::move(backend);
    }
    
    IBackend* get_backend(BackendType type) {
        auto it = backends_.find(type);
        return it != backends_.end() ? it->second.get() : nullptr;
    }
    
    IBackend* get_backend(const std::string& name) {
        for (auto& [type, backend] : backends_) {
            if (backend->get_name() == name) {
                return backend.get();
            }
        }
        return nullptr;
    }
    
    std::vector<BackendType> get_available_backends() const {
        std::vector<BackendType> available;
        for (const auto& [type, backend] : backends_) {
            if (backend->is_available()) {
                available.push_back(type);
            }
        }
        return available;
    }
    
    BackendType get_default_backend() const {
        // Priority: CUDA > ROCm > MPS > MACA > Ascend > CPU
        std::vector<BackendType> priority = {
            BackendType::CUDA, BackendType::ROCM, BackendType::MPS,
            BackendType::MACA, BackendType::ASCEND, BackendType::CPU
        };
        
        for (auto type : priority) {
            auto it = backends_.find(type);
            if (it != backends_.end() && it->second->is_available()) {
                return type;
            }
        }
        return BackendType::CPU;
    }
    
    void clear() { backends_.clear(); }
    
private:
    BackendRegistry() = default;
    std::map<BackendType, std::unique_ptr<IBackend>> backends_;
};

}  // namespace backend
}  // namespace yirage

using namespace yirage::backend;

// =============================================================================
// Test Fixtures
// =============================================================================

class BackendRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto& registry = BackendRegistry::get_instance();
        registry.clear();
        
        // Register CPU backend (always available)
        registry.register_backend(std::make_unique<CPUBackend>());
        
        // Register CUDA backend (not available by default)
        registry.register_backend(std::make_unique<CUDABackend>());
    }
    
    void TearDown() override {
        BackendRegistry::get_instance().clear();
    }
};

// =============================================================================
// Backend Registry Tests
// =============================================================================

TEST_F(BackendRegistryTest, Singleton) {
    auto& registry1 = BackendRegistry::get_instance();
    auto& registry2 = BackendRegistry::get_instance();
    
    EXPECT_EQ(&registry1, &registry2);
}

TEST_F(BackendRegistryTest, HasAvailableBackends) {
    auto& registry = BackendRegistry::get_instance();
    auto available = registry.get_available_backends();
    
    // At minimum, CPU should always be available
    EXPECT_FALSE(available.empty());
}

TEST_F(BackendRegistryTest, CPUAlwaysAvailable) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    ASSERT_NE(cpu, nullptr);
    EXPECT_TRUE(cpu->is_available());
    EXPECT_STREQ(cpu->get_name(), "cpu");
}

TEST_F(BackendRegistryTest, GetByType) {
    auto& registry = BackendRegistry::get_instance();
    
    auto* backend = registry.get_backend(BackendType::CPU);
    ASSERT_NE(backend, nullptr);
    EXPECT_STREQ(backend->get_name(), "cpu");
}

TEST_F(BackendRegistryTest, GetByName) {
    auto& registry = BackendRegistry::get_instance();
    
    auto* cpu = registry.get_backend("cpu");
    ASSERT_NE(cpu, nullptr);
    
    auto* nonexistent = registry.get_backend("nonexistent");
    EXPECT_EQ(nonexistent, nullptr);
}

TEST_F(BackendRegistryTest, DefaultBackend) {
    auto& registry = BackendRegistry::get_instance();
    auto default_type = registry.get_default_backend();
    
    auto* backend = registry.get_backend(default_type);
    ASSERT_NE(backend, nullptr);
    EXPECT_TRUE(backend->is_available());
}

// =============================================================================
// CPU Backend Tests
// =============================================================================

TEST_F(BackendRegistryTest, CPUBackendBasicInfo) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    ASSERT_NE(cpu, nullptr);
    EXPECT_TRUE(cpu->is_available());
    EXPECT_EQ(cpu->get_type(), BackendType::CPU);
    
    auto info = cpu->get_info();
    EXPECT_FALSE(info.requires_gpu);
}

TEST_F(BackendRegistryTest, CPUBackendMemoryInfo) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    ASSERT_NE(cpu, nullptr);
    
    size_t max_memory = cpu->get_max_memory();
    EXPECT_GT(max_memory, 0u);
    
    int device_count = cpu->get_device_count();
    EXPECT_GE(device_count, 1);
}

TEST_F(BackendRegistryTest, CPUBackendDataTypeSupport) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    ASSERT_NE(cpu, nullptr);
    
    EXPECT_TRUE(cpu->supports_data_type(DataType::FLOAT32));
    EXPECT_TRUE(cpu->supports_data_type(DataType::FLOAT16));
    EXPECT_TRUE(cpu->supports_data_type(DataType::INT32));
}

TEST_F(BackendRegistryTest, CPUBackendComputeUnits) {
    auto& registry = BackendRegistry::get_instance();
    auto* cpu = registry.get_backend("cpu");
    
    ASSERT_NE(cpu, nullptr);
    
    int units = cpu->get_num_compute_units();
    EXPECT_GT(units, 0);
}

// =============================================================================
// CUDA Backend Tests
// =============================================================================

TEST_F(BackendRegistryTest, CUDABackendRegistration) {
    auto& registry = BackendRegistry::get_instance();
    auto* cuda = registry.get_backend("cuda");
    
    ASSERT_NE(cuda, nullptr);
    EXPECT_EQ(cuda->get_type(), BackendType::CUDA);
}

TEST_F(BackendRegistryTest, CUDABackendAvailability) {
    auto& registry = BackendRegistry::get_instance();
    auto* cuda = registry.get_backend("cuda");
    
    ASSERT_NE(cuda, nullptr);
    
    // By default, CUDA is not available in mock
    EXPECT_FALSE(cuda->is_available());
}

TEST_F(BackendRegistryTest, CUDABackendInfo) {
    auto& registry = BackendRegistry::get_instance();
    auto* cuda = registry.get_backend("cuda");
    
    ASSERT_NE(cuda, nullptr);
    
    auto info = cuda->get_info();
    EXPECT_EQ(info.warp_size, 32);
    EXPECT_EQ(info.max_threads_per_block, 1024);
    EXPECT_TRUE(info.requires_gpu);
}

// =============================================================================
// Cross-Backend Tests
// =============================================================================

TEST_F(BackendRegistryTest, AllRegisteredBackendsHaveNames) {
    auto& registry = BackendRegistry::get_instance();
    auto available = registry.get_available_backends();
    
    for (auto type : available) {
        auto* backend = registry.get_backend(type);
        ASSERT_NE(backend, nullptr);
        
        const char* name = backend->get_name();
        EXPECT_NE(name, nullptr);
        EXPECT_GT(strlen(name), 0u);
    }
}

TEST_F(BackendRegistryTest, AllBackendsHaveDisplayNames) {
    auto& registry = BackendRegistry::get_instance();
    auto available = registry.get_available_backends();
    
    for (auto type : available) {
        auto* backend = registry.get_backend(type);
        ASSERT_NE(backend, nullptr);
        
        const char* display = backend->get_display_name();
        EXPECT_NE(display, nullptr);
        EXPECT_GT(strlen(display), 0u);
    }
}

TEST_F(BackendRegistryTest, DataTypeSupportConsistency) {
    auto& registry = BackendRegistry::get_instance();
    auto available = registry.get_available_backends();
    
    for (auto type : available) {
        auto* backend = registry.get_backend(type);
        if (!backend || !backend->is_available()) continue;
        
        // All backends should support FP32
        EXPECT_TRUE(backend->supports_data_type(DataType::FLOAT32));
    }
}

// =============================================================================
// Backend Priority Tests
// =============================================================================

TEST_F(BackendRegistryTest, GPUPreferredOverCPU) {
    auto& registry = BackendRegistry::get_instance();
    
    // Make CUDA available
    auto* cuda = dynamic_cast<CUDABackend*>(registry.get_backend("cuda"));
    if (cuda) {
        cuda->set_available(true);
    }
    
    auto default_type = registry.get_default_backend();
    
    // If CUDA is available, it should be the default
    if (cuda && cuda->is_available()) {
        EXPECT_EQ(default_type, BackendType::CUDA);
    }
}

// =============================================================================
// Parameterized Tests
// =============================================================================

class BackendDataTypeTest : public ::testing::TestWithParam<DataType> {};

TEST_P(BackendDataTypeTest, CPUSupportsBasicTypes) {
    auto& registry = BackendRegistry::get_instance();
    registry.clear();
    registry.register_backend(std::make_unique<CPUBackend>());
    
    auto* cpu = registry.get_backend("cpu");
    ASSERT_NE(cpu, nullptr);
    
    DataType dtype = GetParam();
    
    // CPU should support FLOAT32, FLOAT16, INT32
    if (dtype == DataType::FLOAT32 || 
        dtype == DataType::FLOAT16 || 
        dtype == DataType::INT32) {
        EXPECT_TRUE(cpu->supports_data_type(dtype));
    }
}

INSTANTIATE_TEST_SUITE_P(
    DataTypes,
    BackendDataTypeTest,
    ::testing::Values(
        DataType::FLOAT32,
        DataType::FLOAT16,
        DataType::BFLOAT16,
        DataType::INT32,
        DataType::INT8
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
