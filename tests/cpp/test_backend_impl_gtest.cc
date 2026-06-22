// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_backend_impl_gtest.cc
 * @brief Comprehensive Backend Implementation Tests (Google Test version)
 *
 * Tests for YiRage backend implementations: registry, interface, and
 * individual backend implementations (CPU, CUDA, ROCm, MPS, etc.)
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <cstring>

namespace yirage {
namespace type {

// Backend type enumeration (mirrors actual implementation)
enum BackendType {
    BT_UNKNOWN = 0,
    BT_CPU = 1,
    BT_CUDA = 2,
    BT_ROCM = 3,
    BT_MPS = 4,
    BT_ASCEND = 5,
    BT_MACA = 6,
    BT_TPU = 7,
    BT_XPU = 8,
    BT_FPGA = 9,
    BT_TRITON = 10,
    BT_NKI = 11,
    BT_MLIR = 12,
    BT_CUDNN = 13,
    BT_MKL = 14,
};

// Data type enumeration
enum DataType {
    DT_UNKNOWN = 0,
    DT_FLOAT16 = 1,
    DT_BFLOAT16 = 2,
    DT_FLOAT32 = 3,
    DT_DOUBLE = 4,
    DT_INT8 = 5,
    DT_INT16 = 6,
    DT_INT32 = 7,
    DT_INT64 = 8,
    DT_UINT8 = 9,
    DT_UINT16 = 10,
    DT_UINT32 = 11,
    DT_UINT64 = 12,
    DT_FLOAT8 = 13,
    DT_FLOAT4 = 14,
    DT_INT4 = 15,
    DT_UINT4 = 16,
};

struct BackendInfo {
    BackendType type = BT_UNKNOWN;
    std::string name;
    std::string display_name;
    bool requires_gpu = false;
    std::vector<std::string> required_libs;
};

}  // namespace type

namespace backend {

// Compile context structure
struct CompileContext {
    std::string source_code;
    std::string output_path;
    std::vector<std::string> include_dirs;
    std::vector<std::string> compile_flags;
    bool debug_mode = false;
    int optimization_level = 2;
};

// Backend interface (abstract)
class BackendInterface {
public:
    virtual ~BackendInterface() = default;

    // Backend Information
    virtual type::BackendType get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_display_name() const = 0;
    virtual bool is_available() const = 0;
    virtual type::BackendInfo get_info() const = 0;

    // Compilation
    virtual bool compile(CompileContext const& ctx) = 0;
    virtual std::string get_compile_flags() const = 0;
    virtual std::vector<std::string> get_include_dirs() const = 0;
    virtual std::vector<std::string> get_library_dirs() const = 0;
    virtual std::vector<std::string> get_link_libraries() const = 0;

    // Memory Management
    virtual void* allocate_memory(size_t size) = 0;
    virtual void free_memory(void* ptr) = 0;
    virtual bool copy_to_device(void* dst, void const* src, size_t size) = 0;
    virtual bool copy_to_host(void* dst, void const* src, size_t size) = 0;
    virtual bool copy_device_to_device(void* dst, void const* src, size_t size) = 0;

    // Synchronization
    virtual void synchronize() = 0;

    // Capability Query
    virtual size_t get_max_memory() const = 0;
    virtual size_t get_max_shared_memory() const = 0;
    virtual bool supports_data_type(type::DataType dt) const = 0;
    virtual int get_compute_capability() const = 0;
    virtual int get_num_compute_units() const = 0;

    // Device Management
    virtual bool set_device(int device_id) = 0;
    virtual int get_device() const = 0;
    virtual int get_device_count() const = 0;
};

// CPU Backend implementation (mock for testing)
class CPUBackend : public BackendInterface {
public:
    CPUBackend() : num_cores_(8), total_memory_(16ULL * 1024 * 1024 * 1024) {}

    type::BackendType get_type() const override { return type::BT_CPU; }
    std::string get_name() const override { return "cpu"; }
    std::string get_display_name() const override { return "CPU"; }
    bool is_available() const override { return true; }

    type::BackendInfo get_info() const override {
        type::BackendInfo info;
        info.type = type::BT_CPU;
        info.name = "cpu";
        info.display_name = "CPU";
        info.requires_gpu = false;
        info.required_libs = {};
        return info;
    }

    bool compile(CompileContext const& ctx) override { return true; }
    std::string get_compile_flags() const override {
        return "-std=c++17 -O2 -march=native -fopenmp";
    }
    std::vector<std::string> get_include_dirs() const override { return {}; }
    std::vector<std::string> get_library_dirs() const override { return {}; }
    std::vector<std::string> get_link_libraries() const override {
        return {"gomp", "pthread"};
    }

    void* allocate_memory(size_t size) override {
        return std::malloc(size);
    }
    void free_memory(void* ptr) override {
        if (ptr) std::free(ptr);
    }
    bool copy_to_device(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size);
        return true;
    }
    bool copy_to_host(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size);
        return true;
    }
    bool copy_device_to_device(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size);
        return true;
    }

    void synchronize() override {}

    size_t get_max_memory() const override { return total_memory_; }
    size_t get_max_shared_memory() const override { return 8 * 1024 * 1024; }
    bool supports_data_type(type::DataType dt) const override {
        return dt >= type::DT_FLOAT16 && dt <= type::DT_UINT64;
    }
    int get_compute_capability() const override { return 100; }
    int get_num_compute_units() const override { return num_cores_; }

    bool set_device(int device_id) override { return device_id == 0; }
    int get_device() const override { return 0; }
    int get_device_count() const override { return 1; }

private:
    int num_cores_;
    size_t total_memory_;
};

// CUDA Backend implementation (mock for testing)
class CUDABackend : public BackendInterface {
public:
    CUDABackend() 
        : is_available_(true), current_device_(0), device_count_(1),
          total_memory_(24ULL * 1024 * 1024 * 1024),
          shared_memory_(48 * 1024), sm_count_(84), compute_capability_(86) {}

    type::BackendType get_type() const override { return type::BT_CUDA; }
    std::string get_name() const override { return "cuda"; }
    std::string get_display_name() const override { return "CUDA"; }
    bool is_available() const override { return is_available_; }

    type::BackendInfo get_info() const override {
        type::BackendInfo info;
        info.type = type::BT_CUDA;
        info.name = "cuda";
        info.display_name = "CUDA";
        info.requires_gpu = true;
        info.required_libs = {"cudart", "cuda", "cudadevrt"};
        return info;
    }

    bool compile(CompileContext const& ctx) override { return true; }
    std::string get_compile_flags() const override {
        return "-std=c++17 -O2 -Xcompiler=-fPIC";
    }
    std::vector<std::string> get_include_dirs() const override {
        return {"/usr/local/cuda/include"};
    }
    std::vector<std::string> get_library_dirs() const override {
        return {"/usr/local/cuda/lib64"};
    }
    std::vector<std::string> get_link_libraries() const override {
        return {"cudart", "cuda", "cudadevrt", "cublas"};
    }

    void* allocate_memory(size_t size) override {
        return std::malloc(size);  // Mock
    }
    void free_memory(void* ptr) override {
        if (ptr) std::free(ptr);
    }
    bool copy_to_device(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size);
        return true;
    }
    bool copy_to_host(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size);
        return true;
    }
    bool copy_device_to_device(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size);
        return true;
    }

    void synchronize() override {}

    size_t get_max_memory() const override { return total_memory_; }
    size_t get_max_shared_memory() const override { return shared_memory_; }
    bool supports_data_type(type::DataType dt) const override {
        if (dt >= type::DT_FLOAT16 && dt <= type::DT_UINT64) return true;
        // Newer data types require compute capability >= 8.0
        if (compute_capability_ >= 80) {
            return dt == type::DT_FLOAT8 || dt == type::DT_FLOAT4 ||
                   dt == type::DT_INT4 || dt == type::DT_UINT4;
        }
        return false;
    }
    int get_compute_capability() const override { return compute_capability_; }
    int get_num_compute_units() const override { return sm_count_; }

    bool set_device(int device_id) override {
        if (device_id >= 0 && device_id < device_count_) {
            current_device_ = device_id;
            return true;
        }
        return false;
    }
    int get_device() const override { return current_device_; }
    int get_device_count() const override { return device_count_; }

private:
    bool is_available_;
    int current_device_;
    int device_count_;
    size_t total_memory_;
    size_t shared_memory_;
    int sm_count_;
    int compute_capability_;
};

// ROCm Backend implementation (mock for testing)
class ROCmBackend : public BackendInterface {
public:
    ROCmBackend() 
        : is_available_(true), current_device_(0), device_count_(1) {}

    type::BackendType get_type() const override { return type::BT_ROCM; }
    std::string get_name() const override { return "rocm"; }
    std::string get_display_name() const override { return "ROCm"; }
    bool is_available() const override { return is_available_; }

    type::BackendInfo get_info() const override {
        type::BackendInfo info;
        info.type = type::BT_ROCM;
        info.name = "rocm";
        info.display_name = "ROCm";
        info.requires_gpu = true;
        info.required_libs = {"hip", "amdhip64", "rocblas"};
        return info;
    }

    bool compile(CompileContext const& ctx) override { return true; }
    std::string get_compile_flags() const override {
        return "-std=c++17 -O2 -D__HIP_PLATFORM_AMD__";
    }
    std::vector<std::string> get_include_dirs() const override {
        return {"/opt/rocm/include"};
    }
    std::vector<std::string> get_library_dirs() const override {
        return {"/opt/rocm/lib"};
    }
    std::vector<std::string> get_link_libraries() const override {
        return {"hip", "amdhip64", "rocblas"};
    }

    void* allocate_memory(size_t size) override { return std::malloc(size); }
    void free_memory(void* ptr) override { if (ptr) std::free(ptr); }
    bool copy_to_device(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size); return true;
    }
    bool copy_to_host(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size); return true;
    }
    bool copy_device_to_device(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size); return true;
    }

    void synchronize() override {}

    size_t get_max_memory() const override { return 16ULL * 1024 * 1024 * 1024; }
    size_t get_max_shared_memory() const override { return 64 * 1024; }
    bool supports_data_type(type::DataType dt) const override {
        return dt >= type::DT_FLOAT16 && dt <= type::DT_UINT64;
    }
    int get_compute_capability() const override { return 908; }  // gfx908
    int get_num_compute_units() const override { return 120; }

    bool set_device(int device_id) override {
        if (device_id >= 0 && device_id < device_count_) {
            current_device_ = device_id;
            return true;
        }
        return false;
    }
    int get_device() const override { return current_device_; }
    int get_device_count() const override { return device_count_; }

private:
    bool is_available_;
    int current_device_;
    int device_count_;
};

// MPS Backend implementation (mock for testing)
class MPSBackend : public BackendInterface {
public:
    MPSBackend() : is_available_(true), unified_memory_(32ULL * 1024 * 1024 * 1024) {}

    type::BackendType get_type() const override { return type::BT_MPS; }
    std::string get_name() const override { return "mps"; }
    std::string get_display_name() const override { return "Metal"; }
    bool is_available() const override { return is_available_; }

    type::BackendInfo get_info() const override {
        type::BackendInfo info;
        info.type = type::BT_MPS;
        info.name = "mps";
        info.display_name = "Metal";
        info.requires_gpu = true;
        info.required_libs = {"Metal", "MetalPerformanceShaders"};
        return info;
    }

    bool compile(CompileContext const& ctx) override { return true; }
    std::string get_compile_flags() const override {
        return "-std=c++17 -O2 -framework Metal -framework Foundation";
    }
    std::vector<std::string> get_include_dirs() const override { return {}; }
    std::vector<std::string> get_library_dirs() const override { return {}; }
    std::vector<std::string> get_link_libraries() const override {
        return {"Metal", "MetalPerformanceShaders"};
    }

    void* allocate_memory(size_t size) override { return std::malloc(size); }
    void free_memory(void* ptr) override { if (ptr) std::free(ptr); }
    bool copy_to_device(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size); return true;
    }
    bool copy_to_host(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size); return true;
    }
    bool copy_device_to_device(void* dst, void const* src, size_t size) override {
        std::memcpy(dst, src, size); return true;
    }

    void synchronize() override {}

    size_t get_max_memory() const override { return unified_memory_; }
    size_t get_max_shared_memory() const override { return 32 * 1024; }
    bool supports_data_type(type::DataType dt) const override {
        // MPS has limited data type support
        return dt == type::DT_FLOAT16 || dt == type::DT_FLOAT32 ||
               dt == type::DT_INT8 || dt == type::DT_INT16 ||
               dt == type::DT_INT32 || dt == type::DT_UINT8;
    }
    int get_compute_capability() const override { return 100; }
    int get_num_compute_units() const override { return 40; }  // GPU cores

    bool set_device(int device_id) override { return device_id == 0; }
    int get_device() const override { return 0; }
    int get_device_count() const override { return 1; }

private:
    bool is_available_;
    size_t unified_memory_;
};

// Backend Registry implementation (mock)
class BackendRegistry {
public:
    static BackendRegistry& get_instance() {
        static BackendRegistry instance;
        return instance;
    }

    bool register_backend(std::unique_ptr<BackendInterface> backend) {
        if (!backend) return false;
        type::BackendType type = backend->get_type();
        if (backends_.find(type) != backends_.end()) return false;
        name_to_type_[backend->get_name()] = type;
        backends_[type] = std::move(backend);
        if (default_backend_ == type::BT_UNKNOWN) {
            default_backend_ = type;
        }
        return true;
    }

    BackendInterface* get_backend(type::BackendType type) {
        auto it = backends_.find(type);
        return it != backends_.end() ? it->second.get() : nullptr;
    }

    BackendInterface* get_backend(std::string const& name) {
        auto it = name_to_type_.find(name);
        if (it != name_to_type_.end()) {
            return get_backend(it->second);
        }
        return nullptr;
    }

    std::vector<type::BackendType> get_registered_backends() const {
        std::vector<type::BackendType> types;
        for (auto const& pair : backends_) {
            types.push_back(pair.first);
        }
        return types;
    }

    std::vector<type::BackendType> get_available_backends() const {
        std::vector<type::BackendType> types;
        for (auto const& pair : backends_) {
            if (pair.second->is_available()) {
                types.push_back(pair.first);
            }
        }
        return types;
    }

    bool is_backend_registered(type::BackendType type) const {
        return backends_.find(type) != backends_.end();
    }

    bool is_backend_available(type::BackendType type) const {
        auto it = backends_.find(type);
        return it != backends_.end() && it->second->is_available();
    }

    type::BackendType get_default_backend() const {
        if (default_backend_ != type::BT_UNKNOWN &&
            is_backend_available(default_backend_)) {
            return default_backend_;
        }
        if (is_backend_available(type::BT_CUDA)) {
            return type::BT_CUDA;
        }
        auto available = get_available_backends();
        return !available.empty() ? available[0] : type::BT_UNKNOWN;
    }

    bool set_default_backend(type::BackendType type) {
        if (!is_backend_available(type)) return false;
        default_backend_ = type;
        return true;
    }

    bool unregister_backend(type::BackendType type) {
        auto it = backends_.find(type);
        if (it == backends_.end()) return false;
        name_to_type_.erase(it->second->get_name());
        backends_.erase(it);
        if (default_backend_ == type) {
            default_backend_ = get_default_backend();
        }
        return true;
    }

    void clear() {
        backends_.clear();
        name_to_type_.clear();
        default_backend_ = type::BT_UNKNOWN;
    }

private:
    BackendRegistry() : default_backend_(type::BT_UNKNOWN) {}

    std::map<type::BackendType, std::unique_ptr<BackendInterface>> backends_;
    std::map<std::string, type::BackendType> name_to_type_;
    type::BackendType default_backend_;
};

}  // namespace backend
}  // namespace yirage

using namespace yirage;
using namespace yirage::backend;

// =============================================================================
// Backend Registry Tests
// =============================================================================

class BackendRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        BackendRegistry::get_instance().clear();
    }

    void TearDown() override {
        BackendRegistry::get_instance().clear();
    }
};

TEST_F(BackendRegistryTest, Singleton) {
    auto& instance1 = BackendRegistry::get_instance();
    auto& instance2 = BackendRegistry::get_instance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(BackendRegistryTest, RegisterCPUBackend) {
    auto& registry = BackendRegistry::get_instance();
    bool result = registry.register_backend(std::make_unique<CPUBackend>());
    EXPECT_TRUE(result);
    EXPECT_TRUE(registry.is_backend_registered(type::BT_CPU));
}

TEST_F(BackendRegistryTest, RegisterCUDABackend) {
    auto& registry = BackendRegistry::get_instance();
    bool result = registry.register_backend(std::make_unique<CUDABackend>());
    EXPECT_TRUE(result);
    EXPECT_TRUE(registry.is_backend_registered(type::BT_CUDA));
}

TEST_F(BackendRegistryTest, PreventDuplicateRegistration) {
    auto& registry = BackendRegistry::get_instance();
    registry.register_backend(std::make_unique<CPUBackend>());
    bool result = registry.register_backend(std::make_unique<CPUBackend>());
    EXPECT_FALSE(result);
}

TEST_F(BackendRegistryTest, GetBackendByType) {
    auto& registry = BackendRegistry::get_instance();
    registry.register_backend(std::make_unique<CPUBackend>());
    
    auto* backend = registry.get_backend(type::BT_CPU);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->get_type(), type::BT_CPU);
}

TEST_F(BackendRegistryTest, GetBackendByName) {
    auto& registry = BackendRegistry::get_instance();
    registry.register_backend(std::make_unique<CPUBackend>());
    
    auto* backend = registry.get_backend("cpu");
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->get_name(), "cpu");
}

TEST_F(BackendRegistryTest, GetNonExistentBackend) {
    auto& registry = BackendRegistry::get_instance();
    EXPECT_EQ(registry.get_backend(type::BT_CUDA), nullptr);
    EXPECT_EQ(registry.get_backend("nonexistent"), nullptr);
}

TEST_F(BackendRegistryTest, GetRegisteredBackends) {
    auto& registry = BackendRegistry::get_instance();
    registry.register_backend(std::make_unique<CPUBackend>());
    registry.register_backend(std::make_unique<CUDABackend>());
    registry.register_backend(std::make_unique<ROCmBackend>());
    
    auto backends = registry.get_registered_backends();
    EXPECT_EQ(backends.size(), 3u);
}

TEST_F(BackendRegistryTest, GetAvailableBackends) {
    auto& registry = BackendRegistry::get_instance();
    registry.register_backend(std::make_unique<CPUBackend>());
    registry.register_backend(std::make_unique<CUDABackend>());
    
    auto available = registry.get_available_backends();
    EXPECT_GE(available.size(), 1u);  // At least CPU
}

TEST_F(BackendRegistryTest, DefaultBackendSelection) {
    auto& registry = BackendRegistry::get_instance();
    registry.register_backend(std::make_unique<CPUBackend>());
    
    auto default_type = registry.get_default_backend();
    EXPECT_EQ(default_type, type::BT_CPU);
}

TEST_F(BackendRegistryTest, SetDefaultBackend) {
    auto& registry = BackendRegistry::get_instance();
    registry.register_backend(std::make_unique<CPUBackend>());
    registry.register_backend(std::make_unique<CUDABackend>());
    
    bool result = registry.set_default_backend(type::BT_CUDA);
    EXPECT_TRUE(result);
    EXPECT_EQ(registry.get_default_backend(), type::BT_CUDA);
}

TEST_F(BackendRegistryTest, UnregisterBackend) {
    auto& registry = BackendRegistry::get_instance();
    registry.register_backend(std::make_unique<CPUBackend>());
    
    bool result = registry.unregister_backend(type::BT_CPU);
    EXPECT_TRUE(result);
    EXPECT_FALSE(registry.is_backend_registered(type::BT_CPU));
}

TEST_F(BackendRegistryTest, ClearRegistry) {
    auto& registry = BackendRegistry::get_instance();
    registry.register_backend(std::make_unique<CPUBackend>());
    registry.register_backend(std::make_unique<CUDABackend>());
    
    registry.clear();
    EXPECT_TRUE(registry.get_registered_backends().empty());
}

// =============================================================================
// CPU Backend Tests
// =============================================================================

class CPUBackendTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(CPUBackendTest, BasicInfo) {
    EXPECT_EQ(backend.get_type(), type::BT_CPU);
    EXPECT_EQ(backend.get_name(), "cpu");
    EXPECT_EQ(backend.get_display_name(), "CPU");
    EXPECT_TRUE(backend.is_available());
}

TEST_F(CPUBackendTest, BackendInfo) {
    auto info = backend.get_info();
    EXPECT_EQ(info.type, type::BT_CPU);
    EXPECT_EQ(info.name, "cpu");
    EXPECT_FALSE(info.requires_gpu);
    EXPECT_TRUE(info.required_libs.empty());
}

TEST_F(CPUBackendTest, CompileFlags) {
    auto flags = backend.get_compile_flags();
    EXPECT_NE(flags.find("-std=c++17"), std::string::npos);
    EXPECT_NE(flags.find("-O2"), std::string::npos);
}

TEST_F(CPUBackendTest, LinkLibraries) {
    auto libs = backend.get_link_libraries();
    EXPECT_FALSE(libs.empty());
}

TEST_F(CPUBackendTest, MemoryAllocation) {
    void* ptr = backend.allocate_memory(1024);
    ASSERT_NE(ptr, nullptr);
    backend.free_memory(ptr);
}

TEST_F(CPUBackendTest, MemoryCopy) {
    const size_t size = 256;
    std::vector<float> src(size / sizeof(float), 1.5f);
    std::vector<float> dst(size / sizeof(float), 0.0f);

    EXPECT_TRUE(backend.copy_to_device(dst.data(), src.data(), size));
    EXPECT_EQ(dst[0], 1.5f);
}

TEST_F(CPUBackendTest, DataTypeSupport) {
    EXPECT_TRUE(backend.supports_data_type(type::DT_FLOAT32));
    EXPECT_TRUE(backend.supports_data_type(type::DT_FLOAT16));
    EXPECT_TRUE(backend.supports_data_type(type::DT_INT32));
    EXPECT_TRUE(backend.supports_data_type(type::DT_INT8));
}

TEST_F(CPUBackendTest, DeviceManagement) {
    EXPECT_EQ(backend.get_device(), 0);
    EXPECT_EQ(backend.get_device_count(), 1);
    EXPECT_TRUE(backend.set_device(0));
    EXPECT_FALSE(backend.set_device(1));
}

TEST_F(CPUBackendTest, ComputeUnits) {
    EXPECT_GT(backend.get_num_compute_units(), 0);
}

// =============================================================================
// CUDA Backend Tests
// =============================================================================

class CUDABackendTest : public ::testing::Test {
protected:
    CUDABackend backend;
};

TEST_F(CUDABackendTest, BasicInfo) {
    EXPECT_EQ(backend.get_type(), type::BT_CUDA);
    EXPECT_EQ(backend.get_name(), "cuda");
    EXPECT_EQ(backend.get_display_name(), "CUDA");
}

TEST_F(CUDABackendTest, BackendInfo) {
    auto info = backend.get_info();
    EXPECT_EQ(info.type, type::BT_CUDA);
    EXPECT_TRUE(info.requires_gpu);
    EXPECT_FALSE(info.required_libs.empty());
}

TEST_F(CUDABackendTest, CompileFlags) {
    auto flags = backend.get_compile_flags();
    EXPECT_NE(flags.find("-std=c++17"), std::string::npos);
}

TEST_F(CUDABackendTest, IncludeDirs) {
    auto dirs = backend.get_include_dirs();
    EXPECT_FALSE(dirs.empty());
}

TEST_F(CUDABackendTest, LibraryDirs) {
    auto dirs = backend.get_library_dirs();
    EXPECT_FALSE(dirs.empty());
}

TEST_F(CUDABackendTest, LinkLibraries) {
    auto libs = backend.get_link_libraries();
    EXPECT_FALSE(libs.empty());
    // Should have common CUDA libs
    std::set<std::string> lib_set(libs.begin(), libs.end());
    EXPECT_TRUE(lib_set.count("cudart") > 0);
    EXPECT_TRUE(lib_set.count("cublas") > 0);
}

TEST_F(CUDABackendTest, ComputeCapability) {
    int cc = backend.get_compute_capability();
    EXPECT_GE(cc, 50);  // Minimum supported
}

TEST_F(CUDABackendTest, SharedMemory) {
    size_t smem = backend.get_max_shared_memory();
    EXPECT_GT(smem, 0u);
}

TEST_F(CUDABackendTest, DataTypeSupport) {
    EXPECT_TRUE(backend.supports_data_type(type::DT_FLOAT32));
    EXPECT_TRUE(backend.supports_data_type(type::DT_FLOAT16));
    // FP8 support depends on compute capability
    if (backend.get_compute_capability() >= 80) {
        EXPECT_TRUE(backend.supports_data_type(type::DT_FLOAT8));
    }
}

TEST_F(CUDABackendTest, DeviceManagement) {
    EXPECT_GE(backend.get_device_count(), 1);
    EXPECT_TRUE(backend.set_device(0));
    EXPECT_EQ(backend.get_device(), 0);
}

// =============================================================================
// ROCm Backend Tests
// =============================================================================

class ROCmBackendTest : public ::testing::Test {
protected:
    ROCmBackend backend;
};

TEST_F(ROCmBackendTest, BasicInfo) {
    EXPECT_EQ(backend.get_type(), type::BT_ROCM);
    EXPECT_EQ(backend.get_name(), "rocm");
    EXPECT_EQ(backend.get_display_name(), "ROCm");
}

TEST_F(ROCmBackendTest, BackendInfo) {
    auto info = backend.get_info();
    EXPECT_EQ(info.type, type::BT_ROCM);
    EXPECT_TRUE(info.requires_gpu);
}

TEST_F(ROCmBackendTest, CompileFlags) {
    auto flags = backend.get_compile_flags();
    EXPECT_NE(flags.find("__HIP_PLATFORM_AMD__"), std::string::npos);
}

TEST_F(ROCmBackendTest, LinkLibraries) {
    auto libs = backend.get_link_libraries();
    std::set<std::string> lib_set(libs.begin(), libs.end());
    EXPECT_TRUE(lib_set.count("hip") > 0 || lib_set.count("amdhip64") > 0);
}

// =============================================================================
// MPS Backend Tests
// =============================================================================

class MPSBackendTest : public ::testing::Test {
protected:
    MPSBackend backend;
};

TEST_F(MPSBackendTest, BasicInfo) {
    EXPECT_EQ(backend.get_type(), type::BT_MPS);
    EXPECT_EQ(backend.get_name(), "mps");
    EXPECT_EQ(backend.get_display_name(), "Metal");
}

TEST_F(MPSBackendTest, BackendInfo) {
    auto info = backend.get_info();
    EXPECT_EQ(info.type, type::BT_MPS);
    EXPECT_TRUE(info.requires_gpu);
}

TEST_F(MPSBackendTest, CompileFlags) {
    auto flags = backend.get_compile_flags();
    EXPECT_NE(flags.find("Metal"), std::string::npos);
}

TEST_F(MPSBackendTest, DataTypeSupport) {
    EXPECT_TRUE(backend.supports_data_type(type::DT_FLOAT32));
    EXPECT_TRUE(backend.supports_data_type(type::DT_FLOAT16));
    // MPS doesn't support all types
    EXPECT_FALSE(backend.supports_data_type(type::DT_DOUBLE));
}

TEST_F(MPSBackendTest, UnifiedMemory) {
    size_t max_mem = backend.get_max_memory();
    EXPECT_GT(max_mem, 0u);
}

// =============================================================================
// Backend Interface Tests (Parameterized)
// =============================================================================

struct BackendTestParam {
    type::BackendType type;
    std::string name;
    std::string display_name;
    bool requires_gpu;
};

class BackendInterfaceParamTest 
    : public ::testing::TestWithParam<BackendTestParam> {};

TEST_P(BackendInterfaceParamTest, BasicInterface) {
    auto param = GetParam();
    std::unique_ptr<BackendInterface> backend;

    switch (param.type) {
        case type::BT_CPU:
            backend = std::make_unique<CPUBackend>();
            break;
        case type::BT_CUDA:
            backend = std::make_unique<CUDABackend>();
            break;
        case type::BT_ROCM:
            backend = std::make_unique<ROCmBackend>();
            break;
        case type::BT_MPS:
            backend = std::make_unique<MPSBackend>();
            break;
        default:
            GTEST_SKIP() << "Unknown backend type";
    }

    EXPECT_EQ(backend->get_type(), param.type);
    EXPECT_EQ(backend->get_name(), param.name);
    EXPECT_EQ(backend->get_display_name(), param.display_name);

    auto info = backend->get_info();
    EXPECT_EQ(info.requires_gpu, param.requires_gpu);
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    BackendInterfaceParamTest,
    ::testing::Values(
        BackendTestParam{type::BT_CPU, "cpu", "CPU", false},
        BackendTestParam{type::BT_CUDA, "cuda", "CUDA", true},
        BackendTestParam{type::BT_ROCM, "rocm", "ROCm", true},
        BackendTestParam{type::BT_MPS, "mps", "Metal", true}
    )
);

// =============================================================================
// Data Type Support Tests (Parameterized)
// =============================================================================

class DataTypeSupportTest 
    : public ::testing::TestWithParam<std::pair<type::BackendType, type::DataType>> {
protected:
    std::unique_ptr<BackendInterface> get_backend(type::BackendType type) {
        switch (type) {
            case type::BT_CPU: return std::make_unique<CPUBackend>();
            case type::BT_CUDA: return std::make_unique<CUDABackend>();
            case type::BT_ROCM: return std::make_unique<ROCmBackend>();
            case type::BT_MPS: return std::make_unique<MPSBackend>();
            default: return nullptr;
        }
    }
};

TEST_P(DataTypeSupportTest, CommonDataTypes) {
    auto [backend_type, data_type] = GetParam();
    auto backend = get_backend(backend_type);
    if (!backend) {
        GTEST_SKIP() << "Backend not available";
    }

    // All backends should support float32
    if (data_type == type::DT_FLOAT32) {
        EXPECT_TRUE(backend->supports_data_type(data_type));
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsFloat32,
    DataTypeSupportTest,
    ::testing::Values(
        std::make_pair(type::BT_CPU, type::DT_FLOAT32),
        std::make_pair(type::BT_CUDA, type::DT_FLOAT32),
        std::make_pair(type::BT_ROCM, type::DT_FLOAT32),
        std::make_pair(type::BT_MPS, type::DT_FLOAT32)
    )
);

// =============================================================================
// Memory Operations Tests
// =============================================================================

class MemoryOperationsTest : public ::testing::Test {
protected:
    CPUBackend backend;  // Use CPU for reliable testing
};

TEST_F(MemoryOperationsTest, AllocateAndFree) {
    void* ptr = backend.allocate_memory(4096);
    ASSERT_NE(ptr, nullptr);
    backend.free_memory(ptr);
}

TEST_F(MemoryOperationsTest, ZeroSizeAllocation) {
    // Implementation-defined behavior
    void* ptr = backend.allocate_memory(0);
    // Either nullptr or valid pointer is acceptable
    backend.free_memory(ptr);
}

TEST_F(MemoryOperationsTest, LargeAllocation) {
    // Allocate 1MB
    void* ptr = backend.allocate_memory(1024 * 1024);
    ASSERT_NE(ptr, nullptr);
    backend.free_memory(ptr);
}

TEST_F(MemoryOperationsTest, CopyRoundTrip) {
    const size_t count = 100;
    std::vector<int> src(count);
    for (size_t i = 0; i < count; ++i) {
        src[i] = static_cast<int>(i * 2);
    }

    std::vector<int> dst(count, 0);
    
    // Copy to "device" (same for CPU)
    EXPECT_TRUE(backend.copy_to_device(dst.data(), src.data(), count * sizeof(int)));
    
    // Verify
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST_F(MemoryOperationsTest, DeviceToDeviceCopy) {
    const size_t size = 256;
    void* src = backend.allocate_memory(size);
    void* dst = backend.allocate_memory(size);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    std::memset(src, 0xAB, size);
    
    EXPECT_TRUE(backend.copy_device_to_device(dst, src, size));
    EXPECT_EQ(std::memcmp(src, dst, size), 0);
    
    backend.free_memory(src);
    backend.free_memory(dst);
}

// =============================================================================
// Compile Context Tests
// =============================================================================

class CompileContextTest : public ::testing::Test {};

TEST_F(CompileContextTest, DefaultValues) {
    CompileContext ctx;
    EXPECT_FALSE(ctx.debug_mode);
    EXPECT_EQ(ctx.optimization_level, 2);
    EXPECT_TRUE(ctx.source_code.empty());
}

TEST_F(CompileContextTest, SetValues) {
    CompileContext ctx;
    ctx.source_code = "int main() { return 0; }";
    ctx.output_path = "/tmp/output";
    ctx.debug_mode = true;
    ctx.optimization_level = 3;
    ctx.include_dirs.push_back("/usr/include");
    ctx.compile_flags.push_back("-Wall");

    EXPECT_FALSE(ctx.source_code.empty());
    EXPECT_EQ(ctx.output_path, "/tmp/output");
    EXPECT_TRUE(ctx.debug_mode);
    EXPECT_EQ(ctx.optimization_level, 3);
    EXPECT_EQ(ctx.include_dirs.size(), 1u);
    EXPECT_EQ(ctx.compile_flags.size(), 1u);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
