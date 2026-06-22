// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_pk_backend_interface_gtest.cc
 * @brief Persistent Kernel Backend Interface Unit Tests
 *
 * Tests for backend interface components:
 *   - PKBackendType enum
 *   - PKMode enum
 *   - PKDataType enum
 *   - PKCapabilities structure
 *   - PKRuntimeConfig structure
 *   - PKBackendInterface abstract class
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// Enumerations
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

enum class PKDataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    INT4 = 4,
    FP8_E4M3 = 5,
    FP8_E5M2 = 6,
    NUM_TYPES
};

// =============================================================================
// Utility Functions
// =============================================================================

inline const char* pk_backend_type_to_name(PKBackendType type) {
    switch (type) {
        case PKBackendType::CUDA: return "cuda";
        case PKBackendType::CPU: return "cpu";
        case PKBackendType::MPS: return "mps";
        case PKBackendType::ASCEND: return "ascend";
        case PKBackendType::MACA: return "maca";
        case PKBackendType::TRITON: return "triton";
        case PKBackendType::NKI: return "nki";
        case PKBackendType::ROCM: return "rocm";
        default: return "unknown";
    }
}

inline const char* pk_mode_to_name(PKMode mode) {
    switch (mode) {
        case PKMode::OFFLINE: return "offline";
        case PKMode::ONLINE: return "online";
        case PKMode::ONEPASS: return "onepass";
        case PKMode::EAGER: return "eager";
        case PKMode::GRAPH: return "graph";
        case PKMode::STREAMING: return "streaming";
        default: return "unknown";
    }
}

inline const char* pk_datatype_to_name(PKDataType dtype) {
    switch (dtype) {
        case PKDataType::FP32: return "fp32";
        case PKDataType::FP16: return "fp16";
        case PKDataType::BF16: return "bf16";
        case PKDataType::INT8: return "int8";
        case PKDataType::INT4: return "int4";
        case PKDataType::FP8_E4M3: return "fp8_e4m3";
        case PKDataType::FP8_E5M2: return "fp8_e5m2";
        default: return "unknown";
    }
}

inline size_t pk_datatype_size(PKDataType dtype) {
    switch (dtype) {
        case PKDataType::FP32: return 4;
        case PKDataType::FP16:
        case PKDataType::BF16: return 2;
        case PKDataType::INT8:
        case PKDataType::FP8_E4M3:
        case PKDataType::FP8_E5M2: return 1;
        case PKDataType::INT4: return 1;  // Packed, 2 per byte
        default: return 0;
    }
}

// =============================================================================
// PKCapabilities
// =============================================================================

struct PKCapabilities {
    bool supports_tma = false;
    bool supports_tensor_cores = false;
    bool supports_async_copy = false;
    bool supports_nvshmem = false;
    bool supports_fp8 = false;

    size_t max_shared_memory = 48 * 1024;
    size_t max_global_memory = 0;
    size_t max_threads_per_block = 1024;
    size_t max_blocks_per_sm = 32;

    int compute_major = 0;
    int compute_minor = 0;

    std::vector<PKMode> supported_modes;
    std::vector<PKDataType> supported_dtypes;

    bool supports_mode(PKMode mode) const {
        return std::find(supported_modes.begin(), supported_modes.end(), mode) !=
               supported_modes.end();
    }

    bool supports_dtype(PKDataType dtype) const {
        return std::find(supported_dtypes.begin(), supported_dtypes.end(), dtype) !=
               supported_dtypes.end();
    }

    int get_compute_capability() const {
        return compute_major * 10 + compute_minor;
    }
};

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

    size_t total_pages_memory() const {
        return max_num_pages * page_size;
    }

    size_t max_batch_tokens() const {
        return max_num_batched_requests * max_seq_length;
    }
};

// =============================================================================
// PKBackendInterface (Abstract)
// =============================================================================

class PKBackendInterface {
public:
    virtual ~PKBackendInterface() = default;

    virtual PKBackendType get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_display_name() const = 0;
    virtual bool is_available() const = 0;
    virtual PKCapabilities get_capabilities() const = 0;

    virtual bool supports_mode(PKMode mode) const = 0;
    virtual PKMode get_default_mode() const = 0;
    virtual std::vector<PKMode> get_supported_modes() const = 0;

    virtual bool initialize(const PKRuntimeConfig& config) = 0;
    virtual void finalize() = 0;
    virtual void reset() = 0;

    virtual void* create_stream() = 0;
    virtual void destroy_stream(void* stream) = 0;
    virtual void synchronize_stream(void* stream) = 0;
    virtual void synchronize() = 0;

    virtual bool set_device(int device_id) = 0;
    virtual int get_device() const = 0;
    virtual int get_device_count() const = 0;
};

// =============================================================================
// Mock Backend Implementation
// =============================================================================

class MockPKBackend : public PKBackendInterface {
public:
    MockPKBackend(PKBackendType type, bool available = true)
        : type_(type), available_(available), device_id_(0), initialized_(false) {
        setup_capabilities();
    }

    PKBackendType get_type() const override { return type_; }

    std::string get_name() const override {
        return pk_backend_type_to_name(type_);
    }

    std::string get_display_name() const override {
        switch (type_) {
            case PKBackendType::CUDA: return "NVIDIA CUDA";
            case PKBackendType::CPU: return "CPU";
            case PKBackendType::MPS: return "Apple Metal";
            case PKBackendType::ASCEND: return "Huawei Ascend";
            case PKBackendType::MACA: return "MetaX MACA";
            case PKBackendType::ROCM: return "AMD ROCm";
            default: return "Unknown";
        }
    }

    bool is_available() const override { return available_; }

    PKCapabilities get_capabilities() const override { return caps_; }

    bool supports_mode(PKMode mode) const override {
        return caps_.supports_mode(mode);
    }

    PKMode get_default_mode() const override {
        switch (type_) {
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

    std::vector<PKMode> get_supported_modes() const override {
        return caps_.supported_modes;
    }

    bool initialize(const PKRuntimeConfig& config) override {
        if (!available_) return false;
        config_ = config;
        initialized_ = true;
        return true;
    }

    void finalize() override {
        initialized_ = false;
    }

    void reset() override {
        // Reset state
    }

    void* create_stream() override {
        return reinterpret_cast<void*>(++stream_counter_);
    }

    void destroy_stream(void* stream) override {
        // No-op for mock
    }

    void synchronize_stream(void* stream) override {
        // No-op for mock
    }

    void synchronize() override {
        // No-op for mock
    }

    bool set_device(int device_id) override {
        if (device_id < 0 || device_id >= device_count_) return false;
        device_id_ = device_id;
        return true;
    }

    int get_device() const override { return device_id_; }

    int get_device_count() const override { return device_count_; }

    bool is_initialized() const { return initialized_; }

private:
    void setup_capabilities() {
        switch (type_) {
            case PKBackendType::CUDA:
                caps_.supports_tma = true;
                caps_.supports_tensor_cores = true;
                caps_.supports_async_copy = true;
                caps_.supports_nvshmem = true;
                caps_.supports_fp8 = true;
                caps_.max_shared_memory = 228 * 1024;  // Hopper
                caps_.compute_major = 9;
                caps_.compute_minor = 0;
                caps_.supported_modes = {PKMode::OFFLINE, PKMode::ONLINE, PKMode::ONEPASS, PKMode::GRAPH};
                caps_.supported_dtypes = {PKDataType::FP32, PKDataType::FP16, PKDataType::BF16,
                                          PKDataType::INT8, PKDataType::FP8_E4M3, PKDataType::FP8_E5M2};
                device_count_ = 4;
                break;

            case PKBackendType::ROCM:
                caps_.supports_tensor_cores = true;  // Matrix Cores
                caps_.supports_async_copy = true;
                caps_.max_shared_memory = 64 * 1024;
                caps_.compute_major = 9;
                caps_.compute_minor = 4;
                caps_.supported_modes = {PKMode::OFFLINE, PKMode::ONLINE, PKMode::ONEPASS};
                caps_.supported_dtypes = {PKDataType::FP32, PKDataType::FP16, PKDataType::BF16, PKDataType::INT8};
                device_count_ = 2;
                break;

            case PKBackendType::CPU:
                caps_.max_shared_memory = 32 * 1024;  // L1 cache
                caps_.max_threads_per_block = 256;
                caps_.supported_modes = {PKMode::EAGER, PKMode::GRAPH, PKMode::OFFLINE};
                caps_.supported_dtypes = {PKDataType::FP32, PKDataType::FP16, PKDataType::INT8};
                device_count_ = 1;
                break;

            case PKBackendType::MPS:
                caps_.supports_tensor_cores = true;  // Apple AMX
                caps_.max_shared_memory = 64 * 1024;
                caps_.supported_modes = {PKMode::EAGER, PKMode::GRAPH};
                caps_.supported_dtypes = {PKDataType::FP32, PKDataType::FP16};
                device_count_ = 1;
                break;

            case PKBackendType::ASCEND:
                caps_.supports_tensor_cores = true;  // AI Cores
                caps_.supports_async_copy = true;
                caps_.max_shared_memory = 128 * 1024;
                caps_.supported_modes = {PKMode::OFFLINE, PKMode::ONLINE, PKMode::GRAPH};
                caps_.supported_dtypes = {PKDataType::FP32, PKDataType::FP16, PKDataType::INT8};
                device_count_ = 8;
                break;

            default:
                caps_.supported_modes = {PKMode::OFFLINE};
                caps_.supported_dtypes = {PKDataType::FP32};
                device_count_ = 1;
                break;
        }
    }

    PKBackendType type_;
    bool available_;
    int device_id_;
    int device_count_ = 1;
    bool initialized_;
    PKCapabilities caps_;
    PKRuntimeConfig config_;
    size_t stream_counter_ = 0;
};

}  // namespace persistent_kernel
}  // namespace yirage

using namespace yirage::persistent_kernel;

// =============================================================================
// PKBackendType Tests
// =============================================================================

class PKBackendTypeTest : public ::testing::Test {};

TEST_F(PKBackendTypeTest, EnumValues) {
    EXPECT_EQ(static_cast<int>(PKBackendType::CUDA), 0);
    EXPECT_EQ(static_cast<int>(PKBackendType::CPU), 1);
    EXPECT_EQ(static_cast<int>(PKBackendType::MPS), 2);
    EXPECT_EQ(static_cast<int>(PKBackendType::ASCEND), 3);
    EXPECT_EQ(static_cast<int>(PKBackendType::ROCM), 7);
}

TEST_F(PKBackendTypeTest, BackendTypeToName) {
    EXPECT_STREQ(pk_backend_type_to_name(PKBackendType::CUDA), "cuda");
    EXPECT_STREQ(pk_backend_type_to_name(PKBackendType::CPU), "cpu");
    EXPECT_STREQ(pk_backend_type_to_name(PKBackendType::MPS), "mps");
    EXPECT_STREQ(pk_backend_type_to_name(PKBackendType::ASCEND), "ascend");
    EXPECT_STREQ(pk_backend_type_to_name(PKBackendType::ROCM), "rocm");
    EXPECT_STREQ(pk_backend_type_to_name(PKBackendType::TRITON), "triton");
}

TEST_F(PKBackendTypeTest, NumBackends) {
    EXPECT_EQ(static_cast<int>(PKBackendType::NUM_BACKENDS), 8);
}

// =============================================================================
// PKMode Tests
// =============================================================================

class PKModeTest : public ::testing::Test {};

TEST_F(PKModeTest, EnumValues) {
    EXPECT_EQ(static_cast<int>(PKMode::OFFLINE), 0);
    EXPECT_EQ(static_cast<int>(PKMode::ONLINE), 1);
    EXPECT_EQ(static_cast<int>(PKMode::ONEPASS), 2);
    EXPECT_EQ(static_cast<int>(PKMode::EAGER), 3);
    EXPECT_EQ(static_cast<int>(PKMode::GRAPH), 4);
    EXPECT_EQ(static_cast<int>(PKMode::STREAMING), 5);
}

TEST_F(PKModeTest, ModeToName) {
    EXPECT_STREQ(pk_mode_to_name(PKMode::OFFLINE), "offline");
    EXPECT_STREQ(pk_mode_to_name(PKMode::ONLINE), "online");
    EXPECT_STREQ(pk_mode_to_name(PKMode::ONEPASS), "onepass");
    EXPECT_STREQ(pk_mode_to_name(PKMode::EAGER), "eager");
    EXPECT_STREQ(pk_mode_to_name(PKMode::GRAPH), "graph");
    EXPECT_STREQ(pk_mode_to_name(PKMode::STREAMING), "streaming");
}

TEST_F(PKModeTest, NumModes) {
    EXPECT_EQ(static_cast<int>(PKMode::NUM_MODES), 6);
}

// =============================================================================
// PKDataType Tests
// =============================================================================

class PKDataTypeTest : public ::testing::Test {};

TEST_F(PKDataTypeTest, EnumValues) {
    EXPECT_EQ(static_cast<int>(PKDataType::FP32), 0);
    EXPECT_EQ(static_cast<int>(PKDataType::FP16), 1);
    EXPECT_EQ(static_cast<int>(PKDataType::BF16), 2);
    EXPECT_EQ(static_cast<int>(PKDataType::INT8), 3);
    EXPECT_EQ(static_cast<int>(PKDataType::FP8_E4M3), 5);
}

TEST_F(PKDataTypeTest, DataTypeToName) {
    EXPECT_STREQ(pk_datatype_to_name(PKDataType::FP32), "fp32");
    EXPECT_STREQ(pk_datatype_to_name(PKDataType::FP16), "fp16");
    EXPECT_STREQ(pk_datatype_to_name(PKDataType::BF16), "bf16");
    EXPECT_STREQ(pk_datatype_to_name(PKDataType::INT8), "int8");
}

TEST_F(PKDataTypeTest, DataTypeSize) {
    EXPECT_EQ(pk_datatype_size(PKDataType::FP32), 4u);
    EXPECT_EQ(pk_datatype_size(PKDataType::FP16), 2u);
    EXPECT_EQ(pk_datatype_size(PKDataType::BF16), 2u);
    EXPECT_EQ(pk_datatype_size(PKDataType::INT8), 1u);
    EXPECT_EQ(pk_datatype_size(PKDataType::FP8_E4M3), 1u);
}

// =============================================================================
// PKCapabilities Tests
// =============================================================================

class PKCapabilitiesTest : public ::testing::Test {};

TEST_F(PKCapabilitiesTest, DefaultValues) {
    PKCapabilities caps;
    EXPECT_FALSE(caps.supports_tma);
    EXPECT_FALSE(caps.supports_tensor_cores);
    EXPECT_EQ(caps.max_shared_memory, 48u * 1024u);
    EXPECT_EQ(caps.compute_major, 0);
}

TEST_F(PKCapabilitiesTest, SupportsMode) {
    PKCapabilities caps;
    caps.supported_modes = {PKMode::ONLINE, PKMode::OFFLINE};

    EXPECT_TRUE(caps.supports_mode(PKMode::ONLINE));
    EXPECT_TRUE(caps.supports_mode(PKMode::OFFLINE));
    EXPECT_FALSE(caps.supports_mode(PKMode::EAGER));
}

TEST_F(PKCapabilitiesTest, SupportsDtype) {
    PKCapabilities caps;
    caps.supported_dtypes = {PKDataType::FP32, PKDataType::FP16};

    EXPECT_TRUE(caps.supports_dtype(PKDataType::FP32));
    EXPECT_TRUE(caps.supports_dtype(PKDataType::FP16));
    EXPECT_FALSE(caps.supports_dtype(PKDataType::INT8));
}

TEST_F(PKCapabilitiesTest, GetComputeCapability) {
    PKCapabilities caps;
    caps.compute_major = 9;
    caps.compute_minor = 0;

    EXPECT_EQ(caps.get_compute_capability(), 90);
}

// =============================================================================
// PKRuntimeConfig Tests
// =============================================================================

class PKRuntimeConfigTest : public ::testing::Test {};

TEST_F(PKRuntimeConfigTest, DefaultValues) {
    PKRuntimeConfig config;
    EXPECT_EQ(config.mode, PKMode::ONLINE);
    EXPECT_EQ(config.num_workers, 4);
    EXPECT_EQ(config.threads_per_worker, 256);
    EXPECT_EQ(config.max_seq_length, 2048u);
}

TEST_F(PKRuntimeConfigTest, TotalPagesMemory) {
    PKRuntimeConfig config;
    config.max_num_pages = 1024;
    config.page_size = 64;

    EXPECT_EQ(config.total_pages_memory(), 1024u * 64u);
}

TEST_F(PKRuntimeConfigTest, MaxBatchTokens) {
    PKRuntimeConfig config;
    config.max_num_batched_requests = 16;
    config.max_seq_length = 2048;

    EXPECT_EQ(config.max_batch_tokens(), 16u * 2048u);
}

TEST_F(PKRuntimeConfigTest, MultiGPUConfig) {
    PKRuntimeConfig config;
    config.num_gpus = 8;
    config.my_gpu_id = 3;

    EXPECT_EQ(config.num_gpus, 8);
    EXPECT_EQ(config.my_gpu_id, 3);
}

// =============================================================================
// PKBackendInterface Tests (via Mock)
// =============================================================================

class PKBackendInterfaceTest : public ::testing::Test {};

TEST_F(PKBackendInterfaceTest, CUDABackendProperties) {
    MockPKBackend backend(PKBackendType::CUDA);

    EXPECT_EQ(backend.get_type(), PKBackendType::CUDA);
    EXPECT_EQ(backend.get_name(), "cuda");
    EXPECT_EQ(backend.get_display_name(), "NVIDIA CUDA");
    EXPECT_TRUE(backend.is_available());
}

TEST_F(PKBackendInterfaceTest, CUDACapabilities) {
    MockPKBackend backend(PKBackendType::CUDA);
    auto caps = backend.get_capabilities();

    EXPECT_TRUE(caps.supports_tma);
    EXPECT_TRUE(caps.supports_tensor_cores);
    EXPECT_TRUE(caps.supports_async_copy);
    EXPECT_TRUE(caps.supports_fp8);
    EXPECT_EQ(caps.compute_major, 9);
}

TEST_F(PKBackendInterfaceTest, CPUBackendProperties) {
    MockPKBackend backend(PKBackendType::CPU);

    EXPECT_EQ(backend.get_type(), PKBackendType::CPU);
    EXPECT_EQ(backend.get_name(), "cpu");
    EXPECT_EQ(backend.get_default_mode(), PKMode::EAGER);
}

TEST_F(PKBackendInterfaceTest, InitializeAndFinalize) {
    MockPKBackend backend(PKBackendType::CUDA);
    PKRuntimeConfig config;

    EXPECT_FALSE(backend.is_initialized());
    EXPECT_TRUE(backend.initialize(config));
    EXPECT_TRUE(backend.is_initialized());

    backend.finalize();
    EXPECT_FALSE(backend.is_initialized());
}

TEST_F(PKBackendInterfaceTest, InitializeUnavailable) {
    MockPKBackend backend(PKBackendType::CUDA, false);  // Not available
    PKRuntimeConfig config;

    EXPECT_FALSE(backend.is_available());
    EXPECT_FALSE(backend.initialize(config));
}

TEST_F(PKBackendInterfaceTest, StreamManagement) {
    MockPKBackend backend(PKBackendType::CUDA);

    void* stream1 = backend.create_stream();
    void* stream2 = backend.create_stream();

    EXPECT_NE(stream1, nullptr);
    EXPECT_NE(stream2, nullptr);
    EXPECT_NE(stream1, stream2);

    backend.synchronize_stream(stream1);
    backend.destroy_stream(stream1);
    backend.destroy_stream(stream2);
}

TEST_F(PKBackendInterfaceTest, DeviceManagement) {
    MockPKBackend backend(PKBackendType::CUDA);

    EXPECT_EQ(backend.get_device_count(), 4);
    EXPECT_EQ(backend.get_device(), 0);

    EXPECT_TRUE(backend.set_device(2));
    EXPECT_EQ(backend.get_device(), 2);

    EXPECT_FALSE(backend.set_device(10));  // Invalid device
    EXPECT_EQ(backend.get_device(), 2);    // Unchanged
}

TEST_F(PKBackendInterfaceTest, SupportedModes) {
    MockPKBackend backend(PKBackendType::CUDA);

    EXPECT_TRUE(backend.supports_mode(PKMode::ONLINE));
    EXPECT_TRUE(backend.supports_mode(PKMode::OFFLINE));
    EXPECT_FALSE(backend.supports_mode(PKMode::EAGER));

    auto modes = backend.get_supported_modes();
    EXPECT_FALSE(modes.empty());
}

// =============================================================================
// Parameterized Backend Tests
// =============================================================================

struct BackendTestParam {
    PKBackendType type;
    PKMode expected_default_mode;
    bool has_tensor_cores;
};

class BackendParameterizedTest : public ::testing::TestWithParam<BackendTestParam> {};

TEST_P(BackendParameterizedTest, BackendProperties) {
    auto param = GetParam();
    MockPKBackend backend(param.type);

    EXPECT_EQ(backend.get_default_mode(), param.expected_default_mode);
    EXPECT_EQ(backend.get_capabilities().supports_tensor_cores, param.has_tensor_cores);
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    BackendParameterizedTest,
    ::testing::Values(
        BackendTestParam{PKBackendType::CUDA, PKMode::ONLINE, true},
        BackendTestParam{PKBackendType::ROCM, PKMode::ONLINE, true},
        BackendTestParam{PKBackendType::CPU, PKMode::EAGER, false},
        BackendTestParam{PKBackendType::MPS, PKMode::EAGER, true},
        BackendTestParam{PKBackendType::ASCEND, PKMode::ONLINE, true}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
