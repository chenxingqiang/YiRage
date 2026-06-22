// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_pk_task_gtest.cc
 * @brief Persistent Kernel Task Module Unit Tests
 *
 * Tests for task management components:
 *   - PKTaskType enum
 *   - PKTensorDesc structure
 *   - PKTaskDesc structure
 *   - PKTaskExecutor interface
 *   - PKMemoryAllocator interface
 *   - PKAtomicOps interface
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <unordered_map>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// Data Type Enum
// =============================================================================

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
// PKTaskType Enum
// =============================================================================

enum class PKTaskType {
    TERMINATE = 0,
    BEGIN_TASK_GRAPH = 10,
    
    // Compute tasks
    EMBEDDING = 101,
    RMS_NORM = 102,
    RMS_NORM_LINEAR = 103,
    LINEAR = 104,
    LINEAR_RESIDUAL = 105,
    ATTENTION = 106,
    PAGED_ATTENTION = 107,
    SILU_MUL = 108,
    SILU_MUL_LINEAR = 109,
    ARGMAX = 110,
    ROTARY_EMBEDDING = 111,
    
    // MOE tasks
    MOE_GATE = 120,
    MOE_LINEAR = 121,
    
    // Communication tasks
    ALLREDUCE = 200,
    REDUCE = 201,
    NVSHMEM_COPY = 202,
    
    // Custom task
    CUSTOM = 999,
};

inline const char* pk_task_type_to_name(PKTaskType type) {
    switch (type) {
        case PKTaskType::TERMINATE: return "terminate";
        case PKTaskType::BEGIN_TASK_GRAPH: return "begin_task_graph";
        case PKTaskType::EMBEDDING: return "embedding";
        case PKTaskType::RMS_NORM: return "rms_norm";
        case PKTaskType::RMS_NORM_LINEAR: return "rms_norm_linear";
        case PKTaskType::LINEAR: return "linear";
        case PKTaskType::LINEAR_RESIDUAL: return "linear_residual";
        case PKTaskType::ATTENTION: return "attention";
        case PKTaskType::PAGED_ATTENTION: return "paged_attention";
        case PKTaskType::SILU_MUL: return "silu_mul";
        case PKTaskType::SILU_MUL_LINEAR: return "silu_mul_linear";
        case PKTaskType::ARGMAX: return "argmax";
        case PKTaskType::ROTARY_EMBEDDING: return "rotary_embedding";
        case PKTaskType::MOE_GATE: return "moe_gate";
        case PKTaskType::MOE_LINEAR: return "moe_linear";
        case PKTaskType::ALLREDUCE: return "allreduce";
        case PKTaskType::REDUCE: return "reduce";
        case PKTaskType::NVSHMEM_COPY: return "nvshmem_copy";
        case PKTaskType::CUSTOM: return "custom";
        default: return "unknown";
    }
}

inline bool pk_is_compute_task(PKTaskType type) {
    int t = static_cast<int>(type);
    return t >= 100 && t < 200;
}

inline bool pk_is_communication_task(PKTaskType type) {
    int t = static_cast<int>(type);
    return t >= 200 && t < 300;
}

// =============================================================================
// PKTensorDesc
// =============================================================================

struct PKTensorDesc {
    void* data = nullptr;
    PKDataType dtype = PKDataType::FP32;
    int num_dims = 0;
    int64_t dims[8] = {0};
    int64_t strides[8] = {0};
    
    size_t num_elements() const {
        if (num_dims == 0) return 0;
        size_t elements = 1;
        for (int i = 0; i < num_dims; ++i) {
            elements *= dims[i];
        }
        return elements;
    }
    
    size_t size_bytes() const {
        size_t element_size = 4;  // Default FP32
        switch (dtype) {
            case PKDataType::FP32: element_size = 4; break;
            case PKDataType::FP16:
            case PKDataType::BF16: element_size = 2; break;
            case PKDataType::INT8:
            case PKDataType::FP8_E4M3:
            case PKDataType::FP8_E5M2: element_size = 1; break;
            default: element_size = 4; break;
        }
        return num_elements() * element_size;
    }
    
    bool is_contiguous() const {
        if (num_dims == 0) return true;
        
        int64_t expected_stride = 1;
        for (int i = num_dims - 1; i >= 0; --i) {
            if (strides[i] != expected_stride) return false;
            expected_stride *= dims[i];
        }
        return true;
    }
    
    void compute_strides() {
        if (num_dims == 0) return;
        
        strides[num_dims - 1] = 1;
        for (int i = num_dims - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
    }
};

// =============================================================================
// PKTaskDesc
// =============================================================================

struct PKTaskDesc {
    PKTaskType type = PKTaskType::TERMINATE;
    int task_id = 0;
    
    int num_inputs = 0;
    int num_outputs = 0;
    PKTensorDesc inputs[8];
    PKTensorDesc outputs[4];
    
    void* config = nullptr;
    size_t config_size = 0;
    
    void* params = nullptr;
    size_t params_size = 0;
    
    static PKTaskDesc create_linear(int task_id, void* input, void* weight, void* output,
                                    int batch, int in_features, int out_features) {
        PKTaskDesc desc;
        desc.type = PKTaskType::LINEAR;
        desc.task_id = task_id;
        desc.num_inputs = 2;
        desc.num_outputs = 1;
        
        // Input tensor
        desc.inputs[0].data = input;
        desc.inputs[0].dtype = PKDataType::FP16;
        desc.inputs[0].num_dims = 2;
        desc.inputs[0].dims[0] = batch;
        desc.inputs[0].dims[1] = in_features;
        desc.inputs[0].compute_strides();
        
        // Weight tensor
        desc.inputs[1].data = weight;
        desc.inputs[1].dtype = PKDataType::FP16;
        desc.inputs[1].num_dims = 2;
        desc.inputs[1].dims[0] = out_features;
        desc.inputs[1].dims[1] = in_features;
        desc.inputs[1].compute_strides();
        
        // Output tensor
        desc.outputs[0].data = output;
        desc.outputs[0].dtype = PKDataType::FP16;
        desc.outputs[0].num_dims = 2;
        desc.outputs[0].dims[0] = batch;
        desc.outputs[0].dims[1] = out_features;
        desc.outputs[0].compute_strides();
        
        return desc;
    }
    
    static PKTaskDesc create_attention(int task_id, void* q, void* k, void* v, void* output,
                                       int batch, int seq_len, int num_heads, int head_dim) {
        PKTaskDesc desc;
        desc.type = PKTaskType::ATTENTION;
        desc.task_id = task_id;
        desc.num_inputs = 3;
        desc.num_outputs = 1;
        
        // Q, K, V tensors (same shape)
        for (int i = 0; i < 3; ++i) {
            desc.inputs[i].dtype = PKDataType::FP16;
            desc.inputs[i].num_dims = 4;
            desc.inputs[i].dims[0] = batch;
            desc.inputs[i].dims[1] = num_heads;
            desc.inputs[i].dims[2] = seq_len;
            desc.inputs[i].dims[3] = head_dim;
            desc.inputs[i].compute_strides();
        }
        desc.inputs[0].data = q;
        desc.inputs[1].data = k;
        desc.inputs[2].data = v;
        
        // Output
        desc.outputs[0].data = output;
        desc.outputs[0].dtype = PKDataType::FP16;
        desc.outputs[0].num_dims = 4;
        desc.outputs[0].dims[0] = batch;
        desc.outputs[0].dims[1] = num_heads;
        desc.outputs[0].dims[2] = seq_len;
        desc.outputs[0].dims[3] = head_dim;
        desc.outputs[0].compute_strides();
        
        return desc;
    }
};

// =============================================================================
// PKMemoryAllocator Interface
// =============================================================================

class PKMemoryAllocator {
public:
    virtual ~PKMemoryAllocator() = default;
    
    virtual void* allocate(size_t size) = 0;
    virtual void free(void* ptr) = 0;
    virtual void copy_h2d(void* dst, const void* src, size_t size) = 0;
    virtual void copy_d2h(void* dst, const void* src, size_t size) = 0;
    virtual void copy_d2d(void* dst, const void* src, size_t size) = 0;
    virtual void copy_h2d_async(void* dst, const void* src, size_t size, void* stream) = 0;
    virtual void memset(void* ptr, int value, size_t size) = 0;
    virtual size_t get_total_memory() const = 0;
    virtual size_t get_free_memory() const = 0;
};

// Mock Memory Allocator
class MockMemoryAllocator : public PKMemoryAllocator {
public:
    MockMemoryAllocator(size_t total_memory = 16ULL * 1024 * 1024 * 1024)
        : total_memory_(total_memory), used_memory_(0) {}
    
    void* allocate(size_t size) override {
        if (used_memory_ + size > total_memory_) return nullptr;
        
        void* ptr = std::malloc(size);
        if (ptr) {
            allocations_[ptr] = size;
            used_memory_ += size;
        }
        return ptr;
    }
    
    void free(void* ptr) override {
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            used_memory_ -= it->second;
            allocations_.erase(it);
            std::free(ptr);
        }
    }
    
    void copy_h2d(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
        ++copy_count_;
    }
    
    void copy_d2h(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
        ++copy_count_;
    }
    
    void copy_d2d(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
        ++copy_count_;
    }
    
    void copy_h2d_async(void* dst, const void* src, size_t size, void* stream) override {
        std::memcpy(dst, src, size);
        ++async_copy_count_;
    }
    
    void memset(void* ptr, int value, size_t size) override {
        std::memset(ptr, value, size);
    }
    
    size_t get_total_memory() const override { return total_memory_; }
    size_t get_free_memory() const override { return total_memory_ - used_memory_; }
    
    size_t get_allocation_count() const { return allocations_.size(); }
    size_t get_copy_count() const { return copy_count_; }
    size_t get_async_copy_count() const { return async_copy_count_; }
    
private:
    size_t total_memory_;
    size_t used_memory_;
    std::unordered_map<void*, size_t> allocations_;
    size_t copy_count_ = 0;
    size_t async_copy_count_ = 0;
};

// =============================================================================
// PKAtomicOps Interface
// =============================================================================

class PKAtomicOps {
public:
    virtual ~PKAtomicOps() = default;
    
    virtual uint64_t fetch_add_u64(uint64_t* addr, uint64_t val) = 0;
    virtual uint64_t fetch_sub_u64(uint64_t* addr, uint64_t val) = 0;
    virtual uint64_t compare_exchange_u64(uint64_t* addr, uint64_t expected, uint64_t desired) = 0;
    virtual void store_release_u64(uint64_t* addr, uint64_t val) = 0;
    virtual uint64_t load_acquire_u64(uint64_t* addr) = 0;
    
    virtual uint32_t fetch_add_u32(uint32_t* addr, uint32_t val) = 0;
    virtual uint32_t fetch_sub_u32(uint32_t* addr, uint32_t val) = 0;
    virtual uint32_t compare_exchange_u32(uint32_t* addr, uint32_t expected, uint32_t desired) = 0;
    
    virtual void memory_fence() = 0;
    virtual void thread_fence() = 0;
};

// Mock Atomic Ops
class MockAtomicOps : public PKAtomicOps {
public:
    uint64_t fetch_add_u64(uint64_t* addr, uint64_t val) override {
        uint64_t old = *addr;
        *addr += val;
        return old;
    }
    
    uint64_t fetch_sub_u64(uint64_t* addr, uint64_t val) override {
        uint64_t old = *addr;
        *addr -= val;
        return old;
    }
    
    uint64_t compare_exchange_u64(uint64_t* addr, uint64_t expected, uint64_t desired) override {
        uint64_t old = *addr;
        if (*addr == expected) {
            *addr = desired;
        }
        return old;
    }
    
    void store_release_u64(uint64_t* addr, uint64_t val) override {
        *addr = val;
    }
    
    uint64_t load_acquire_u64(uint64_t* addr) override {
        return *addr;
    }
    
    uint32_t fetch_add_u32(uint32_t* addr, uint32_t val) override {
        uint32_t old = *addr;
        *addr += val;
        return old;
    }
    
    uint32_t fetch_sub_u32(uint32_t* addr, uint32_t val) override {
        uint32_t old = *addr;
        *addr -= val;
        return old;
    }
    
    uint32_t compare_exchange_u32(uint32_t* addr, uint32_t expected, uint32_t desired) override {
        uint32_t old = *addr;
        if (*addr == expected) {
            *addr = desired;
        }
        return old;
    }
    
    void memory_fence() override {}
    void thread_fence() override {}
};

// =============================================================================
// PKTaskExecutor Interface
// =============================================================================

class PKTaskExecutor {
public:
    virtual ~PKTaskExecutor() = default;
    
    virtual bool supports_task(PKTaskType type) const = 0;
    virtual void execute(const PKTaskDesc& desc, void* shared_memory, size_t shared_memory_size) = 0;
    virtual size_t get_shared_memory_size(PKTaskType type) const = 0;
    virtual const char* get_task_name(PKTaskType type) const = 0;
};

// Mock Task Executor
class MockTaskExecutor : public PKTaskExecutor {
public:
    MockTaskExecutor() {
        supported_tasks_ = {
            PKTaskType::LINEAR,
            PKTaskType::ATTENTION,
            PKTaskType::RMS_NORM,
            PKTaskType::SILU_MUL,
            PKTaskType::EMBEDDING,
            PKTaskType::ALLREDUCE,
        };
    }
    
    bool supports_task(PKTaskType type) const override {
        return supported_tasks_.find(type) != supported_tasks_.end();
    }
    
    void execute(const PKTaskDesc& desc, void* shared_memory, size_t shared_memory_size) override {
        executed_tasks_.push_back(desc.type);
        ++execution_count_;
    }
    
    size_t get_shared_memory_size(PKTaskType type) const override {
        switch (type) {
            case PKTaskType::ATTENTION: return 64 * 1024;  // 64KB
            case PKTaskType::LINEAR: return 32 * 1024;
            case PKTaskType::RMS_NORM: return 16 * 1024;
            default: return 8 * 1024;
        }
    }
    
    const char* get_task_name(PKTaskType type) const override {
        return pk_task_type_to_name(type);
    }
    
    size_t get_execution_count() const { return execution_count_; }
    std::vector<PKTaskType> const& get_executed_tasks() const { return executed_tasks_; }
    
private:
    std::set<PKTaskType> supported_tasks_;
    std::vector<PKTaskType> executed_tasks_;
    size_t execution_count_ = 0;
};

}  // namespace persistent_kernel
}  // namespace yirage

using namespace yirage::persistent_kernel;

// =============================================================================
// PKTaskType Tests
// =============================================================================

class PKTaskTypeTest : public ::testing::Test {};

TEST_F(PKTaskTypeTest, ComputeTasks) {
    EXPECT_TRUE(pk_is_compute_task(PKTaskType::EMBEDDING));
    EXPECT_TRUE(pk_is_compute_task(PKTaskType::LINEAR));
    EXPECT_TRUE(pk_is_compute_task(PKTaskType::ATTENTION));
    EXPECT_TRUE(pk_is_compute_task(PKTaskType::RMS_NORM));
    EXPECT_TRUE(pk_is_compute_task(PKTaskType::SILU_MUL));
}

TEST_F(PKTaskTypeTest, CommunicationTasks) {
    EXPECT_TRUE(pk_is_communication_task(PKTaskType::ALLREDUCE));
    EXPECT_TRUE(pk_is_communication_task(PKTaskType::REDUCE));
    EXPECT_TRUE(pk_is_communication_task(PKTaskType::NVSHMEM_COPY));
}

TEST_F(PKTaskTypeTest, ControlTasks) {
    EXPECT_FALSE(pk_is_compute_task(PKTaskType::TERMINATE));
    EXPECT_FALSE(pk_is_communication_task(PKTaskType::TERMINATE));
    EXPECT_FALSE(pk_is_compute_task(PKTaskType::BEGIN_TASK_GRAPH));
}

TEST_F(PKTaskTypeTest, TaskTypeToName) {
    EXPECT_STREQ(pk_task_type_to_name(PKTaskType::LINEAR), "linear");
    EXPECT_STREQ(pk_task_type_to_name(PKTaskType::ATTENTION), "attention");
    EXPECT_STREQ(pk_task_type_to_name(PKTaskType::RMS_NORM), "rms_norm");
    EXPECT_STREQ(pk_task_type_to_name(PKTaskType::ALLREDUCE), "allreduce");
}

// =============================================================================
// PKTensorDesc Tests
// =============================================================================

class PKTensorDescTest : public ::testing::Test {};

TEST_F(PKTensorDescTest, DefaultValues) {
    PKTensorDesc desc;
    EXPECT_EQ(desc.data, nullptr);
    EXPECT_EQ(desc.dtype, PKDataType::FP32);
    EXPECT_EQ(desc.num_dims, 0);
}

TEST_F(PKTensorDescTest, NumElements2D) {
    PKTensorDesc desc;
    desc.num_dims = 2;
    desc.dims[0] = 128;
    desc.dims[1] = 256;
    
    EXPECT_EQ(desc.num_elements(), 128u * 256u);
}

TEST_F(PKTensorDescTest, NumElements4D) {
    PKTensorDesc desc;
    desc.num_dims = 4;
    desc.dims[0] = 8;   // batch
    desc.dims[1] = 32;  // heads
    desc.dims[2] = 128; // seq_len
    desc.dims[3] = 64;  // head_dim
    
    EXPECT_EQ(desc.num_elements(), 8u * 32u * 128u * 64u);
}

TEST_F(PKTensorDescTest, SizeBytesFP32) {
    PKTensorDesc desc;
    desc.dtype = PKDataType::FP32;
    desc.num_dims = 2;
    desc.dims[0] = 100;
    desc.dims[1] = 100;
    
    EXPECT_EQ(desc.size_bytes(), 100u * 100u * 4u);
}

TEST_F(PKTensorDescTest, SizeBytesFP16) {
    PKTensorDesc desc;
    desc.dtype = PKDataType::FP16;
    desc.num_dims = 2;
    desc.dims[0] = 100;
    desc.dims[1] = 100;
    
    EXPECT_EQ(desc.size_bytes(), 100u * 100u * 2u);
}

TEST_F(PKTensorDescTest, ComputeStrides) {
    PKTensorDesc desc;
    desc.num_dims = 3;
    desc.dims[0] = 8;
    desc.dims[1] = 32;
    desc.dims[2] = 64;
    desc.compute_strides();
    
    EXPECT_EQ(desc.strides[2], 1);
    EXPECT_EQ(desc.strides[1], 64);
    EXPECT_EQ(desc.strides[0], 32 * 64);
}

TEST_F(PKTensorDescTest, IsContiguous) {
    PKTensorDesc desc;
    desc.num_dims = 2;
    desc.dims[0] = 128;
    desc.dims[1] = 256;
    desc.compute_strides();
    
    EXPECT_TRUE(desc.is_contiguous());
    
    // Make non-contiguous
    desc.strides[0] = 512;  // Padded
    EXPECT_FALSE(desc.is_contiguous());
}

// =============================================================================
// PKTaskDesc Tests
// =============================================================================

class PKTaskDescTest : public ::testing::Test {};

TEST_F(PKTaskDescTest, DefaultValues) {
    PKTaskDesc desc;
    EXPECT_EQ(desc.type, PKTaskType::TERMINATE);
    EXPECT_EQ(desc.task_id, 0);
    EXPECT_EQ(desc.num_inputs, 0);
    EXPECT_EQ(desc.num_outputs, 0);
}

TEST_F(PKTaskDescTest, CreateLinear) {
    float input[128 * 256];
    float weight[512 * 256];
    float output[128 * 512];
    
    auto desc = PKTaskDesc::create_linear(1, input, weight, output, 128, 256, 512);
    
    EXPECT_EQ(desc.type, PKTaskType::LINEAR);
    EXPECT_EQ(desc.task_id, 1);
    EXPECT_EQ(desc.num_inputs, 2);
    EXPECT_EQ(desc.num_outputs, 1);
    
    EXPECT_EQ(desc.inputs[0].dims[0], 128);
    EXPECT_EQ(desc.inputs[0].dims[1], 256);
    EXPECT_EQ(desc.inputs[1].dims[0], 512);
    EXPECT_EQ(desc.inputs[1].dims[1], 256);
    EXPECT_EQ(desc.outputs[0].dims[0], 128);
    EXPECT_EQ(desc.outputs[0].dims[1], 512);
}

TEST_F(PKTaskDescTest, CreateAttention) {
    float q[8 * 32 * 128 * 64];
    float k[8 * 32 * 128 * 64];
    float v[8 * 32 * 128 * 64];
    float output[8 * 32 * 128 * 64];
    
    auto desc = PKTaskDesc::create_attention(2, q, k, v, output, 8, 128, 32, 64);
    
    EXPECT_EQ(desc.type, PKTaskType::ATTENTION);
    EXPECT_EQ(desc.task_id, 2);
    EXPECT_EQ(desc.num_inputs, 3);
    EXPECT_EQ(desc.num_outputs, 1);
    
    // Check Q shape
    EXPECT_EQ(desc.inputs[0].num_dims, 4);
    EXPECT_EQ(desc.inputs[0].dims[0], 8);   // batch
    EXPECT_EQ(desc.inputs[0].dims[1], 32);  // heads
    EXPECT_EQ(desc.inputs[0].dims[2], 128); // seq_len
    EXPECT_EQ(desc.inputs[0].dims[3], 64);  // head_dim
}

// =============================================================================
// PKMemoryAllocator Tests
// =============================================================================

class PKMemoryAllocatorTest : public ::testing::Test {
protected:
    MockMemoryAllocator allocator{1024 * 1024};  // 1MB
};

TEST_F(PKMemoryAllocatorTest, AllocateAndFree) {
    void* ptr = allocator.allocate(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(allocator.get_allocation_count(), 1u);
    
    allocator.free(ptr);
    EXPECT_EQ(allocator.get_allocation_count(), 0u);
}

TEST_F(PKMemoryAllocatorTest, AllocateTooMuch) {
    void* ptr = allocator.allocate(2 * 1024 * 1024);  // More than 1MB
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(PKMemoryAllocatorTest, GetFreeMemory) {
    EXPECT_EQ(allocator.get_free_memory(), 1024u * 1024u);
    
    void* ptr = allocator.allocate(256 * 1024);
    EXPECT_EQ(allocator.get_free_memory(), 1024u * 1024u - 256u * 1024u);
    
    allocator.free(ptr);
    EXPECT_EQ(allocator.get_free_memory(), 1024u * 1024u);
}

TEST_F(PKMemoryAllocatorTest, CopyOperations) {
    char src[100] = "Hello, World!";
    char dst[100] = {0};
    
    allocator.copy_h2d(dst, src, 14);
    EXPECT_STREQ(dst, "Hello, World!");
    EXPECT_EQ(allocator.get_copy_count(), 1u);
}

TEST_F(PKMemoryAllocatorTest, AsyncCopy) {
    char src[100] = "Async";
    char dst[100] = {0};
    
    allocator.copy_h2d_async(dst, src, 6, nullptr);
    EXPECT_EQ(allocator.get_async_copy_count(), 1u);
}

TEST_F(PKMemoryAllocatorTest, Memset) {
    char buffer[100] = {0};
    allocator.memset(buffer, 0xFF, 50);
    
    for (int i = 0; i < 50; ++i) {
        EXPECT_EQ(static_cast<unsigned char>(buffer[i]), 0xFF);
    }
    for (int i = 50; i < 100; ++i) {
        EXPECT_EQ(buffer[i], 0);
    }
}

// =============================================================================
// PKAtomicOps Tests
// =============================================================================

class PKAtomicOpsTest : public ::testing::Test {
protected:
    MockAtomicOps atomic_ops;
};

TEST_F(PKAtomicOpsTest, FetchAddU64) {
    uint64_t value = 100;
    uint64_t old = atomic_ops.fetch_add_u64(&value, 50);
    
    EXPECT_EQ(old, 100u);
    EXPECT_EQ(value, 150u);
}

TEST_F(PKAtomicOpsTest, FetchSubU64) {
    uint64_t value = 100;
    uint64_t old = atomic_ops.fetch_sub_u64(&value, 30);
    
    EXPECT_EQ(old, 100u);
    EXPECT_EQ(value, 70u);
}

TEST_F(PKAtomicOpsTest, CompareExchangeU64Success) {
    uint64_t value = 100;
    uint64_t old = atomic_ops.compare_exchange_u64(&value, 100, 200);
    
    EXPECT_EQ(old, 100u);
    EXPECT_EQ(value, 200u);  // Changed
}

TEST_F(PKAtomicOpsTest, CompareExchangeU64Failure) {
    uint64_t value = 100;
    uint64_t old = atomic_ops.compare_exchange_u64(&value, 50, 200);  // Expected != value
    
    EXPECT_EQ(old, 100u);
    EXPECT_EQ(value, 100u);  // Unchanged
}

TEST_F(PKAtomicOpsTest, StoreAndLoad) {
    uint64_t value = 0;
    atomic_ops.store_release_u64(&value, 42);
    
    uint64_t loaded = atomic_ops.load_acquire_u64(&value);
    EXPECT_EQ(loaded, 42u);
}

TEST_F(PKAtomicOpsTest, FetchAddU32) {
    uint32_t value = 100;
    uint32_t old = atomic_ops.fetch_add_u32(&value, 25);
    
    EXPECT_EQ(old, 100u);
    EXPECT_EQ(value, 125u);
}

// =============================================================================
// PKTaskExecutor Tests
// =============================================================================

class PKTaskExecutorTest : public ::testing::Test {
protected:
    MockTaskExecutor executor;
};

TEST_F(PKTaskExecutorTest, SupportsTask) {
    EXPECT_TRUE(executor.supports_task(PKTaskType::LINEAR));
    EXPECT_TRUE(executor.supports_task(PKTaskType::ATTENTION));
    EXPECT_TRUE(executor.supports_task(PKTaskType::ALLREDUCE));
    EXPECT_FALSE(executor.supports_task(PKTaskType::CUSTOM));
}

TEST_F(PKTaskExecutorTest, Execute) {
    PKTaskDesc desc;
    desc.type = PKTaskType::LINEAR;
    desc.task_id = 1;
    
    executor.execute(desc, nullptr, 0);
    
    EXPECT_EQ(executor.get_execution_count(), 1u);
    EXPECT_EQ(executor.get_executed_tasks().size(), 1u);
    EXPECT_EQ(executor.get_executed_tasks()[0], PKTaskType::LINEAR);
}

TEST_F(PKTaskExecutorTest, GetSharedMemorySize) {
    EXPECT_EQ(executor.get_shared_memory_size(PKTaskType::ATTENTION), 64u * 1024u);
    EXPECT_EQ(executor.get_shared_memory_size(PKTaskType::LINEAR), 32u * 1024u);
    EXPECT_EQ(executor.get_shared_memory_size(PKTaskType::RMS_NORM), 16u * 1024u);
}

TEST_F(PKTaskExecutorTest, GetTaskName) {
    EXPECT_STREQ(executor.get_task_name(PKTaskType::LINEAR), "linear");
    EXPECT_STREQ(executor.get_task_name(PKTaskType::ATTENTION), "attention");
}

// =============================================================================
// Parameterized Task Tests
// =============================================================================

struct TaskTestParam {
    PKTaskType type;
    const char* expected_name;
    bool is_compute;
    bool is_communication;
};

class TaskParameterizedTest : public ::testing::TestWithParam<TaskTestParam> {};

TEST_P(TaskParameterizedTest, TaskProperties) {
    auto param = GetParam();
    
    EXPECT_STREQ(pk_task_type_to_name(param.type), param.expected_name);
    EXPECT_EQ(pk_is_compute_task(param.type), param.is_compute);
    EXPECT_EQ(pk_is_communication_task(param.type), param.is_communication);
}

INSTANTIATE_TEST_SUITE_P(
    AllTaskTypes,
    TaskParameterizedTest,
    ::testing::Values(
        TaskTestParam{PKTaskType::LINEAR, "linear", true, false},
        TaskTestParam{PKTaskType::ATTENTION, "attention", true, false},
        TaskTestParam{PKTaskType::RMS_NORM, "rms_norm", true, false},
        TaskTestParam{PKTaskType::SILU_MUL, "silu_mul", true, false},
        TaskTestParam{PKTaskType::ALLREDUCE, "allreduce", false, true},
        TaskTestParam{PKTaskType::REDUCE, "reduce", false, true},
        TaskTestParam{PKTaskType::TERMINATE, "terminate", false, false}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
