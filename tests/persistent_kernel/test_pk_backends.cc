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

/**
 * @file test_pk_backends.cc
 * @brief Comprehensive tests for persistent kernel multi-backend support
 */

#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>
#include <string>

#include "persistent_kernel/backends/pk_backends.h"
#include "persistent_kernel/pk_runtime_adapter.h"
#include "persistent_kernel/pk_utils.h"

using namespace yirage::persistent_kernel;

// =============================================================================
// Test Utilities
// =============================================================================

#define TEST_CASE(name) \
    static void test_##name(); \
    static struct TestRegister_##name { \
        TestRegister_##name() { \
            test_cases.push_back({#name, test_##name}); \
        } \
    } test_register_##name; \
    static void test_##name()

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("Assertion failed"); \
        } \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))
#define ASSERT_NE(a, b) ASSERT_TRUE((a) != (b))
#define ASSERT_GT(a, b) ASSERT_TRUE((a) > (b))
#define ASSERT_GE(a, b) ASSERT_TRUE((a) >= (b))
#define ASSERT_NOT_NULL(ptr) ASSERT_TRUE((ptr) != nullptr)

struct TestCase {
    std::string name;
    void (*func)();
};

static std::vector<TestCase> test_cases;

// =============================================================================
// Backend Factory Tests
// =============================================================================

TEST_CASE(backend_factory_cpu) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    ASSERT_TRUE(backend->is_available());
    ASSERT_EQ(backend->get_type(), PKBackendType::CPU);
    ASSERT_EQ(backend->get_name(), "cpu");
}

TEST_CASE(backend_factory_cuda) {
    auto backend = create_pk_backend(PKBackendType::CUDA, 0);
    ASSERT_NOT_NULL(backend.get());
    // CUDA may or may not be available
    if (backend->is_available()) {
        ASSERT_EQ(backend->get_type(), PKBackendType::CUDA);
        ASSERT_EQ(backend->get_name(), "cuda");
    }
}

TEST_CASE(backend_factory_ascend) {
    auto backend = create_pk_backend(PKBackendType::ASCEND, 0);
    ASSERT_NOT_NULL(backend.get());
    ASSERT_EQ(backend->get_type(), PKBackendType::ASCEND);
    ASSERT_EQ(backend->get_name(), "ascend");
}

TEST_CASE(backend_factory_maca) {
    auto backend = create_pk_backend(PKBackendType::MACA, 0);
    ASSERT_NOT_NULL(backend.get());
    ASSERT_EQ(backend->get_type(), PKBackendType::MACA);
    ASSERT_EQ(backend->get_name(), "maca");
}

TEST_CASE(get_available_backends) {
    auto backends = get_available_pk_backends();
    ASSERT_FALSE(backends.empty());
    
    // CPU should always be in the list
    bool found_cpu = false;
    for (auto type : backends) {
        if (type == PKBackendType::CPU) {
            found_cpu = true;
            break;
        }
    }
    ASSERT_TRUE(found_cpu);
}

// =============================================================================
// Backend Capabilities Tests
// =============================================================================

TEST_CASE(cpu_capabilities) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    
    auto caps = backend->get_capabilities();
    
    // CPU should not support GPU features
    ASSERT_FALSE(caps.supports_tma);
    ASSERT_FALSE(caps.supports_tensor_cores);
    ASSERT_FALSE(caps.supports_nvshmem);
    
    // CPU should have positive memory
    ASSERT_GT(caps.max_global_memory, 0UL);
    
    // CPU should support some modes
    ASSERT_FALSE(caps.supported_modes.empty());
}

TEST_CASE(cpu_supported_modes) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    
    // CPU supports EAGER, GRAPH, OFFLINE (per implementation)
    ASSERT_TRUE(backend->supports_mode(PKMode::OFFLINE));
    ASSERT_TRUE(backend->supports_mode(PKMode::EAGER));
    ASSERT_TRUE(backend->supports_mode(PKMode::GRAPH));
    
    // CPU does not support ONLINE, ONEPASS, STREAMING
    ASSERT_FALSE(backend->supports_mode(PKMode::ONEPASS));
    ASSERT_FALSE(backend->supports_mode(PKMode::STREAMING));
}

TEST_CASE(backend_mode_support_matrix) {
    // Test mode support across backends (per pk_backend_factory.cc implementation)
    struct TestCase {
        PKBackendType backend;
        PKMode mode;
        bool expected;
    };
    
    std::vector<TestCase> cases = {
        // CPU: EAGER, GRAPH, OFFLINE
        {PKBackendType::CPU, PKMode::OFFLINE, true},
        {PKBackendType::CPU, PKMode::EAGER, true},
        {PKBackendType::CPU, PKMode::GRAPH, true},
        {PKBackendType::CPU, PKMode::ONEPASS, false},
        {PKBackendType::CPU, PKMode::STREAMING, false},
        // CUDA: OFFLINE, ONLINE, ONEPASS, GRAPH
        {PKBackendType::CUDA, PKMode::OFFLINE, true},
        {PKBackendType::CUDA, PKMode::ONLINE, true},
        {PKBackendType::CUDA, PKMode::ONEPASS, true},
        {PKBackendType::CUDA, PKMode::GRAPH, true},
        {PKBackendType::CUDA, PKMode::STREAMING, false},
        // Ascend: OFFLINE, ONLINE, GRAPH
        {PKBackendType::ASCEND, PKMode::OFFLINE, true},
        {PKBackendType::ASCEND, PKMode::ONLINE, true},
        {PKBackendType::ASCEND, PKMode::GRAPH, true},
        {PKBackendType::ASCEND, PKMode::STREAMING, false},
    };
    
    for (const auto& tc : cases) {
        bool result = pk_is_mode_supported(tc.backend, tc.mode);
        ASSERT_EQ(result, tc.expected);
    }
}

// =============================================================================
// Memory Allocator Tests
// =============================================================================

TEST_CASE(cpu_memory_allocation) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    
    auto* allocator = backend->get_allocator();
    ASSERT_NOT_NULL(allocator);
    
    // Allocate memory
    size_t size = 1024 * 1024;  // 1MB
    void* ptr = allocator->allocate(size);
    ASSERT_NOT_NULL(ptr);
    
    // Test memset
    allocator->memset(ptr, 0, size);
    
    // Test copy
    std::vector<char> src(1024, 'A');
    std::vector<char> dst(1024, 0);
    
    allocator->copy_h2d(ptr, src.data(), 1024);
    allocator->copy_d2h(dst.data(), ptr, 1024);
    
    ASSERT_EQ(memcmp(src.data(), dst.data(), 1024), 0);
    
    // Free memory
    allocator->free(ptr);
}

TEST_CASE(cpu_memory_info) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    
    auto* allocator = backend->get_allocator();
    ASSERT_NOT_NULL(allocator);
    
    size_t total = allocator->get_total_memory();
    size_t free = allocator->get_free_memory();
    
    ASSERT_GT(total, 0UL);
    ASSERT_GT(free, 0UL);
    ASSERT_GE(total, free);
}

// =============================================================================
// Initialization and Lifecycle Tests
// =============================================================================

TEST_CASE(cpu_backend_lifecycle) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    
    PKRuntimeConfig config;
    config.mode = PKMode::OFFLINE;
    config.num_workers = 4;
    
    // Initialize
    bool success = backend->initialize(config);
    ASSERT_TRUE(success);
    
    // Reset
    backend->reset();
    
    // Synchronize
    backend->synchronize();
    
    // Finalize
    backend->finalize();
}

TEST_CASE(cpu_device_management) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    
    // CPU has exactly 1 device
    ASSERT_EQ(backend->get_device_count(), 1);
    ASSERT_EQ(backend->get_device(), 0);
    
    // Setting device 0 should succeed
    ASSERT_TRUE(backend->set_device(0));
}

// =============================================================================
// Compile Flags Tests
// =============================================================================

TEST_CASE(cpu_compile_flags) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    
    auto flags = backend->get_compile_flags(PKMode::OFFLINE);
    ASSERT_FALSE(flags.empty());
    
    // Should have MODE_OFFLINE flag
    bool found_mode = false;
    for (const auto& flag : flags) {
        if (flag.find("MODE_OFFLINE") != std::string::npos) {
            found_mode = true;
            break;
        }
    }
    ASSERT_TRUE(found_mode);
}

TEST_CASE(cuda_compile_flags) {
    auto backend = create_pk_backend(PKBackendType::CUDA, 0);
    ASSERT_NOT_NULL(backend.get());
    
    auto flags = backend->get_compile_flags(PKMode::ONLINE);
    ASSERT_FALSE(flags.empty());
    
    // Should have arch flag
    bool found_arch = false;
    for (const auto& flag : flags) {
        if (flag.find("-arch") != std::string::npos || 
            flag.find("YPK_TARGET_CC") != std::string::npos) {
            found_arch = true;
            break;
        }
    }
    ASSERT_TRUE(found_arch);
}

// =============================================================================
// Task Queue Tests
// =============================================================================

TEST_CASE(task_queue_basic) {
    PKTaskQueue queue(10);
    
    ASSERT_TRUE(queue.empty());
    ASSERT_EQ(queue.size(), 0UL);
    
    // Push task
    PKTaskDesc task;
    task.type = PKTaskType::LINEAR;
    task.task_id = 1;
    
    ASSERT_TRUE(queue.push(task));
    ASSERT_FALSE(queue.empty());
    ASSERT_EQ(queue.size(), 1UL);
    
    // Pop task
    PKTaskDesc popped;
    ASSERT_TRUE(queue.pop(popped, 0));
    ASSERT_EQ(popped.type, PKTaskType::LINEAR);
    ASSERT_EQ(popped.task_id, 1);
    
    ASSERT_TRUE(queue.empty());
}

TEST_CASE(task_queue_timeout) {
    PKTaskQueue queue(10);
    
    // Pop from empty queue with timeout
    PKTaskDesc task;
    ASSERT_FALSE(queue.pop(task, 10));  // 10ms timeout
}

// =============================================================================
// Batch Manager Tests
// =============================================================================

TEST_CASE(batch_manager_basic) {
    PKBatchConfig config;
    config.max_batch_size = 4;
    config.max_seq_length = 128;
    config.max_tokens_per_batch = 32;
    config.page_size = 16;
    config.max_pages = 64;
    config.eos_token_id = 2;
    
    PKBatchManager manager(config);
    
    // Add request
    std::vector<int64_t> tokens = {1, 2, 3, 4, 5};
    int req_id = manager.add_request(tokens);
    ASSERT_GE(req_id, 0);
    
    // Prepare batch
    std::vector<int64_t> input_tokens;
    std::vector<int> qo_indptr, kv_indptr, kv_indices, kv_last_page_len;
    
    int num_active = manager.prepare_batch(
        input_tokens, qo_indptr, kv_indptr, kv_indices, kv_last_page_len
    );
    
    ASSERT_EQ(num_active, 1);
    ASSERT_FALSE(input_tokens.empty());
}

// =============================================================================
// Mode Adapter Tests
// =============================================================================

TEST_CASE(offline_mode_adapter) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    
    if (!backend->supports_mode(PKMode::OFFLINE)) {
        return;  // Skip if not supported
    }
    
    auto adapter = create_mode_adapter(backend.get(), PKMode::OFFLINE);
    ASSERT_NOT_NULL(adapter.get());
    ASSERT_EQ(adapter->get_mode(), PKMode::OFFLINE);
}

TEST_CASE(onepass_mode_adapter) {
    auto backend = create_pk_backend(PKBackendType::CPU, 0);
    ASSERT_NOT_NULL(backend.get());
    
    if (!backend->supports_mode(PKMode::ONEPASS)) {
        return;
    }
    
    auto adapter = create_mode_adapter(backend.get(), PKMode::ONEPASS);
    ASSERT_NOT_NULL(adapter.get());
    ASSERT_EQ(adapter->get_mode(), PKMode::ONEPASS);
}

// =============================================================================
// Utility Tests
// =============================================================================

TEST_CASE(backend_priority) {
    // CUDA should have highest priority
    ASSERT_GT(get_backend_priority(PKBackendType::CUDA), 
              get_backend_priority(PKBackendType::CPU));
    
    // MACA should have lower than CUDA but higher than CPU
    ASSERT_GT(get_backend_priority(PKBackendType::MACA),
              get_backend_priority(PKBackendType::CPU));
}

TEST_CASE(backend_selection_by_name) {
    ASSERT_EQ(select_backend_by_name("cuda"), PKBackendType::CUDA);
    ASSERT_EQ(select_backend_by_name("CUDA"), PKBackendType::CUDA);
    ASSERT_EQ(select_backend_by_name("cpu"), PKBackendType::CPU);
    ASSERT_EQ(select_backend_by_name("ascend"), PKBackendType::ASCEND);
    ASSERT_EQ(select_backend_by_name("huawei"), PKBackendType::ASCEND);
    ASSERT_EQ(select_backend_by_name("maca"), PKBackendType::MACA);
    ASSERT_EQ(select_backend_by_name("metax"), PKBackendType::MACA);
}

TEST_CASE(mode_parsing) {
    ASSERT_EQ(parse_mode("offline"), PKMode::OFFLINE);
    ASSERT_EQ(parse_mode("OFFLINE"), PKMode::OFFLINE);
    ASSERT_EQ(parse_mode("online"), PKMode::ONLINE);
    ASSERT_EQ(parse_mode("onepass"), PKMode::ONEPASS);
    ASSERT_EQ(parse_mode("eager"), PKMode::EAGER);
    ASSERT_EQ(parse_mode("graph"), PKMode::GRAPH);
    ASSERT_EQ(parse_mode("streaming"), PKMode::STREAMING);
}

TEST_CASE(config_builder) {
    auto config = PKRuntimeConfigBuilder()
        .mode(PKMode::ONLINE)
        .workers(8)
        .schedulers(2, 1)
        .max_seq_length(4096)
        .batch_config(32, 128)
        .paging(2048, 128)
        .eos_token(2)
        .multi_gpu(4, 0)
        .profiling(true)
        .build();
    
    ASSERT_EQ(config.mode, PKMode::ONLINE);
    ASSERT_EQ(config.num_workers, 8);
    ASSERT_EQ(config.num_local_schedulers, 2);
    ASSERT_EQ(config.num_remote_schedulers, 1);
    ASSERT_EQ(config.max_seq_length, 4096UL);
    ASSERT_EQ(config.max_num_batched_requests, 32UL);
    ASSERT_EQ(config.max_num_batched_tokens, 128UL);
    ASSERT_EQ(config.max_num_pages, 2048UL);
    ASSERT_EQ(config.page_size, 128UL);
    ASSERT_EQ(config.eos_token_id, 2);
    ASSERT_EQ(config.num_gpus, 4);
    ASSERT_EQ(config.my_gpu_id, 0);
    ASSERT_TRUE(config.profiling_enabled);
}

TEST_CASE(profiler) {
    PKProfiler profiler;
    profiler.enable();
    
    profiler.start();
    // Do some work
    for (volatile int i = 0; i < 1000000; ++i) {}
    profiler.stop();
    
    ASSERT_GT(profiler.get_total_ms(), 0.0);
    ASSERT_EQ(profiler.get_count(), 1UL);
    
    profiler.reset();
    ASSERT_EQ(profiler.get_count(), 0UL);
}

// =============================================================================
// Name Conversion Tests
// =============================================================================

TEST_CASE(backend_type_to_name) {
    ASSERT_EQ(std::string(pk_backend_type_to_name(PKBackendType::CUDA)), "cuda");
    ASSERT_EQ(std::string(pk_backend_type_to_name(PKBackendType::CPU)), "cpu");
    ASSERT_EQ(std::string(pk_backend_type_to_name(PKBackendType::ASCEND)), "ascend");
    ASSERT_EQ(std::string(pk_backend_type_to_name(PKBackendType::MACA)), "maca");
}

TEST_CASE(mode_to_name) {
    ASSERT_EQ(std::string(pk_mode_to_name(PKMode::OFFLINE)), "offline");
    ASSERT_EQ(std::string(pk_mode_to_name(PKMode::ONLINE)), "online");
    ASSERT_EQ(std::string(pk_mode_to_name(PKMode::ONEPASS)), "onepass");
    ASSERT_EQ(std::string(pk_mode_to_name(PKMode::EAGER)), "eager");
    ASSERT_EQ(std::string(pk_mode_to_name(PKMode::GRAPH)), "graph");
    ASSERT_EQ(std::string(pk_mode_to_name(PKMode::STREAMING)), "streaming");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "Running Persistent Kernel Backend Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& tc : test_cases) {
        std::cout << "Running " << tc.name << "... ";
        std::cout.flush();
        
        try {
            tc.func();
            std::cout << "PASSED" << std::endl;
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    
    return failed > 0 ? 1 : 0;
}
