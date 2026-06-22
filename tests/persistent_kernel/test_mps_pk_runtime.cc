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
 * @file test_mps_pk_runtime.cc
 * @brief Tests for MPS persistent kernel runtime
 */

#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>

// Include MPS runtime
#include "persistent_kernel/backends/mps_pk_runtime.h"

using namespace yirage::persistent_kernel;
using namespace yirage::persistent_kernel::mps;
using namespace yirage::persistent_kernel::runtime;

// =============================================================================
// Test Helpers
// =============================================================================

#define TEST_CASE(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { \
            std::cout << "Running: " << #name << "..." << std::endl; \
            test_##name(); \
            std::cout << "  PASSED" << std::endl; \
        } \
    } runner_##name; \
    void test_##name()

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "ASSERTION FAILED: " << #cond << std::endl; \
            std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_TRUE(a) ASSERT(a)
#define ASSERT_FALSE(a) ASSERT(!(a))

// =============================================================================
// MPS Runtime Configuration Tests
// =============================================================================

TEST_CASE(mps_runtime_config_defaults) {
    MpsRuntimeConfig config;
    
    // Check default values
    ASSERT_EQ(config.num_workers, 0);
    ASSERT_EQ(config.num_local_schedulers, 0);
    ASSERT_EQ(config.num_remote_schedulers, 0);
    ASSERT_EQ(config.mtl_device, nullptr);
    ASSERT_EQ(config.command_queue, nullptr);
    ASSERT_EQ(config.mps_graph, nullptr);
    ASSERT_FALSE(config.use_graph_mode);
    ASSERT_EQ(config.num_encoder_threads, 1);
}

TEST_CASE(mps_runtime_config_setup) {
    MpsRuntimeConfig config;
    
    // Configure for testing
    config.num_workers = 2;
    config.num_local_schedulers = 1;
    config.num_remote_schedulers = 0;
    config.per_worker_queue_len = 256;
    config.per_sched_queue_len = 128;
    config.use_graph_mode = false;
    
    ASSERT_EQ(config.num_workers, 2);
    ASSERT_EQ(config.num_local_schedulers, 1);
    ASSERT_EQ(config.per_worker_queue_len, 256);
}

// =============================================================================
// Task Descriptor Tests
// =============================================================================

TEST_CASE(mps_task_desc_creation) {
    PKTaskDesc task;
    
    // Default values
    ASSERT_EQ(task.task_type, PK_TASK_TERMINATE);
    ASSERT_EQ(task.trigger_event, EVENT_INVALID_ID);
    ASSERT_EQ(task.dependent_event, EVENT_INVALID_ID);
    
    // Set values
    task.task_type = PK_TASK_EMBEDDING;
    task.variant_id = 1;
    task.input_ptrs[0] = reinterpret_cast<void*>(0x1000);
    task.output_ptrs[0] = reinterpret_cast<void*>(0x2000);
    
    ASSERT_EQ(task.task_type, PK_TASK_EMBEDDING);
    ASSERT_EQ(task.variant_id, 1);
    ASSERT_NE(task.input_ptrs[0], nullptr);
}

TEST_CASE(mps_event_desc_creation) {
    PKEventDesc event;
    
    // Default values
    ASSERT_EQ(event.event_type, PK_EVENT_INVALID);
    ASSERT_EQ(event.num_triggers, 0);
    ASSERT_EQ(event.first_task_id, TASK_INVALID_ID);
    
    // Create with values
    PKEventDesc launch_event(PK_EVENT_LAUNCH_TASKS, 4, 10, 20);
    
    ASSERT_EQ(launch_event.event_type, PK_EVENT_LAUNCH_TASKS);
    ASSERT_EQ(launch_event.num_triggers, 4);
    ASSERT_EQ(launch_event.first_task_id, 10);
    ASSERT_EQ(launch_event.last_task_id, 20);
}

// =============================================================================
// Task ID Utilities Tests
// =============================================================================

TEST_CASE(mps_task_id_utilities) {
    // Test compute_task_id
    TaskId task_id = compute_task_id(5, 100);
    
    ASSERT_EQ(get_task_iteration_num(task_id), 5);
    ASSERT_EQ(get_task_position_index(task_id), 100);
    
    // Test with larger values
    TaskId task_id2 = compute_task_id(1000, 50000);
    
    ASSERT_EQ(get_task_iteration_num(task_id2), 1000);
    ASSERT_EQ(get_task_position_index(task_id2), 50000);
}

TEST_CASE(mps_event_id_utilities) {
    // Event ID format: bits 32-47 = device_id, bits 0-31 = position_index
    EventId event_id = 0x0001000000001234ULL;
    
    ASSERT_EQ(get_event_position_index(event_id), 0x1234);
    // device_id is bits 32-47: (0x00010000 >> 0) & 0xffff = 0
    ASSERT_EQ(get_event_device_id(event_id), 0);
    
    // Test with device_id = 1 (must be in bits 32-47)
    EventId event_id_dev1 = 0x0000000100001234ULL;
    ASSERT_EQ(get_event_device_id(event_id_dev1), 1);
    
    // Test NVSHMEM tag
    ASSERT_FALSE(is_nvshmem_event(event_id));
    
    EventId nvshmem_event = EVENT_NVSHMEM_TAG | event_id;
    ASSERT_TRUE(is_nvshmem_event(nvshmem_event));
}

// =============================================================================
// Batch Preparation Tests (without Metal device)
// =============================================================================

TEST_CASE(mps_batch_preparation_empty) {
    PKRuntimeConfig config;
    
    // Allocate required buffers
    config.request_ids = new int[YPK_MAX_NUM_BATCHED_REQUESTS + 1];
    config.step = new int[16];
    config.tokens = new int64_t[16 * YPK_MAX_SEQ_LENGTH];
    config.input_tokens = new int64_t[YPK_MAX_NUM_BATCHED_TOKENS];
    config.output_tokens = new int64_t[YPK_MAX_NUM_BATCHED_TOKENS];
    config.prompt_length = new int[16];
    config.qo_indptr_buffer = new int[YPK_MAX_NUM_BATCHED_REQUESTS + 1];
    config.paged_kv_indptr_buffer = new int[YPK_MAX_NUM_BATCHED_REQUESTS + 1];
    config.paged_kv_indices_buffer = new int[YPK_MAX_NUM_PAGES];
    config.paged_kv_last_page_len_buffer = new int[YPK_MAX_NUM_BATCHED_REQUESTS];
    config.page_queue = new int[YPK_MAX_NUM_PAGES];
    config.page_queue_head = new int(0);
    config.page_queue_tail = new int(YPK_MAX_NUM_PAGES);
    config.next_request_id = new int(0);
    config.total_num_requests = 0;
    config.max_seq_length = 512;
    config.eos_token_id = 2;
    
    // Initialize page queue
    for (int i = 0; i < YPK_MAX_NUM_PAGES; ++i) {
        config.page_queue[i] = i;
    }
    
    // Initialize request IDs to -1 (empty)
    for (int i = 0; i < YPK_MAX_NUM_BATCHED_REQUESTS; ++i) {
        config.request_ids[i] = -1;
    }
    
    // No requests, should return false
    bool has_batch = mps_prepare_next_batch(config);
    ASSERT_FALSE(has_batch);
    
    // Cleanup
    delete[] config.request_ids;
    delete[] config.step;
    delete[] config.tokens;
    delete[] config.input_tokens;
    delete[] config.output_tokens;
    delete[] config.prompt_length;
    delete[] config.qo_indptr_buffer;
    delete[] config.paged_kv_indptr_buffer;
    delete[] config.paged_kv_indices_buffer;
    delete[] config.paged_kv_last_page_len_buffer;
    delete[] config.page_queue;
    delete config.page_queue_head;
    delete config.page_queue_tail;
    delete config.next_request_id;
}

TEST_CASE(mps_batch_preparation_with_request) {
    PKRuntimeConfig config;
    
    // Allocate buffers
    config.request_ids = new int[YPK_MAX_NUM_BATCHED_REQUESTS + 1];
    config.step = new int[16];
    config.tokens = new int64_t[16 * YPK_MAX_SEQ_LENGTH];
    config.input_tokens = new int64_t[YPK_MAX_NUM_BATCHED_TOKENS];
    config.output_tokens = new int64_t[YPK_MAX_NUM_BATCHED_TOKENS];
    config.prompt_length = new int[16];
    config.qo_indptr_buffer = new int[YPK_MAX_NUM_BATCHED_REQUESTS + 1];
    config.paged_kv_indptr_buffer = new int[YPK_MAX_NUM_BATCHED_REQUESTS + 1];
    config.paged_kv_indices_buffer = new int[YPK_MAX_NUM_PAGES];
    config.paged_kv_last_page_len_buffer = new int[YPK_MAX_NUM_BATCHED_REQUESTS];
    config.page_queue = new int[YPK_MAX_NUM_PAGES];
    config.page_queue_head = new int(0);
    config.page_queue_tail = new int(YPK_MAX_NUM_PAGES);
    config.next_request_id = new int(0);
    config.total_num_requests = 1;  // Only 1 request
    config.max_seq_length = 512;
    config.eos_token_id = 2;
    
    // Initialize all arrays
    for (int i = 0; i < YPK_MAX_NUM_PAGES; ++i) {
        config.page_queue[i] = i;
    }
    for (int i = 0; i < YPK_MAX_NUM_BATCHED_REQUESTS; ++i) {
        config.request_ids[i] = -1;
    }
    for (int i = 0; i < 16; ++i) {
        config.step[i] = 0;
        config.prompt_length[i] = 0;
    }
    memset(config.tokens, 0, 16 * YPK_MAX_SEQ_LENGTH * sizeof(int64_t));
    
    // Setup request 0
    config.prompt_length[0] = 10;
    config.step[0] = 0;
    for (int i = 0; i < 10; ++i) {
        config.tokens[i] = 100 + i;
    }
    
    // Should pick up the request
    bool has_batch = mps_prepare_next_batch(config);
    ASSERT_TRUE(has_batch);
    ASSERT_EQ(config.request_ids[0], 0);
    ASSERT_EQ(*config.next_request_id, 1);
    
    // Cleanup
    delete[] config.request_ids;
    delete[] config.step;
    delete[] config.tokens;
    delete[] config.input_tokens;
    delete[] config.output_tokens;
    delete[] config.prompt_length;
    delete[] config.qo_indptr_buffer;
    delete[] config.paged_kv_indptr_buffer;
    delete[] config.paged_kv_indices_buffer;
    delete[] config.paged_kv_last_page_len_buffer;
    delete[] config.page_queue;
    delete config.page_queue_head;
    delete config.page_queue_tail;
    delete config.next_request_id;
}

// =============================================================================
// MPS Runtime Lifecycle Tests (without actual Metal device)
// =============================================================================

TEST_CASE(mps_runtime_creation) {
    MpsPKRuntime runtime;
    
    // Runtime should not be initialized yet
    // (No Metal device available in test environment)
    
    MpsRuntimeConfig config;
    config.num_workers = 2;
    config.num_local_schedulers = 1;
    
    // Without Metal device, initialization will fail on non-Apple platforms
#ifdef __APPLE__
    // Would need actual Metal device for full test
    // bool result = runtime.initialize(config);
#endif
}

TEST_CASE(mps_task_executor_creation) {
    MpsTaskExecutor executor;
    
    // Without Metal device, initialization will return false
#ifndef __APPLE__
    bool result = executor.initialize(nullptr);
    ASSERT_FALSE(result);
#endif
}

// =============================================================================
// Metal Shader Source Tests
// =============================================================================

TEST_CASE(mps_kernel_source_exists) {
    // Verify kernel source is defined
    ASSERT_NE(MPS_KERNEL_SOURCE, nullptr);
    ASSERT_TRUE(strlen(MPS_KERNEL_SOURCE) > 0);
    
    // Check for key kernel names
    std::string source(MPS_KERNEL_SOURCE);
    ASSERT_TRUE(source.find("embedding_kernel") != std::string::npos);
    ASSERT_TRUE(source.find("rms_norm_kernel") != std::string::npos);
    ASSERT_TRUE(source.find("silu_mul_kernel") != std::string::npos);
    ASSERT_TRUE(source.find("gemm_kernel") != std::string::npos);
    ASSERT_TRUE(source.find("attention_score_kernel") != std::string::npos);
    ASSERT_TRUE(source.find("softmax_kernel") != std::string::npos);
    ASSERT_TRUE(source.find("rotary_embedding_kernel") != std::string::npos);
    ASSERT_TRUE(source.find("argmax_kernel") != std::string::npos);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== MPS Persistent Kernel Runtime Tests ===" << std::endl;
    std::cout << std::endl;
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
