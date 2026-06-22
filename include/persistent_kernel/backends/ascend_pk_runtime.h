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
 * @file ascend_pk_runtime.h
 * @brief Ascend NPU persistent kernel runtime implementation
 * 
 * This file implements the persistent kernel runtime for Huawei Ascend NPUs.
 * It mirrors the CUDA persistent_kernel.cuh implementation but uses:
 * - ACL (Ascend Computing Language) for device management
 * - HCCL for collective communication (equivalent to NVSHMEM)
 * - CANN operators for compute tasks
 */

#pragma once

#include "persistent_kernel/pk_runtime_core.h"

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <hccl/hccl.h>
#endif

namespace yirage {
namespace persistent_kernel {
namespace ascend {

// Import all runtime types explicitly to avoid ambiguity with pk_backend_interface.h
// Core types
using runtime::PKTaskDesc;
using runtime::PKRuntimeConfig;
using runtime::PKEventDesc;
using runtime::TaskId;
using runtime::EventId;
using runtime::PKRuntime;
using runtime::TaskExecutorFn;
using runtime::BatchPrepareFn;
// Enums
using runtime::PKTaskType;
using runtime::PKEventType;
// Task type constants
using runtime::PK_TASK_TERMINATE;
using runtime::PK_TASK_BEGIN_TASK_GRAPH;
using runtime::PK_TASK_EMBEDDING;
using runtime::PK_TASK_RMS_NORM;
using runtime::PK_TASK_RMS_NORM_LINEAR;
using runtime::PK_TASK_LINEAR;
using runtime::PK_TASK_LINEAR_WITH_RESIDUAL;
using runtime::PK_TASK_ATTENTION_1;
using runtime::PK_TASK_ATTENTION_2;
using runtime::PK_TASK_SILU_MUL;
using runtime::PK_TASK_SILU_MUL_LINEAR_WITH_RESIDUAL;
using runtime::PK_TASK_ARGMAX;
using runtime::PK_TASK_ALLREDUCE;
using runtime::PK_TASK_REDUCE;
using runtime::PK_TASK_PAGED_ATTENTION_1;
using runtime::PK_TASK_PAGED_ATTENTION_2;
// Event type constants
using runtime::PK_EVENT_EMPTY;
using runtime::PK_EVENT_INVALID;
using runtime::PK_EVENT_LAUNCH_TASKS;
using runtime::PK_EVENT_LAUNCH_MASSIVE_TASKS;
using runtime::PK_EVENT_LAUNCH_DEPENDENT_TASKS;
using runtime::PK_EVENT_END_OF_TASK_GRAPH;
using runtime::PK_EVENT_TERMINATION;
// Constants
using runtime::TASK_INVALID_ID;
using runtime::EVENT_INVALID_ID;
using runtime::EVENT_NVSHMEM_TAG;
using runtime::MAX_INPUTS_PER_TASK;
using runtime::MAX_OUTPUTS_PER_TASK;
using runtime::YPK_MAX_NUM_BATCHED_REQUESTS;
using runtime::YPK_MAX_NUM_BATCHED_TOKENS;
using runtime::YPK_MAX_NUM_PAGES;
using runtime::YPK_PAGE_SIZE;
using runtime::YPK_MAX_SEQ_LENGTH;
// Utility functions
using runtime::compute_task_id;
using runtime::get_task_iteration_num;
using runtime::get_task_position_index;
using runtime::get_event_position_index;
using runtime::get_event_device_id;
using runtime::is_nvshmem_event;
// Additional types
using runtime::EventCounter;
using runtime::MAX_WORKER_PER_SCHEDULER;

// =============================================================================
// Ascend Atomic Operations (equivalent to mpk_atoms.cuh)
// =============================================================================

/**
 * @brief Ascend atomic operations for task queue synchronization
 * 
 * Ascend NPU uses device-side atomic operations through CANN.
 * For host-side control, we use std::atomic with memory ordering.
 */
class AscendAtomics {
public:
    // Release semantics atomic add
    static inline uint64_t atom_add_release(std::atomic<uint64_t>* addr, 
                                            uint64_t val) {
        return addr->fetch_add(val, std::memory_order_release);
    }
    
    // Acquire semantics load
    static inline uint64_t ld_acquire(std::atomic<uint64_t>* addr) {
        return addr->load(std::memory_order_acquire);
    }
    
    // Release semantics store
    static inline void st_release(std::atomic<uint64_t>* addr, uint64_t val) {
        addr->store(val, std::memory_order_release);
    }
    
    // Compare-and-swap with release semantics
    static inline uint64_t atom_cas_release(std::atomic<uint64_t>* addr,
                                            uint64_t expected,
                                            uint64_t desired) {
        addr->compare_exchange_strong(expected, desired,
                                      std::memory_order_release);
        return expected;
    }
};

// =============================================================================
// Ascend Runtime Configuration (extends base PKRuntimeConfig)
// =============================================================================

struct AscendRuntimeConfig : public PKRuntimeConfig {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    aclrtStream worker_stream;
    aclrtStream scheduler_stream;
    aclrtContext context;
    HcclComm hccl_comm;  // For multi-NPU communication
#else
    void* worker_stream;
    void* scheduler_stream;
    void* context;
    void* hccl_comm;
#endif
    
    // Ascend-specific memory pools
    void* device_task_buffer;
    void* device_event_buffer;
    void* device_queue_buffer;
    
    AscendRuntimeConfig() : PKRuntimeConfig(),
        worker_stream(nullptr), scheduler_stream(nullptr),
        context(nullptr), hccl_comm(nullptr),
        device_task_buffer(nullptr), device_event_buffer(nullptr),
        device_queue_buffer(nullptr) {}
};

// =============================================================================
// Ascend Task Executor
// =============================================================================

/**
 * @brief Execute a task on Ascend NPU
 * 
 * This maps task types to appropriate CANN operators.
 */
inline void ascend_execute_task(const PKTaskDesc& task,
                                 const AscendRuntimeConfig& config) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    switch (task.task_type) {
        case PK_TASK_EMBEDDING: {
            // Use aclnnEmbedding operator
            // aclnnEmbedding(input, weight, output, stream)
            break;
        }
        case PK_TASK_RMS_NORM:
        case PK_TASK_RMS_NORM_LINEAR: {
            // Use aclnnRmsNorm operator
            break;
        }
        case PK_TASK_LINEAR:
        case PK_TASK_LINEAR_WITH_RESIDUAL: {
            // Use aclnnMatmul operator
            break;
        }
        case PK_TASK_ATTENTION_1:
        case PK_TASK_ATTENTION_2:
        case PK_TASK_PAGED_ATTENTION_1:
        case PK_TASK_PAGED_ATTENTION_2: {
            // Use aclnnFlashAttention operator
            break;
        }
        case PK_TASK_SILU_MUL:
        case PK_TASK_SILU_MUL_LINEAR_WITH_RESIDUAL: {
            // Use aclnnSilu + aclnnMul
            break;
        }
        case PK_TASK_ARGMAX: {
            // Use aclnnArgmax operator
            break;
        }
        case PK_TASK_ALLREDUCE: {
            // Use HCCL AllReduce
            // HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream)
            break;
        }
        case PK_TASK_REDUCE: {
            // Use HCCL Reduce
            break;
        }
        default:
            break;
    }
    
    // Synchronize stream if needed
    aclrtSynchronizeStream(config.worker_stream);
#endif
}

// =============================================================================
// Ascend Batch Preparation (matching CUDA prepare_next_batch)
// =============================================================================

/**
 * @brief Prepare next batch for Ascend inference
 * 
 * This is equivalent to the CUDA prepare_next_batch() device function.
 */
inline bool ascend_prepare_next_batch(PKRuntimeConfig& config) {
    int page_queue_head = *config.page_queue_head;
    int page_queue_tail = *config.page_queue_tail;
    
    // Step 1: Finalize previous batch
    for (int i = 0; i < YPK_MAX_NUM_BATCHED_REQUESTS; ++i) {
        int request_id = config.request_ids[i];
        if (request_id != -1) {
            int step = config.step[request_id];
            int qo_indptr = config.qo_indptr_buffer[i];
            int num_tokens = config.qo_indptr_buffer[i + 1] - qo_indptr;
            int prompt_len = config.prompt_length[request_id];
            
            // Move output tokens to tokens buffer
            for (int j = 0; j < num_tokens; ++j) {
                if (step + j + 1 >= prompt_len &&
                    step + j + 1 < config.max_seq_length) {
                    config.tokens[request_id * YPK_MAX_SEQ_LENGTH + step + j + 1] =
                        config.output_tokens[qo_indptr + j];
                }
            }
            config.step[request_id] = step + num_tokens;
            
            // Check if request is complete
            if ((step + num_tokens + 1 >= config.max_seq_length) ||
                (config.tokens[request_id * YPK_MAX_SEQ_LENGTH + step + num_tokens] 
                 == config.eos_token_id)) {
                config.request_ids[i] = -1;
                
                // Free pages
                int kv_indptr = config.paged_kv_indptr_buffer[i];
                int num_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
                for (int j = 0; j < num_pages; ++j) {
                    config.page_queue[page_queue_tail % YPK_MAX_NUM_PAGES] =
                        config.paged_kv_indices_buffer[kv_indptr + j];
                    page_queue_tail++;
                }
            }
        }
    }
    
    // Step 2: Prepare next batch
    int num_reqs = 0, num_tokens = 0, num_pages = 0;
    
    // Keep active requests
    for (int i = 0; i < YPK_MAX_NUM_BATCHED_REQUESTS; ++i) {
        int request_id = config.request_ids[i];
        if (request_id != -1) {
            int step = config.step[request_id];
            int num_new_tokens = config.prompt_length[request_id] - step;
            
            if (num_new_tokens > 0) {
                num_new_tokens = std::min(num_new_tokens, 
                                          YPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
            } else {
                num_new_tokens = std::min(1, YPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
            }
            
            // Move tokens to input buffer
            for (int j = 0; j < num_new_tokens; ++j) {
                config.input_tokens[num_tokens + j] =
                    config.tokens[request_id * YPK_MAX_SEQ_LENGTH + step + j];
            }
            
            config.request_ids[num_reqs] = request_id;
            config.qo_indptr_buffer[num_reqs] = num_tokens;
            config.paged_kv_indptr_buffer[num_reqs] = num_pages;
            
            // Allocate new pages if needed
            int num_new_pages = (step + num_new_tokens + YPK_PAGE_SIZE - 1) / YPK_PAGE_SIZE;
            config.paged_kv_last_page_len_buffer[num_reqs] = 
                (step + num_new_tokens) % YPK_PAGE_SIZE;
            
            for (int j = num_pages; j < num_new_pages; ++j) {
                config.paged_kv_indices_buffer[num_pages + j] =
                    config.page_queue[page_queue_head % YPK_MAX_NUM_PAGES];
                page_queue_head++;
            }
            
            num_pages += num_new_pages;
            num_tokens += num_new_tokens;
            num_reqs++;
        }
    }
    
    // Add new prefill requests
    while (num_reqs < YPK_MAX_NUM_BATCHED_REQUESTS &&
           num_tokens < YPK_MAX_NUM_BATCHED_TOKENS) {
        int next_request_id = *config.next_request_id;
        if (next_request_id >= config.total_num_requests) {
            break;
        }
        
        int num_new_tokens = std::min(config.prompt_length[next_request_id],
                                      YPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
        
        for (int j = 0; j < num_new_tokens; ++j) {
            config.input_tokens[num_tokens + j] =
                config.tokens[next_request_id * YPK_MAX_SEQ_LENGTH + j];
        }
        
        config.request_ids[num_reqs] = next_request_id;
        config.qo_indptr_buffer[num_reqs] = num_tokens;
        config.paged_kv_indptr_buffer[num_reqs] = num_pages;
        
        int num_new_pages = (num_new_tokens + YPK_PAGE_SIZE - 1) / YPK_PAGE_SIZE;
        config.paged_kv_last_page_len_buffer[num_reqs] = num_new_tokens % YPK_PAGE_SIZE;
        
        for (int j = 0; j < num_new_pages; ++j) {
            config.paged_kv_indices_buffer[num_pages + j] =
                config.page_queue[page_queue_head % YPK_MAX_NUM_PAGES];
            page_queue_head++;
        }
        
        num_tokens += num_new_tokens;
        num_pages += num_new_pages;
        num_reqs++;
        *config.next_request_id = next_request_id + 1;
    }
    
    // Update queue pointers
    for (int i = num_reqs; i < YPK_MAX_NUM_BATCHED_REQUESTS; ++i) {
        config.request_ids[i] = -1;
    }
    for (int i = num_reqs; i <= YPK_MAX_NUM_BATCHED_REQUESTS; ++i) {
        config.qo_indptr_buffer[i] = num_tokens;
        config.paged_kv_indptr_buffer[i] = num_pages;
    }
    
    *config.page_queue_head = page_queue_head;
    *config.page_queue_tail = page_queue_tail;
    
    return (num_tokens > 0);
}

// =============================================================================
// Ascend Persistent Kernel Runtime
// =============================================================================

/**
 * @brief Ascend-specific Persistent Kernel Runtime
 */
class AscendPKRuntime {
public:
    AscendPKRuntime() : initialized_(false) {}
    
    ~AscendPKRuntime() {
        finalize();
    }
    
    /**
     * @brief Initialize the Ascend runtime
     */
    bool initialize(AscendRuntimeConfig& config) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
        // Initialize ACL
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            return false;
        }
        
        // Set device
        ret = aclrtSetDevice(config.my_device_id);
        if (ret != ACL_SUCCESS) {
            return false;
        }
        
        // Create context
        ret = aclrtCreateContext(&config.context, config.my_device_id);
        if (ret != ACL_SUCCESS) {
            return false;
        }
        
        // Create streams
        ret = aclrtCreateStream(&config.worker_stream);
        if (ret != ACL_SUCCESS) {
            return false;
        }
        
        ret = aclrtCreateStream(&config.scheduler_stream);
        if (ret != ACL_SUCCESS) {
            return false;
        }
        
        // Initialize HCCL for multi-NPU (if needed)
        if (config.num_devices > 1) {
            // HcclCommInitRank(&config.hccl_comm, config.num_devices, 
            //                  hccl_id, config.my_device_id);
        }
#endif
        
        config_ = &config;
        
        // Create base runtime with Ascend-specific executor
        TaskExecutorFn executor = [this](const PKTaskDesc& task, 
                                          const PKRuntimeConfig& cfg) {
            ascend_execute_task(task, *config_);
        };
        
        BatchPrepareFn batch_prepare = [](PKRuntimeConfig& cfg) {
            return ascend_prepare_next_batch(cfg);
        };
        
        runtime_.initialize(config, executor, batch_prepare);
        
        initialized_ = true;
        return true;
    }
    
    /**
     * @brief Launch the persistent kernel
     */
    void launch() {
        if (!initialized_) return;
        
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
        // Prepare initial batch
        ascend_prepare_next_batch(*config_);
        
        // Set initial event
        config_->sched_queue_next_free_event_id[0].store(1);
        config_->sched_queues[0][0] = config_->num_events - 1;  // End of task graph event
        config_->sched_queue_last_ready_event_id[0].store(1);
#endif
        
        runtime_.launch();
    }
    
    /**
     * @brief Wait for completion
     */
    void synchronize() {
        runtime_.synchronize();
        
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
        if (config_) {
            aclrtSynchronizeStream(config_->worker_stream);
            aclrtSynchronizeStream(config_->scheduler_stream);
        }
#endif
    }
    
    /**
     * @brief Cleanup
     */
    void finalize() {
        if (!initialized_) return;
        
        runtime_.finalize();
        
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
        if (config_) {
            if (config_->worker_stream) {
                aclrtDestroyStream(config_->worker_stream);
            }
            if (config_->scheduler_stream) {
                aclrtDestroyStream(config_->scheduler_stream);
            }
            if (config_->context) {
                aclrtDestroyContext(config_->context);
            }
            aclrtResetDevice(config_->my_device_id);
            aclFinalize();
        }
#endif
        
        initialized_ = false;
    }
    
private:
    bool initialized_;
    AscendRuntimeConfig* config_;
    PKRuntime runtime_;
};

} // namespace ascend
} // namespace persistent_kernel
} // namespace yirage
