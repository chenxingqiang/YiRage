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
 * @file cpu_pk_runtime.h
 * @brief CPU persistent kernel runtime implementation
 * 
 * This file implements the persistent kernel runtime for CPU execution.
 * It uses C++ threads and atomics to replicate the worker-scheduler model
 * from the CUDA implementation.
 * 
 * Key differences from GPU implementation:
 * - Workers run as separate threads instead of thread blocks
 * - Uses std::atomic for synchronization instead of GPU atomics
 * - Task execution uses OpenMP for parallelization within tasks
 * - No device memory management needed
 */

#pragma once

#include "persistent_kernel/pk_runtime_core.h"
#include <thread>
#include <vector>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace yirage {
namespace persistent_kernel {
namespace cpu {

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
// CPU Task Execution Functions (equivalent to CUDA _execute_task)
// =============================================================================

// Forward declarations for CPU kernels
void cpu_embedding(const PKTaskDesc& task, const PKRuntimeConfig& config);
void cpu_rms_norm(const PKTaskDesc& task, const PKRuntimeConfig& config);
void cpu_rms_norm_linear(const PKTaskDesc& task, const PKRuntimeConfig& config);
void cpu_linear(const PKTaskDesc& task, const PKRuntimeConfig& config);
void cpu_linear_with_residual(const PKTaskDesc& task, const PKRuntimeConfig& config);
void cpu_attention(const PKTaskDesc& task, const PKRuntimeConfig& config);
void cpu_silu_mul(const PKTaskDesc& task, const PKRuntimeConfig& config);
void cpu_silu_mul_linear(const PKTaskDesc& task, const PKRuntimeConfig& config);
void cpu_argmax(const PKTaskDesc& task, const PKRuntimeConfig& config);
void cpu_allreduce(const PKTaskDesc& task, const PKRuntimeConfig& config);

/**
 * @brief Execute a task on CPU
 * 
 * This function dispatches to the appropriate CPU kernel based on task type.
 * Each kernel uses OpenMP for parallel execution.
 */
inline void cpu_execute_task(const PKTaskDesc& task, 
                              const PKRuntimeConfig& config) {
    switch (task.task_type) {
        case PK_TASK_EMBEDDING:
            cpu_embedding(task, config);
            break;
        case PK_TASK_RMS_NORM:
            cpu_rms_norm(task, config);
            break;
        case PK_TASK_RMS_NORM_LINEAR:
            cpu_rms_norm_linear(task, config);
            break;
        case PK_TASK_LINEAR:
            cpu_linear(task, config);
            break;
        case PK_TASK_LINEAR_WITH_RESIDUAL:
            cpu_linear_with_residual(task, config);
            break;
        case PK_TASK_ATTENTION_1:
        case PK_TASK_ATTENTION_2:
            cpu_attention(task, config);
            break;
        case PK_TASK_SILU_MUL:
            cpu_silu_mul(task, config);
            break;
        case PK_TASK_SILU_MUL_LINEAR_WITH_RESIDUAL:
            cpu_silu_mul_linear(task, config);
            break;
        case PK_TASK_ARGMAX:
            cpu_argmax(task, config);
            break;
        case PK_TASK_ALLREDUCE:
            cpu_allreduce(task, config);
            break;
        default:
            break;
    }
}

// =============================================================================
// CPU Batch Preparation (equivalent to CUDA prepare_next_batch)
// =============================================================================

/**
 * @brief Prepare next batch for CPU inference
 */
inline bool cpu_prepare_next_batch(PKRuntimeConfig& config) {
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
            
            // Move output tokens
            for (int j = 0; j < num_tokens; ++j) {
                if (step + j + 1 >= prompt_len &&
                    step + j + 1 < config.max_seq_length) {
                    config.tokens[request_id * YPK_MAX_SEQ_LENGTH + step + j + 1] =
                        config.output_tokens[qo_indptr + j];
                }
            }
            config.step[request_id] = step + num_tokens;
            
            // Check completion
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
            
            for (int j = 0; j < num_new_tokens; ++j) {
                config.input_tokens[num_tokens + j] =
                    config.tokens[request_id * YPK_MAX_SEQ_LENGTH + step + j];
            }
            
            config.request_ids[num_reqs] = request_id;
            config.qo_indptr_buffer[num_reqs] = num_tokens;
            config.paged_kv_indptr_buffer[num_reqs] = num_pages;
            
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
    
    // Update state
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
// CPU Persistent Kernel Runtime
// =============================================================================

/**
 * @brief CPU Persistent Kernel Runtime Configuration
 */
struct CpuRuntimeConfig : public PKRuntimeConfig {
    int num_threads;  // Number of OpenMP threads per task
    
    CpuRuntimeConfig() : PKRuntimeConfig(), num_threads(0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }
};

/**
 * @brief CPU-specific Persistent Kernel Runtime
 * 
 * This runtime uses the base PKRuntime with CPU-specific executor.
 */
class CpuPKRuntime {
public:
    CpuPKRuntime() : initialized_(false) {}
    
    ~CpuPKRuntime() {
        finalize();
    }
    
    bool initialize(CpuRuntimeConfig& config) {
        config_ = &config;
        
#ifdef _OPENMP
        omp_set_num_threads(config.num_threads);
#endif
        
        // Create runtime with CPU executor
        TaskExecutorFn executor = [](const PKTaskDesc& task, 
                                     const PKRuntimeConfig& cfg) {
            cpu_execute_task(task, cfg);
        };
        
        BatchPrepareFn batch_prepare = [](PKRuntimeConfig& cfg) {
            return cpu_prepare_next_batch(cfg);
        };
        
        runtime_.initialize(config, executor, batch_prepare);
        
        initialized_ = true;
        return true;
    }
    
    void launch() {
        if (!initialized_) return;
        
        // Prepare initial batch
        cpu_prepare_next_batch(*config_);
        
        // Set initial event
        config_->sched_queue_next_free_event_id[0].store(1);
        config_->sched_queues[0][0] = config_->num_events - 1;
        config_->sched_queue_last_ready_event_id[0].store(1);
        
        runtime_.launch();
    }
    
    void synchronize() {
        runtime_.synchronize();
    }
    
    void finalize() {
        if (!initialized_) return;
        runtime_.finalize();
        initialized_ = false;
    }
    
private:
    bool initialized_;
    CpuRuntimeConfig* config_;
    PKRuntime runtime_;
};

// =============================================================================
// CPU Kernel Implementations (skeletal)
// =============================================================================

inline void cpu_embedding(const PKTaskDesc& task, const PKRuntimeConfig& config) {
    // Get tensor pointers from task
    const int64_t* input_ids = static_cast<const int64_t*>(task.input_ptrs[0]);
    const float* embedding_table = static_cast<const float*>(task.input_ptrs[1]);
    float* output = static_cast<float*>(task.output_ptrs[0]);
    
    // Get batch dimensions from config
    int num_tokens = config.qo_indptr_buffer[YPK_MAX_NUM_BATCHED_REQUESTS];
    // hidden_dim would come from tensor metadata
    int hidden_dim = 4096;  // Example
    
    #pragma omp parallel for
    for (int i = 0; i < num_tokens; ++i) {
        int token_id = static_cast<int>(input_ids[i]);
        const float* emb_row = embedding_table + token_id * hidden_dim;
        float* out_row = output + i * hidden_dim;
        
        for (int j = 0; j < hidden_dim; ++j) {
            out_row[j] = emb_row[j];
        }
    }
}

inline void cpu_rms_norm(const PKTaskDesc& task, const PKRuntimeConfig& config) {
    const float* input = static_cast<const float*>(task.input_ptrs[0]);
    const float* weight = static_cast<const float*>(task.input_ptrs[1]);
    float* output = static_cast<float*>(task.output_ptrs[0]);
    
    int num_tokens = config.qo_indptr_buffer[YPK_MAX_NUM_BATCHED_REQUESTS];
    int hidden_dim = 4096;
    float eps = 1e-6f;
    
    #pragma omp parallel for
    for (int i = 0; i < num_tokens; ++i) {
        const float* in_row = input + i * hidden_dim;
        float* out_row = output + i * hidden_dim;
        
        // Compute RMS
        float sum_sq = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            sum_sq += in_row[j] * in_row[j];
        }
        float rms = std::sqrt(sum_sq / hidden_dim + eps);
        float inv_rms = 1.0f / rms;
        
        // Normalize
        for (int j = 0; j < hidden_dim; ++j) {
            out_row[j] = in_row[j] * inv_rms * weight[j];
        }
    }
}

inline void cpu_rms_norm_linear(const PKTaskDesc& task, const PKRuntimeConfig& config) {
    // Combined RMS norm + linear projection
    cpu_rms_norm(task, config);
    // Then linear...
}

inline void cpu_linear(const PKTaskDesc& task, const PKRuntimeConfig& config) {
    const float* input = static_cast<const float*>(task.input_ptrs[0]);
    const float* weight = static_cast<const float*>(task.input_ptrs[1]);
    float* output = static_cast<float*>(task.output_ptrs[0]);
    
    int num_tokens = config.qo_indptr_buffer[YPK_MAX_NUM_BATCHED_REQUESTS];
    int in_dim = 4096;
    int out_dim = 4096;
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_tokens; ++i) {
        for (int o = 0; o < out_dim; ++o) {
            float sum = 0.0f;
            const float* in_row = input + i * in_dim;
            const float* w_row = weight + o * in_dim;
            
            for (int k = 0; k < in_dim; ++k) {
                sum += in_row[k] * w_row[k];
            }
            output[i * out_dim + o] = sum;
        }
    }
}

inline void cpu_linear_with_residual(const PKTaskDesc& task, 
                                      const PKRuntimeConfig& config) {
    cpu_linear(task, config);
    // Add residual...
}

inline void cpu_attention(const PKTaskDesc& task, const PKRuntimeConfig& config) {
    // Simplified attention
}

inline void cpu_silu_mul(const PKTaskDesc& task, const PKRuntimeConfig& config) {
    const float* gate = static_cast<const float*>(task.input_ptrs[0]);
    const float* up = static_cast<const float*>(task.input_ptrs[1]);
    float* output = static_cast<float*>(task.output_ptrs[0]);
    
    int num_tokens = config.qo_indptr_buffer[YPK_MAX_NUM_BATCHED_REQUESTS];
    int hidden_dim = 4096;
    int total = num_tokens * hidden_dim;
    
    #pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        float g = gate[i];
        float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
        output[i] = (g * sigmoid_g) * up[i];
    }
}

inline void cpu_silu_mul_linear(const PKTaskDesc& task, 
                                 const PKRuntimeConfig& config) {
    cpu_silu_mul(task, config);
    // Then linear...
}

inline void cpu_argmax(const PKTaskDesc& task, const PKRuntimeConfig& config) {
    const float* logits = static_cast<const float*>(task.input_ptrs[0]);
    int32_t* output = static_cast<int32_t*>(task.output_ptrs[0]);
    
    int num_tokens = config.qo_indptr_buffer[YPK_MAX_NUM_BATCHED_REQUESTS];
    int vocab_size = 32000;
    
    #pragma omp parallel for
    for (int i = 0; i < num_tokens; ++i) {
        const float* row = logits + i * vocab_size;
        int max_idx = 0;
        float max_val = row[0];
        
        for (int v = 1; v < vocab_size; ++v) {
            if (row[v] > max_val) {
                max_val = row[v];
                max_idx = v;
            }
        }
        output[i] = max_idx;
    }
}

inline void cpu_allreduce(const PKTaskDesc& task, const PKRuntimeConfig& config) {
    // Single-process: no-op
}

} // namespace cpu
} // namespace persistent_kernel
} // namespace yirage
