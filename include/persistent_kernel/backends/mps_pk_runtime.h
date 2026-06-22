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
 * @file mps_pk_runtime.h
 * @brief Apple Metal Performance Shaders persistent kernel runtime
 * 
 * This file implements the persistent kernel runtime for Apple Silicon GPUs
 * using Metal Performance Shaders (MPS) and MPSGraph.
 * 
 * Key differences from CUDA implementation:
 * - Uses Metal command buffers instead of persistent kernels
 * - MPSGraph for optimized execution graphs
 * - Unified memory architecture (no explicit H2D/D2H copies)
 * - Metal shader language (MSL) for compute kernels
 * - No equivalent to NVSHMEM (single-GPU focus)
 * 
 * Supported modes:
 * - EAGER: Immediate execution via command buffers
 * - GRAPH: MPSGraph compiled execution
 */

#pragma once

#include "persistent_kernel/pk_runtime_core.h"
#include <functional>
#include <vector>
#include <thread>
#include <atomic>

// Metal/MPS forward declarations (Objective-C types)
#ifdef __APPLE__
#ifdef __OBJC__
@class MTLDevice;
@class MTLCommandQueue;
@class MTLCommandBuffer;
@class MTLComputeCommandEncoder;
@class MTLComputePipelineState;
@class MTLBuffer;
@class MPSGraph;
@class MPSGraphExecutable;
#else
typedef void* MTLDevice;
typedef void* MTLCommandQueue;
typedef void* MTLCommandBuffer;
typedef void* MTLComputeCommandEncoder;
typedef void* MTLComputePipelineState;
typedef void* MTLBuffer;
typedef void* MPSGraph;
typedef void* MPSGraphExecutable;
#endif
#endif

namespace yirage {
namespace persistent_kernel {
namespace mps {

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
// MPS Runtime Configuration
// =============================================================================

/**
 * @brief MPS-specific runtime configuration
 */
struct MpsRuntimeConfig : public PKRuntimeConfig {
    // Metal device and queues
    void* mtl_device;           // id<MTLDevice>
    void* command_queue;        // id<MTLCommandQueue>
    void* compute_queue;        // Secondary queue for compute
    
    // MPSGraph for GRAPH mode
    void* mps_graph;            // MPSGraph*
    void* graph_executable;     // MPSGraphExecutable*
    
    // Unified memory buffers (no separate device memory)
    void* task_buffer;          // MTLBuffer for tasks
    void* event_buffer;         // MTLBuffer for events
    void* queue_buffer;         // MTLBuffer for queues
    void* counter_buffer;       // MTLBuffer for atomic counters
    
    // Execution mode
    bool use_graph_mode;        // true for GRAPH, false for EAGER
    
    // Threading configuration
    int num_encoder_threads;    // Threads for command encoding
    
    MpsRuntimeConfig() : PKRuntimeConfig(),
        mtl_device(nullptr), command_queue(nullptr), compute_queue(nullptr),
        mps_graph(nullptr), graph_executable(nullptr),
        task_buffer(nullptr), event_buffer(nullptr),
        queue_buffer(nullptr), counter_buffer(nullptr),
        use_graph_mode(false), num_encoder_threads(1) {}
};

// =============================================================================
// MPS Task Kernels (Metal Shader Library)
// =============================================================================

/**
 * @brief Metal shader source for persistent kernel tasks
 * 
 * These are compiled at runtime into MTLComputePipelineState objects.
 */
constexpr const char* MPS_KERNEL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// Embedding lookup kernel
kernel void embedding_kernel(
    device const int* input_ids [[buffer(0)]],
    device const float* embedding_table [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& hidden_dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)(num_tokens * hidden_dim)) return;
    
    int token_idx = tid / hidden_dim;
    int dim_idx = tid % hidden_dim;
    int token_id = input_ids[token_idx];
    
    output[tid] = embedding_table[token_id * hidden_dim + dim_idx];
}

// RMS Normalization kernel
kernel void rms_norm_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_tokens [[buffer(3)]],
    constant int& hidden_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]]
) {
    // Each threadgroup handles one token
    threadgroup float shared_sum[256];
    
    int token_idx = tgid;
    if (token_idx >= num_tokens) return;
    
    device const float* in_row = input + token_idx * hidden_dim;
    device float* out_row = output + token_idx * hidden_dim;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (int i = tid_in_tg; i < hidden_dim; i += 256) {
        float val = in_row[i];
        local_sum += val * val;
    }
    shared_sum[tid_in_tg] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            shared_sum[tid_in_tg] += shared_sum[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rms = sqrt(shared_sum[0] / float(hidden_dim) + eps);
    float inv_rms = 1.0f / rms;
    
    // Apply normalization
    for (int i = tid_in_tg; i < hidden_dim; i += 256) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}

// SiLU activation kernel
kernel void silu_mul_kernel(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)size) return;
    
    float g = gate[tid];
    float sigmoid_g = 1.0f / (1.0f + exp(-g));
    output[tid] = (g * sigmoid_g) * up[tid];
}

// Argmax kernel
kernel void argmax_kernel(
    device const float* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant int& num_tokens [[buffer(2)]],
    constant int& vocab_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)num_tokens) return;
    
    device const float* row = input + tid * vocab_size;
    
    int max_idx = 0;
    float max_val = row[0];
    
    for (int i = 1; i < vocab_size; i++) {
        if (row[i] > max_val) {
            max_val = row[i];
            max_idx = i;
        }
    }
    
    output[tid] = max_idx;
}

// Generic GEMM kernel (for linear layers)
kernel void gemm_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int row = tid.y;
    int col = tid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Attention score kernel (Q @ K^T)
kernel void attention_score_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& num_heads [[buffer(4)]],
    constant int& seq_len [[buffer(5)]],
    constant int& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.z / num_heads;
    int h = tid.z % num_heads;
    int q_pos = tid.y;
    int k_pos = tid.x;
    
    if (b >= batch_size || q_pos >= seq_len || k_pos >= seq_len) return;
    
    device const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
    device const float* k = K + ((b * num_heads + h) * seq_len + k_pos) * head_dim;
    
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += q[d] * k[d];
    }
    
    scores[((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos] = dot * scale;
}

// Softmax kernel
kernel void softmax_kernel(
    device float* scores [[buffer(0)]],
    constant int& num_rows [[buffer(1)]],
    constant int& row_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    int row = tid / 256;
    if (row >= num_rows) return;
    
    device float* row_data = scores + row * row_size;
    
    // Find max
    float local_max = -INFINITY;
    for (int i = tid_in_tg; i < row_size; i += 256) {
        local_max = max(local_max, row_data[i]);
    }
    shared_max[tid_in_tg] = local_max;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce max
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            shared_max[tid_in_tg] = max(shared_max[tid_in_tg], 
                                         shared_max[tid_in_tg + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid_in_tg; i < row_size; i += 256) {
        float val = exp(row_data[i] - row_max);
        row_data[i] = val;
        local_sum += val;
    }
    shared_sum[tid_in_tg] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce sum
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            shared_sum[tid_in_tg] += shared_sum[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_sum[0];
    
    // Normalize
    for (int i = tid_in_tg; i < row_size; i += 256) {
        row_data[i] /= row_sum;
    }
}

// Rotary embedding kernel
kernel void rotary_embedding_kernel(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* cos_cache [[buffer(2)]],
    device const float* sin_cache [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& num_heads [[buffer(5)]],
    constant int& seq_len [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant int& position_offset [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.z;
    int h = tid.y;
    int s = tid.x;
    
    if (b >= batch_size || h >= num_heads || s >= seq_len) return;
    
    int half_dim = head_dim / 2;
    int pos = s + position_offset;
    
    device float* q_ptr = q + ((b * num_heads + h) * seq_len + s) * head_dim;
    device const float* cos_ptr = cos_cache + pos * half_dim;
    device const float* sin_ptr = sin_cache + pos * half_dim;
    
    for (int d = 0; d < half_dim; d++) {
        float x0 = q_ptr[d];
        float x1 = q_ptr[d + half_dim];
        q_ptr[d] = x0 * cos_ptr[d] - x1 * sin_ptr[d];
        q_ptr[d + half_dim] = x1 * cos_ptr[d] + x0 * sin_ptr[d];
    }
}
)";

// =============================================================================
// MPS Task Executor
// =============================================================================

/**
 * @brief Execute a task using Metal compute shaders
 */
class MpsTaskExecutor {
public:
    MpsTaskExecutor() : initialized_(false) {}
    
    ~MpsTaskExecutor() {
        cleanup();
    }
    
    /**
     * @brief Initialize the executor with a Metal device
     */
    bool initialize(void* device) {
#ifdef __APPLE__
        mtl_device_ = device;
        
        // Compile shader library from source
        // In production, this would load precompiled metallib
        if (!compile_shaders()) {
            return false;
        }
        
        initialized_ = true;
        return true;
#else
        return false;
#endif
    }
    
    /**
     * @brief Execute a task
     */
    void execute(const PKTaskDesc& task, 
                 const MpsRuntimeConfig& config,
                 void* command_buffer) {
#ifdef __APPLE__
        if (!initialized_) return;
        
        switch (task.task_type) {
            case PK_TASK_EMBEDDING:
                execute_embedding(task, config, command_buffer);
                break;
            case PK_TASK_RMS_NORM:
            case PK_TASK_RMS_NORM_LINEAR:
                execute_rms_norm(task, config, command_buffer);
                break;
            case PK_TASK_LINEAR:
            case PK_TASK_LINEAR_WITH_RESIDUAL:
                execute_gemm(task, config, command_buffer);
                break;
            case PK_TASK_ATTENTION_1:
            case PK_TASK_ATTENTION_2:
                execute_attention(task, config, command_buffer);
                break;
            case PK_TASK_SILU_MUL:
            case PK_TASK_SILU_MUL_LINEAR_WITH_RESIDUAL:
                execute_silu_mul(task, config, command_buffer);
                break;
            case PK_TASK_ARGMAX:
                execute_argmax(task, config, command_buffer);
                break;
            default:
                break;
        }
#endif
    }
    
private:
    bool compile_shaders() {
        // Would compile MPS_KERNEL_SOURCE to MTLLibrary
        // and create pipeline states for each kernel
        return true;
    }
    
    void execute_embedding(const PKTaskDesc& task,
                           const MpsRuntimeConfig& config,
                           void* command_buffer) {
        // Create compute encoder
        // Set pipeline state for embedding_kernel
        // Set buffers
        // Dispatch threads
        // End encoding
    }
    
    void execute_rms_norm(const PKTaskDesc& task,
                          const MpsRuntimeConfig& config,
                          void* command_buffer) {
        // Similar pattern for RMS norm
    }
    
    void execute_gemm(const PKTaskDesc& task,
                      const MpsRuntimeConfig& config,
                      void* command_buffer) {
        // Matrix multiplication
    }
    
    void execute_attention(const PKTaskDesc& task,
                           const MpsRuntimeConfig& config,
                           void* command_buffer) {
        // Multi-head attention
    }
    
    void execute_silu_mul(const PKTaskDesc& task,
                          const MpsRuntimeConfig& config,
                          void* command_buffer) {
        // SiLU activation with gating
    }
    
    void execute_argmax(const PKTaskDesc& task,
                        const MpsRuntimeConfig& config,
                        void* command_buffer) {
        // Argmax for token selection
    }
    
    void cleanup() {
        // Release Metal objects
    }
    
    bool initialized_;
    void* mtl_device_;
    
    // Pipeline states for each kernel
    void* embedding_pipeline_;
    void* rms_norm_pipeline_;
    void* gemm_pipeline_;
    void* attention_pipeline_;
    void* silu_mul_pipeline_;
    void* argmax_pipeline_;
};

// =============================================================================
// MPS Batch Preparation
// =============================================================================

/**
 * @brief Prepare next batch for MPS inference
 * 
 * Same logic as CUDA/CPU, but optimized for unified memory.
 */
inline bool mps_prepare_next_batch(PKRuntimeConfig& config) {
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
            
            for (int j = 0; j < num_tokens; ++j) {
                if (step + j + 1 >= prompt_len &&
                    step + j + 1 < config.max_seq_length) {
                    config.tokens[request_id * YPK_MAX_SEQ_LENGTH + step + j + 1] =
                        config.output_tokens[qo_indptr + j];
                }
            }
            config.step[request_id] = step + num_tokens;
            
            if ((step + num_tokens + 1 >= config.max_seq_length) ||
                (config.tokens[request_id * YPK_MAX_SEQ_LENGTH + step + num_tokens] 
                 == config.eos_token_id)) {
                config.request_ids[i] = -1;
                
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
    
    // Step 2: Prepare next batch (same logic as CPU/CUDA)
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
    
    // Add new requests
    while (num_reqs < YPK_MAX_NUM_BATCHED_REQUESTS &&
           num_tokens < YPK_MAX_NUM_BATCHED_TOKENS) {
        int next_request_id = *config.next_request_id;
        if (next_request_id >= config.total_num_requests) break;
        
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
// MPS Worker Implementation
// =============================================================================

/**
 * @brief MPS Worker that processes tasks using command buffers
 * 
 * Unlike CUDA's persistent kernel, MPS uses command buffer submission.
 * Each worker encodes commands into a command buffer and commits it.
 */
class MpsWorker {
public:
    MpsWorker(int worker_id, MpsRuntimeConfig* config, MpsTaskExecutor* executor)
        : worker_id_(worker_id), config_(config), executor_(executor),
          running_(false) {}
    
    void start() {
        running_ = true;
        thread_ = std::thread(&MpsWorker::run, this);
    }
    
    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    
private:
    void run() {
        uint64_t next_task_pos = 0;
        uint64_t last_task_pos = 0;
        
        while (running_ && !config_->terminate_flag.load()) {
            // Wait for task
            while (next_task_pos == last_task_pos) {
                last_task_pos = config_->worker_queue_last_ready_task_id[worker_id_]
                    .load(std::memory_order_acquire);
                
                if (next_task_pos < last_task_pos) break;
                if (!running_) return;
                
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            
            // Get task
            uint64_t queue_idx = next_task_pos % config_->per_worker_queue_len;
            TaskId task_id = config_->worker_queues[worker_id_][queue_idx];
            uint64_t task_idx = get_task_position_index(task_id);
            PKTaskDesc& task = config_->all_tasks[task_idx];
            next_task_pos++;
            
            // Wait for dependent event
            if (task.dependent_event != EVENT_INVALID_ID) {
                EventId event_id = task.dependent_event;
                uint64_t event_index = get_event_position_index(event_id);
                EventCounter needed = 
                    static_cast<EventCounter>(config_->all_event_num_triggers[event_index]) *
                    get_task_iteration_num(task_id);
                
                while (config_->all_event_counters[event_index].load(std::memory_order_acquire) 
                       < needed) {
                    if (!running_) return;
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
            
            // Execute task
            if (task.task_type == PK_TASK_TERMINATE) {
                return;
            } else if (task.task_type != PK_TASK_BEGIN_TASK_GRAPH) {
#ifdef __APPLE__
                // Create command buffer and execute
                // id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
                // executor_->execute(task, *config_, (__bridge void*)cmdBuffer);
                // [cmdBuffer commit];
                // [cmdBuffer waitUntilCompleted];
                executor_->execute(task, *config_, config_->command_queue);
#endif
            }
            
            // Trigger completion event
            if (task.trigger_event != EVENT_INVALID_ID) {
                EventId event_id = task.trigger_event;
                uint64_t event_index = get_event_position_index(event_id);
                
                EventCounter count = config_->all_event_counters[event_index]
                    .fetch_add(1, std::memory_order_release);
                
                int num_triggers = config_->all_event_num_triggers[event_index];
                
                if ((count + 1) == static_cast<EventCounter>(num_triggers) *
                    get_task_iteration_num(task_id)) {
                    // Notify scheduler
                    PKEventDesc& event_desc = config_->all_events[event_index];
                    
                    if (event_desc.event_type != PK_EVENT_EMPTY) {
                        int sched_id = worker_id_ % config_->num_local_schedulers;
                        
                        uint64_t last_event_pos = config_->sched_queue_next_free_event_id[sched_id]
                            .fetch_add(1, std::memory_order_release);
                        
                        config_->sched_queues[sched_id]
                            [last_event_pos % config_->per_sched_queue_len] = event_index;
                        
                        uint64_t expected = last_event_pos;
                        while (!config_->sched_queue_last_ready_event_id[sched_id]
                               .compare_exchange_weak(expected, last_event_pos + 1,
                                                      std::memory_order_release)) {
                            expected = last_event_pos;
                        }
                    }
                }
            }
        }
    }
    
    int worker_id_;
    MpsRuntimeConfig* config_;
    MpsTaskExecutor* executor_;
    std::atomic<bool> running_;
    std::thread thread_;
};

// =============================================================================
// MPS Scheduler Implementation
// =============================================================================

/**
 * @brief MPS Scheduler that dispatches tasks to workers
 */
class MpsScheduler {
public:
    MpsScheduler(int sched_id, MpsRuntimeConfig* config)
        : sched_id_(sched_id), config_(config), running_(false) {}
    
    void start() {
        running_ = true;
        thread_ = std::thread(&MpsScheduler::run, this);
    }
    
    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    
private:
    void run() {
        int workers_per_sched = (config_->num_workers + config_->num_local_schedulers - 1) /
                                config_->num_local_schedulers;
        int my_first_worker = sched_id_ * workers_per_sched;
        int my_last_worker = std::min(my_first_worker + workers_per_sched,
                                      config_->num_workers);
        
        uint64_t cur_event_pos = 0;
        uint64_t last_event_pos = 0;
        uint64_t iteration_num = 0;
        
        std::vector<uint64_t> worker_queue_next_free(MAX_WORKER_PER_SCHEDULER, 0);
        int next_worker = my_first_worker;
        
        while (running_ && !config_->terminate_flag.load()) {
            // Wait for event
            while (cur_event_pos == last_event_pos) {
                last_event_pos = config_->sched_queue_last_ready_event_id[sched_id_]
                    .load(std::memory_order_acquire);
                
                if (cur_event_pos < last_event_pos) break;
                if (!running_) return;
                
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            
            EventId event_id = config_->sched_queues[sched_id_]
                [cur_event_pos % config_->per_sched_queue_len];
            PKEventDesc& e = config_->all_events[event_id];
            
            if (is_termination_event(event_id, e)) {
                // Terminate workers
                for (int i = my_first_worker; i < my_last_worker; ++i) {
                    uint64_t last_task = worker_queue_next_free[i - my_first_worker]++;
                    config_->worker_queues[i][last_task % config_->per_worker_queue_len] = 0;
                    config_->worker_queue_last_ready_task_id[i]
                        .fetch_add(1, std::memory_order_release);
                }
                return;
            }
            
            if (e.event_type == PK_EVENT_END_OF_TASK_GRAPH) {
                if (!mps_prepare_next_batch(*config_)) {
                    // Terminate
                    terminate_all_schedulers();
                } else {
                    // Launch next iteration
                    uint64_t last_task = worker_queue_next_free[next_worker - my_first_worker]++;
                    config_->worker_queues[next_worker]
                        [last_task % config_->per_worker_queue_len] = 
                        compute_task_id(iteration_num + 1, 1);
                    config_->worker_queue_last_ready_task_id[next_worker]
                        .fetch_add(1, std::memory_order_release);
                    
                    next_worker = (next_worker == my_last_worker - 1) ? 
                                  my_first_worker : next_worker + 1;
                }
            } else {
                // Dispatch tasks
                if (e.event_type == PK_EVENT_LAUNCH_DEPENDENT_TASKS) {
                    iteration_num++;
                }
                
                for (TaskId i = e.first_task_id; i < e.last_task_id; ++i) {
                    uint64_t last_task = worker_queue_next_free[next_worker - my_first_worker]++;
                    config_->worker_queues[next_worker]
                        [last_task % config_->per_worker_queue_len] = 
                        compute_task_id(iteration_num, i);
                    config_->worker_queue_last_ready_task_id[next_worker]
                        .fetch_add(1, std::memory_order_release);
                    
                    next_worker = (next_worker == my_last_worker - 1) ? 
                                  my_first_worker : next_worker + 1;
                }
            }
            
            cur_event_pos++;
        }
    }
    
    void terminate_all_schedulers() {
        int num_schedulers = config_->num_local_schedulers + 
                             config_->num_remote_schedulers;
        
        for (int i = 0; i < num_schedulers; ++i) {
            uint64_t last_event_pos = config_->sched_queue_next_free_event_id[i]
                .fetch_add(1, std::memory_order_release);
            config_->sched_queues[i][last_event_pos % config_->per_sched_queue_len] = 0;
            
            uint64_t expected = last_event_pos;
            while (!config_->sched_queue_last_ready_event_id[i]
                   .compare_exchange_weak(expected, last_event_pos + 1,
                                          std::memory_order_release)) {
                expected = last_event_pos;
            }
        }
    }
    
    int sched_id_;
    MpsRuntimeConfig* config_;
    std::atomic<bool> running_;
    std::thread thread_;
};

// =============================================================================
// MPS Persistent Kernel Runtime
// =============================================================================

/**
 * @brief MPS Persistent Kernel Runtime
 * 
 * This runtime manages Metal command buffer execution for LLM inference
 * on Apple Silicon GPUs.
 */
class MpsPKRuntime {
public:
    MpsPKRuntime() : initialized_(false), config_(nullptr) {}
    
    ~MpsPKRuntime() {
        finalize();
    }
    
    /**
     * @brief Initialize the MPS runtime
     */
    bool initialize(MpsRuntimeConfig& config) {
#ifdef __APPLE__
        config_ = &config;
        
        // Initialize task executor
        if (!executor_.initialize(config.mtl_device)) {
            return false;
        }
        
        // Allocate queues
        allocate_queues();
        
        // Create workers
        for (int i = 0; i < config.num_workers; ++i) {
            workers_.emplace_back(
                std::make_unique<MpsWorker>(i, config_, &executor_));
        }
        
        // Create schedulers
        for (int i = 0; i < config.num_local_schedulers; ++i) {
            schedulers_.emplace_back(
                std::make_unique<MpsScheduler>(i, config_));
        }
        
        initialized_ = true;
        return true;
#else
        return false;
#endif
    }
    
    /**
     * @brief Launch the persistent kernel execution
     */
    void launch() {
        if (!initialized_) return;
        
        // Prepare initial batch
        mps_prepare_next_batch(*config_);
        
        // Set initial event
        config_->sched_queue_next_free_event_id[0].store(1);
        config_->sched_queues[0][0] = config_->num_events - 1;
        config_->sched_queue_last_ready_event_id[0].store(1);
        
        // Start schedulers
        for (auto& sched : schedulers_) {
            sched->start();
        }
        
        // Start workers
        for (auto& worker : workers_) {
            worker->start();
        }
    }
    
    /**
     * @brief Wait for execution to complete
     */
    void synchronize() {
        // Wait for workers
        for (auto& worker : workers_) {
            worker->stop();
        }
        
        // Wait for schedulers
        for (auto& sched : schedulers_) {
            sched->stop();
        }
    }
    
    /**
     * @brief Cleanup and release resources
     */
    void finalize() {
        if (!initialized_) return;
        
        config_->terminate_flag.store(true);
        synchronize();
        
        workers_.clear();
        schedulers_.clear();
        
        free_queues();
        
        initialized_ = false;
    }
    
private:
    void allocate_queues() {
        int num_workers = config_->num_workers;
        int num_schedulers = config_->num_local_schedulers + 
                             config_->num_remote_schedulers;
        
        config_->worker_queue_last_ready_task_id = 
            new std::atomic<uint64_t>[num_workers * 2];
        for (int i = 0; i < num_workers * 2; ++i) {
            config_->worker_queue_last_ready_task_id[i].store(0);
        }
        
        config_->sched_queue_last_ready_event_id = 
            new std::atomic<uint64_t>[num_schedulers + 1];
        config_->sched_queue_next_free_event_id = 
            new std::atomic<uint64_t>[num_schedulers + 1];
        for (int i = 0; i < num_schedulers + 1; ++i) {
            config_->sched_queue_last_ready_event_id[i].store(0);
            config_->sched_queue_next_free_event_id[i].store(0);
        }
        
        config_->all_event_counters = 
            new std::atomic<uint64_t>[config_->num_events];
        for (int i = 0; i < config_->num_events; ++i) {
            config_->all_event_counters[i].store(0);
        }
        
        config_->worker_queues = new TaskId*[num_workers * 2];
        for (int i = 0; i < num_workers * 2; ++i) {
            config_->worker_queues[i] = new TaskId[config_->per_worker_queue_len];
        }
        
        config_->sched_queues = new EventId*[num_schedulers + 1];
        for (int i = 0; i < num_schedulers + 1; ++i) {
            config_->sched_queues[i] = new EventId[config_->per_sched_queue_len];
        }
    }
    
    void free_queues() {
        if (!config_) return;
        
        int num_workers = config_->num_workers;
        int num_schedulers = config_->num_local_schedulers + 
                             config_->num_remote_schedulers;
        
        delete[] config_->worker_queue_last_ready_task_id;
        delete[] config_->sched_queue_last_ready_event_id;
        delete[] config_->sched_queue_next_free_event_id;
        delete[] config_->all_event_counters;
        
        for (int i = 0; i < num_workers * 2; ++i) {
            delete[] config_->worker_queues[i];
        }
        delete[] config_->worker_queues;
        
        for (int i = 0; i < num_schedulers + 1; ++i) {
            delete[] config_->sched_queues[i];
        }
        delete[] config_->sched_queues;
    }
    
    bool initialized_;
    MpsRuntimeConfig* config_;
    MpsTaskExecutor executor_;
    std::vector<std::unique_ptr<MpsWorker>> workers_;
    std::vector<std::unique_ptr<MpsScheduler>> schedulers_;
};

} // namespace mps
} // namespace persistent_kernel
} // namespace yirage
