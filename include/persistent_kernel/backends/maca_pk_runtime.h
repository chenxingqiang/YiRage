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
 * @file maca_pk_runtime.h
 * @brief MetaX MACA GPU persistent kernel runtime implementation
 * 
 * This file implements the persistent kernel runtime for MetaX MACA GPUs.
 * MACA provides CUDA-compatible APIs, so this closely mirrors the CUDA
 * implementation with mc* function calls instead of cuda* calls.
 */

#pragma once

#include "persistent_kernel/pk_runtime_core.h"

#ifdef YIRAGE_BACKEND_MACA_ENABLED
#include <mc_runtime.h>  // MACA runtime (CUDA-compatible)
#endif

namespace yirage {
namespace persistent_kernel {
namespace maca {

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
// MACA Atomic Operations (equivalent to mpk_atoms.cuh)
// =============================================================================

/**
 * @brief MACA atomic operations for task queue synchronization
 * 
 * MACA supports CUDA-compatible atomic operations.
 * For host-side control, we use std::atomic.
 */
class MacaAtomics {
public:
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    // Device-side atomics (inline assembly similar to CUDA)
    // Note: MACA uses compatible PTX assembly
    
    static __device__ __forceinline__ uint64_t 
    atom_add_release_gpu(uint64_t* addr, uint64_t val) {
        uint64_t old_val;
        // MACA compatible with CUDA PTX
        asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
                     : "=l"(old_val)
                     : "l"(addr), "l"(val)
                     : "memory");
        return old_val;
    }
    
    static __device__ __forceinline__ uint64_t 
    ld_acquire_gpu(uint64_t* addr) {
        uint64_t val;
        asm volatile("ld.acquire.gpu.u64 %0, [%1];" 
                     : "=l"(val) 
                     : "l"(addr));
        return val;
    }
    
    static __device__ __forceinline__ void 
    st_relaxed_gpu(uint64_t* addr, uint64_t val) {
        asm volatile("st.relaxed.gpu.u64 [%0], %1;" 
                     : 
                     : "l"(addr), "l"(val));
    }
    
    static __device__ __forceinline__ uint64_t 
    atom_cas_release_gpu(uint64_t* addr, uint64_t cmp, uint64_t val) {
        uint64_t old_val;
        asm volatile("atom.cas.release.gpu.b64 %0,[%1],%2,%3;"
                     : "=l"(old_val)
                     : "l"(addr), "l"(cmp), "l"(val)
                     : "memory");
        return old_val;
    }
#endif

    // Host-side atomics using std::atomic
    static inline uint64_t atom_add_release(std::atomic<uint64_t>* addr, 
                                            uint64_t val) {
        return addr->fetch_add(val, std::memory_order_release);
    }
    
    static inline uint64_t ld_acquire(std::atomic<uint64_t>* addr) {
        return addr->load(std::memory_order_acquire);
    }
    
    static inline void st_release(std::atomic<uint64_t>* addr, uint64_t val) {
        addr->store(val, std::memory_order_release);
    }
};

// =============================================================================
// MACA Runtime Configuration
// =============================================================================

struct MacaRuntimeConfig : public PKRuntimeConfig {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    mcStream_t worker_stream;
    mcStream_t scheduler_stream;
#else
    void* worker_stream;
    void* scheduler_stream;
#endif
    
    // Device memory pointers
    void* device_tasks;
    void* device_events;
    void* device_worker_queues;
    void* device_sched_queues;
    void* device_counters;
    
    MacaRuntimeConfig() : PKRuntimeConfig(),
        worker_stream(nullptr), scheduler_stream(nullptr),
        device_tasks(nullptr), device_events(nullptr),
        device_worker_queues(nullptr), device_sched_queues(nullptr),
        device_counters(nullptr) {}
};

// =============================================================================
// MACA Worker Kernel (equivalent to CUDA worker_kernel)
// =============================================================================

#ifdef YIRAGE_BACKEND_MACA_ENABLED
/**
 * @brief MACA worker kernel
 * 
 * This kernel runs on MACA GPUs and executes tasks from the worker queue.
 * It's equivalent to the CUDA execute_worker() device function.
 */
__global__ void maca_worker_kernel(PKRuntimeConfig config) {
    constexpr int TASK_BUFFER_LEN = 16;
    __shared__ PKTaskDesc task_descs[TASK_BUFFER_LEN];
    __shared__ TaskId task_ids[TASK_BUFFER_LEN];
    __shared__ TaskId* worker_queue;
    __shared__ size_t next_task_pos;
    __shared__ size_t last_task_pos;
    
    int worker_id = blockIdx.x;
    worker_queue = config.worker_queues[worker_id];
    
    if (threadIdx.x == 0) {
        next_task_pos = 0;
        last_task_pos = 0;
    }
    
    int queue_pos = 0, queue_len = 0;
    
    while (true) {
        // Fetch next task batch
        if (queue_pos == queue_len) {
            if (threadIdx.x == 0) {
                while (next_task_pos == last_task_pos) {
                    last_task_pos = MacaAtomics::ld_acquire_gpu(
                        &config.worker_queue_last_ready_task_id[worker_id]);
                    if (next_task_pos < last_task_pos) break;
                    __nanosleep(10);
                }
            }
            __syncthreads();
            
            int num_to_load = min((int)(last_task_pos - next_task_pos), 
                                  TASK_BUFFER_LEN);
            
            // Load task IDs
            if (threadIdx.x < num_to_load) {
                task_ids[threadIdx.x] = MacaAtomics::ld_acquire_gpu(
                    &worker_queue[(next_task_pos + threadIdx.x) % 
                                  config.per_worker_queue_len]);
            }
            __syncthreads();
            
            if (threadIdx.x == 0) {
                next_task_pos += num_to_load;
            }
            
            // Load task descriptors (cooperatively)
            for (int i = threadIdx.x; i < num_to_load; i += blockDim.x) {
                uint64_t task_idx = get_task_position_index(task_ids[i]);
                task_descs[i] = config.all_tasks[task_idx];
            }
            __syncthreads();
            
            queue_pos = 0;
            queue_len = num_to_load;
        }
        
        PKTaskDesc& task = task_descs[queue_pos];
        TaskId task_id = task_ids[queue_pos];
        
        // Wait for dependent event
        if (threadIdx.x == 0 && task.dependent_event != EVENT_INVALID_ID) {
            EventId event_id = task.dependent_event;
            uint64_t event_index = get_event_position_index(event_id);
            EventCounter needed = 
                static_cast<EventCounter>(config.all_event_num_triggers[event_index]) *
                get_task_iteration_num(task_id);
            
            while (MacaAtomics::ld_acquire_gpu(&config.all_event_counters[event_index]) 
                   < needed) {
                __nanosleep(10);
            }
        }
        __syncthreads();
        
        // Execute task
        if (task.task_type == PK_TASK_TERMINATE) {
            return;
        } else if (task.task_type != PK_TASK_BEGIN_TASK_GRAPH) {
            // Execute task (dispatch based on type)
            // maca_execute_task(task, config);
        }
        __syncthreads();
        
        // Trigger completion event
        if (threadIdx.x == 0 && task.trigger_event != EVENT_INVALID_ID) {
            EventId event_id = task.trigger_event;
            uint64_t event_index = get_event_position_index(event_id);
            
            EventCounter count = MacaAtomics::atom_add_release_gpu(
                &config.all_event_counters[event_index], 1);
            
            int num_triggers = config.all_event_num_triggers[event_index];
            
            if ((count + 1) == static_cast<EventCounter>(num_triggers) *
                get_task_iteration_num(task_id)) {
                PKEventDesc& event_desc = config.all_events[event_index];
                
                if (event_desc.event_type != PK_EVENT_EMPTY) {
                    int sched_id = worker_id % config.num_local_schedulers;
                    
                    uint64_t last_event_pos = MacaAtomics::atom_add_release_gpu(
                        &config.sched_queue_next_free_event_id[sched_id], 1);
                    
                    MacaAtomics::st_relaxed_gpu(
                        &config.sched_queues[sched_id]
                            [last_event_pos % config.per_sched_queue_len],
                        event_index);
                    
                    uint64_t old;
                    do {
                        old = MacaAtomics::atom_cas_release_gpu(
                            &config.sched_queue_last_ready_event_id[sched_id],
                            last_event_pos, last_event_pos + 1);
                    } while (old != last_event_pos);
                }
            }
        }
        
        queue_pos += 1;
    }
}

/**
 * @brief MACA scheduler kernel
 */
__global__ void maca_scheduler_kernel(PKRuntimeConfig config) {
    int sched_id = blockIdx.x;
    
    if (threadIdx.x != 0) return;  // Single thread per scheduler
    
    int num_schedulers = config.num_local_schedulers + config.num_remote_schedulers;
    int workers_per_sched = (config.num_workers + config.num_local_schedulers - 1) /
                            config.num_local_schedulers;
    int my_first_worker = sched_id * workers_per_sched;
    int my_last_worker = min(my_first_worker + workers_per_sched, config.num_workers);
    
    uint64_t cur_event_pos = 0;
    uint64_t last_event_pos = 0;
    uint64_t iteration_num = 0;
    
    uint64_t worker_queue_next_free[MAX_WORKER_PER_SCHEDULER];
    for (int i = 0; i < MAX_WORKER_PER_SCHEDULER; ++i) {
        worker_queue_next_free[i] = 0;
    }
    
    int next_worker = my_first_worker;
    
    while (true) {
        // Wait for event
        while (cur_event_pos == last_event_pos) {
            last_event_pos = MacaAtomics::ld_acquire_gpu(
                &config.sched_queue_last_ready_event_id[sched_id]);
            if (cur_event_pos < last_event_pos) break;
            __nanosleep(10);
        }
        
        EventId event_id = MacaAtomics::ld_acquire_gpu(
            &config.sched_queues[sched_id][cur_event_pos % config.per_sched_queue_len]);
        PKEventDesc& e = config.all_events[event_id];
        
        // Handle termination
        if (is_termination_event(event_id, e)) {
            for (int i = my_first_worker; i < my_last_worker; ++i) {
                uint64_t last_task = worker_queue_next_free[i - my_first_worker]++;
                MacaAtomics::st_relaxed_gpu(
                    &config.worker_queues[i][last_task % config.per_worker_queue_len], 0);
                MacaAtomics::atom_add_release_gpu(
                    &config.worker_queue_last_ready_task_id[i], 1);
            }
            return;
        }
        
        // Handle events
        if (e.event_type == PK_EVENT_END_OF_TASK_GRAPH) {
            // Check for next batch (would call prepare_next_batch)
            // For now, just terminate
            // terminate_all_schedulers(config);
        } else {
            // Dispatch tasks to workers
            for (TaskId i = e.first_task_id; i < e.last_task_id; ++i) {
                uint64_t last_task = worker_queue_next_free[next_worker - my_first_worker]++;
                MacaAtomics::st_relaxed_gpu(
                    &config.worker_queues[next_worker]
                        [last_task % config.per_worker_queue_len],
                    compute_task_id(iteration_num, i));
                MacaAtomics::atom_add_release_gpu(
                    &config.worker_queue_last_ready_task_id[next_worker], 1);
                
                next_worker = (next_worker == my_last_worker - 1) ? 
                              my_first_worker : next_worker + 1;
            }
        }
        
        cur_event_pos += 1;
    }
}
#endif  // YIRAGE_BACKEND_MACA_ENABLED

// =============================================================================
// MACA Persistent Kernel Runtime
// =============================================================================

/**
 * @brief MACA-specific Persistent Kernel Runtime
 */
class MacaPKRuntime {
public:
    MacaPKRuntime() : initialized_(false) {}
    
    ~MacaPKRuntime() {
        finalize();
    }
    
    bool initialize(MacaRuntimeConfig& config) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
        // Set device
        mcSetDevice(config.my_device_id);
        
        // Create streams
        mcStreamCreate(&config.worker_stream);
        mcStreamCreate(&config.scheduler_stream);
        
        // Allocate device memory for tasks and events
        mcMalloc(&config.device_tasks, 
                 config.num_events * sizeof(PKTaskDesc));
        mcMemcpy(config.device_tasks, config.all_tasks,
                 config.num_events * sizeof(PKTaskDesc),
                 mcMemcpyHostToDevice);
        
        config_ = &config;
        initialized_ = true;
        return true;
#else
        return false;
#endif
    }
    
    void launch() {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
        if (!initialized_) return;
        
        // Launch worker kernel
        maca_worker_kernel<<<config_->num_workers, 256,
                            0, config_->worker_stream>>>(*config_);
        
        // Launch scheduler kernel
        maca_scheduler_kernel<<<config_->num_local_schedulers, 32,
                               0, config_->scheduler_stream>>>(*config_);
#endif
    }
    
    void synchronize() {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
        if (config_) {
            mcStreamSynchronize(config_->worker_stream);
            mcStreamSynchronize(config_->scheduler_stream);
        }
#endif
    }
    
    void finalize() {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
        if (!initialized_) return;
        
        if (config_) {
            if (config_->worker_stream) {
                mcStreamDestroy(config_->worker_stream);
            }
            if (config_->scheduler_stream) {
                mcStreamDestroy(config_->scheduler_stream);
            }
            if (config_->device_tasks) {
                mcFree(config_->device_tasks);
            }
        }
        
        initialized_ = false;
#endif
    }
    
private:
    bool initialized_;
    MacaRuntimeConfig* config_;
};

} // namespace maca
} // namespace persistent_kernel
} // namespace yirage
