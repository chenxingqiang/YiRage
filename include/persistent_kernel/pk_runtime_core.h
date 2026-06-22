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
 * @file pk_runtime_core.h
 * @brief Core persistent kernel runtime abstractions for multi-backend support
 * 
 * This header provides backend-agnostic abstractions for the persistent kernel
 * runtime, mirroring the CUDA implementation in persistent_kernel.cuh but with
 * support for CPU, Ascend, MACA, and MPS backends.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <functional>
#include <thread>
#include <condition_variable>
#include <mutex>

namespace yirage {
namespace persistent_kernel {
namespace runtime {

// =============================================================================
// Configuration Constants (matching CUDA implementation)
// =============================================================================

constexpr int MAX_INPUTS_PER_TASK = 7;
constexpr int MAX_OUTPUTS_PER_TASK = 3;
constexpr int MAX_NUM_WORKERS = 128;
constexpr int MAX_WORKER_PER_SCHEDULER = 32;
constexpr int YPK_MAX_NUM_BATCHED_REQUESTS = 16;
constexpr int YPK_MAX_NUM_BATCHED_TOKENS = 64;
constexpr int YPK_MAX_NUM_PAGES = 1024;
constexpr int YPK_PAGE_SIZE = 64;
constexpr int YPK_MAX_SEQ_LENGTH = 4096;

// =============================================================================
// Type Aliases (matching CUDA runtime_header.h)
// =============================================================================

using TaskId = uint64_t;
using EventId = uint64_t;
using EventCounter = uint64_t;

constexpr TaskId TASK_INVALID_ID = 0x7fffffffffffffffULL;
constexpr EventId EVENT_INVALID_ID = 0x7ffffffffffffffeULL;
constexpr EventId EVENT_NVSHMEM_TAG = 0x1e00000000000000ULL;

// =============================================================================
// Task Types (matching CUDA implementation)
// =============================================================================

enum PKTaskType {
    PK_TASK_TERMINATE = 0,
    PK_TASK_BEGIN_TASK_GRAPH = 10,
    // Compute tasks
    PK_TASK_EMBEDDING = 101,
    PK_TASK_RMS_NORM_LINEAR = 102,
    PK_TASK_ATTENTION_1 = 103,
    PK_TASK_ATTENTION_2 = 104,
    PK_TASK_SILU_MUL_LINEAR_WITH_RESIDUAL = 105,
    PK_TASK_ALLREDUCE = 106,
    PK_TASK_REDUCE = 107,
    PK_TASK_LINEAR_WITH_RESIDUAL = 108,
    PK_TASK_ARGMAX = 109,
    PK_TASK_LINEAR = 120,
    PK_TASK_RMS_NORM = 119,
    PK_TASK_SILU_MUL = 118,
    PK_TASK_PAGED_ATTENTION_1 = 116,
    PK_TASK_PAGED_ATTENTION_2 = 117,
};

enum PKEventType {
    PK_EVENT_EMPTY = 900,
    PK_EVENT_LAUNCH_TASKS = 901,
    PK_EVENT_LAUNCH_MASSIVE_TASKS = 902,
    PK_EVENT_LAUNCH_DEPENDENT_TASKS = 903,
    PK_EVENT_END_OF_TASK_GRAPH = 910,
    PK_EVENT_TERMINATION = 911,
    PK_EVENT_INVALID = 999,
};

// =============================================================================
// Tensor Descriptor (matching CUDA TensorDesc)
// =============================================================================

struct PKTensorDesc {
    int num_dims;
    void* base_ptr;
    int data_type;
    int dim[8];    // MAX_TENSOR_DIMS
    int stride[8];
};

// =============================================================================
// Event Descriptor (matching CUDA EventDesc)
// =============================================================================

struct PKEventDesc {
    PKEventType event_type;
    int num_triggers;
    TaskId first_task_id;
    TaskId last_task_id;
    
    PKEventDesc()
        : event_type(PK_EVENT_INVALID), num_triggers(0),
          first_task_id(TASK_INVALID_ID), last_task_id(TASK_INVALID_ID) {}
    
    PKEventDesc(PKEventType type, int nt, TaskId f, TaskId l)
        : event_type(type), num_triggers(nt), first_task_id(f), last_task_id(l) {}
};

// =============================================================================
// Task Descriptor (matching CUDA TaskDesc)
// =============================================================================

struct PKTaskDesc {
    PKTaskType task_type;
    unsigned variant_id;
    EventId trigger_event;
    EventId dependent_event;
    void* input_ptrs[MAX_INPUTS_PER_TASK];
    void* output_ptrs[MAX_OUTPUTS_PER_TASK];
    union {
        struct {
            int request_id;
            int head_group;
        };
        int expert_offset;
        size_t xfer_size_in_bytes;
    };
    
    PKTaskDesc() : task_type(PK_TASK_TERMINATE), variant_id(0),
                   trigger_event(EVENT_INVALID_ID), 
                   dependent_event(EVENT_INVALID_ID),
                   request_id(-1), head_group(-1) {
        for (int i = 0; i < MAX_INPUTS_PER_TASK; ++i) input_ptrs[i] = nullptr;
        for (int i = 0; i < MAX_OUTPUTS_PER_TASK; ++i) output_ptrs[i] = nullptr;
    }
};

// =============================================================================
// Runtime Configuration (matching CUDA RuntimeConfig)
// =============================================================================

struct PKRuntimeConfig {
    // Worker/Scheduler counts
    int num_workers;
    int num_local_schedulers;
    int num_remote_schedulers;
    int num_graphs;
    int num_devices;
    int my_device_id;
    int num_events;
    
    // Queue parameters
    uint64_t per_worker_queue_len;
    uint64_t per_sched_queue_len;
    
    // Atomic counters (host-side for CPU, device-side for GPU)
    std::atomic<uint64_t>* worker_queue_last_ready_task_id;
    std::atomic<uint64_t>* sched_queue_last_ready_event_id;
    std::atomic<uint64_t>* sched_queue_next_free_event_id;
    std::atomic<uint64_t>* all_event_counters;
    int* all_event_num_triggers;
    
    // Task and event arrays
    PKTaskDesc* all_tasks;
    PKEventDesc* all_events;
    TaskId** worker_queues;
    EventId** sched_queues;
    TaskId* first_tasks;
    
    // LLM serving metadata
    int* step;
    int64_t* tokens;
    int64_t* input_tokens;
    int64_t* output_tokens;
    int64_t eos_token_id;
    int max_seq_length;
    int* new_token_nums;
    int* qo_indptr_buffer;
    int* paged_kv_indptr_buffer;
    int* paged_kv_indices_buffer;
    int* paged_kv_last_page_len_buffer;
    
    // Offline/Online mode metadata
    int* prompt_length;
    int* request_ids;
    int* page_queue;
    int* page_queue_head;
    int* page_queue_tail;
    int* next_request_id;
    int total_num_requests;
    
    // Profiling
    void* profiler_buffer;
    
    // Execution control
    bool split_worker_scheduler;
    std::atomic<bool> terminate_flag;
    
    PKRuntimeConfig() 
        : num_workers(0), num_local_schedulers(0), num_remote_schedulers(0),
          num_graphs(1), num_devices(1), my_device_id(0), num_events(0),
          per_worker_queue_len(1024), per_sched_queue_len(1024),
          worker_queue_last_ready_task_id(nullptr),
          sched_queue_last_ready_event_id(nullptr),
          sched_queue_next_free_event_id(nullptr),
          all_event_counters(nullptr), all_event_num_triggers(nullptr),
          all_tasks(nullptr), all_events(nullptr),
          worker_queues(nullptr), sched_queues(nullptr), first_tasks(nullptr),
          step(nullptr), tokens(nullptr), input_tokens(nullptr),
          output_tokens(nullptr), eos_token_id(0), max_seq_length(0),
          new_token_nums(nullptr), qo_indptr_buffer(nullptr),
          paged_kv_indptr_buffer(nullptr), paged_kv_indices_buffer(nullptr),
          paged_kv_last_page_len_buffer(nullptr), prompt_length(nullptr),
          request_ids(nullptr), page_queue(nullptr),
          page_queue_head(nullptr), page_queue_tail(nullptr),
          next_request_id(nullptr), total_num_requests(0),
          profiler_buffer(nullptr), split_worker_scheduler(true),
          terminate_flag(false) {}
};

// =============================================================================
// Task ID / Event ID Utilities (matching CUDA implementation)
// =============================================================================

inline uint64_t get_task_iteration_num(TaskId task_id) {
    return (task_id >> 32);
}

inline uint64_t get_task_position_index(TaskId task_id) {
    return (task_id & 0xffffffffULL);
}

inline TaskId compute_task_id(uint64_t iteration_num, uint64_t position_index) {
    return ((iteration_num << 32) | position_index);
}

inline bool is_nvshmem_event(EventId event_id) {
    return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

inline uint64_t get_event_device_id(EventId event_id) {
    return ((event_id >> 32) & 0xffffULL);
}

inline uint64_t get_event_position_index(EventId event_id) {
    return (event_id & 0xffffffffULL);
}

inline bool is_termination_event(uint64_t event_loc, const PKEventDesc& e) {
    return (event_loc == 0);
}

// =============================================================================
// Backend-Agnostic Worker Implementation
// =============================================================================

/**
 * @brief Task executor function type
 */
using TaskExecutorFn = std::function<void(const PKTaskDesc&, const PKRuntimeConfig&)>;

/**
 * @brief CPU Worker Thread Implementation
 * 
 * This mirrors the CUDA execute_worker() device function but runs on CPU.
 */
class PKWorker {
public:
    PKWorker(int worker_id, PKRuntimeConfig* config, TaskExecutorFn executor)
        : worker_id_(worker_id), config_(config), executor_(executor),
          running_(false) {}
    
    void start() {
        running_ = true;
        thread_ = std::thread(&PKWorker::run, this);
    }
    
    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    
    void run() {
        const int queue_buffer_size = 16;
        std::vector<TaskId> task_ids(queue_buffer_size);
        std::vector<PKTaskDesc> task_descs(queue_buffer_size);
        
        uint64_t next_task_pos = 0;
        uint64_t last_task_pos = 0;
        
        int queue_pos = 0, queue_len = 0;
        
        while (running_ && !config_->terminate_flag.load()) {
            // Fetch next task batch if buffer empty
            if (queue_pos == queue_len) {
                // Wait for tasks
                while (next_task_pos == last_task_pos) {
                    last_task_pos = config_->worker_queue_last_ready_task_id[worker_id_]
                        .load(std::memory_order_acquire);
                    
                    if (next_task_pos < last_task_pos) {
                        break;
                    }
                    
                    if (!running_ || config_->terminate_flag.load()) {
                        return;
                    }
                    
                    // Yield to avoid busy spinning
                    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
                }
                
                // Load task IDs from queue
                int num_to_load = std::min(
                    static_cast<int>(last_task_pos - next_task_pos),
                    queue_buffer_size
                );
                
                for (int i = 0; i < num_to_load; ++i) {
                    uint64_t queue_idx = (next_task_pos + i) % config_->per_worker_queue_len;
                    task_ids[i] = config_->worker_queues[worker_id_][queue_idx];
                }
                
                // Load task descriptors
                for (int i = 0; i < num_to_load; ++i) {
                    uint64_t task_idx = get_task_position_index(task_ids[i]);
                    task_descs[i] = config_->all_tasks[task_idx];
                }
                
                next_task_pos += num_to_load;
                queue_pos = 0;
                queue_len = num_to_load;
            }
            
            PKTaskDesc& task = task_descs[queue_pos];
            TaskId task_id = task_ids[queue_pos];
            
            // Wait for dependent event if needed
            if (task.dependent_event != EVENT_INVALID_ID) {
                EventId event_id = task.dependent_event;
                uint64_t event_index = get_event_position_index(event_id);
                EventCounter needed_counts = 
                    static_cast<EventCounter>(config_->all_event_num_triggers[event_index]) *
                    get_task_iteration_num(task_id);
                
                while (config_->all_event_counters[event_index].load(std::memory_order_acquire) 
                       < needed_counts) {
                    if (!running_ || config_->terminate_flag.load()) {
                        return;
                    }
                    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
                }
            }
            
            // Execute task
            if (task.task_type == PK_TASK_TERMINATE) {
                return;
            } else if (task.task_type != PK_TASK_BEGIN_TASK_GRAPH) {
                executor_(task, *config_);
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
                    // Event completed, notify scheduler
                    PKEventDesc& event_desc = config_->all_events[event_index];
                    
                    if (event_desc.event_type != PK_EVENT_EMPTY) {
                        int sched_id = worker_id_ % config_->num_local_schedulers;
                        
                        uint64_t last_event_pos = config_->sched_queue_next_free_event_id[sched_id]
                            .fetch_add(1, std::memory_order_release);
                        
                        config_->sched_queues[sched_id][last_event_pos % config_->per_sched_queue_len] = event_index;
                        
                        // Update ready count
                        uint64_t expected = last_event_pos;
                        while (!config_->sched_queue_last_ready_event_id[sched_id]
                               .compare_exchange_weak(expected, last_event_pos + 1,
                                                      std::memory_order_release)) {
                            expected = last_event_pos;
                        }
                    }
                }
            }
            
            queue_pos += 1;
        }
    }
    
private:
    int worker_id_;
    PKRuntimeConfig* config_;
    TaskExecutorFn executor_;
    std::atomic<bool> running_;
    std::thread thread_;
};

// =============================================================================
// Backend-Agnostic Scheduler Implementation
// =============================================================================

/**
 * @brief Batch preparation function type
 */
using BatchPrepareFn = std::function<bool(PKRuntimeConfig&)>;

/**
 * @brief CPU Scheduler Thread Implementation
 * 
 * This mirrors the CUDA execute_scheduler() device function.
 */
class PKScheduler {
public:
    PKScheduler(int sched_id, PKRuntimeConfig* config, BatchPrepareFn batch_prepare)
        : sched_id_(sched_id), config_(config), batch_prepare_(batch_prepare),
          running_(false) {}
    
    void start() {
        running_ = true;
        thread_ = std::thread(&PKScheduler::run, this);
    }
    
    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    
    void run() {
        int num_schedulers = config_->num_local_schedulers + 
                             config_->num_remote_schedulers;
        
        // Calculate worker range for this scheduler
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
                
                if (cur_event_pos < last_event_pos) {
                    break;
                }
                
                if (!running_ || config_->terminate_flag.load()) {
                    return;
                }
                
                std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            }
            
            // Get event from queue
            EventId event_id = config_->sched_queues[sched_id_]
                [cur_event_pos % config_->per_sched_queue_len];
            PKEventDesc& e = config_->all_events[event_id];
            
            // Check for termination
            if (is_termination_event(event_id, e)) {
                // Terminate all workers
                for (int i = my_first_worker; i < my_last_worker; ++i) {
                    uint64_t last_task_id = worker_queue_next_free[i - my_first_worker]++;
                    config_->worker_queues[i][last_task_id % config_->per_worker_queue_len] = 0;
                    config_->worker_queue_last_ready_task_id[i]
                        .fetch_add(1, std::memory_order_release);
                }
                return;
            }
            
            // Handle end of task graph
            if (e.event_type == PK_EVENT_END_OF_TASK_GRAPH) {
                if (!batch_prepare_(*config_)) {
                    // No more batches, terminate
                    terminate_all_schedulers();
                } else {
                    // Launch begin_task_graph for next iteration
                    uint64_t last_task_id = worker_queue_next_free[next_worker - my_first_worker]++;
                    config_->worker_queues[next_worker]
                        [last_task_id % config_->per_worker_queue_len] = 
                        compute_task_id(iteration_num + 1, 1);
                    config_->worker_queue_last_ready_task_id[next_worker]
                        .fetch_add(1, std::memory_order_release);
                    
                    next_worker = (next_worker == my_last_worker - 1) ? 
                                  my_first_worker : next_worker + 1;
                }
            } else if (e.event_type == PK_EVENT_LAUNCH_DEPENDENT_TASKS ||
                       e.event_type == PK_EVENT_LAUNCH_MASSIVE_TASKS ||
                       e.event_type == PK_EVENT_LAUNCH_TASKS) {
                // Dispatch tasks to workers
                iteration_num = (e.event_type == PK_EVENT_LAUNCH_DEPENDENT_TASKS) ?
                                iteration_num + 1 : iteration_num;
                
                for (TaskId i = e.first_task_id; i < e.last_task_id; ++i) {
                    uint64_t last_task_id = worker_queue_next_free[next_worker - my_first_worker]++;
                    config_->worker_queues[next_worker]
                        [last_task_id % config_->per_worker_queue_len] = 
                        compute_task_id(iteration_num, i);
                    config_->worker_queue_last_ready_task_id[next_worker]
                        .fetch_add(1, std::memory_order_release);
                    
                    next_worker = (next_worker == my_last_worker - 1) ? 
                                  my_first_worker : next_worker + 1;
                }
            }
            
            cur_event_pos += 1;
        }
    }
    
private:
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
    PKRuntimeConfig* config_;
    BatchPrepareFn batch_prepare_;
    std::atomic<bool> running_;
    std::thread thread_;
};

// =============================================================================
// Persistent Kernel Runtime Manager
// =============================================================================

/**
 * @brief Multi-backend Persistent Kernel Runtime
 * 
 * This class manages the worker-scheduler execution model across
 * different hardware backends.
 */
class PKRuntime {
public:
    PKRuntime() : initialized_(false) {}
    
    ~PKRuntime() {
        finalize();
    }
    
    /**
     * @brief Initialize the runtime with given configuration
     */
    bool initialize(PKRuntimeConfig& config, 
                    TaskExecutorFn executor,
                    BatchPrepareFn batch_prepare) {
        if (initialized_) {
            return false;
        }
        
        config_ = &config;
        executor_ = executor;
        batch_prepare_ = batch_prepare;
        
        // Allocate queues
        allocate_queues();
        
        // Create workers
        for (int i = 0; i < config.num_workers; ++i) {
            workers_.emplace_back(
                std::make_unique<PKWorker>(i, config_, executor_));
        }
        
        // Create schedulers
        for (int i = 0; i < config.num_local_schedulers; ++i) {
            schedulers_.emplace_back(
                std::make_unique<PKScheduler>(i, config_, batch_prepare_));
        }
        
        initialized_ = true;
        return true;
    }
    
    /**
     * @brief Start the persistent kernel execution
     */
    void launch() {
        if (!initialized_) return;
        
        // Start schedulers first
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
     * @brief Terminate and cleanup
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
        
        // Allocate worker queue counters
        config_->worker_queue_last_ready_task_id = 
            new std::atomic<uint64_t>[num_workers * 2];
        for (int i = 0; i < num_workers * 2; ++i) {
            config_->worker_queue_last_ready_task_id[i].store(0);
        }
        
        // Allocate scheduler queue counters
        config_->sched_queue_last_ready_event_id = 
            new std::atomic<uint64_t>[num_schedulers + 1];
        config_->sched_queue_next_free_event_id = 
            new std::atomic<uint64_t>[num_schedulers + 1];
        for (int i = 0; i < num_schedulers + 1; ++i) {
            config_->sched_queue_last_ready_event_id[i].store(0);
            config_->sched_queue_next_free_event_id[i].store(0);
        }
        
        // Allocate event counters
        config_->all_event_counters = 
            new std::atomic<uint64_t>[config_->num_events];
        for (int i = 0; i < config_->num_events; ++i) {
            config_->all_event_counters[i].store(0);
        }
        
        // Allocate worker queues
        config_->worker_queues = new TaskId*[num_workers * 2];
        for (int i = 0; i < num_workers * 2; ++i) {
            config_->worker_queues[i] = new TaskId[config_->per_worker_queue_len];
        }
        
        // Allocate scheduler queues
        config_->sched_queues = new EventId*[num_schedulers + 1];
        for (int i = 0; i < num_schedulers + 1; ++i) {
            config_->sched_queues[i] = new EventId[config_->per_sched_queue_len];
        }
    }
    
    void free_queues() {
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
    PKRuntimeConfig* config_;
    TaskExecutorFn executor_;
    BatchPrepareFn batch_prepare_;
    std::vector<std::unique_ptr<PKWorker>> workers_;
    std::vector<std::unique_ptr<PKScheduler>> schedulers_;
};

} // namespace runtime
} // namespace persistent_kernel
} // namespace yirage
