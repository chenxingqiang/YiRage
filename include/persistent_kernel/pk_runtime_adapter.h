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

#pragma once

#include "persistent_kernel/pk_backend_interface.h"
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// Task Queue for Mode Execution
// =============================================================================

/**
 * @brief Thread-safe task queue for persistent kernel execution
 */
class PKTaskQueue {
public:
    PKTaskQueue(size_t max_size = 1024);
    ~PKTaskQueue();
    
    /**
     * @brief Push a task to the queue
     * @return true if successful, false if queue is full
     */
    bool push(const PKTaskDesc& task);
    
    /**
     * @brief Pop a task from the queue
     * @param task Output task descriptor
     * @param timeout_ms Timeout in milliseconds (-1 for infinite)
     * @return true if task retrieved, false if timeout or empty
     */
    bool pop(PKTaskDesc& task, int timeout_ms = -1);
    
    /**
     * @brief Check if queue is empty
     */
    bool empty() const;
    
    /**
     * @brief Get current queue size
     */
    size_t size() const;
    
    /**
     * @brief Clear the queue
     */
    void clear();
    
    /**
     * @brief Signal termination to waiting threads
     */
    void terminate();
    
private:
    std::queue<PKTaskDesc> queue_;
    size_t max_size_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool terminated_;
};

// =============================================================================
// Batch Manager
// =============================================================================

/**
 * @brief Batch configuration for LLM serving
 */
struct PKBatchConfig {
    int max_batch_size;
    int max_seq_length;
    int max_tokens_per_batch;
    int page_size;
    int max_pages;
    int64_t eos_token_id;
    
    PKBatchConfig()
        : max_batch_size(16),
          max_seq_length(2048),
          max_tokens_per_batch(64),
          page_size(64),
          max_pages(1024),
          eos_token_id(-1) {}
};

/**
 * @brief Request state for batch processing
 */
struct PKRequestState {
    int request_id;
    int step;
    int prompt_length;
    bool completed;
    std::vector<int64_t> tokens;
    std::vector<int> page_indices;
    
    PKRequestState()
        : request_id(-1), step(0), prompt_length(0), completed(false) {}
};

/**
 * @brief Batch manager for LLM request handling
 */
class PKBatchManager {
public:
    explicit PKBatchManager(const PKBatchConfig& config);
    ~PKBatchManager();
    
    /**
     * @brief Add a new request to be processed
     * @param tokens Input token IDs
     * @return Request ID, or -1 if queue is full
     */
    int add_request(const std::vector<int64_t>& tokens);
    
    /**
     * @brief Prepare next batch for processing
     * @param input_tokens Output: tokens to process
     * @param qo_indptr Output: query offset pointers
     * @param kv_indptr Output: key-value offset pointers
     * @param kv_indices Output: page indices
     * @param kv_last_page_len Output: last page lengths
     * @return Number of active requests in batch
     */
    int prepare_batch(
        std::vector<int64_t>& input_tokens,
        std::vector<int>& qo_indptr,
        std::vector<int>& kv_indptr,
        std::vector<int>& kv_indices,
        std::vector<int>& kv_last_page_len
    );
    
    /**
     * @brief Update batch with generated tokens
     * @param output_tokens Generated token IDs
     */
    void update_batch(const std::vector<int64_t>& output_tokens);
    
    /**
     * @brief Get completed requests
     * @return Vector of completed request IDs
     */
    std::vector<int> get_completed_requests();
    
    /**
     * @brief Get output tokens for a completed request
     */
    std::vector<int64_t> get_request_output(int request_id);
    
    /**
     * @brief Check if all requests are processed
     */
    bool all_done() const;
    
    /**
     * @brief Reset manager for new session
     */
    void reset();
    
private:
    PKBatchConfig config_;
    std::vector<PKRequestState> active_requests_;
    std::queue<std::vector<int64_t>> pending_requests_;
    std::vector<int> completed_requests_;
    std::queue<int> free_pages_;
    int next_request_id_;
    mutable std::mutex mutex_;
    
    int allocate_pages(int num_pages, std::vector<int>& page_indices);
    void free_pages(const std::vector<int>& page_indices);
};

// =============================================================================
// Mode-Specific Runtime Adapters
// =============================================================================

/**
 * @brief Base class for mode-specific runtime adaptation
 */
class PKModeAdapter {
public:
    PKModeAdapter(PKBackendInterface* backend, PKMode mode);
    virtual ~PKModeAdapter();
    
    /**
     * @brief Initialize the adapter
     */
    virtual bool initialize(const PKRuntimeConfig& config) = 0;
    
    /**
     * @brief Finalize the adapter
     */
    virtual void finalize() = 0;
    
    /**
     * @brief Execute one step/iteration
     * @return true if more work to do, false if complete
     */
    virtual bool step() = 0;
    
    /**
     * @brief Get the mode this adapter handles
     */
    PKMode get_mode() const { return mode_; }
    
    /**
     * @brief Get the backend being used
     */
    PKBackendInterface* get_backend() const { return backend_; }
    
protected:
    PKBackendInterface* backend_;
    PKMode mode_;
    PKRuntimeConfig config_;
};

/**
 * @brief OFFLINE mode adapter - batch processing
 */
class PKOfflineModeAdapter : public PKModeAdapter {
public:
    PKOfflineModeAdapter(PKBackendInterface* backend);
    ~PKOfflineModeAdapter() override;
    
    bool initialize(const PKRuntimeConfig& config) override;
    void finalize() override;
    bool step() override;
    
    /**
     * @brief Set the batch of requests to process
     */
    void set_requests(const std::vector<std::vector<int64_t>>& requests);
    
    /**
     * @brief Get results for all requests
     */
    std::vector<std::vector<int64_t>> get_results();
    
private:
    std::unique_ptr<PKBatchManager> batch_manager_;
    bool processing_;
};

/**
 * @brief ONLINE mode adapter - single request streaming
 */
class PKOnlineModeAdapter : public PKModeAdapter {
public:
    PKOnlineModeAdapter(PKBackendInterface* backend);
    ~PKOnlineModeAdapter() override;
    
    bool initialize(const PKRuntimeConfig& config) override;
    void finalize() override;
    bool step() override;
    
    /**
     * @brief Set input tokens for current request
     */
    void set_input(const std::vector<int64_t>& tokens);
    
    /**
     * @brief Get next generated token
     * @return Token ID, or -1 if no token available yet
     */
    int64_t get_next_token();
    
    /**
     * @brief Check if generation is complete
     */
    bool is_complete() const;
    
private:
    std::vector<int64_t> input_tokens_;
    std::vector<int64_t> output_tokens_;
    int current_step_;
    bool complete_;
};

/**
 * @brief ONEPASS mode adapter - single forward pass
 */
class PKOnepassModeAdapter : public PKModeAdapter {
public:
    PKOnepassModeAdapter(PKBackendInterface* backend);
    ~PKOnepassModeAdapter() override;
    
    bool initialize(const PKRuntimeConfig& config) override;
    void finalize() override;
    bool step() override;
    
    /**
     * @brief Execute a single forward pass
     * @param inputs Input tensors
     * @param outputs Output tensors (preallocated)
     */
    void forward(
        const std::vector<PKTensorDesc>& inputs,
        std::vector<PKTensorDesc>& outputs
    );
    
private:
    bool executed_;
};

/**
 * @brief EAGER mode adapter - immediate execution
 */
class PKEagerModeAdapter : public PKModeAdapter {
public:
    PKEagerModeAdapter(PKBackendInterface* backend);
    ~PKEagerModeAdapter() override;
    
    bool initialize(const PKRuntimeConfig& config) override;
    void finalize() override;
    bool step() override;
    
    /**
     * @brief Execute a task immediately
     */
    void execute_task(const PKTaskDesc& task);
    
private:
    std::unique_ptr<PKTaskQueue> task_queue_;
};

/**
 * @brief GRAPH mode adapter - graph-based execution
 */
class PKGraphModeAdapter : public PKModeAdapter {
public:
    PKGraphModeAdapter(PKBackendInterface* backend);
    ~PKGraphModeAdapter() override;
    
    bool initialize(const PKRuntimeConfig& config) override;
    void finalize() override;
    bool step() override;
    
    /**
     * @brief Begin recording a task graph
     */
    void begin_capture();
    
    /**
     * @brief End recording and return graph handle
     */
    void* end_capture();
    
    /**
     * @brief Execute a captured graph
     */
    void execute_graph(void* graph);
    
    /**
     * @brief Destroy a captured graph
     */
    void destroy_graph(void* graph);
    
private:
    bool capturing_;
    void* current_graph_;
    std::vector<PKTaskDesc> recorded_tasks_;
};

// =============================================================================
// Factory Function
// =============================================================================

/**
 * @brief Create a mode adapter for the given backend and mode
 */
std::unique_ptr<PKModeAdapter> create_mode_adapter(
    PKBackendInterface* backend,
    PKMode mode
);

} // namespace persistent_kernel
} // namespace yirage
