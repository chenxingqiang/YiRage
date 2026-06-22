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

#include "persistent_kernel/pk_runtime_adapter.h"
#include <chrono>
#include <algorithm>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// PKTaskQueue Implementation
// =============================================================================

PKTaskQueue::PKTaskQueue(size_t max_size)
    : max_size_(max_size), terminated_(false) {}

PKTaskQueue::~PKTaskQueue() {
    terminate();
}

bool PKTaskQueue::push(const PKTaskDesc& task) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (terminated_) return false;
    
    // Wait if queue is full
    not_full_.wait(lock, [this]() {
        return queue_.size() < max_size_ || terminated_;
    });
    
    if (terminated_) return false;
    
    queue_.push(task);
    not_empty_.notify_one();
    return true;
}

bool PKTaskQueue::pop(PKTaskDesc& task, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (timeout_ms < 0) {
        // Infinite wait
        not_empty_.wait(lock, [this]() {
            return !queue_.empty() || terminated_;
        });
    } else {
        // Timed wait
        auto timeout = std::chrono::milliseconds(timeout_ms);
        if (!not_empty_.wait_for(lock, timeout, [this]() {
            return !queue_.empty() || terminated_;
        })) {
            return false;  // Timeout
        }
    }
    
    if (terminated_ && queue_.empty()) return false;
    
    task = queue_.front();
    queue_.pop();
    not_full_.notify_one();
    return true;
}

bool PKTaskQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

size_t PKTaskQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void PKTaskQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
    not_full_.notify_all();
}

void PKTaskQueue::terminate() {
    std::lock_guard<std::mutex> lock(mutex_);
    terminated_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
}

// =============================================================================
// PKBatchManager Implementation
// =============================================================================

PKBatchManager::PKBatchManager(const PKBatchConfig& config)
    : config_(config), next_request_id_(0) {
    // Initialize free page queue
    for (int i = 0; i < config_.max_pages; ++i) {
        free_pages_.push(i);
    }
}

PKBatchManager::~PKBatchManager() = default;

int PKBatchManager::add_request(const std::vector<int64_t>& tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_requests_.push(tokens);
    return next_request_id_++;
}

int PKBatchManager::prepare_batch(
    std::vector<int64_t>& input_tokens,
    std::vector<int>& qo_indptr,
    std::vector<int>& kv_indptr,
    std::vector<int>& kv_indices,
    std::vector<int>& kv_last_page_len
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    input_tokens.clear();
    qo_indptr.clear();
    kv_indptr.clear();
    kv_indices.clear();
    kv_last_page_len.clear();
    
    qo_indptr.push_back(0);
    kv_indptr.push_back(0);
    
    int num_active = 0;
    int total_tokens = 0;
    int total_pages = 0;
    
    // Process active requests
    for (auto& req : active_requests_) {
        if (req.completed) continue;
        
        int tokens_to_add = 0;
        if (req.step < req.prompt_length) {
            // Prefill
            tokens_to_add = std::min(
                req.prompt_length - req.step,
                config_.max_tokens_per_batch - total_tokens
            );
        } else {
            // Decode
            tokens_to_add = 1;
        }
        
        if (tokens_to_add == 0) continue;
        
        // Add tokens
        for (int i = 0; i < tokens_to_add; ++i) {
            input_tokens.push_back(req.tokens[req.step + i]);
        }
        total_tokens += tokens_to_add;
        qo_indptr.push_back(total_tokens);
        
        // Add page info
        int num_pages = (req.step + tokens_to_add + config_.page_size - 1) / 
                        config_.page_size;
        
        // Allocate new pages if needed
        while (req.page_indices.size() < static_cast<size_t>(num_pages)) {
            if (free_pages_.empty()) break;
            int page = free_pages_.front();
            free_pages_.pop();
            req.page_indices.push_back(page);
        }
        
        for (int page : req.page_indices) {
            kv_indices.push_back(page);
        }
        total_pages += req.page_indices.size();
        kv_indptr.push_back(total_pages);
        
        int last_page_len = (req.step + tokens_to_add) % config_.page_size;
        if (last_page_len == 0) last_page_len = config_.page_size;
        kv_last_page_len.push_back(last_page_len);
        
        ++num_active;
        
        if (total_tokens >= config_.max_tokens_per_batch) break;
    }
    
    // Add pending requests
    while (num_active < config_.max_batch_size && 
           total_tokens < config_.max_tokens_per_batch &&
           !pending_requests_.empty()) {
        auto tokens = pending_requests_.front();
        pending_requests_.pop();
        
        PKRequestState req;
        req.request_id = next_request_id_++;
        req.prompt_length = tokens.size();
        req.tokens = std::move(tokens);
        req.tokens.resize(config_.max_seq_length);  // Reserve space
        
        int tokens_to_add = std::min(
            req.prompt_length,
            config_.max_tokens_per_batch - total_tokens
        );
        
        for (int i = 0; i < tokens_to_add; ++i) {
            input_tokens.push_back(req.tokens[i]);
        }
        total_tokens += tokens_to_add;
        qo_indptr.push_back(total_tokens);
        
        // Allocate pages
        int num_pages = (tokens_to_add + config_.page_size - 1) / config_.page_size;
        allocate_pages(num_pages, req.page_indices);
        
        for (int page : req.page_indices) {
            kv_indices.push_back(page);
        }
        total_pages += req.page_indices.size();
        kv_indptr.push_back(total_pages);
        
        int last_page_len = tokens_to_add % config_.page_size;
        if (last_page_len == 0) last_page_len = config_.page_size;
        kv_last_page_len.push_back(last_page_len);
        
        active_requests_.push_back(std::move(req));
        ++num_active;
    }
    
    return num_active;
}

void PKBatchManager::update_batch(const std::vector<int64_t>& output_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t token_idx = 0;
    for (auto& req : active_requests_) {
        if (req.completed) continue;
        
        // Get number of tokens for this request
        int num_tokens = 1;  // Simplified; should match prepare_batch
        
        for (int i = 0; i < num_tokens && token_idx < output_tokens.size(); ++i) {
            int64_t new_token = output_tokens[token_idx++];
            int new_pos = req.step + i + 1;
            
            if (new_pos < config_.max_seq_length) {
                req.tokens[new_pos] = new_token;
            }
            
            // Check for completion
            if (new_token == config_.eos_token_id || 
                new_pos >= config_.max_seq_length - 1) {
                req.completed = true;
                completed_requests_.push_back(req.request_id);
                free_pages(req.page_indices);
                break;
            }
        }
        
        req.step += num_tokens;
    }
    
    // Remove completed requests
    active_requests_.erase(
        std::remove_if(active_requests_.begin(), active_requests_.end(),
            [](const PKRequestState& req) { return req.completed; }),
        active_requests_.end()
    );
}

std::vector<int> PKBatchManager::get_completed_requests() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = completed_requests_;
    completed_requests_.clear();
    return result;
}

std::vector<int64_t> PKBatchManager::get_request_output(int request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    // In a real implementation, this would return stored output
    return {};
}

bool PKBatchManager::all_done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_requests_.empty() && pending_requests_.empty();
}

void PKBatchManager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    active_requests_.clear();
    while (!pending_requests_.empty()) pending_requests_.pop();
    completed_requests_.clear();
    while (!free_pages_.empty()) free_pages_.pop();
    for (int i = 0; i < config_.max_pages; ++i) {
        free_pages_.push(i);
    }
    next_request_id_ = 0;
}

int PKBatchManager::allocate_pages(int num_pages, std::vector<int>& page_indices) {
    int allocated = 0;
    while (allocated < num_pages && !free_pages_.empty()) {
        page_indices.push_back(free_pages_.front());
        free_pages_.pop();
        ++allocated;
    }
    return allocated;
}

void PKBatchManager::free_pages(const std::vector<int>& page_indices) {
    for (int page : page_indices) {
        free_pages_.push(page);
    }
}

// =============================================================================
// PKModeAdapter Base Implementation
// =============================================================================

PKModeAdapter::PKModeAdapter(PKBackendInterface* backend, PKMode mode)
    : backend_(backend), mode_(mode) {}

PKModeAdapter::~PKModeAdapter() = default;

// =============================================================================
// PKOfflineModeAdapter Implementation
// =============================================================================

PKOfflineModeAdapter::PKOfflineModeAdapter(PKBackendInterface* backend)
    : PKModeAdapter(backend, PKMode::OFFLINE), processing_(false) {}

PKOfflineModeAdapter::~PKOfflineModeAdapter() {
    finalize();
}

bool PKOfflineModeAdapter::initialize(const PKRuntimeConfig& config) {
    config_ = config;
    
    PKBatchConfig batch_config;
    batch_config.max_batch_size = config.max_num_batched_requests;
    batch_config.max_seq_length = config.max_seq_length;
    batch_config.max_tokens_per_batch = config.max_num_batched_tokens;
    batch_config.page_size = config.page_size;
    batch_config.max_pages = config.max_num_pages;
    batch_config.eos_token_id = config.eos_token_id;
    
    batch_manager_ = std::make_unique<PKBatchManager>(batch_config);
    
    return backend_->initialize(config);
}

void PKOfflineModeAdapter::finalize() {
    backend_->finalize();
    batch_manager_.reset();
}

bool PKOfflineModeAdapter::step() {
    if (!processing_ || batch_manager_->all_done()) {
        return false;
    }
    
    std::vector<int64_t> input_tokens;
    std::vector<int> qo_indptr, kv_indptr, kv_indices, kv_last_page_len;
    
    int num_active = batch_manager_->prepare_batch(
        input_tokens, qo_indptr, kv_indptr, kv_indices, kv_last_page_len
    );
    
    if (num_active == 0) {
        processing_ = false;
        return false;
    }
    
    // Execute forward pass
    backend_->launch_worker_kernel(config_, config_.num_workers, 
                                    config_.threads_per_worker);
    backend_->synchronize();
    
    // Simulated output (in real impl, would come from kernel output)
    std::vector<int64_t> output_tokens(input_tokens.size());
    for (size_t i = 0; i < output_tokens.size(); ++i) {
        output_tokens[i] = i % 100;  // Placeholder
    }
    
    batch_manager_->update_batch(output_tokens);
    
    return !batch_manager_->all_done();
}

void PKOfflineModeAdapter::set_requests(
    const std::vector<std::vector<int64_t>>& requests) {
    for (const auto& req : requests) {
        batch_manager_->add_request(req);
    }
    processing_ = true;
}

std::vector<std::vector<int64_t>> PKOfflineModeAdapter::get_results() {
    // Return collected results
    return {};
}

// =============================================================================
// PKOnlineModeAdapter Implementation
// =============================================================================

PKOnlineModeAdapter::PKOnlineModeAdapter(PKBackendInterface* backend)
    : PKModeAdapter(backend, PKMode::ONLINE),
      current_step_(0), complete_(false) {}

PKOnlineModeAdapter::~PKOnlineModeAdapter() {
    finalize();
}

bool PKOnlineModeAdapter::initialize(const PKRuntimeConfig& config) {
    config_ = config;
    return backend_->initialize(config);
}

void PKOnlineModeAdapter::finalize() {
    backend_->finalize();
}

bool PKOnlineModeAdapter::step() {
    if (complete_ || input_tokens_.empty()) {
        return false;
    }
    
    // Execute one decode step
    backend_->launch_worker_kernel(config_, config_.num_workers,
                                    config_.threads_per_worker);
    backend_->synchronize();
    
    // Simulated output token
    int64_t new_token = current_step_ % 100;  // Placeholder
    output_tokens_.push_back(new_token);
    
    ++current_step_;
    
    if (new_token == config_.eos_token_id || 
        current_step_ >= static_cast<int>(config_.max_seq_length)) {
        complete_ = true;
        return false;
    }
    
    return true;
}

void PKOnlineModeAdapter::set_input(const std::vector<int64_t>& tokens) {
    input_tokens_ = tokens;
    output_tokens_.clear();
    current_step_ = tokens.size();
    complete_ = false;
}

int64_t PKOnlineModeAdapter::get_next_token() {
    if (output_tokens_.empty()) {
        return -1;
    }
    int64_t token = output_tokens_.front();
    output_tokens_.erase(output_tokens_.begin());
    return token;
}

bool PKOnlineModeAdapter::is_complete() const {
    return complete_;
}

// =============================================================================
// PKOnepassModeAdapter Implementation
// =============================================================================

PKOnepassModeAdapter::PKOnepassModeAdapter(PKBackendInterface* backend)
    : PKModeAdapter(backend, PKMode::ONEPASS), executed_(false) {}

PKOnepassModeAdapter::~PKOnepassModeAdapter() {
    finalize();
}

bool PKOnepassModeAdapter::initialize(const PKRuntimeConfig& config) {
    config_ = config;
    return backend_->initialize(config);
}

void PKOnepassModeAdapter::finalize() {
    backend_->finalize();
}

bool PKOnepassModeAdapter::step() {
    if (executed_) {
        return false;
    }
    
    backend_->launch_worker_kernel(config_, config_.num_workers,
                                    config_.threads_per_worker);
    backend_->synchronize();
    
    executed_ = true;
    return false;  // One-pass is done after single execution
}

void PKOnepassModeAdapter::forward(
    const std::vector<PKTensorDesc>& inputs,
    std::vector<PKTensorDesc>& outputs) {
    
    executed_ = false;
    step();
}

// =============================================================================
// PKEagerModeAdapter Implementation
// =============================================================================

PKEagerModeAdapter::PKEagerModeAdapter(PKBackendInterface* backend)
    : PKModeAdapter(backend, PKMode::EAGER) {}

PKEagerModeAdapter::~PKEagerModeAdapter() {
    finalize();
}

bool PKEagerModeAdapter::initialize(const PKRuntimeConfig& config) {
    config_ = config;
    task_queue_ = std::make_unique<PKTaskQueue>();
    return backend_->initialize(config);
}

void PKEagerModeAdapter::finalize() {
    if (task_queue_) {
        task_queue_->terminate();
    }
    backend_->finalize();
}

bool PKEagerModeAdapter::step() {
    PKTaskDesc task;
    if (!task_queue_->pop(task, 0)) {
        return false;  // No tasks
    }
    
    execute_task(task);
    return !task_queue_->empty();
}

void PKEagerModeAdapter::execute_task(const PKTaskDesc& task) {
    auto* executor = backend_->get_executor();
    if (executor && executor->supports_task(task.type)) {
        executor->execute(task, config_, nullptr, 0);
    }
    backend_->synchronize();
}

// =============================================================================
// PKGraphModeAdapter Implementation
// =============================================================================

PKGraphModeAdapter::PKGraphModeAdapter(PKBackendInterface* backend)
    : PKModeAdapter(backend, PKMode::GRAPH),
      capturing_(false), current_graph_(nullptr) {}

PKGraphModeAdapter::~PKGraphModeAdapter() {
    finalize();
}

bool PKGraphModeAdapter::initialize(const PKRuntimeConfig& config) {
    config_ = config;
    return backend_->initialize(config);
}

void PKGraphModeAdapter::finalize() {
    if (current_graph_) {
        destroy_graph(current_graph_);
        current_graph_ = nullptr;
    }
    backend_->finalize();
}

bool PKGraphModeAdapter::step() {
    if (!current_graph_) {
        return false;
    }
    
    execute_graph(current_graph_);
    return false;  // Single graph execution per step
}

void PKGraphModeAdapter::begin_capture() {
    recorded_tasks_.clear();
    capturing_ = true;
}

void* PKGraphModeAdapter::end_capture() {
    capturing_ = false;
    
    // Create a graph from recorded tasks
    // In a real implementation, this would create a CUDA graph or equivalent
    void* graph = new std::vector<PKTaskDesc>(recorded_tasks_);
    recorded_tasks_.clear();
    
    return graph;
}

void PKGraphModeAdapter::execute_graph(void* graph) {
    auto* tasks = static_cast<std::vector<PKTaskDesc>*>(graph);
    
    auto* executor = backend_->get_executor();
    for (const auto& task : *tasks) {
        if (executor && executor->supports_task(task.type)) {
            executor->execute(task, config_, nullptr, 0);
        }
    }
    
    backend_->synchronize();
}

void PKGraphModeAdapter::destroy_graph(void* graph) {
    auto* tasks = static_cast<std::vector<PKTaskDesc>*>(graph);
    delete tasks;
}

// =============================================================================
// Factory Function
// =============================================================================

std::unique_ptr<PKModeAdapter> create_mode_adapter(
    PKBackendInterface* backend,
    PKMode mode) {
    
    if (!backend || !backend->supports_mode(mode)) {
        return nullptr;
    }
    
    switch (mode) {
        case PKMode::OFFLINE:
            return std::make_unique<PKOfflineModeAdapter>(backend);
        case PKMode::ONLINE:
            return std::make_unique<PKOnlineModeAdapter>(backend);
        case PKMode::ONEPASS:
            return std::make_unique<PKOnepassModeAdapter>(backend);
        case PKMode::EAGER:
            return std::make_unique<PKEagerModeAdapter>(backend);
        case PKMode::GRAPH:
            return std::make_unique<PKGraphModeAdapter>(backend);
        default:
            return nullptr;
    }
}

} // namespace persistent_kernel
} // namespace yirage
