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
 * @file pk_multi_backend_demo.cc
 * @brief Multi-backend Persistent Kernel Demo
 * 
 * This demo shows how to use the YiRage Persistent Kernel framework
 * across different hardware backends (CUDA, CPU, Ascend, MACA, MPS).
 * 
 * The demo creates a simple LLM inference task graph and runs it
 * on the best available backend.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

// Include all backends
#include "persistent_kernel/backends/pk_backends.h"

using namespace yirage::persistent_kernel;
using namespace yirage::persistent_kernel::runtime;

// =============================================================================
// Demo Configuration
// =============================================================================

struct DemoConfig {
    int batch_size = 1;
    int seq_len = 128;
    int hidden_dim = 4096;
    int num_heads = 32;
    int head_dim = 128;
    int vocab_size = 32000;
    int num_layers = 32;
    int num_workers = 4;
    int num_schedulers = 1;
    int max_seq_length = 512;
    int num_requests = 4;
};

// =============================================================================
// Task Graph Builder
// =============================================================================

/**
 * @brief Build a simple LLM task graph
 * 
 * Task graph for one transformer layer:
 * 1. Embedding (first layer only)
 * 2. RMS Norm
 * 3. Attention (Q, K, V projections + attention + output projection)
 * 4. Residual Add
 * 5. RMS Norm
 * 6. MLP (gate + up + SiLU*gate + down)
 * 7. Residual Add
 * 8. Argmax (last layer only)
 */
class TaskGraphBuilder {
public:
    TaskGraphBuilder(const DemoConfig& config) : config_(config) {}
    
    void build(std::vector<PKTaskDesc>& tasks, 
               std::vector<PKEventDesc>& events) {
        tasks.clear();
        events.clear();
        
        // Task 0: Termination sentinel
        PKTaskDesc terminate;
        terminate.task_type = PK_TASK_TERMINATE;
        tasks.push_back(terminate);
        
        // Task 1: Begin task graph
        PKTaskDesc begin;
        begin.task_type = PK_TASK_BEGIN_TASK_GRAPH;
        begin.trigger_event = 1;  // Triggers embedding
        tasks.push_back(begin);
        
        // Task 2: Embedding
        PKTaskDesc embedding;
        embedding.task_type = PK_TASK_EMBEDDING;
        embedding.trigger_event = 2;
        embedding.dependent_event = 1;
        tasks.push_back(embedding);
        
        // Task 3-N: Transformer layers
        int task_id = 3;
        for (int layer = 0; layer < config_.num_layers; ++layer) {
            // RMS Norm 1
            PKTaskDesc rms1;
            rms1.task_type = PK_TASK_RMS_NORM;
            rms1.trigger_event = task_id;
            rms1.dependent_event = task_id - 1;
            tasks.push_back(rms1);
            task_id++;
            
            // Attention
            PKTaskDesc attn;
            attn.task_type = PK_TASK_ATTENTION_1;
            attn.trigger_event = task_id;
            attn.dependent_event = task_id - 1;
            tasks.push_back(attn);
            task_id++;
            
            // Linear (attention output)
            PKTaskDesc attn_out;
            attn_out.task_type = PK_TASK_LINEAR_WITH_RESIDUAL;
            attn_out.trigger_event = task_id;
            attn_out.dependent_event = task_id - 1;
            tasks.push_back(attn_out);
            task_id++;
            
            // RMS Norm 2
            PKTaskDesc rms2;
            rms2.task_type = PK_TASK_RMS_NORM;
            rms2.trigger_event = task_id;
            rms2.dependent_event = task_id - 1;
            tasks.push_back(rms2);
            task_id++;
            
            // MLP: Gate + Up
            PKTaskDesc mlp_gate;
            mlp_gate.task_type = PK_TASK_LINEAR;
            mlp_gate.trigger_event = task_id;
            mlp_gate.dependent_event = task_id - 1;
            tasks.push_back(mlp_gate);
            task_id++;
            
            // MLP: SiLU * Gate
            PKTaskDesc silu;
            silu.task_type = PK_TASK_SILU_MUL;
            silu.trigger_event = task_id;
            silu.dependent_event = task_id - 1;
            tasks.push_back(silu);
            task_id++;
            
            // MLP: Down projection with residual
            PKTaskDesc mlp_down;
            mlp_down.task_type = PK_TASK_LINEAR_WITH_RESIDUAL;
            mlp_down.trigger_event = task_id;
            mlp_down.dependent_event = task_id - 1;
            tasks.push_back(mlp_down);
            task_id++;
        }
        
        // Final RMS Norm
        PKTaskDesc final_rms;
        final_rms.task_type = PK_TASK_RMS_NORM;
        final_rms.trigger_event = task_id;
        final_rms.dependent_event = task_id - 1;
        tasks.push_back(final_rms);
        task_id++;
        
        // LM Head (linear to vocab)
        PKTaskDesc lm_head;
        lm_head.task_type = PK_TASK_LINEAR;
        lm_head.trigger_event = task_id;
        lm_head.dependent_event = task_id - 1;
        tasks.push_back(lm_head);
        task_id++;
        
        // Argmax
        PKTaskDesc argmax;
        argmax.task_type = PK_TASK_ARGMAX;
        argmax.trigger_event = task_id;
        argmax.dependent_event = task_id - 1;
        tasks.push_back(argmax);
        task_id++;
        
        // Build events
        // Event 0: Termination
        events.push_back(PKEventDesc(PK_EVENT_TERMINATION, 0, 0, 0));
        
        // Event for each task
        for (int i = 1; i < task_id; ++i) {
            events.push_back(PKEventDesc(PK_EVENT_LAUNCH_TASKS, 1, i, i + 1));
        }
        
        // End of task graph event
        events.push_back(PKEventDesc(PK_EVENT_END_OF_TASK_GRAPH, 1, 0, 0));
        
        std::cout << "Built task graph with " << tasks.size() << " tasks and "
                  << events.size() << " events" << std::endl;
    }
    
private:
    DemoConfig config_;
};

// =============================================================================
// Backend Runner
// =============================================================================

/**
 * @brief Run the demo on a specific backend
 */
class BackendRunner {
public:
    BackendRunner(PKBackendType backend_type, int device_id = 0)
        : backend_type_(backend_type), device_id_(device_id) {}
    
    bool run(const DemoConfig& config,
             const std::vector<PKTaskDesc>& tasks,
             const std::vector<PKEventDesc>& events) {
        
        std::cout << "\n=== Running on " << pk_backend_type_to_name(backend_type_) 
                  << " ===" << std::endl;
        
        // Check backend availability
        auto backend = create_pk_backend(backend_type_, device_id_);
        if (!backend) {
            std::cout << "Backend not available: " 
                      << pk_backend_type_to_name(backend_type_) << std::endl;
            return false;
        }
        
        if (!backend->is_available()) {
            std::cout << "Backend not available on this system" << std::endl;
            return false;
        }
        
        // Print capabilities
        auto caps = backend->get_capabilities();
        std::cout << "Capabilities:" << std::endl;
        std::cout << "  TMA: " << (caps.supports_tma ? "Yes" : "No") << std::endl;
        std::cout << "  Tensor Cores: " << (caps.supports_tensor_cores ? "Yes" : "No") << std::endl;
        std::cout << "  Max Shared Memory: " << caps.max_shared_memory / 1024 << " KB" << std::endl;
        
        // Select execution mode
        PKMode mode = backend->get_default_mode();
        std::cout << "Mode: " << pk_mode_to_name(mode) << std::endl;
        
        // Run based on backend type
        auto start = std::chrono::high_resolution_clock::now();
        
        switch (backend_type_) {
            case PKBackendType::CPU:
                run_cpu(config, tasks, events);
                break;
            case PKBackendType::CUDA:
            case PKBackendType::MACA:
            case PKBackendType::ASCEND:
            case PKBackendType::MPS:
                run_generic(config, tasks, events, backend.get());
                break;
            default:
                std::cout << "Backend not yet implemented" << std::endl;
                return false;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
        
        return true;
    }
    
private:
    void run_cpu(const DemoConfig& config,
                 const std::vector<PKTaskDesc>& tasks,
                 const std::vector<PKEventDesc>& events) {
        
        cpu::CpuRuntimeConfig rt_config;
        rt_config.num_workers = config.num_workers;
        rt_config.num_local_schedulers = config.num_schedulers;
        rt_config.num_remote_schedulers = 0;
        rt_config.num_events = static_cast<int>(events.size());
        rt_config.max_seq_length = config.max_seq_length;
        rt_config.total_num_requests = config.num_requests;
        
        // Allocate buffers
        allocate_runtime_buffers(rt_config, config);
        
        // Copy tasks and events
        rt_config.all_tasks = new PKTaskDesc[tasks.size()];
        std::memcpy(rt_config.all_tasks, tasks.data(), 
                   tasks.size() * sizeof(PKTaskDesc));
        
        rt_config.all_events = new PKEventDesc[events.size()];
        std::memcpy(rt_config.all_events, events.data(),
                   events.size() * sizeof(PKEventDesc));
        
        rt_config.all_event_num_triggers = new int[events.size()];
        for (size_t i = 0; i < events.size(); ++i) {
            rt_config.all_event_num_triggers[i] = events[i].num_triggers;
        }
        
        // Initialize runtime
        cpu::CpuPKRuntime runtime;
        if (runtime.initialize(rt_config)) {
            std::cout << "CPU runtime initialized" << std::endl;
            
            // Launch
            runtime.launch();
            
            // Wait for completion
            runtime.synchronize();
            
            std::cout << "CPU runtime completed" << std::endl;
        }
        
        runtime.finalize();
        
        // Cleanup
        free_runtime_buffers(rt_config);
        delete[] rt_config.all_tasks;
        delete[] rt_config.all_events;
        delete[] rt_config.all_event_num_triggers;
    }
    
    void run_generic(const DemoConfig& config,
                     const std::vector<PKTaskDesc>& tasks,
                     const std::vector<PKEventDesc>& events,
                     PKBackendInterface* backend) {
        
        // Initialize backend
        PKRuntimeConfig rt_config;
        rt_config.mode = backend->get_default_mode();
        rt_config.num_workers = config.num_workers;
        
        if (backend->initialize(rt_config)) {
            std::cout << "Backend initialized: " << backend->get_display_name() << std::endl;
            
            // Launch workers
            backend->launch_worker_kernel(rt_config, config.num_workers, 256);
            
            // Synchronize
            backend->synchronize();
            
            std::cout << "Backend execution completed" << std::endl;
        }
        
        backend->finalize();
    }
    
    void allocate_runtime_buffers(PKRuntimeConfig& config, const DemoConfig& demo) {
        int max_requests = YPK_MAX_NUM_BATCHED_REQUESTS;
        int max_seq = YPK_MAX_SEQ_LENGTH;
        
        config.request_ids = new int[max_requests + 1];
        config.step = new int[demo.num_requests];
        config.tokens = new int64_t[demo.num_requests * max_seq];
        config.input_tokens = new int64_t[YPK_MAX_NUM_BATCHED_TOKENS];
        config.output_tokens = new int64_t[YPK_MAX_NUM_BATCHED_TOKENS];
        config.prompt_length = new int[demo.num_requests];
        config.qo_indptr_buffer = new int[max_requests + 1];
        config.paged_kv_indptr_buffer = new int[max_requests + 1];
        config.paged_kv_indices_buffer = new int[YPK_MAX_NUM_PAGES];
        config.paged_kv_last_page_len_buffer = new int[max_requests];
        config.page_queue = new int[YPK_MAX_NUM_PAGES];
        config.page_queue_head = new int(0);
        config.page_queue_tail = new int(YPK_MAX_NUM_PAGES);
        config.next_request_id = new int(0);
        config.new_token_nums = new int[max_requests];
        
        // Initialize
        for (int i = 0; i < max_requests; ++i) {
            config.request_ids[i] = -1;
        }
        for (int i = 0; i < demo.num_requests; ++i) {
            config.step[i] = 0;
            config.prompt_length[i] = 32;  // Example prompt length
        }
        for (int i = 0; i < YPK_MAX_NUM_PAGES; ++i) {
            config.page_queue[i] = i;
        }
        
        // Initialize tokens with dummy data
        for (int r = 0; r < demo.num_requests; ++r) {
            for (int t = 0; t < config.prompt_length[r]; ++t) {
                config.tokens[r * max_seq + t] = (t + 100) % demo.vocab_size;
            }
        }
        
        config.eos_token_id = 2;
    }
    
    void free_runtime_buffers(PKRuntimeConfig& config) {
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
        delete[] config.new_token_nums;
    }
    
    PKBackendType backend_type_;
    int device_id_;
};

// =============================================================================
// Main
// =============================================================================

void print_usage() {
    std::cout << "Usage: pk_multi_backend_demo [backend]" << std::endl;
    std::cout << "  backend: cuda, cpu, ascend, maca, mps, auto" << std::endl;
    std::cout << "  (default: auto - uses best available)" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== YiRage Persistent Kernel Multi-Backend Demo ===" << std::endl;
    std::cout << std::endl;
    
    // Parse command line
    std::string backend_name = "auto";
    if (argc > 1) {
        backend_name = argv[1];
    }
    
    // Configuration
    DemoConfig config;
    config.num_layers = 2;  // Reduced for demo
    config.num_workers = 2;
    config.num_requests = 2;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Sequence length: " << config.seq_len << std::endl;
    std::cout << "  Hidden dim: " << config.hidden_dim << std::endl;
    std::cout << "  Num layers: " << config.num_layers << std::endl;
    std::cout << "  Num workers: " << config.num_workers << std::endl;
    std::cout << "  Num requests: " << config.num_requests << std::endl;
    
    // Build task graph
    TaskGraphBuilder builder(config);
    std::vector<PKTaskDesc> tasks;
    std::vector<PKEventDesc> events;
    builder.build(tasks, events);
    
    // Get available backends
    std::cout << "\nAvailable backends:" << std::endl;
    auto available = get_available_pk_backends();
    for (auto b : available) {
        std::cout << "  - " << pk_backend_type_to_name(b) << std::endl;
    }
    
    // Run on selected backend(s)
    if (backend_name == "auto") {
        // Run on best available
        PKBackendType best = get_best_available_backend();
        std::cout << "\nBest available: " << pk_backend_type_to_name(best) << std::endl;
        
        BackendRunner runner(best);
        runner.run(config, tasks, events);
    } else if (backend_name == "all") {
        // Run on all available backends
        for (auto b : available) {
            BackendRunner runner(b);
            runner.run(config, tasks, events);
        }
    } else {
        // Run on specified backend
        PKBackendType type = PKBackendType::CPU;
        if (backend_name == "cuda") type = PKBackendType::CUDA;
        else if (backend_name == "cpu") type = PKBackendType::CPU;
        else if (backend_name == "ascend") type = PKBackendType::ASCEND;
        else if (backend_name == "maca") type = PKBackendType::MACA;
        else if (backend_name == "mps") type = PKBackendType::MPS;
        else {
            std::cerr << "Unknown backend: " << backend_name << std::endl;
            print_usage();
            return 1;
        }
        
        BackendRunner runner(type);
        if (!runner.run(config, tasks, events)) {
            std::cout << "Backend " << backend_name << " not available, falling back to CPU" << std::endl;
            BackendRunner cpu_runner(PKBackendType::CPU);
            cpu_runner.run(config, tasks, events);
        }
    }
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    return 0;
}
