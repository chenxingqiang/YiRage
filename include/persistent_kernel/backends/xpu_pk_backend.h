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
 *
 * Intel XPU Persistent Kernel Backend
 */

#pragma once

#include "persistent_kernel/pk_backend.h"

#ifdef YIRAGE_BACKEND_XPU_ENABLED
#include <sycl/sycl.hpp>
#endif

namespace yirage {
namespace pk {

/**
 * @brief Intel XPU Persistent Kernel Backend
 * 
 * Architecture features:
 * - SYCL/DPC++ programming model
 * - Xe Matrix Extensions (XMX)
 * - Shared Local Memory (SLM)
 * - Sub-groups for SIMD operations
 */
class XPUPKBackend : public PKBackend {
public:
    XPUPKBackend();
    ~XPUPKBackend() override;

    // ========== Initialization ==========
    bool initialize(int device_id = 0) override;
    void shutdown() override;
    bool is_initialized() const override;

    // ========== Memory Management ==========
    void* allocate_device(size_t size);
    void* allocate_shared(size_t size);  // USM shared allocation
    void free_memory(void* ptr) override;
    
    bool copy_to_device(void* dst, const void* src, size_t size) override;
    bool copy_to_host(void* dst, const void* src, size_t size) override;

    // ========== Kernel Execution ==========
    bool launch_kernel(const PKKernelConfig& config) override;
    void synchronize() override;

    // ========== XPU-specific ==========
    
    /**
     * @brief Get preferred sub-group size
     */
    int get_subgroup_size() const;
    
    /**
     * @brief Check if XMX is available
     */
    bool has_xmx() const;
    
    /**
     * @brief Get SLM size per workgroup
     */
    size_t get_slm_size() const;

#ifdef YIRAGE_BACKEND_XPU_ENABLED
    /**
     * @brief Get SYCL queue
     */
    sycl::queue& get_queue();
#endif

private:
    bool is_initialized_;
    int device_id_;
    
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    std::unique_ptr<sycl::queue> queue_;
    sycl::device device_;
#endif
};

// =============================================================================
// XPU-specific PK Configuration
// =============================================================================

struct XPUPKConfig {
    int workgroup_size_x = 256;
    int workgroup_size_y = 1;
    int workgroup_size_z = 1;
    int subgroup_size = 16;       // 16 or 32 for Intel
    size_t slm_size = 0;          // SLM required
    bool use_xmx = true;          // Use XMX if available
    bool use_prefetch = true;     // Use memory prefetching
};

}  // namespace pk
}  // namespace yirage
