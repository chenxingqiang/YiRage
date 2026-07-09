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

#include "persistent_kernel/pk_backend_interface.h"
#include "persistent_kernel/backends/cuda_pk_backend.h"
#include "persistent_kernel/backends/cpu_pk_backend.h"
#include "persistent_kernel/backends/ascend_pk_backend.h"
#ifdef YIRAGE_PK_MACA_BACKEND_ENABLED
#include "persistent_kernel/backends/maca_pk_backend.h"
#endif
#include "persistent_kernel/backends/mps_pk_backend.h"

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// Backend Name Utilities
// =============================================================================

const char* pk_backend_type_to_name(PKBackendType type) {
    switch (type) {
        case PKBackendType::CUDA:   return "cuda";
        case PKBackendType::CPU:    return "cpu";
        case PKBackendType::MPS:    return "mps";
        case PKBackendType::ASCEND: return "ascend";
        case PKBackendType::MACA:   return "maca";
        case PKBackendType::TRITON: return "triton";
        case PKBackendType::NKI:    return "nki";
        default:                    return "unknown";
    }
}

const char* pk_mode_to_name(PKMode mode) {
    switch (mode) {
        case PKMode::OFFLINE:   return "offline";
        case PKMode::ONLINE:    return "online";
        case PKMode::ONEPASS:   return "onepass";
        case PKMode::EAGER:     return "eager";
        case PKMode::GRAPH:     return "graph";
        case PKMode::STREAMING: return "streaming";
        default:                return "unknown";
    }
}

// =============================================================================
// Backend Mode Support Matrix
// =============================================================================

/**
 * @brief Backend Mode Support Matrix (aligned with workload plan)
 * 
 * Backend     | Workload | Supported Modes
 * ------------|----------|---------------------------
 * CUDA        | 1 week   | OFFLINE, ONLINE, ONEPASS, GRAPH
 * CPU         | 1 week   | EAGER, GRAPH, OFFLINE
 * Ascend      | 2 weeks  | OFFLINE, ONLINE, GRAPH
 * MACA        | 2 weeks  | OFFLINE, ONLINE, ONEPASS
 * MPS         | 2 weeks  | EAGER, GRAPH
 */
bool pk_is_mode_supported(PKBackendType backend, PKMode mode) {
    switch (backend) {
        case PKBackendType::CUDA:
            // CUDA: OFFLINE, ONLINE, ONEPASS, GRAPH
            return (mode == PKMode::OFFLINE || 
                    mode == PKMode::ONLINE || 
                    mode == PKMode::ONEPASS ||
                    mode == PKMode::GRAPH);
            
        case PKBackendType::CPU:
            // CPU: EAGER, GRAPH, OFFLINE
            return (mode == PKMode::EAGER || 
                    mode == PKMode::GRAPH ||
                    mode == PKMode::OFFLINE);
            
        case PKBackendType::MPS:
            // MPS: EAGER, GRAPH
            return (mode == PKMode::EAGER || mode == PKMode::GRAPH);
            
        case PKBackendType::ASCEND:
            // Ascend: OFFLINE, ONLINE, GRAPH
            return (mode == PKMode::OFFLINE || 
                    mode == PKMode::ONLINE ||
                    mode == PKMode::GRAPH);
            
        case PKBackendType::MACA:
            // MACA: OFFLINE, ONLINE, ONEPASS
            return (mode == PKMode::OFFLINE || 
                    mode == PKMode::ONLINE ||
                    mode == PKMode::ONEPASS);
            
        case PKBackendType::TRITON:
            // Triton: OFFLINE, ONEPASS, EAGER
            return (mode == PKMode::OFFLINE ||
                    mode == PKMode::ONEPASS ||
                    mode == PKMode::EAGER);
            
        case PKBackendType::NKI:
            // NKI: OFFLINE, ONEPASS
            return (mode == PKMode::OFFLINE || mode == PKMode::ONEPASS);
            
        default:
            return false;
    }
}

// =============================================================================
// Backend Factory
// =============================================================================

std::unique_ptr<PKBackendInterface> create_pk_backend(
    PKBackendType type, 
    int device_id) {
    
    switch (type) {
        case PKBackendType::CUDA:
            return std::make_unique<CudaPKBackend>(device_id);
            
        case PKBackendType::CPU:
            return std::make_unique<CpuPKBackend>();
            
        case PKBackendType::ASCEND:
            return std::make_unique<AscendPKBackend>(device_id);
            
        case PKBackendType::MACA:
#ifdef YIRAGE_PK_MACA_BACKEND_ENABLED
            return std::make_unique<MacaPKBackend>(device_id);
#else
            return nullptr;
#endif
            
        case PKBackendType::MPS:
            return std::make_unique<MpsPKBackend>(device_id);
            
        case PKBackendType::TRITON:
            // Triton backend not yet implemented
            return nullptr;
            
        case PKBackendType::NKI:
            // NKI backend not yet implemented
            return nullptr;
            
        default:
            return nullptr;
    }
}

std::vector<PKBackendType> get_available_pk_backends() {
    std::vector<PKBackendType> available;
    
    // Check CUDA availability
    {
        auto cuda = create_pk_backend(PKBackendType::CUDA, 0);
        if (cuda && cuda->is_available()) {
            available.push_back(PKBackendType::CUDA);
        }
    }
    
    // Check MACA availability
#ifdef YIRAGE_PK_MACA_BACKEND_ENABLED
    {
        auto maca = create_pk_backend(PKBackendType::MACA, 0);
        if (maca && maca->is_available()) {
            available.push_back(PKBackendType::MACA);
        }
    }
#endif
    
    // Check Ascend availability
    {
        auto ascend = create_pk_backend(PKBackendType::ASCEND, 0);
        if (ascend && ascend->is_available()) {
            available.push_back(PKBackendType::ASCEND);
        }
    }
    
    // Check MPS availability (Apple Silicon only)
    {
        auto mps = create_pk_backend(PKBackendType::MPS, 0);
        if (mps && mps->is_available()) {
            available.push_back(PKBackendType::MPS);
        }
    }
    
    // CPU is always available
    available.push_back(PKBackendType::CPU);
    
    return available;
}

} // namespace persistent_kernel
} // namespace yirage
