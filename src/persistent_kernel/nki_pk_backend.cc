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
 * AWS NKI Persistent Kernel Backend Implementation
 */

#include "persistent_kernel/backends/nki_pk_backend.h"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>

// Neuron Runtime headers would be included here
#ifdef YIRAGE_BACKEND_NKI_ENABLED
// #include "nrt/nrt.h"
#endif

namespace yirage {
namespace pk {

// =============================================================================
// Constructor / Destructor
// =============================================================================

NKIPKBackend::NKIPKBackend()
    : is_initialized_(false), device_id_(0), neuron_core_count_(0),
      chip_type_(UNKNOWN_CHIP), nrt_model_(nullptr) {}

NKIPKBackend::~NKIPKBackend() {
    shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool NKIPKBackend::initialize(int device_id) {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    if (is_initialized_) {
        return true;
    }
    
    device_id_ = device_id;
    
    // Initialize Neuron Runtime
    // nrt_status_t status = nrt_init();
    // if (status != NRT_SUCCESS) return false;
    
    // Detect chip type and core count
    const char* instance_type = getenv("AWS_NEURON_INSTANCE_TYPE");
    if (instance_type) {
        if (strstr(instance_type, "trn1")) {
            chip_type_ = TRAINIUM_V1;
            neuron_core_count_ = 32;  // trn1.32xlarge
        } else if (strstr(instance_type, "trn2")) {
            chip_type_ = TRAINIUM_V2;
            neuron_core_count_ = 32;
        } else if (strstr(instance_type, "inf2")) {
            chip_type_ = INFERENTIA_V2;
            neuron_core_count_ = 12;
        } else if (strstr(instance_type, "inf1")) {
            chip_type_ = INFERENTIA_V1;
            neuron_core_count_ = 16;
        }
    }
    
    is_initialized_ = true;
    return true;
#else
    return false;
#endif
}

void NKIPKBackend::shutdown() {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    if (nrt_model_) {
        // nrt_unload(nrt_model_);
        nrt_model_ = nullptr;
    }
    // nrt_close();
    is_initialized_ = false;
#endif
}

bool NKIPKBackend::is_initialized() const {
    return is_initialized_;
}

// =============================================================================
// Memory Management
// =============================================================================

void* NKIPKBackend::allocate_hbm(size_t size) {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    // Allocate in HBM via Neuron Runtime
    // void* ptr = nullptr;
    // nrt_tensor_allocate(&ptr, size, NRT_MEMORY_HBM);
    // return ptr;
    return nullptr;
#else
    return nullptr;
#endif
}

void* NKIPKBackend::allocate_sbuf(size_t size) {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    // SBUF is managed by the compiler, not directly allocatable
    // This is for reference
    return nullptr;
#else
    return nullptr;
#endif
}

void NKIPKBackend::free_memory(void* ptr) {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    if (ptr) {
        // nrt_tensor_free(ptr);
    }
#endif
}

bool NKIPKBackend::copy_to_device(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    // nrt_memcpy(dst, src, size, NRT_MEMCPY_HOST_TO_DEVICE);
    return true;
#else
    return false;
#endif
}

bool NKIPKBackend::copy_to_host(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    // nrt_memcpy(dst, src, size, NRT_MEMCPY_DEVICE_TO_HOST);
    return true;
#else
    return false;
#endif
}

// =============================================================================
// Kernel Execution
// =============================================================================

bool NKIPKBackend::launch_kernel(const PKKernelConfig& config) {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    // Launch compiled NKI kernel
    // nrt_execute(nrt_model_, inputs, outputs);
    return true;
#else
    return false;
#endif
}

void NKIPKBackend::synchronize() {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    // nrt_wait();
#endif
}

// =============================================================================
// NKI-specific
// =============================================================================

bool NKIPKBackend::compile_nki_kernel(const std::string& kernel_code,
                                      const std::string& kernel_name) {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    // Write kernel to file
    std::string filename = "/tmp/" + kernel_name + ".py";
    std::ofstream file(filename);
    file << kernel_code;
    file.close();
    
    // Compile with neuronx-cc
    std::string cmd = "neuronx-cc compile --target=trn1 " + 
                     filename + " -o /tmp/" + kernel_name + ".neff";
    int result = system(cmd.c_str());
    
    if (result == 0) {
        // Load compiled model
        // nrt_load(&nrt_model_, ("/tmp/" + kernel_name + ".neff").c_str());
        
        threadblock::nki::NKIKernelRegistry::instance()
            .register_kernel(kernel_name, kernel_code);
        return true;
    }
    
    return false;
#else
    return false;
#endif
}

bool NKIPKBackend::launch_nki_kernel(const std::string& kernel_name,
                                     void** args,
                                     int num_args,
                                     const threadblock::nki::NKITileConfig& config) {
#ifdef YIRAGE_BACKEND_NKI_ENABLED
    if (!nrt_model_) {
        return false;
    }
    
    // Set kernel arguments and launch
    // nrt_execute(nrt_model_, args, ...);
    
    return true;
#else
    return false;
#endif
}

int NKIPKBackend::get_neuron_core_count() const {
    return neuron_core_count_;
}

NKIPKBackend::NeuronChipType NKIPKBackend::get_chip_type() const {
    return chip_type_;
}

// =============================================================================
// Factory Registration
// =============================================================================

#ifdef YIRAGE_BACKEND_NKI_ENABLED
namespace {
    struct NKIPKBackendRegistrar {
        NKIPKBackendRegistrar() {
            // PKBackendFactory::register_backend(PKBackendType::NKI, ...);
        }
    };
    static NKIPKBackendRegistrar nki_pk_registrar;
}
#endif

}  // namespace pk
}  // namespace yirage
