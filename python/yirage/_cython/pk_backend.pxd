# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cython declarations for the Persistent Kernel multi-backend interface.

NOTE: C++11 scoped enums (enum class) require special handling.
We use a C++ wrapper header with inline functions for conversions.
"""

from libcpp cimport bool as cpp_bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libc.stdint cimport uint32_t, uint64_t, int64_t


# =============================================================================
# C++ Wrapper Header (inline) for enum class handling
# =============================================================================

cdef extern from *:
    """
    // Inline C++ header for Cython enum class interoperability
    #include "persistent_kernel/pk_backend_interface.h"
    #include <vector>
    #include <memory>
    
    namespace yirage_cython {
    
    // Type aliases for simpler Cython declarations
    using PKBackendType = yirage::persistent_kernel::PKBackendType;
    using PKMode = yirage::persistent_kernel::PKMode;
    using PKDataType = yirage::persistent_kernel::PKDataType;
    using PKTaskType = yirage::persistent_kernel::PKTaskType;
    using PKCapabilities = yirage::persistent_kernel::PKCapabilities;
    using PKRuntimeConfig = yirage::persistent_kernel::PKRuntimeConfig;
    using PKBackendInterface = yirage::persistent_kernel::PKBackendInterface;
    
    // =========================================================================
    // Conversion helpers: int <-> enum class
    // =========================================================================
    
    inline PKBackendType int_to_backend_type(int v) {
        return static_cast<PKBackendType>(v);
    }
    
    inline PKMode int_to_pk_mode(int v) {
        return static_cast<PKMode>(v);
    }
    
    inline PKDataType int_to_pk_dtype(int v) {
        return static_cast<PKDataType>(v);
    }
    
    inline PKTaskType int_to_pk_task_type(int v) {
        return static_cast<PKTaskType>(v);
    }
    
    inline int backend_type_to_int(PKBackendType v) {
        return static_cast<int>(v);
    }
    
    inline int pk_mode_to_int(PKMode v) {
        return static_cast<int>(v);
    }
    
    inline int pk_dtype_to_int(PKDataType v) {
        return static_cast<int>(v);
    }
    
    inline int pk_task_type_to_int(PKTaskType v) {
        return static_cast<int>(v);
    }
    
    // =========================================================================
    // Wrapper functions that take int and call the real functions
    // =========================================================================
    
    inline std::unique_ptr<PKBackendInterface> create_pk_backend_int(int backend_type, int device_id) {
        return yirage::persistent_kernel::create_pk_backend(
            static_cast<PKBackendType>(backend_type), device_id);
    }
    
    inline std::vector<int> get_available_pk_backends_int() {
        auto backends = yirage::persistent_kernel::get_available_pk_backends();
        std::vector<int> result;
        result.reserve(backends.size());
        for (auto b : backends) {
            result.push_back(static_cast<int>(b));
        }
        return result;
    }
    
    inline const char* pk_backend_type_to_name_int(int backend_type) {
        return yirage::persistent_kernel::pk_backend_type_to_name(
            static_cast<PKBackendType>(backend_type));
    }
    
    inline const char* pk_mode_to_name_int(int mode) {
        return yirage::persistent_kernel::pk_mode_to_name(
            static_cast<PKMode>(mode));
    }
    
    inline bool pk_is_mode_supported_int(int backend, int mode) {
        return yirage::persistent_kernel::pk_is_mode_supported(
            static_cast<PKBackendType>(backend),
            static_cast<PKMode>(mode));
    }
    
    // =========================================================================
    // PKRuntimeConfig with int mode
    // =========================================================================
    
    struct PKRuntimeConfigInt {
        int mode;
        int num_workers;
        int num_local_schedulers;
        int num_remote_schedulers;
        int threads_per_worker;
        size_t max_seq_length;
        size_t max_num_batched_requests;
        size_t max_num_batched_tokens;
        size_t max_num_pages;
        size_t page_size;
        int64_t eos_token_id;
        int num_gpus;
        int my_gpu_id;
        void* backend_context;
        void* stream_handle;
        bool profiling_enabled;
        void* profiler_buffer;
        
        PKRuntimeConfig to_cpp() const {
            PKRuntimeConfig config;
            config.mode = static_cast<PKMode>(mode);
            config.num_workers = num_workers;
            config.num_local_schedulers = num_local_schedulers;
            config.num_remote_schedulers = num_remote_schedulers;
            config.threads_per_worker = threads_per_worker;
            config.max_seq_length = max_seq_length;
            config.max_num_batched_requests = max_num_batched_requests;
            config.max_num_batched_tokens = max_num_batched_tokens;
            config.max_num_pages = max_num_pages;
            config.page_size = page_size;
            config.eos_token_id = eos_token_id;
            config.num_gpus = num_gpus;
            config.my_gpu_id = my_gpu_id;
            config.backend_context = backend_context;
            config.stream_handle = stream_handle;
            config.profiling_enabled = profiling_enabled;
            config.profiler_buffer = profiler_buffer;
            return config;
        }
    };
    
    // =========================================================================
    // PKCapabilities with int vectors
    // =========================================================================
    
    struct PKCapabilitiesInt {
        bool supports_tma;
        bool supports_tensor_cores;
        bool supports_async_copy;
        bool supports_nvshmem;
        bool supports_fp8;
        size_t max_shared_memory;
        size_t max_global_memory;
        size_t max_threads_per_block;
        size_t max_blocks_per_sm;
        int compute_major;
        int compute_minor;
        std::vector<int> supported_modes;
        std::vector<int> supported_dtypes;
        
        static PKCapabilitiesInt from_cpp(const PKCapabilities& caps) {
            PKCapabilitiesInt result;
            result.supports_tma = caps.supports_tma;
            result.supports_tensor_cores = caps.supports_tensor_cores;
            result.supports_async_copy = caps.supports_async_copy;
            result.supports_nvshmem = caps.supports_nvshmem;
            result.supports_fp8 = caps.supports_fp8;
            result.max_shared_memory = caps.max_shared_memory;
            result.max_global_memory = caps.max_global_memory;
            result.max_threads_per_block = caps.max_threads_per_block;
            result.max_blocks_per_sm = caps.max_blocks_per_sm;
            result.compute_major = caps.compute_major;
            result.compute_minor = caps.compute_minor;
            for (auto m : caps.supported_modes) {
                result.supported_modes.push_back(static_cast<int>(m));
            }
            for (auto d : caps.supported_dtypes) {
                result.supported_dtypes.push_back(static_cast<int>(d));
            }
            return result;
        }
    };
    
    // =========================================================================
    // Backend wrapper class with int-based interface
    // =========================================================================
    
    class PKBackendWrapper {
    public:
        std::unique_ptr<PKBackendInterface> backend;
        
        PKBackendWrapper(std::unique_ptr<PKBackendInterface> b) 
            : backend(std::move(b)) {}
        
        bool is_valid() const { return backend != nullptr; }
        
        int get_type() const {
            return backend ? static_cast<int>(backend->get_type()) : -1;
        }
        
        std::string get_name() const {
            return backend ? backend->get_name() : "";
        }
        
        std::string get_display_name() const {
            return backend ? backend->get_display_name() : "";
        }
        
        bool is_available() const {
            return backend && backend->is_available();
        }
        
        PKCapabilitiesInt get_capabilities() const {
            if (!backend) return PKCapabilitiesInt{};
            return PKCapabilitiesInt::from_cpp(backend->get_capabilities());
        }
        
        bool supports_mode(int mode) const {
            return backend && backend->supports_mode(static_cast<PKMode>(mode));
        }
        
        int get_default_mode() const {
            return backend ? static_cast<int>(backend->get_default_mode()) : 0;
        }
        
        std::vector<int> get_supported_modes() const {
            std::vector<int> result;
            if (backend) {
                for (auto m : backend->get_supported_modes()) {
                    result.push_back(static_cast<int>(m));
                }
            }
            return result;
        }
        
        bool initialize(const PKRuntimeConfigInt& config) {
            return backend && backend->initialize(config.to_cpp());
        }
        
        void finalize() {
            if (backend) backend->finalize();
        }
        
        void reset() {
            if (backend) backend->reset();
        }
        
        void synchronize() {
            if (backend) backend->synchronize();
        }
        
        bool set_device(int device_id) {
            return backend && backend->set_device(device_id);
        }
        
        int get_device() const {
            return backend ? backend->get_device() : -1;
        }
        
        int get_device_count() const {
            return backend ? backend->get_device_count() : 0;
        }
        
        std::vector<std::string> get_compile_flags(int mode) const {
            std::vector<std::string> result;
            if (backend) {
                for (auto& s : backend->get_compile_flags(static_cast<PKMode>(mode))) {
                    result.push_back(s);
                }
            }
            return result;
        }
        
        std::vector<std::string> get_include_dirs() const {
            std::vector<std::string> result;
            if (backend) {
                for (auto& s : backend->get_include_dirs()) {
                    result.push_back(s);
                }
            }
            return result;
        }
    };
    
    inline PKBackendWrapper* create_backend_wrapper(int backend_type, int device_id) {
        auto backend = create_pk_backend_int(backend_type, device_id);
        if (!backend) return nullptr;
        return new PKBackendWrapper(std::move(backend));
    }
    
    } // namespace yirage_cython
    """
    pass


# =============================================================================
# Cython declarations for the wrapper types
# =============================================================================

cdef extern from * namespace "yirage_cython":
    
    # =========================================================================
    # Simple wrapper structures with int-based enums
    # =========================================================================
    
    cdef struct PKRuntimeConfigInt:
        int mode
        int num_workers
        int num_local_schedulers
        int num_remote_schedulers
        int threads_per_worker
        size_t max_seq_length
        size_t max_num_batched_requests
        size_t max_num_batched_tokens
        size_t max_num_pages
        size_t page_size
        int64_t eos_token_id
        int num_gpus
        int my_gpu_id
        void* backend_context
        void* stream_handle
        cpp_bool profiling_enabled
        void* profiler_buffer
    
    cdef struct PKCapabilitiesInt:
        cpp_bool supports_tma
        cpp_bool supports_tensor_cores
        cpp_bool supports_async_copy
        cpp_bool supports_nvshmem
        cpp_bool supports_fp8
        size_t max_shared_memory
        size_t max_global_memory
        size_t max_threads_per_block
        size_t max_blocks_per_sm
        int compute_major
        int compute_minor
        vector[int] supported_modes
        vector[int] supported_dtypes
    
    # =========================================================================
    # Backend wrapper class
    # =========================================================================
    
    cdef cppclass PKBackendWrapper:
        cpp_bool is_valid()
        int get_type()
        string get_name()
        string get_display_name()
        cpp_bool is_available()
        PKCapabilitiesInt get_capabilities()
        cpp_bool supports_mode(int mode)
        int get_default_mode()
        vector[int] get_supported_modes()
        cpp_bool initialize(const PKRuntimeConfigInt& config)
        void finalize()
        void reset()
        void synchronize()
        cpp_bool set_device(int device_id)
        int get_device()
        int get_device_count()
        vector[string] get_compile_flags(int mode)
        vector[string] get_include_dirs()
    
    # =========================================================================
    # Factory and utility functions (all using int for enums)
    # =========================================================================
    
    PKBackendWrapper* create_backend_wrapper(int backend_type, int device_id) except +
    vector[int] get_available_pk_backends_int() except +
    const char* pk_backend_type_to_name_int(int backend_type)
    const char* pk_mode_to_name_int(int mode)
    cpp_bool pk_is_mode_supported_int(int backend, int mode)


# =============================================================================
# Runtime Core Declarations (from pk_runtime_core.h)
# These are less affected by enum class issues
# =============================================================================

cdef extern from "persistent_kernel/pk_runtime_core.h" namespace "yirage::persistent_kernel::runtime":
    
    # Task/Event ID types
    ctypedef unsigned long long TaskId
    ctypedef unsigned long long EventId
    ctypedef unsigned long long EventCounter
    
    # Constants
    cdef TaskId TASK_INVALID_ID
    cdef EventId EVENT_INVALID_ID
    cdef EventId EVENT_NVSHMEM_TAG
    
    cdef int YPK_MAX_NUM_BATCHED_REQUESTS
    cdef int YPK_MAX_NUM_BATCHED_TOKENS
    cdef int YPK_MAX_NUM_PAGES
    cdef int YPK_PAGE_SIZE
    cdef int YPK_MAX_SEQ_LENGTH
