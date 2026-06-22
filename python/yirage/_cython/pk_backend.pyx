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
Cython bindings for the Persistent Kernel multi-backend interface.

This module provides Python access to the C++ persistent kernel backends,
enabling multi-backend kernel execution from Python.
"""

from libc.stdint cimport uint32_t, uint64_t, int64_t
from libcpp cimport bool as cpp_bool
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport pk_backend

# Check if C++ bindings are available
CPP_PK_AVAILABLE = True
try:
    _test_backends = pk_backend.get_available_pk_backends_int()
except:
    CPP_PK_AVAILABLE = False


# =============================================================================
# Python Enums
# =============================================================================

class PyPKBackendType:
    """Python enum for PK backend types."""
    CUDA = 0
    CPU = 1
    MPS = 2
    ASCEND = 3
    MACA = 4
    TRITON = 5
    NKI = 6
    
    @staticmethod
    def to_string(backend_type):
        names = {0: "cuda", 1: "cpu", 2: "mps", 3: "ascend", 
                 4: "maca", 5: "triton", 6: "nki"}
        return names.get(backend_type, "unknown")
    
    @staticmethod
    def from_string(name):
        types = {"cuda": 0, "cpu": 1, "mps": 2, "ascend": 3,
                 "maca": 4, "triton": 5, "nki": 6}
        return types.get(name.lower(), -1)


class PyPKMode:
    """Python enum for PK execution modes."""
    OFFLINE = 0
    ONLINE = 1
    ONEPASS = 2
    EAGER = 3
    GRAPH = 4
    STREAMING = 5
    
    @staticmethod
    def to_string(mode):
        names = {0: "offline", 1: "online", 2: "onepass",
                 3: "eager", 4: "graph", 5: "streaming"}
        return names.get(mode, "unknown")
    
    @staticmethod
    def from_string(name):
        modes = {"offline": 0, "online": 1, "onepass": 2,
                 "eager": 3, "graph": 4, "streaming": 5}
        return modes.get(name.lower(), -1)


class PyPKDataType:
    """Python enum for PK data types."""
    FP32 = 0
    FP16 = 1
    BF16 = 2
    INT8 = 3
    INT4 = 4
    FP8_E4M3 = 5
    FP8_E5M2 = 6


# =============================================================================
# Python Data Classes
# =============================================================================

class PyPKCapabilities:
    """Python wrapper for PK backend capabilities."""
    
    def __init__(self):
        self.supports_tma = False
        self.supports_tensor_cores = False
        self.supports_async_copy = False
        self.supports_nvshmem = False
        self.supports_fp8 = False
        self.max_shared_memory = 0
        self.max_global_memory = 0
        self.max_threads_per_block = 0
        self.max_blocks_per_sm = 0
        self.compute_major = 0
        self.compute_minor = 0
        self.supported_modes = []
        self.supported_dtypes = []
    
    def to_dict(self):
        return {
            "supports_tma": self.supports_tma,
            "supports_tensor_cores": self.supports_tensor_cores,
            "supports_async_copy": self.supports_async_copy,
            "supports_nvshmem": self.supports_nvshmem,
            "supports_fp8": self.supports_fp8,
            "max_shared_memory": self.max_shared_memory,
            "max_global_memory": self.max_global_memory,
            "max_threads_per_block": self.max_threads_per_block,
            "max_blocks_per_sm": self.max_blocks_per_sm,
            "compute_major": self.compute_major,
            "compute_minor": self.compute_minor,
            "supported_modes": [PyPKMode.to_string(m) for m in self.supported_modes],
            "supported_dtypes": self.supported_dtypes,
        }


class PyPKRuntimeConfig:
    """Python wrapper for PK runtime configuration."""
    
    def __init__(self):
        self.mode = PyPKMode.ONLINE
        self.num_workers = 4
        self.num_local_schedulers = 1
        self.num_remote_schedulers = 0
        self.threads_per_worker = 256
        self.max_seq_length = 2048
        self.max_num_batched_requests = 16
        self.max_num_batched_tokens = 64
        self.max_num_pages = 1024
        self.page_size = 64
        self.eos_token_id = -1
        self.num_gpus = 1
        self.my_gpu_id = 0
        self.profiling_enabled = False
    
    def to_dict(self):
        return {
            "mode": PyPKMode.to_string(self.mode),
            "num_workers": self.num_workers,
            "num_local_schedulers": self.num_local_schedulers,
            "num_remote_schedulers": self.num_remote_schedulers,
            "threads_per_worker": self.threads_per_worker,
            "max_seq_length": self.max_seq_length,
            "max_num_batched_requests": self.max_num_batched_requests,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_pages": self.max_num_pages,
            "page_size": self.page_size,
            "eos_token_id": self.eos_token_id,
            "num_gpus": self.num_gpus,
            "my_gpu_id": self.my_gpu_id,
            "profiling_enabled": self.profiling_enabled,
        }
    
    @staticmethod
    def from_dict(config_dict):
        config = PyPKRuntimeConfig()
        if "mode" in config_dict:
            config.mode = PyPKMode.from_string(config_dict["mode"])
        if "num_workers" in config_dict:
            config.num_workers = config_dict["num_workers"]
        if "num_local_schedulers" in config_dict:
            config.num_local_schedulers = config_dict["num_local_schedulers"]
        if "num_remote_schedulers" in config_dict:
            config.num_remote_schedulers = config_dict["num_remote_schedulers"]
        if "threads_per_worker" in config_dict:
            config.threads_per_worker = config_dict["threads_per_worker"]
        if "max_seq_length" in config_dict:
            config.max_seq_length = config_dict["max_seq_length"]
        if "max_num_batched_requests" in config_dict:
            config.max_num_batched_requests = config_dict["max_num_batched_requests"]
        if "max_num_batched_tokens" in config_dict:
            config.max_num_batched_tokens = config_dict["max_num_batched_tokens"]
        if "max_num_pages" in config_dict:
            config.max_num_pages = config_dict["max_num_pages"]
        if "page_size" in config_dict:
            config.page_size = config_dict["page_size"]
        if "eos_token_id" in config_dict:
            config.eos_token_id = config_dict["eos_token_id"]
        if "num_gpus" in config_dict:
            config.num_gpus = config_dict["num_gpus"]
        if "my_gpu_id" in config_dict:
            config.my_gpu_id = config_dict["my_gpu_id"]
        if "profiling_enabled" in config_dict:
            config.profiling_enabled = config_dict["profiling_enabled"]
        return config


# =============================================================================
# Cython to Python Conversion Helpers
# =============================================================================

cdef object _convert_capabilities(pk_backend.PKCapabilitiesInt cpp_caps):
    """Convert C++ PKCapabilitiesInt to Python PyPKCapabilities."""
    caps = PyPKCapabilities()
    caps.supports_tma = cpp_caps.supports_tma
    caps.supports_tensor_cores = cpp_caps.supports_tensor_cores
    caps.supports_async_copy = cpp_caps.supports_async_copy
    caps.supports_nvshmem = cpp_caps.supports_nvshmem
    caps.supports_fp8 = cpp_caps.supports_fp8
    caps.max_shared_memory = cpp_caps.max_shared_memory
    caps.max_global_memory = cpp_caps.max_global_memory
    caps.max_threads_per_block = cpp_caps.max_threads_per_block
    caps.max_blocks_per_sm = cpp_caps.max_blocks_per_sm
    caps.compute_major = cpp_caps.compute_major
    caps.compute_minor = cpp_caps.compute_minor
    caps.supported_modes = list(cpp_caps.supported_modes)
    caps.supported_dtypes = list(cpp_caps.supported_dtypes)
    return caps


cdef pk_backend.PKRuntimeConfigInt _convert_to_cpp_config(object py_config):
    """Convert Python PyPKRuntimeConfig to C++ PKRuntimeConfigInt."""
    cdef pk_backend.PKRuntimeConfigInt cpp_config
    cpp_config.mode = py_config.mode
    cpp_config.num_workers = py_config.num_workers
    cpp_config.num_local_schedulers = py_config.num_local_schedulers
    cpp_config.num_remote_schedulers = py_config.num_remote_schedulers
    cpp_config.threads_per_worker = py_config.threads_per_worker
    cpp_config.max_seq_length = py_config.max_seq_length
    cpp_config.max_num_batched_requests = py_config.max_num_batched_requests
    cpp_config.max_num_batched_tokens = py_config.max_num_batched_tokens
    cpp_config.max_num_pages = py_config.max_num_pages
    cpp_config.page_size = py_config.page_size
    cpp_config.eos_token_id = py_config.eos_token_id
    cpp_config.num_gpus = py_config.num_gpus
    cpp_config.my_gpu_id = py_config.my_gpu_id
    cpp_config.profiling_enabled = py_config.profiling_enabled
    cpp_config.backend_context = NULL
    cpp_config.stream_handle = NULL
    cpp_config.profiler_buffer = NULL
    return cpp_config


# =============================================================================
# Python Wrapper for PKBackendWrapper
# =============================================================================

cdef class PyPKBackend:
    """
    Python wrapper for C++ PKBackendInterface.
    
    Provides access to persistent kernel backend functionality from Python.
    
    Example:
        backend = create_pk_backend("cuda", device_id=0)
        if backend and backend.is_available():
            caps = backend.get_capabilities()
            print(f"Backend: {backend.get_display_name()}")
            print(f"Supports TMA: {caps.supports_tma}")
            
            config = PyPKRuntimeConfig()
            config.mode = PyPKMode.ONLINE
            config.num_workers = 4
            
            if backend.initialize(config):
                backend.synchronize()
                backend.finalize()
    """
    cdef pk_backend.PKBackendWrapper* _backend
    cdef bint _initialized
    
    def __cinit__(self):
        self._backend = NULL
        self._initialized = False
    
    def __dealloc__(self):
        if self._backend != NULL:
            if self._initialized:
                self._backend.finalize()
            del self._backend
            self._backend = NULL
    
    @staticmethod
    cdef PyPKBackend _wrap(pk_backend.PKBackendWrapper* backend):
        """Create wrapper from C++ pointer."""
        cdef PyPKBackend wrapper = PyPKBackend()
        wrapper._backend = backend
        return wrapper
    
    def get_type(self):
        """Get backend type as integer."""
        if self._backend == NULL:
            return -1
        return self._backend.get_type()
    
    def get_name(self):
        """Get backend name (e.g., 'cuda', 'cpu')."""
        if self._backend == NULL:
            return ""
        return self._backend.get_name().decode('utf-8')
    
    def get_display_name(self):
        """Get human-readable backend name."""
        if self._backend == NULL:
            return ""
        return self._backend.get_display_name().decode('utf-8')
    
    def is_available(self):
        """Check if backend is available on this system."""
        if self._backend == NULL:
            return False
        return self._backend.is_available()
    
    def get_capabilities(self):
        """Get backend capabilities."""
        if self._backend == NULL:
            return PyPKCapabilities()
        return _convert_capabilities(self._backend.get_capabilities())
    
    def supports_mode(self, int mode):
        """Check if backend supports a specific execution mode."""
        if self._backend == NULL:
            return False
        return self._backend.supports_mode(mode)
    
    def get_default_mode(self):
        """Get default execution mode for this backend."""
        if self._backend == NULL:
            return PyPKMode.OFFLINE
        return self._backend.get_default_mode()
    
    def get_supported_modes(self):
        """Get list of supported execution modes."""
        if self._backend == NULL:
            return []
        cdef vector[int] modes = self._backend.get_supported_modes()
        return list(modes)
    
    def initialize(self, config=None):
        """Initialize the backend with configuration."""
        if self._backend == NULL:
            return False
        
        if config is None:
            config = PyPKRuntimeConfig()
        
        cdef pk_backend.PKRuntimeConfigInt cpp_config = _convert_to_cpp_config(config)
        cdef bint success = self._backend.initialize(cpp_config)
        
        if success:
            self._initialized = True
        
        return success
    
    def finalize(self):
        """Finalize the backend and release resources."""
        if self._backend == NULL:
            return
        self._backend.finalize()
        self._initialized = False
    
    def reset(self):
        """Reset backend for new session."""
        if self._backend == NULL:
            return
        self._backend.reset()
    
    def synchronize(self):
        """Synchronize all streams/operations."""
        if self._backend == NULL:
            return
        self._backend.synchronize()
    
    def set_device(self, int device_id):
        """Set current device."""
        if self._backend == NULL:
            return False
        return self._backend.set_device(device_id)
    
    def get_device(self):
        """Get current device."""
        if self._backend == NULL:
            return -1
        return self._backend.get_device()
    
    def get_device_count(self):
        """Get number of available devices."""
        if self._backend == NULL:
            return 0
        return self._backend.get_device_count()
    
    def get_compile_flags(self, int mode):
        """Get compile flags for specified mode."""
        if self._backend == NULL:
            return []
        cdef vector[string] flags = self._backend.get_compile_flags(mode)
        return [f.decode('utf-8') for f in flags]
    
    def get_include_dirs(self):
        """Get include directories for compilation."""
        if self._backend == NULL:
            return []
        cdef vector[string] dirs = self._backend.get_include_dirs()
        return [d.decode('utf-8') for d in dirs]


# =============================================================================
# Module-Level Functions
# =============================================================================

def create_pk_backend(backend_type, int device_id=0):
    """
    Create a persistent kernel backend.
    
    Args:
        backend_type: Backend type (int or string, e.g., "cuda", "cpu")
        device_id: Device ID for GPU backends (default 0)
    
    Returns:
        PyPKBackend instance or None if creation failed
    """
    if not CPP_PK_AVAILABLE:
        return None
    
    cdef int backend_int
    if isinstance(backend_type, str):
        backend_int = PyPKBackendType.from_string(backend_type)
        if backend_int < 0:
            return None
    else:
        backend_int = backend_type
    
    cdef pk_backend.PKBackendWrapper* cpp_backend
    cpp_backend = pk_backend.create_backend_wrapper(backend_int, device_id)
    
    if cpp_backend == NULL:
        return None
    
    return PyPKBackend._wrap(cpp_backend)


def get_available_backends():
    """
    Get list of available backends on this system.
    
    Returns:
        List of backend type integers
    """
    if not CPP_PK_AVAILABLE:
        return [PyPKBackendType.CPU]  # CPU always available
    
    cdef vector[int] backends = pk_backend.get_available_pk_backends_int()
    return list(backends)


def get_available_backend_names():
    """
    Get list of available backend names.
    
    Returns:
        List of backend name strings
    """
    return [PyPKBackendType.to_string(b) for b in get_available_backends()]


def backend_type_to_name(int backend_type):
    """Convert backend type integer to name string."""
    if not CPP_PK_AVAILABLE:
        return PyPKBackendType.to_string(backend_type)
    return pk_backend.pk_backend_type_to_name_int(backend_type).decode('utf-8')


def mode_to_name(int mode):
    """Convert mode integer to name string."""
    if not CPP_PK_AVAILABLE:
        return PyPKMode.to_string(mode)
    return pk_backend.pk_mode_to_name_int(mode).decode('utf-8')


def is_mode_supported(int backend_type, int mode):
    """
    Check if a mode is supported by a backend type.
    
    Mode support matrix (aligned with workload plan):
    - CUDA: OFFLINE, ONLINE, ONEPASS, GRAPH
    - CPU: EAGER, GRAPH, OFFLINE
    - Ascend: OFFLINE, ONLINE, GRAPH
    - MACA: OFFLINE, ONLINE, ONEPASS
    - MPS: EAGER, GRAPH
    """
    if not CPP_PK_AVAILABLE:
        # Fallback: Python-side mode matrix
        mode_matrix = {
            PyPKBackendType.CUDA: [PyPKMode.OFFLINE, PyPKMode.ONLINE, 
                                   PyPKMode.ONEPASS, PyPKMode.GRAPH],
            PyPKBackendType.CPU: [PyPKMode.EAGER, PyPKMode.GRAPH, 
                                  PyPKMode.OFFLINE],
            PyPKBackendType.ASCEND: [PyPKMode.OFFLINE, PyPKMode.ONLINE, 
                                     PyPKMode.GRAPH],
            PyPKBackendType.MACA: [PyPKMode.OFFLINE, PyPKMode.ONLINE, 
                                   PyPKMode.ONEPASS],
            PyPKBackendType.MPS: [PyPKMode.EAGER, PyPKMode.GRAPH],
        }
        supported = mode_matrix.get(backend_type, [])
        return mode in supported
    return pk_backend.pk_is_mode_supported_int(backend_type, mode)


def get_best_backend(int device_id=0):
    """
    Create the best available backend.
    
    Priority: CUDA > MACA > Ascend > MPS > CPU
    
    Args:
        device_id: Device ID for GPU backends
    
    Returns:
        PyPKBackend instance for best available backend
    """
    available = get_available_backends()
    
    # Priority order
    for backend_type in [PyPKBackendType.CUDA, 
                         PyPKBackendType.MACA,
                         PyPKBackendType.ASCEND,
                         PyPKBackendType.MPS,
                         PyPKBackendType.CPU]:
        if backend_type in available:
            backend = create_pk_backend(backend_type, device_id)
            if backend and backend.is_available():
                return backend
    
    # Fallback to CPU
    return create_pk_backend(PyPKBackendType.CPU, 0)
