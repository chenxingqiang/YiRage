# YiRage YPK Multi-Backend Support Design Document

## 1. Design Goals

Add multi-backend support for YiRage Persistent Kernel (YPK) to achieve:
- Support for all major PyTorch backends
- Compile-time selection of multiple backends simultaneously
- Runtime dynamic backend selection
- Maintain backward compatibility
- Minimize code modifications

## 2. Supported Backend List

Referencing `torch.backends`, the following backends are planned:

| Backend Type | Priority | Description |
|-------------|----------|-------------|
| CUDA | P0 | Supported, NVIDIA GPU |
| CPU | P0 | CPU execution |
| MPS | P1 | Apple Metal Performance Shaders |
| CUDNN | P1 | CUDA Deep Neural Network library |
| MKL | P2 | Intel Math Kernel Library |
| MKLDNN | P2 | Intel oneDNN |
| OpenMP | P2 | OpenMP parallelization |
| NKI | P2 | Supported, AWS Neuron |
| Triton | P2 | Python layer support |
| cuSPARSELt | P3 | CUDA sparse matrix |
| MHA | P3 | Multi-Head Attention |
| NNPACK | P3 | Neural Network PACKage |
| opt_einsum | P3 | Optimized einsum |
| Xeon | P3 | Intel Xeon optimization |

## 3. Architecture Design

### 3.1 Backend Abstraction Layer Architecture

```
┌─────────────────────────────────────────────────┐
│           Python API Layer                      │
│   PersistentKernel(backend="cuda", ...)        │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Backend Manager (C++)                   │
│  - Backend Registration                         │
│  - Backend Selection                            │
│  - Capability Query                             │
└────────────────┬────────────────────────────────┘
                 │
     ┌───────────┴───────────┬──────────┬─────────┐
     │                       │          │         │
┌────▼──────┐  ┌─────▼──────┐  ┌──▼───┐  ┌──▼───┐
│CUDA       │  │CPU         │  │MPS   │  │ ...  │
│Backend    │  │Backend     │  │Backend│ │      │
└───────────┘  └────────────┘  └──────┘  └──────┘
```

### 3.2 Core Interface Definition

#### BackendInterface (C++)
```cpp
class BackendInterface {
public:
    virtual ~BackendInterface() = default;
    
    // Backend information
    virtual BackendType get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual bool is_available() const = 0;
    
    // Compilation
    virtual bool compile(const CompileContext& ctx) = 0;
    virtual std::string get_compile_flags() const = 0;
    
    // Runtime
    virtual void* allocate_memory(size_t size) = 0;
    virtual void free_memory(void* ptr) = 0;
    virtual void synchronize() = 0;
    
    // Capability query
    virtual size_t get_max_memory() const = 0;
    virtual size_t get_max_shared_memory() const = 0;
    virtual bool supports_data_type(DataType dt) const = 0;
};
```

#### BackendRegistry (C++)
```cpp
class BackendRegistry {
public:
    static BackendRegistry& instance();
    
    void register_backend(std::unique_ptr<BackendInterface> backend);
    BackendInterface* get_backend(BackendType type);
    BackendInterface* get_backend(const std::string& name);
    
    std::vector<BackendType> get_available_backends() const;
    bool is_backend_available(BackendType type) const;
    
private:
    std::unordered_map<BackendType, std::unique_ptr<BackendInterface>> backends_;
    std::unordered_map<std::string, BackendType> name_to_type_;
};
```

### 3.3 Backend Type Extension

Modify `include/yirage/type.h`:

```cpp
enum BackendType {
  // GPU Backends
  BT_CUDA = 0,
  BT_MPS = 1,
  BT_CUDNN = 2,
  BT_CUSPARSELT = 3,
  
  // CPU Backends
  BT_CPU = 10,
  BT_MKL = 11,
  BT_MKLDNN = 12,
  BT_OPENMP = 13,
  BT_XEON = 14,
  
  // Specialized Backends
  BT_NKI = 20,
  BT_TRITON = 21,
  BT_MHA = 22,
  BT_NNPACK = 23,
  BT_OPT_EINSUM = 24,
  
  BT_UNKNOWN = 999,
};

// Backend metadata
struct BackendInfo {
    BackendType type;
    std::string name;
    std::string display_name;
    bool requires_gpu;
    std::vector<std::string> required_libs;
};
```

## 4. Build System Modification

### 4.1 config.cmake Modification

Change from mutex selection to multi-selection:

```cmake
# Backend options - multiple selections allowed
set(USE_CUDA ON)
set(USE_CPU ON)
set(USE_MPS OFF)
set(USE_CUDNN OFF)
set(USE_MKL OFF)
set(USE_MKLDNN OFF)
set(USE_OPENMP ON)
set(USE_NKI OFF)
set(USE_TRITON ON)
set(USE_CUSPARSELT OFF)
set(USE_MHA OFF)
set(USE_NNPACK OFF)
set(USE_OPT_EINSUM OFF)
set(USE_XEON OFF)

# Other options
set(BUILD_CPP_EXAMPLES OFF)
set(USE_FORMAL_VERIFIER OFF)
```

### 4.2 CMakeLists.txt Modification

```cmake
# Check and enable backends
if(USE_CUDA)
    add_compile_definitions(YIRAGE_BACKEND_CUDA_ENABLED)
    # CUDA specific setup
endif()

if(USE_CPU)
    add_compile_definitions(YIRAGE_BACKEND_CPU_ENABLED)
    # CPU specific setup
endif()

if(USE_MPS)
    add_compile_definitions(YIRAGE_BACKEND_MPS_ENABLED)
    # MPS specific setup
endif()

# ... other backends
```

### 4.3 setup.py Modification

```python
def get_backend_macros(config_file):
    """Read config.cmake and return all enabled backend macros"""
    backend_flags = {
        "USE_CUDA": None,
        "USE_CPU": None,
        "USE_MPS": None,
        "USE_CUDNN": None,
        "USE_MKL": None,
        "USE_MKLDNN": None,
        "USE_OPENMP": None,
        "USE_NKI": None,
        "USE_TRITON": None,
        # ... other backends
    }
    
    pattern = re.compile(r'^\s*set\s*\(\s*([A-Z_]+)\s+(ON|OFF)\s*\)')
    
    with open(config_file, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                var, val = match.groups()
                if var in backend_flags:
                    backend_flags[var] = (val.upper() == "ON")
    
    macros = []
    # Add macro for each enabled backend
    for flag_name, enabled in backend_flags.items():
        if enabled:
            backend_name = flag_name.replace("USE_", "")
            macros.append((f"YIRAGE_BACKEND_{backend_name}_ENABLED", None))
    
    # At least one backend required
    if not any(backend_flags.values()):
        raise ValueError("At least one backend must be enabled in config.cmake")
    
    return macros
```

## 5. Configuration File Modification

### 5.1 config.h Modification

Remove mutex check, change to on-demand configuration:

```cpp
namespace yirage {
namespace config {

// Configuration for each backend
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    namespace cuda {
        size_t const MAX_DMEM_SIZE = (size_t)2 * 1024 * 1024 * 1024;    // 2 GB
        size_t const MAX_SMEM_SIZE = 96 * 1024;                         // 96 KB
    }
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
    namespace cpu {
        size_t const MAX_DMEM_SIZE = (size_t)64 * 1024 * 1024 * 1024;   // 64 GB
        size_t const MAX_SMEM_SIZE = (size_t)32 * 1024 * 1024;          // 32 MB
    }
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
    namespace mps {
        size_t const MAX_DMEM_SIZE = (size_t)16 * 1024 * 1024 * 1024;   // 16 GB
        size_t const MAX_SMEM_SIZE = (size_t)64 * 1024;                 // 64 KB
    }
#endif

// ... other backend configurations

} // namespace config
} // namespace yirage
```

## 6. Python API Modification

### 6.1 PersistentKernel Extension

```python
class PersistentKernel:
    def __init__(
        self,
        mode: str,
        backend: str = "cuda",  # New: backend selection
        fallback_backends: list = None,  # New: fallback backend list
        world_size: int = 1,
        mpi_rank: int = 0,
        # ... other parameters
    ):
        # Verify backend availability
        available_backends = self._get_available_backends()
        if backend not in available_backends:
            if fallback_backends:
                for fb in fallback_backends:
                    if fb in available_backends:
                        backend = fb
                        break
                else:
                    raise ValueError(f"No available backend from {[backend] + fallback_backends}")
            else:
                raise ValueError(f"Backend '{backend}' not available. Available: {available_backends}")
        
        self.backend = backend
        # ... initialization
    
    @staticmethod
    def _get_available_backends():
        """Query compile-time enabled backends"""
        return core.get_available_backends()
```

### 6.2 New Backend Query API

```python
def get_available_backends() -> list:
    """Return all available backends"""
    return core.get_available_backends()

def is_backend_available(backend: str) -> bool:
    """Check if specified backend is available"""
    return backend in get_available_backends()

def get_backend_info(backend: str) -> dict:
    """Get detailed backend information"""
    return core.get_backend_info(backend)
```

## 7. Implementation Roadmap

### Phase 1: Basic Architecture (Week 1-2)
- [x] Extend BackendType enum
- [x] Implement BackendInterface abstract class
- [x] Implement BackendRegistry singleton
- [x] Modify config.cmake for multi-backend support
- [x] Modify CMakeLists.txt for multi-backend compilation
- [x] Modify setup.py for multi-backend macro definitions

### Phase 2: Core Backend Implementation (Week 3-4)
- [x] Refactor CUDA Backend (existing code)
- [x] Implement CPU Backend
- [x] Implement MPS Backend (macOS)
- [x] Implement CUDNN Backend
- [x] Refactor NKI Backend (existing code)

### Phase 3: Python Bindings (Week 5)
- [x] Update Cython bindings
- [x] Implement Python backend query API
- [x] Update PersistentKernel class
- [x] Update KNGraph.superoptimize

### Phase 4: Extended Backends (Week 6-8)
- [x] Implement MKL Backend
- [x] Implement MKLDNN Backend (as MKL extension)
- [x] Implement OpenMP Backend (integrated with CPU)
- [x] Implement Triton Backend (integrate existing code)
- [x] Implement other backends

### Phase 5: Testing & Documentation (Week 9-10)
- [x] Unit tests
- [x] Integration tests
- [x] Performance tests
- [x] Documentation updates
- [x] Example code

**Status**: ✅ All phases completed

## 8. Backward Compatibility

- Default backend remains CUDA
- Old config.cmake configuration still valid
- Python API default parameters unchanged
- Compile macro `YIRAGE_BACKEND_USE_CUDA` automatically maps to `YIRAGE_BACKEND_CUDA_ENABLED`

## 9. Testing Strategy

### 9.1 Unit Tests
- Independent testing for each Backend implementation
- BackendRegistry registration and query tests
- Multi-backend compilation system tests

### 9.2 Integration Tests
- Multi-backend simultaneous compilation tests
- Runtime backend switching tests
- Fallback mechanism tests

### 9.3 Performance Tests
- Performance benchmarks for each backend
- Backend switching overhead tests

## 10. Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Breaking existing CUDA code | High | Maintain backward compatibility, thorough testing |
| Increased compilation time | Medium | Support selective backend compilation |
| Large binary size | Medium | Modular design, dynamic loading |
| Missing backend libraries | Low | Compile-time detection, graceful degradation |

## 11. Dependencies

| Backend | Required Libraries | Optional Libraries |
|---------|-------------------|-------------------|
| CUDA | CUDA Toolkit, cuBLAS | CUDNN, cuSPARSELt |
| CPU | - | OpenMP |
| MPS | Metal (macOS only) | - |
| MKL | Intel MKL | - |
| MKLDNN | oneDNN | - |
| NKI | AWS Neuron SDK | - |
| Triton | PyTorch, Triton | - |

## 12. Documentation Updates

Documentation requiring updates:
- INSTALL.md - Add installation instructions for each backend
- README.md - Add multi-backend feature description
- API docs - Add backend selection API
- Tutorials - Add multi-backend usage examples

---

**Document Version**: v1.0  
**Created**: 2025-11-21  
**Last Updated**: 2025-11-21  
**Status**: ✅ Implementation Complete
