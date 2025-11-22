# Changelog - Multi-Backend Support

## [Unreleased] - 2025-11-21

### Added

#### Core Architecture
- **Backend Abstraction Layer** (`include/yirage/backend/backend_interface.h`)
  - Abstract `BackendInterface` class defining unified backend API
  - Methods for memory management, compilation, device queries
  - Support for multi-device backends

- **Backend Registry** (`include/yirage/backend/backend_registry.h`)
  - Singleton `BackendRegistry` for managing all backends
  - Thread-safe backend registration and retrieval
  - `REGISTER_BACKEND` macro for automatic registration
  - Default backend selection logic (prefers CUDA > MPS > CPU)

#### Backend Implementations
- **CUDA Backend** (`src/backend/cuda_backend.cc`) - ✅ Fully Implemented
  - Complete CUDA runtime integration
  - Memory allocation/deallocation via `cudaMalloc`/`cudaFree`
  - Device property queries (compute capability, memory, SMs)
  - Multi-device support
  - Data type support including FP16, BF16, INT8

- **CPU Backend** (`src/backend/cpu_backend.cc`) - ✅ Fully Implemented
  - Standard library-based memory management
  - System information queries (cores, memory, cache)
  - Cross-platform support (Linux, macOS, Windows)
  - OpenMP integration ready

- **MPS Backend** (`src/backend/mps_backend.cc`) - ⚠️ Skeleton Implementation
  - Apple Metal Performance Shaders support (macOS only)
  - Basic structure in place, core functions TODO

#### Type System
- **Extended BackendType Enum** (`include/yirage/type.h`)
  - Added 13 new backend types:
    - GPU: `BT_MPS`, `BT_CUDNN`, `BT_CUSPARSELT`
    - CPU: `BT_CPU`, `BT_MKL`, `BT_MKLDNN`, `BT_OPENMP`, `BT_XEON`
    - Specialized: `BT_TRITON`, `BT_MHA`, `BT_NNPACK`, `BT_OPT_EINSUM`
  - Kept existing `BT_CUDA` and `BT_NKI` for compatibility

- **BackendInfo Structure**
  - Metadata about each backend (name, display name, GPU requirement, libraries)

- **Utility Functions**
  - `backend_type_to_string()` - Convert enum to string
  - `string_to_backend_type()` - Convert string to enum

#### Build System
- **Multi-Backend CMake Configuration** (`CMakeLists.txt`)
  - Support for compiling multiple backends simultaneously
  - Individual compile definitions for each backend:
    - `YIRAGE_BACKEND_CUDA_ENABLED`
    - `YIRAGE_BACKEND_CPU_ENABLED`
    - `YIRAGE_BACKEND_MPS_ENABLED`
    - ... etc
  - Backward-compatible macros (`YIRAGE_BACKEND_USE_CUDA`, `YIRAGE_BACKEND_USE_NKI`)
  - Automatic backend source file inclusion
  - OpenMP integration when `USE_OPENMP=ON`

- **Updated config.cmake**
  - Changed from mutually-exclusive to multi-select configuration
  - Added switches for all 14 supported backends
  - Default: CUDA, CPU, OpenMP, Triton enabled

- **Updated setup.py**
  - Modified `get_backend_macros()` to support multiple backends
  - Validates at least one backend is enabled
  - Prints list of enabled backends during build

#### Python API
- **Backend Query Module** (`python/yirage/backend_api.py`)
  - `get_available_backends()` - List all available backends
  - `is_backend_available(backend)` - Check specific backend
  - `get_default_backend()` - Get default backend name
  - `get_backend_info(backend)` - Get detailed backend information
  - `set_default_backend(backend)` - Set default backend
  - `list_backends(verbose)` - Print backend information
  - Convenience aliases: `available_backends`, `default_backend`

- **Updated __init__.py**
  - Exported all backend query functions
  - Integrated with existing API

#### C++ Convenience Functions
- **Backend Utilities** (`include/yirage/backend/backends.h`, `src/backend/backends.cc`)
  - `initialize_backends()` - Initialize all compiled backends (auto-called)
  - `get_available_backend_names()` - Get backend names as strings
  - `get_default_backend()` - Get default backend pointer
  - `is_backend_available(name)` - Check backend availability
  - `get_backend_by_name(name)` - Get backend by name
  - Auto-initialization via static constructor

#### Documentation
- **Design Document** (`docs/ypk/multi_backend_design.md`)
  - Complete architecture design
  - Implementation roadmap
  - Risk analysis and mitigation
  - Dependency requirements

- **Usage Guide** (`docs/ypk/backend_usage.md`)
  - Python and C++ examples
  - Backend selection strategies
  - Performance considerations
  - Troubleshooting guide

- **Implementation Summary** (`docs/ypk/MULTI_BACKEND_IMPLEMENTATION_SUMMARY.md`)
  - Complete file listing
  - Architecture overview
  - Status of each backend
  - Known limitations

#### Tests and Examples
- **C++ Backend Test** (`tests/backend/test_backend_registry.cc`)
  - Tests backend registry singleton
  - Tests backend query functions
  - Tests data type support
  - Tests device properties

- **Python Demo** (`demo/backend_selection_demo.py`)
  - Demonstrates backend queries
  - Shows fallback backend selection
  - Example of backend-specific configuration

### Changed

#### Breaking Changes
**None** - All changes are backward compatible

#### Type System
- `BackendType` enum now uses grouped numbering:
  - GPU backends: 0-9
  - CPU backends: 10-19
  - Specialized backends: 20-29

#### Build System
- `config.cmake` now allows multiple backends to be enabled simultaneously
- CMake no longer enforces mutual exclusion between CUDA and NKI
- Build time may increase when multiple backends are enabled

### Deprecated
- None (old macros still supported)

### Backward Compatibility
- ✅ Old `YIRAGE_BACKEND_USE_CUDA` macro still defined when CUDA enabled
- ✅ Old `YIRAGE_BACKEND_USE_NKI` macro still defined when NKI enabled
- ✅ Old config.cmake format still works (single backend selection)
- ✅ Default backend remains CUDA (if available)
- ✅ All existing code works without modification

### Security
- Thread-safe backend registry using std::mutex
- Null pointer checks in all backend queries
- Safe singleton pattern for registry

### Performance
- Backend registration happens at static initialization (zero runtime cost)
- Backend lookups use `std::unordered_map` (O(1) average)
- No performance impact on existing single-backend workflows

## Future Work

### Planned for Next Release
- [ ] Complete MPS backend implementation
- [ ] Cython bindings for C++ backend API
- [ ] CUDNN backend implementation
- [ ] PersistentKernel backend parameter support
- [ ] Unit tests for Python backend API

### Future Releases
- [ ] MKL/MKLDNN backend
- [ ] OpenMP backend
- [ ] Integrate existing Triton code
- [ ] Migrate existing NKI code
- [ ] Benchmark suite for backend comparison
- [ ] Auto backend selection based on hardware
- [ ] Mixed backend support

---

**Version**: 1.0.0-alpha  
**Date**: 2025-11-21  
**Contributors**: YiRage Team

