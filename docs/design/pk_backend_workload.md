# Persistent Kernel Backend Workload Plan

## Overview

This document tracks the implementation status and workload for each Persistent Kernel (PK) backend in the YiRage framework.

## Backend Summary

| Backend | Workload | Type | Status | Supported Modes |
|---------|----------|------|--------|-----------------|
| CUDA | 1 week | Refactor | ✅ Done | OFFLINE, ONLINE, ONEPASS, GRAPH |
| CPU | 1 week | New | ✅ Done | EAGER, GRAPH, OFFLINE |
| Ascend | 2 weeks | New | ✅ Done | OFFLINE, ONLINE, GRAPH |
| MACA | 2 weeks | New | ✅ Done | OFFLINE, ONLINE, ONEPASS |
| MPS | 2 weeks | New | ✅ Done | EAGER, GRAPH |

## Implementation Details

### CUDA Backend (1 week - Refactor)

**File Locations:**
- Header: `include/persistent_kernel/backends/cuda_pk_backend.h`
- Implementation: `src/persistent_kernel/cuda_pk_backend.cc`

**Capabilities:**
- ✅ TMA (Tensor Memory Accelerator) support for H100+
- ✅ Tensor Cores support
- ✅ Async copy operations
- ✅ NVSHMEM for multi-GPU
- ✅ FP8 on Hopper+

**Supported Modes:**
- `OFFLINE`: Pre-compiled kernel graphs
- `ONLINE`: Persistent kernel loop for LLM serving
- `ONEPASS`: Single-pass execution
- `GRAPH`: CUDA Graph capture and replay

**Default Mode:** `ONLINE`

---

### CPU Backend (1 week - New)

**File Locations:**
- Header: `include/persistent_kernel/backends/cpu_pk_backend.h`
- Implementation: `src/persistent_kernel/cpu_pk_backend.cc`
- Kernels: `src/persistent_kernel/pk_cpu_kernels.cc`

**Capabilities:**
- ✅ AVX/AVX512 SIMD support
- ✅ OpenMP parallel execution
- ✅ Reference implementations for all tasks

**Supported Modes:**
- `EAGER`: Immediate execution (default)
- `GRAPH`: Execution graph for optimization
- `OFFLINE`: Pre-planned execution

**Default Mode:** `EAGER`

---

### Ascend Backend (2 weeks - New)

**File Locations:**
- Header: `include/persistent_kernel/backends/ascend_pk_backend.h`
- Implementation: `src/persistent_kernel/ascend_pk_backend.cc`

**Capabilities:**
- ✅ AI Core / Cube Core for matrix ops
- ✅ HCCL for multi-NPU communication
- ✅ ACL (Ascend Computing Language) integration

**Supported Modes:**
- `OFFLINE`: Operator compilation and caching
- `ONLINE`: Continuous NPU inference
- `GRAPH`: CANN graph optimization

**Default Mode:** `ONLINE`

**Hardware Targets:**
- Ascend 910 (training)
- Ascend 310 (inference)
- Ascend 910B (next-gen)

---

### MACA Backend (2 weeks - New)

**File Locations:**
- Header: `include/persistent_kernel/backends/maca_pk_backend.h`
- Implementation: `src/persistent_kernel/maca_pk_backend.cc`

**Capabilities:**
- ✅ CUDA-compatible API (mc runtime)
- ✅ Tensor Core equivalents
- ✅ Async memory operations

**Supported Modes:**
- `OFFLINE`: Pre-compiled kernels
- `ONLINE`: Persistent kernel execution
- `ONEPASS`: Single execution pass

**Default Mode:** `ONLINE`

**Hardware Targets:**
- MetaX C500 series GPUs

---

### MPS Backend (2 weeks - New)

**File Locations:**
- Header: `include/persistent_kernel/backends/mps_pk_backend.h`
- Implementation: `src/persistent_kernel/mps_pk_backend.cc`

**Capabilities:**
- ✅ Unified memory architecture
- ✅ Metal compute shaders
- ✅ MPSGraph optimization

**Supported Modes:**
- `EAGER`: Command buffer execution (default)
- `GRAPH`: MPSGraph compiled execution

**Default Mode:** `EAGER`

**Hardware Targets:**
- Apple M1/M2/M3 chips
- Apple M1/M2/M3 Pro/Max/Ultra

---

## File Structure

```
include/persistent_kernel/
├── pk_backend_interface.h      # Core interfaces
├── pk_runtime_adapter.h        # Mode adapters
├── pk_utils.h                  # Utilities
├── pk_task_kernels.h           # Task specifications
└── backends/
    ├── pk_backends.h           # Unified header
    ├── cuda_pk_backend.h
    ├── cpu_pk_backend.h
    ├── ascend_pk_backend.h
    ├── maca_pk_backend.h
    └── mps_pk_backend.h

src/persistent_kernel/
├── pk_backend_factory.cc       # Backend factory
├── pk_runtime_adapter.cc       # Mode adapters impl
├── pk_cpu_kernels.cc           # CPU kernel implementations
├── cuda_pk_backend.cc
├── cpu_pk_backend.cc
├── ascend_pk_backend.cc
├── maca_pk_backend.cc
└── mps_pk_backend.cc
```

## Code Statistics

| Component | Lines of Code | Description |
|-----------|---------------|-------------|
| pk_backend_interface.h | ~550 | Core interfaces |
| pk_runtime_adapter.h | ~400 | Mode adapters |
| pk_utils.h | ~420 | Utilities |
| pk_task_kernels.h | ~330 | Task specifications |
| cuda_pk_backend.* | ~780 | CUDA implementation |
| cpu_pk_backend.* | ~580 | CPU implementation |
| ascend_pk_backend.* | ~770 | Ascend implementation |
| maca_pk_backend.* | ~780 | MACA implementation |
| mps_pk_backend.* | ~650 | MPS implementation |
| pk_cpu_kernels.cc | ~360 | CPU reference kernels |

**Total:** ~5,600+ lines of C++ code

## Python Integration

| File | Lines | Description |
|------|-------|-------------|
| pk_backend.pxd | ~200 | Cython declarations |
| pk_backend.pyx | ~540 | Cython wrappers |
| ypk_integration.py | ~600 | Python YPK module |

## Testing

| Test File | Description |
|-----------|-------------|
| tests/persistent_kernel/test_pk_backends.cc | C++ unit tests |
| tests/python/test_pk_backends.py | Python integration tests |

## Mode Matrix

| Mode | CUDA | CPU | Ascend | MACA | MPS |
|------|------|-----|--------|------|-----|
| OFFLINE | ✅ | ✅ | ✅ | ✅ | ❌ |
| ONLINE | ✅ | ❌ | ✅ | ✅ | ❌ |
| ONEPASS | ✅ | ❌ | ❌ | ✅ | ❌ |
| EAGER | ❌ | ✅ | ❌ | ❌ | ✅ |
| GRAPH | ✅ | ✅ | ✅ | ❌ | ✅ |
| STREAMING | ❌ | ❌ | ❌ | ❌ | ❌ |

## Task Support Matrix

| Task | CUDA | CPU | Ascend | MACA | MPS |
|------|------|-----|--------|------|-----|
| EMBEDDING | ✅ | ✅ | ✅ | ✅ | ✅ |
| RMS_NORM | ✅ | ✅ | ✅ | ✅ | ✅ |
| LINEAR | ✅ | ✅ | ✅ | ✅ | ✅ |
| ATTENTION | ✅ | ✅ | ✅ | ✅ | ✅ |
| ROTARY_EMB | ✅ | ✅ | ✅ | ✅ | ✅ |
| SILU_MUL | ✅ | ✅ | ✅ | ✅ | ✅ |
| ARGMAX | ✅ | ✅ | ✅ | ✅ | ✅ |
| PAGED_ATTN | ✅ | ❌ | ✅ | ✅ | ❌ |
| MOE_GATE | ✅ | ✅ | ❌ | ✅ | ❌ |
| MOE_LINEAR | ✅ | ❌ | ❌ | ✅ | ❌ |
| ALLREDUCE | ✅ | ❌ | ✅ | ✅ | ❌ |

## Usage Examples

### C++ Usage

```cpp
#include "persistent_kernel/backends/pk_backends.h"

using namespace yirage::persistent_kernel;

// Get best available backend
auto backend = create_best_backend(0);

// Configure
PKRuntimeConfig config;
config.mode = backend->get_default_mode();
config.num_workers = 4;

// Initialize and run
if (backend->initialize(config)) {
    backend->launch_worker_kernel(config, 4, 256);
    backend->synchronize();
    backend->finalize();
}
```

### Python Usage

```python
from yirage.distributed.ypk_integration import (
    YPKBackend, YPKMode,
    get_available_backends,
    create_ypk_session
)

# Get available backends
backends = get_available_backends()

# Create session with best backend
session = create_ypk_session(
    backend=YPKBackend.CUDA,
    mode=YPKMode.ONLINE,
    device_id=0
)

# Use session for LLM inference
results = session.run(input_tokens)
```

## Persistent Kernel Runtime Implementation

The CUDA `persistent_kernel.cuh` worker-scheduler model has been ported to other backends:

### Core Components (matching CUDA implementation)

| Component | CUDA | CPU | Ascend | MACA |
|-----------|------|-----|--------|------|
| Worker kernel | `worker_kernel()` | `PKWorker` class | ACL operators | `maca_worker_kernel()` |
| Scheduler kernel | `scheduler_kernel()` | `PKScheduler` class | Host thread | `maca_scheduler_kernel()` |
| Atomic ops | PTX assembly | `std::atomic` | `std::atomic` | mc PTX |
| Task queue | Device memory | Host vectors | Host vectors | Device memory |
| Event counter | Device atomic | `atomic<uint64_t>` | `atomic<uint64_t>` | Device atomic |
| Batch prep | Device function | `prepare_next_batch()` | Host function | Device function |

### Runtime Files

| Backend | Runtime Header | Key Classes |
|---------|----------------|-------------|
| Core | `pk_runtime_core.h` | `PKWorker`, `PKScheduler`, `PKRuntime` |
| CPU | `cpu_pk_runtime.h` | `CpuPKRuntime`, `CpuRuntimeConfig` |
| Ascend | `ascend_pk_runtime.h` | `AscendPKRuntime`, `AscendRuntimeConfig` |
| MACA | `maca_pk_runtime.h` | `MacaPKRuntime`, `MacaRuntimeConfig` |
| MPS | `mps_pk_runtime.h` | `MpsPKRuntime`, `MpsRuntimeConfig`, `MpsTaskExecutor` |

### CUDA Reference vs Multi-Backend Implementation

**CUDA Original (`persistent_kernel.cuh`):**
```cuda
// Device-side worker loop
__device__ __forceinline__ void execute_worker(RuntimeConfig config) {
    while (true) {
        // Wait for task in queue
        while (next_task_pos == last_task_pos) {
            last_task_pos = ld_acquire_gpu_u64(&config.worker_queue_last_ready_task_id[...]);
        }
        // Execute task
        _execute_task(task_desc, config);
        // Trigger event
        atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
    }
}
```

**CPU Port (`pk_runtime_core.h`):**
```cpp
// Host-side worker thread
void PKWorker::run() {
    while (running_) {
        // Wait for task in queue
        while (next_task_pos == last_task_pos) {
            last_task_pos = config_->worker_queue_last_ready_task_id[worker_id_]
                .load(std::memory_order_acquire);
        }
        // Execute task
        executor_(task, *config_);
        // Trigger event
        config_->all_event_counters[event_index]
            .fetch_add(1, std::memory_order_release);
    }
}
```

### Key Differences by Backend

| Feature | CUDA | CPU | Ascend | MACA | MPS |
|---------|------|-----|--------|------|-----|
| Parallelism | Thread blocks | std::thread | ACL streams | Thread blocks | Command buffers |
| Memory | Device global | Host heap | NPU HBM | Device global | Unified memory |
| Sync primitive | PTX atomics | C++11 atomics | Host atomics | mc atomics | std::atomic |
| Task dispatch | __device__ | OpenMP | CANN ops | __device__ | Metal shaders |
| Collective | NVSHMEM | N/A | HCCL | N/A | N/A |

### MPS-Specific Features

The MPS backend includes Metal Shader Language (MSL) kernels for:
- `embedding_kernel` - Token embedding lookup
- `rms_norm_kernel` - RMS normalization with threadgroup reduction
- `silu_mul_kernel` - SiLU activation with gating
- `gemm_kernel` - General matrix multiplication
- `attention_score_kernel` - Q·K^T attention scores
- `softmax_kernel` - Row-wise softmax with shared memory reduction
- `rotary_embedding_kernel` - Rotary positional embedding
- `argmax_kernel` - Vocabulary argmax for token selection

## Tooling and Infrastructure

### CMake Build System

The persistent kernel backends are built using CMake with conditional compilation:

**File:** `src/persistent_kernel/CMakeLists.txt`

```cmake
# Backend-specific source files
set(PK_BACKEND_SOURCES
    pk_backend_factory.cc
    pk_runtime_adapter.cc
    pk_cpu_kernels.cc
    cpu_pk_backend.cc
)

# Conditional CUDA backend
if(YIRAGE_BACKEND_CUDA_ENABLED)
    list(APPEND PK_BACKEND_SOURCES cuda_pk_backend.cc)
endif()

# Create library
add_library(yirage_pk_backends STATIC ${PK_BACKEND_SOURCES})
```

**Build flags:**
- `YIRAGE_BACKEND_CUDA_ENABLED` - Enable CUDA backend
- `YIRAGE_BACKEND_ASCEND_ENABLED` - Enable Ascend backend
- `YIRAGE_BACKEND_MACA_ENABLED` - Enable MACA backend
- `YIRAGE_BACKEND_MPS_ENABLED` - Enable MPS backend

### Python Runtime Interface

A high-level Python interface for the multi-backend runtime:

**File:** `python/yirage/pk_runtime.py`

```python
from yirage.pk_runtime import (
    PKRuntime, PKBackendType, PKMode,
    create_runtime, get_available_backends
)

# Create runtime with automatic backend selection
runtime = create_runtime(
    backend=PKBackendType.AUTO,
    num_workers=4
)

# Build and run task graph
with runtime:
    runtime.add_task(PKTaskType.EMBEDDING, inputs=[tokens])
    runtime.add_task(PKTaskType.RMS_NORM, inputs=[embeddings])
    runtime.run()
```

### Cython Bindings

The Cython bindings expose the C++ interfaces to Python:

**Files:**
- `python/yirage/_cython/pk_backend.pxd` - C++ declarations
- `python/yirage/_cython/pk_backend.pyx` - Python wrappers

### Multi-Backend Benchmark

A comprehensive benchmark script for comparing backends:

**File:** `benchmark/pk_multi_backend_benchmark.py`

```bash
# Benchmark all available backends
python benchmark/pk_multi_backend_benchmark.py --all

# Benchmark specific backend
python benchmark/pk_multi_backend_benchmark.py --backend cuda --mode online

# Custom configuration
python benchmark/pk_multi_backend_benchmark.py \
    --backend cpu \
    --batch-sizes 1 4 8 \
    --seq-lengths 128 256 512 \
    --output results.json
```

### C++ Demo

A unified C++ demo for the multi-backend system:

**File:** `examples/pk_multi_backend_demo.cc`

```bash
# Build and run
./pk_multi_backend_demo auto   # Best available
./pk_multi_backend_demo cpu    # Specific backend
./pk_multi_backend_demo all    # All backends
```

## Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Core Interface | ✅ Done | `pk_backend_interface.h` |
| Runtime Core | ✅ Done | `pk_runtime_core.h` |
| CUDA Backend | ✅ Done | `cuda_pk_backend.{h,cc}` |
| CPU Backend | ✅ Done | `cpu_pk_backend.{h,cc}`, `cpu_pk_runtime.h` |
| Ascend Backend | ✅ Done | `ascend_pk_backend.{h,cc}`, `ascend_pk_runtime.h` |
| MACA Backend | ✅ Done | `maca_pk_backend.{h,cc}`, `maca_pk_runtime.h` |
| MPS Backend | ✅ Done | `mps_pk_backend.{h,cc}`, `mps_pk_runtime.h` |
| CMake Config | ✅ Done | `src/persistent_kernel/CMakeLists.txt` |
| Python Runtime | ✅ Done | `python/yirage/pk_runtime.py` |
| Cython Bindings | ✅ Done | `pk_backend.{pxd,pyx}` |
| Benchmark Suite | ✅ Done | `pk_multi_backend_benchmark.py` |
| C++ Demo | ✅ Done | `pk_multi_backend_demo.cc` |
| Tests | ✅ Done | `tests/persistent_kernel/*.cc` |

## Next Steps

1. **STREAMING mode**: Implement multi-node streaming for CUDA
2. **Triton backend**: Add OpenAI Triton transpiler support
3. **NKI backend**: Add AWS Neuron (Trainium/Inferentia) support
4. **Performance tuning**: Profile and optimize each backend
5. **Real-world validation**: Test with actual LLM models
6. **Documentation**: Complete API reference documentation
