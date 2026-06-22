# YPK Multi-Backend C++ Design Document

## Current State Analysis

The YPK (YiRage Persistent Kernel) system is currently **CUDA-only** at the C++ level. The following issues need to be addressed:

### 1. CUDA-Specific Dependencies

```cpp
// persistent_kernel.cuh - Line 62-70
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        // ...
    } while (0)
#endif
```

**All files with CUDA dependencies:**
- `persistent_kernel.cuh` - Core persistent kernel
- `runtime_header.h` - Runtime configuration
- `tma.cuh` - TMA operations (Hopper-specific)
- `tasks/*.cuh` - All task implementations

### 2. Architecture-Specific Task Headers

```cpp
// persistent_kernel.cuh - Lines 32-38
#if defined(YIRAGE_GRACE_HOPPER)
#include "tasks/hopper/task_header.cuh"
#elif defined(YIRAGE_GRACE_BLACKWELL)
#include "tasks/blackwell/task_header.cuh"
#else
#include "tasks/ampere/task_header.cuh"
#endif
```

### 3. Mode-Specific Code

```cpp
// runtime_header.h - Line 253
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  int *prompt_length;
  int *request_ids;
  // ...
#endif
```

---

## Required C++ Changes

### Phase 1: Backend Abstraction Layer

#### 1.1 Create `PersistentKernelBackend` Interface

**File:** `include/persistent_kernel/backend_interface.h`

```cpp
#pragma once

#include <string>
#include <vector>
#include <memory>

namespace yirage {
namespace persistent_kernel {

enum class PKBackendType {
    CUDA,
    CPU,
    MPS,
    ASCEND,
    MACA,
    TRITON,
    NKI
};

enum class PKMode {
    OFFLINE,    // Pre-compile all kernels
    ONLINE,     // JIT compile as needed  
    ONEPASS,    // Single-pass execution
    EAGER,      // Immediate execution
    GRAPH,      // Graph-based execution
    STREAMING   // Streaming/pipelined
};

struct PKCapabilities {
    bool supports_tma;
    bool supports_tensor_cores;
    bool supports_async_copy;
    bool supports_nvshmem;
    size_t max_shared_memory;
    size_t max_threads_per_block;
    std::vector<PKMode> supported_modes;
};

class PersistentKernelBackend {
public:
    virtual ~PersistentKernelBackend() = default;
    
    // Backend info
    virtual PKBackendType get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual bool is_available() const = 0;
    virtual PKCapabilities get_capabilities() const = 0;
    
    // Mode support
    virtual bool supports_mode(PKMode mode) const = 0;
    virtual PKMode get_default_mode() const = 0;
    
    // Memory management (backend-agnostic)
    virtual void* allocate(size_t size) = 0;
    virtual void free(void* ptr) = 0;
    virtual void copy_h2d(void* dst, const void* src, size_t size) = 0;
    virtual void copy_d2h(void* dst, const void* src, size_t size) = 0;
    virtual void copy_d2d(void* dst, const void* src, size_t size) = 0;
    virtual void synchronize() = 0;
    
    // Kernel execution
    virtual void launch_worker_kernel(
        const RuntimeConfig& config,
        int num_workers,
        int threads_per_worker
    ) = 0;
    
    virtual void launch_scheduler_kernel(
        const RuntimeConfig& config
    ) = 0;
    
    // Task registration
    virtual void register_task(
        TaskType type,
        const char* name,
        void* task_func
    ) = 0;
    
    // Compile-time configuration
    virtual std::string get_compile_flags(PKMode mode) const = 0;
};

// Backend factory
std::unique_ptr<PersistentKernelBackend> 
create_pk_backend(PKBackendType type);

} // namespace persistent_kernel
} // namespace yirage
```

### Phase 2: CUDA Backend Implementation

#### 2.1 CUDA-Specific Backend

**File:** `include/persistent_kernel/cuda_pk_backend.h`

```cpp
#pragma once

#include "backend_interface.h"

#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include <cuda_runtime.h>

namespace yirage {
namespace persistent_kernel {

class CudaPKBackend : public PersistentKernelBackend {
public:
    CudaPKBackend(int device_id = 0);
    ~CudaPKBackend() override;
    
    PKBackendType get_type() const override { return PKBackendType::CUDA; }
    std::string get_name() const override { return "cuda"; }
    bool is_available() const override;
    
    PKCapabilities get_capabilities() const override {
        return PKCapabilities{
            .supports_tma = (compute_capability_ >= 90),
            .supports_tensor_cores = (compute_capability_ >= 70),
            .supports_async_copy = true,
            .supports_nvshmem = true,
            .max_shared_memory = max_shared_memory_,
            .max_threads_per_block = 1024,
            .supported_modes = {PKMode::OFFLINE, PKMode::ONLINE, 
                               PKMode::ONEPASS, PKMode::GRAPH}
        };
    }
    
    bool supports_mode(PKMode mode) const override;
    PKMode get_default_mode() const override { return PKMode::ONLINE; }
    
    // Memory operations using CUDA
    void* allocate(size_t size) override;
    void free(void* ptr) override;
    void copy_h2d(void* dst, const void* src, size_t size) override;
    void copy_d2h(void* dst, const void* src, size_t size) override;
    void copy_d2d(void* dst, const void* src, size_t size) override;
    void synchronize() override;
    
    // CUDA kernel launch
    void launch_worker_kernel(
        const RuntimeConfig& config,
        int num_workers,
        int threads_per_worker
    ) override;
    
    void launch_scheduler_kernel(
        const RuntimeConfig& config
    ) override;
    
    void register_task(TaskType type, const char* name, void* task_func) override;
    
    std::string get_compile_flags(PKMode mode) const override;

private:
    int device_id_;
    int compute_capability_;
    size_t max_shared_memory_;
    cudaStream_t worker_stream_;
    cudaStream_t scheduler_stream_;
};

} // namespace persistent_kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDA_ENABLED
```

### Phase 3: CPU Backend Implementation

#### 3.1 CPU Backend for EAGER/GRAPH Modes

**File:** `include/persistent_kernel/cpu_pk_backend.h`

```cpp
#pragma once

#include "backend_interface.h"
#include <thread>
#include <atomic>

namespace yirage {
namespace persistent_kernel {

class CpuPKBackend : public PersistentKernelBackend {
public:
    CpuPKBackend(int num_threads = 0);  // 0 = auto-detect
    ~CpuPKBackend() override;
    
    PKBackendType get_type() const override { return PKBackendType::CPU; }
    std::string get_name() const override { return "cpu"; }
    bool is_available() const override { return true; }
    
    PKCapabilities get_capabilities() const override {
        return PKCapabilities{
            .supports_tma = false,
            .supports_tensor_cores = false,
            .supports_async_copy = true,
            .supports_nvshmem = false,
            .max_shared_memory = 0,
            .max_threads_per_block = num_threads_,
            .supported_modes = {PKMode::EAGER, PKMode::GRAPH, PKMode::OFFLINE}
        };
    }
    
    bool supports_mode(PKMode mode) const override;
    PKMode get_default_mode() const override { return PKMode::EAGER; }
    
    // CPU memory (just malloc/free)
    void* allocate(size_t size) override;
    void free(void* ptr) override;
    void copy_h2d(void* dst, const void* src, size_t size) override;
    void copy_d2h(void* dst, const void* src, size_t size) override;
    void copy_d2d(void* dst, const void* src, size_t size) override;
    void synchronize() override;
    
    // CPU "kernel" launch (thread pool)
    void launch_worker_kernel(
        const RuntimeConfig& config,
        int num_workers,
        int threads_per_worker
    ) override;
    
    void launch_scheduler_kernel(const RuntimeConfig& config) override;
    
    void register_task(TaskType type, const char* name, void* task_func) override;
    
    std::string get_compile_flags(PKMode mode) const override;

private:
    int num_threads_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_;
};

} // namespace persistent_kernel
} // namespace yirage
```

### Phase 4: Ascend NPU Backend

**File:** `include/persistent_kernel/ascend_pk_backend.h`

```cpp
#pragma once

#include "backend_interface.h"

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
#include <acl/acl.h>

namespace yirage {
namespace persistent_kernel {

class AscendPKBackend : public PersistentKernelBackend {
public:
    AscendPKBackend(int device_id = 0);
    ~AscendPKBackend() override;
    
    PKBackendType get_type() const override { return PKBackendType::ASCEND; }
    std::string get_name() const override { return "ascend"; }
    bool is_available() const override;
    
    PKCapabilities get_capabilities() const override {
        return PKCapabilities{
            .supports_tma = false,
            .supports_tensor_cores = true,  // AI Core
            .supports_async_copy = true,
            .supports_nvshmem = false,
            .max_shared_memory = 512 * 1024,  // Ascend 910B
            .max_threads_per_block = 256,
            .supported_modes = {PKMode::OFFLINE, PKMode::ONLINE, PKMode::GRAPH}
        };
    }
    
    bool supports_mode(PKMode mode) const override;
    PKMode get_default_mode() const override { return PKMode::ONLINE; }
    
    // ACL memory operations
    void* allocate(size_t size) override;
    void free(void* ptr) override;
    void copy_h2d(void* dst, const void* src, size_t size) override;
    void copy_d2h(void* dst, const void* src, size_t size) override;
    void copy_d2d(void* dst, const void* src, size_t size) override;
    void synchronize() override;
    
    // Ascend kernel launch
    void launch_worker_kernel(
        const RuntimeConfig& config,
        int num_workers,
        int threads_per_worker
    ) override;
    
    void launch_scheduler_kernel(const RuntimeConfig& config) override;
    
    void register_task(TaskType type, const char* name, void* task_func) override;
    
    std::string get_compile_flags(PKMode mode) const override;

private:
    int device_id_;
    aclrtContext context_;
    aclrtStream stream_;
};

} // namespace persistent_kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED
```

### Phase 5: Task Abstraction

#### 5.1 Backend-Agnostic Task Interface

**File:** `include/persistent_kernel/task_interface.h`

```cpp
#pragma once

#include "backend_interface.h"

namespace yirage {
namespace persistent_kernel {

// Task execution context (backend-agnostic)
struct TaskContext {
    void* shared_memory;
    size_t shared_memory_size;
    int thread_id;
    int block_id;
    int num_threads;
    int num_blocks;
};

// Base task class
class Task {
public:
    virtual ~Task() = default;
    
    virtual TaskType get_type() const = 0;
    virtual const char* get_name() const = 0;
    
    // Execute on current backend
    virtual void execute(
        const TaskContext& ctx,
        const TaskDesc& desc,
        const RuntimeConfig& config
    ) = 0;
    
    // Backend-specific execution
    virtual bool supports_backend(PKBackendType backend) const = 0;
};

// Task registry
class TaskRegistry {
public:
    static TaskRegistry& instance();
    
    void register_task(std::unique_ptr<Task> task);
    Task* get_task(TaskType type) const;
    
    std::vector<TaskType> get_tasks_for_backend(PKBackendType backend) const;
    
private:
    std::unordered_map<TaskType, std::unique_ptr<Task>> tasks_;
};

// Macro for registering tasks
#define REGISTER_PK_TASK(TaskClass) \
    static struct TaskClass##Registrar { \
        TaskClass##Registrar() { \
            TaskRegistry::instance().register_task( \
                std::make_unique<TaskClass>()); \
        } \
    } task_##TaskClass##_registrar;

} // namespace persistent_kernel
} // namespace yirage
```

---

## Implementation Roadmap

### Phase 1: Backend Interface (2 weeks)
- [ ] Create `PersistentKernelBackend` interface
- [ ] Create `Task` interface
- [ ] Create `TaskRegistry`
- [ ] Refactor `RuntimeConfig` to be backend-agnostic

### Phase 2: CUDA Backend (1 week)
- [ ] Implement `CudaPKBackend`
- [ ] Migrate existing CUDA tasks to new interface
- [ ] Test CUDA backend with all modes

### Phase 3: CPU Backend (1 week)
- [ ] Implement `CpuPKBackend`
- [ ] Create CPU versions of core tasks
- [ ] Test CPU EAGER/GRAPH modes

### Phase 4: Ascend/MACA Backends (2 weeks)
- [ ] Implement `AscendPKBackend`
- [ ] Implement `MacaPKBackend`
- [ ] Create backend-specific task variants

### Phase 5: Integration (1 week)
- [ ] Update Python bindings
- [ ] Update compilation pipeline
- [ ] End-to-end testing

---

## Files to Modify

| File | Changes Required |
|------|------------------|
| `runtime_header.h` | Add backend abstraction, remove CUDA-specific code |
| `persistent_kernel.cuh` | Split into backend-specific implementations |
| `tasks/*.cuh` | Convert to backend-agnostic task interface |
| `mpk_atoms.cuh` | Create backend variants |
| `profiler.h` | Add backend-agnostic profiling |

---

## Compile-Time Flags

```cmake
# New CMake options
option(YIRAGE_PK_CUDA "Enable CUDA persistent kernel backend" ON)
option(YIRAGE_PK_CPU "Enable CPU persistent kernel backend" ON)
option(YIRAGE_PK_ASCEND "Enable Ascend persistent kernel backend" OFF)
option(YIRAGE_PK_MACA "Enable MACA persistent kernel backend" OFF)
option(YIRAGE_PK_MPS "Enable MPS persistent kernel backend" OFF)
```

---

## Summary

The current YPK implementation requires significant refactoring to support multiple backends. The key changes are:

1. **Abstract Backend Interface** - Decouple from CUDA
2. **Task Registry** - Register tasks per backend
3. **Memory Abstraction** - Unified memory operations
4. **Mode Support** - Per-backend mode validation
5. **Compile Pipeline** - Backend-specific compilation

This design maintains backward compatibility while enabling new backends.
