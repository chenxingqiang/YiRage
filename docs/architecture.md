# YiRage Architecture Documentation

This document provides comprehensive architecture diagrams for the YiRage multi-backend LLM inference optimization framework.

## System Overview

```mermaid
graph TB
    subgraph "User Interface"
        A[Python API]
        B[CLI Tools]
    end

    subgraph "Core Engine"
        C[Kernel Graph]
        D[Search Engine]
        E[Transpiler]
    end

    subgraph "Backend Layer"
        F[Backend Registry]
        G[Backend Factory]
        H[Strategy Factory]
    end

    subgraph "Hardware Backends"
        I[CUDA]
        J[CPU]
        K[MPS]
        L[Ascend]
        M[MACA]
        N[Triton]
        O[NKI]
        P[cuDNN]
        Q[MKL]
    end

    A --> C
    B --> C
    C --> D --> E
    E --> F
    F --> G --> I & J & K & L & M & N & O & P & Q
    F --> H

    style A fill:#e1f5fe
    style D fill:#fff3e0
    style F fill:#c8e6c9
```

## Backend Architecture

### Multi-Backend Support

```mermaid
graph LR
    subgraph "Python Layer"
        A[yirage.get_available_backends]
        B[yirage.PersistentKernel]
        C[graph.superoptimize]
    end

    subgraph "C++ Backend Manager"
        D[BackendRegistry<br/>Singleton]
        E[BackendInterface<br/>Abstract]
    end

    subgraph "Concrete Backends"
        F[CUDABackend]
        G[CPUBackend]
        H[MPSBackend]
        I[AscendBackend]
        J[MACABackend]
        K[TritonBackend]
        L[NKIBackend]
        M[CuDNNBackend]
        N[MKLBackend]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F & G & H & I & J & K & L & M & N

    style D fill:#fff3e0
    style E fill:#e8f5e9
```

### Backend Selection Flow

```mermaid
flowchart TD
    A[Start] --> B{Check Requested Backend}
    B --> C{Backend Available?}
    C -->|Yes| D[Use Requested Backend]
    C -->|No| E{Fallback Backends Defined?}
    E -->|Yes| F[Try Fallback Backends]
    E -->|No| G[Error: No Backend Available]
    F --> H{Fallback Available?}
    H -->|Yes| I[Use Fallback Backend]
    H -->|No| G
    D --> J[Initialize Backend]
    I --> J
    J --> K[Execute Kernel]
    K --> L[End]

    style D fill:#c8e6c9
    style I fill:#fff9c4
    style G fill:#ffcdd2
```

## Search Engine Architecture

### Superoptimization Pipeline

```mermaid
flowchart LR
    subgraph "Input"
        A[Kernel Graph<br/>User-defined operations]
    end

    subgraph "Search Phase"
        B[Candidate Generator<br/>Fusion possibilities]
        C[Fingerprint Verifier<br/>Correctness check]
        D[Performance Profiler<br/>Timing measurement]
    end

    subgraph "Backend Strategies"
        E[CUDA Strategy<br/>Tensor Core, Warp]
        F[MPS Strategy<br/>SIMD, Threadgroup]
        G[Ascend Strategy<br/>AI Core, Cube]
        H[MACA Strategy<br/>64-thread Warp]
        I[CPU Strategy<br/>AVX, Cache]
    end

    subgraph "Output"
        J[Optimized Kernel<br/>Best configuration]
    end

    A --> B --> C --> D
    D --> E & F & G & H & I
    E & F & G & H & I --> J

    style A fill:#e3f2fd
    style D fill:#fff8e1
    style J fill:#c8e6c9
```

### Fingerprint Verification

```mermaid
sequenceDiagram
    participant G as Graph
    participant FP as Fingerprint Engine
    participant GPU as GPU/NPU
    participant V as Verifier

    G->>FP: Submit candidate graph
    FP->>GPU: Run fingerprint kernel
    GPU-->>FP: Return fingerprint values
    FP->>V: Compare fingerprints
    V-->>FP: Equivalence result
    FP-->>G: Accept/Reject candidate
```

## Hardware Backend Details

### CUDA Backend

```mermaid
graph TB
    subgraph "CUDA Backend"
        A[CUDAOptimizer]
        B[CUDAKernelConfig]
        C[CUDASearchStrategy]
    end

    subgraph "Optimization Dimensions"
        D[Grid/Block Dims]
        E[Shared Memory]
        F[Tensor Core]
        G[Warp Scheduling]
        H[Bank Conflict Avoidance]
    end

    subgraph "Hardware"
        I[NVIDIA GPU<br/>SM, Tensor Cores]
    end

    A --> D & E & F & G & H
    B --> A
    C --> A
    D & E & F & G & H --> I

    style A fill:#76ff03
    style I fill:#ffeb3b
```

### Ascend Backend

```mermaid
graph TB
    subgraph "Ascend Backend"
        A[AscendOptimizer]
        B[AscendKernelConfig]
        C[AscendSearchStrategy]
    end

    subgraph "Code Generation Paths"
        D[Triton Path<br/>Recommended]
        E[Ascend C Path<br/>Optional]
    end

    subgraph "Compiler"
        F[BiSheng Compiler]
        G[ascendc Compiler]
    end

    subgraph "Hardware"
        H[Ascend NPU<br/>AI Core, Cube Unit]
    end

    A --> D & E
    D --> F --> H
    E --> G --> H
    B --> A
    C --> A

    style D fill:#c8e6c9
    style H fill:#ffcdd2
```

### MACA Backend

```mermaid
graph TB
    subgraph "MACA Backend"
        A[MACAOptimizer]
        B[MACAKernelConfig]
        C[MACASearchStrategy]
    end

    subgraph "Key Differences from CUDA"
        D[64-thread Warp<br/>vs CUDA 32]
        E[mxcc Compiler<br/>vs nvcc]
        F[mcruntime<br/>vs cudart]
    end

    subgraph "Hardware"
        G[MetaX GPU<br/>C500, MACA Cores]
    end

    A --> D & E & F
    B --> A
    C --> A
    D & E & F --> G

    style A fill:#e1bee7
    style G fill:#b39ddb
```

### MPS Backend (Apple Silicon)

```mermaid
graph TB
    subgraph "MPS Backend"
        A[MPSOptimizer]
        B[MPSKernelConfig]
        C[MPSSearchStrategy]
    end

    subgraph "Metal Features"
        D[Threadgroup Memory<br/>32 KB]
        E[SIMD Width<br/>32 threads]
        F[Unified Memory<br/>CPU/GPU shared]
    end

    subgraph "Hardware"
        G[Apple Silicon<br/>M1/M2/M3/M4]
    end

    A --> D & E & F
    B --> A
    C --> A
    D & E & F --> G

    style A fill:#80deea
    style G fill:#4dd0e1
```

## Memory Hierarchy

### GPU Memory Model

```mermaid
graph TB
    subgraph "Memory Hierarchy"
        A[Global Memory / HBM<br/>Large, High Latency]
        B[L2 Cache<br/>Shared across SMs]
        C[L1 Cache / Shared Memory<br/>Per SM, Low Latency]
        D[Registers<br/>Per Thread, Fastest]
    end

    A --> B --> C --> D

    style A fill:#ffcdd2
    style B fill:#fff9c4
    style C fill:#c8e6c9
    style D fill:#b3e5fc
```

### Backend Memory Configuration

| Backend | Device Memory | Shared/Local Memory | Notes |
|---------|--------------|---------------------|-------|
| CUDA | 2-80 GB HBM | 96 KB (Ampere+) | Tensor Cores |
| MPS | 8-192 GB Unified | 32 KB Threadgroup | Apple Silicon |
| Ascend | 32-64 GB HBM | 512 KB L1 Buffer | AI Core |
| MACA | 16-64 GB HBM | 64 KB Shared | CUDA-compatible |
| CPU | System RAM | L1/L2/L3 Cache | SIMD |

## Transpiler Architecture

### Code Generation Flow

```mermaid
flowchart LR
    subgraph "Input"
        A[μGraph<br/>Fused Graph]
    end

    subgraph "Transpilers"
        B[CUDA Transpiler]
        C[Triton Transpiler]
        D[NKI Transpiler]
    end

    subgraph "Output Code"
        E[CUDA Kernel<br/>.cu]
        F[Triton Kernel<br/>.py]
        G[NKI Kernel<br/>.py]
    end

    subgraph "Runtime"
        H[cuBLAS/cuDNN]
        I[Triton JIT]
        J[Neuron SDK]
    end

    A --> B --> E --> H
    A --> C --> F --> I
    A --> D --> G --> J

    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style F fill:#fff9c4
    style G fill:#f3e5f5
```

## API Layer

### Python API Structure

```mermaid
classDiagram
    class yirage {
        +new_kernel_graph()
        +get_available_backends()
        +is_backend_available()
        +set_default_backend()
        +PersistentKernel
    }

    class KNGraph {
        +new_input()
        +matmul()
        +add()
        +relu()
        +rms_norm()
        +mark_output()
        +superoptimize()
    }

    class PersistentKernel {
        +backend: str
        +fallback_backends: list
        +world_size: int
        +__call__()
    }

    class BackendConfig {
        +get_mps_search_config()
        +get_ascend_search_config()
        +get_maca_search_config()
    }

    yirage --> KNGraph
    yirage --> PersistentKernel
    yirage --> BackendConfig
```

## Performance Optimization Flow

```mermaid
flowchart TD
    A[User Kernel Graph] --> B[Analyze Operations]
    B --> C[Generate Fusion Candidates]
    C --> D[Verify Correctness<br/>Fingerprint]
    D --> E{Valid?}
    E -->|No| C
    E -->|Yes| F[Profile Performance]
    F --> G[Select Best Configuration]
    G --> H[Generate Optimized Code]
    H --> I[Compile for Target Backend]
    I --> J[Optimized Kernel Ready]

    style A fill:#e3f2fd
    style D fill:#fff8e1
    style J fill:#c8e6c9
```

---

## References

- [YiRage GitHub Repository](https://github.com/chenxingqiang/YiRage)
- [Mirage Paper (OSDI 2025)](https://github.com/mirage-project/mirage)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Ascend CANN Documentation](https://www.hiascend.com/document)
- [MACA SDK Documentation](https://www.metax-tech.com/)

---

*Document Version: 2025-12-18*  
*YiRage Project: https://github.com/chenxingqiang/YiRage*
