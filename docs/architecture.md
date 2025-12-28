# YiRage Architecture Documentation

This document provides comprehensive architecture diagrams for the YiRage multi-backend LLM inference optimization framework.

## System Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
graph TB
    subgraph UI["User Interface"]
        A[Python API]
        B[CLI Tools]
    end

    subgraph CORE["Core Engine"]
        C[Kernel Graph]
        D[Search Engine]
        E[Transpiler]
    end

    subgraph BACKEND["Backend Layer"]
        F[Backend Registry]
        G[Backend Factory]
        H[Strategy Factory]
    end

    subgraph HW["Hardware Backends"]
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

    style UI fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style CORE fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style BACKEND fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
    style HW fill:#c4b5fd,stroke:#7c3aed,stroke-width:2px
```

## Backend Architecture

### Multi-Backend Support

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
graph LR
    subgraph PY["Python Layer"]
        A[yirage.get_available_backends]
        B[yirage.PersistentKernel]
        C[graph.superoptimize]
    end

    subgraph CPP["C++ Backend Manager"]
        D[BackendRegistry<br/>Singleton]
        E[BackendInterface<br/>Abstract]
    end

    subgraph IMPL["Concrete Backends"]
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

    style PY fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style CPP fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style IMPL fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
```

### Backend Selection Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
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

    style A fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style D fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
    style I fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style G fill:#fce4ec,stroke:#7c3aed,stroke-width:2px
    style L fill:#c4b5fd,stroke:#7c3aed,stroke-width:2px
```

## Search Engine Architecture

### Superoptimization Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
flowchart LR
    subgraph IN["Input"]
        A[Kernel Graph<br/>User-defined operations]
    end

    subgraph SEARCH["Search Phase"]
        B[Candidate Generator<br/>Fusion possibilities]
        C[Fingerprint Verifier<br/>Correctness check]
        D[Performance Profiler<br/>Timing measurement]
    end

    subgraph STRAT["Backend Strategies"]
        E[CUDA Strategy<br/>Tensor Core, Warp]
        F[MPS Strategy<br/>SIMD, Threadgroup]
        G[Ascend Strategy<br/>AI Core, Cube]
        H[MACA Strategy<br/>64-thread Warp]
        I[CPU Strategy<br/>AVX, Cache]
    end

    subgraph OUT["Output"]
        J[Optimized Kernel<br/>Best configuration]
    end

    A --> B --> C --> D
    D --> E & F & G & H & I
    E & F & G & H & I --> J

    style IN fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style SEARCH fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style STRAT fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
    style OUT fill:#c4b5fd,stroke:#7c3aed,stroke-width:2px
```

### Fingerprint Verification

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
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
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
graph TB
    subgraph CUDA["CUDA Backend"]
        A[CUDAOptimizer]
        B[CUDAKernelConfig]
        C[CUDASearchStrategy]
    end

    subgraph OPT["Optimization Dimensions"]
        D[Grid/Block Dims]
        E[Shared Memory]
        F[Tensor Core]
        G[Warp Scheduling]
        H[Bank Conflict Avoidance]
    end

    subgraph HW["Hardware"]
        I[NVIDIA GPU<br/>SM, Tensor Cores]
    end

    A --> D & E & F & G & H
    B --> A
    C --> A
    D & E & F & G & H --> I

    style CUDA fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style OPT fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style HW fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
```

### Ascend Backend

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
graph TB
    subgraph ASCEND["Ascend Backend"]
        A[AscendOptimizer]
        B[AscendKernelConfig]
        C[AscendSearchStrategy]
    end

    subgraph PATH["Code Generation Paths"]
        D[Triton Path<br/>Recommended]
        E[Ascend C Path<br/>Optional]
    end

    subgraph COMP["Compiler"]
        F[BiSheng Compiler]
        G[ascendc Compiler]
    end

    subgraph HW["Hardware"]
        H[Ascend NPU<br/>AI Core, Cube Unit]
    end

    A --> D & E
    D --> F --> H
    E --> G --> H
    B --> A
    C --> A

    style ASCEND fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style PATH fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style COMP fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
    style HW fill:#c4b5fd,stroke:#7c3aed,stroke-width:2px
```

### MACA Backend

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
graph TB
    subgraph MACA["MACA Backend"]
        A[MACAOptimizer]
        B[MACAKernelConfig]
        C[MACASearchStrategy]
    end

    subgraph DIFF["Key Differences from CUDA"]
        D[64-thread Warp<br/>vs CUDA 32]
        E[mxcc Compiler<br/>vs nvcc]
        F[mcruntime<br/>vs cudart]
    end

    subgraph HW["Hardware"]
        G[MetaX GPU<br/>C500, MACA Cores]
    end

    A --> D & E & F
    B --> A
    C --> A
    D & E & F --> G

    style MACA fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style DIFF fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style HW fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
```

### MPS Backend (Apple Silicon)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
graph TB
    subgraph MPS["MPS Backend"]
        A[MPSOptimizer]
        B[MPSKernelConfig]
        C[MPSSearchStrategy]
    end

    subgraph METAL["Metal Features"]
        D[Threadgroup Memory<br/>32 KB]
        E[SIMD Width<br/>32 threads]
        F[Unified Memory<br/>CPU/GPU shared]
    end

    subgraph HW["Hardware"]
        G[Apple Silicon<br/>M1/M2/M3/M4]
    end

    A --> D & E & F
    B --> A
    C --> A
    D & E & F --> G

    style MPS fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style METAL fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style HW fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
```

## Memory Hierarchy

### GPU Memory Model

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
graph TB
    subgraph MEM["Memory Hierarchy"]
        A[Global Memory / HBM<br/>Large, High Latency]
        B[L2 Cache<br/>Shared across SMs]
        C[L1 Cache / Shared Memory<br/>Per SM, Low Latency]
        D[Registers<br/>Per Thread, Fastest]
    end

    A --> B --> C --> D

    style MEM fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style A fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style B fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
    style C fill:#c4b5fd,stroke:#7c3aed,stroke-width:2px
    style D fill:#a78bfa,stroke:#7c3aed,stroke-width:2px
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
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
flowchart LR
    subgraph IN["Input"]
        A[μGraph<br/>Fused Graph]
    end

    subgraph TRANS["Transpilers"]
        B[CUDA Transpiler]
        C[Triton Transpiler]
        D[NKI Transpiler]
    end

    subgraph OUT["Output Code"]
        E[CUDA Kernel<br/>.cu]
        F[Triton Kernel<br/>.py]
        G[NKI Kernel<br/>.py]
    end

    subgraph RT["Runtime"]
        H[cuBLAS/cuDNN]
        I[Triton JIT]
        J[Neuron SDK]
    end

    A --> B --> E --> H
    A --> C --> F --> I
    A --> D --> G --> J

    style IN fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style TRANS fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style OUT fill:#ddd6fe,stroke:#7c3aed,stroke-width:2px
    style RT fill:#c4b5fd,stroke:#7c3aed,stroke-width:2px
```

## API Layer

### Python API Structure

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
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
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ede9fe', 'primaryTextColor': '#3b0764', 'lineColor': '#7c3aed'}}}%%
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

    style A fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px
    style D fill:#ede9fe,stroke:#7c3aed,stroke-width:2px
    style J fill:#c4b5fd,stroke:#7c3aed,stroke-width:2px
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
