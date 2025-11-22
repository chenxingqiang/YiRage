# YiRage Architecture Diagram Guide

## Draw.io File

Location: `docs/architecture.drawio`

## How to Edit and Export

### Option 1: Using draw.io Desktop
1. Download from https://www.drawio.com/
2. Open `docs/architecture.drawio`
3. Edit the diagram
4. Export: File → Export as → PNG
5. Save to `docs/images/yirage-architecture.png`

### Option 2: Using draw.io Web
1. Visit https://app.diagrams.net/
2. File → Open → Select `docs/architecture.drawio`
3. Edit as needed
4. File → Export as → PNG
5. Download and save to `docs/images/yirage-architecture.png`

### Option 3: Command Line (requires drawio-desktop)
```bash
# Install drawio (macOS)
brew install --cask drawio

# Export to PNG
/Applications/draw.io.app/Contents/MacOS/draw.io \
    -x -f png -o docs/images/yirage-architecture.png \
    docs/architecture.drawio
```

## Quick Text-Based Architecture

For README, you can also use ASCII art:

```
┌─────────────────────────────────────────────────────┐
│           YiRage Multi-Backend Architecture         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│             Python API Layer                        │
│  yr.get_backends() | Optimizers | Search            │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│        Backend Manager (C++)                        │
│  Registry | Factory | Auto-Init                     │
└────────────────┬────────────────────────────────────┘
                 │
     ┌───────────┴───────────┬──────────┬─────────────┐
     │           │           │          │             │
┌────▼──┐  ┌────▼───┐  ┌───▼───┐  ┌──▼───┐    ┌───▼───┐
│ CUDA  │  │  CPU   │  │  MPS  │  │Triton│ .. │ MKL   │
│       │  │        │  │       │  │      │    │       │
│Tensor │  │ SIMD   │  │Thread │  │Block │    │BLAS   │
│Core   │  │ Cache  │  │Group  │  │Pipe  │    │Thread │
└───┬───┘  └───┬────┘  └───┬───┘  └──┬───┘    └───┬───┘
    │          │           │         │            │
┌───▼──────────▼───────────▼─────────▼────────────▼───┐
│              Target Hardware                         │
│  NVIDIA GPU | x86/ARM CPU | Apple Silicon | AWS     │
└──────────────────────────────────────────────────────┘
```

## Architecture Layers

### Layer 1: Python API
- Backend query functions
- Kernel graph creation
- Optimizer access
- Search strategy selection

### Layer 2: Backend Manager (C++)
- BackendRegistry (singleton)
- Backend factory pattern
- Automatic initialization
- Thread-safe operations

### Layer 3: Backend Implementations
Each backend includes:
- Hardware-specific optimizer
- Search strategy (optional)
- Memory management
- Performance profiling

### Layer 4: Target Hardware
- NVIDIA GPU (CUDA, cuDNN)
- x86/ARM CPU (OpenMP, MKL)
- Apple Silicon (MPS)
- AWS Neuron (NKI)
- Compiler backends (Triton)

## Color Coding

- **Red**: GPU/CUDA backends
- **Green**: CPU backends
- **Blue**: MPS/Apple backends
- **Purple**: Specialized/compiler backends
- **Yellow**: Statistics/metadata

## Using in README

```markdown
# YiRage Architecture

![YiRage Architecture](docs/images/yirage-architecture.png)

Or use the ASCII version for GitHub mobile compatibility.
```

