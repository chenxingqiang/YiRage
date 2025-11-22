# YiRage Multi-Backend Usage Guide

## Overview

YiRage now supports multiple computation backends, allowing you to run your models on different hardware accelerators and CPU platforms. This guide explains how to use the multi-backend system.

## Available Backends

| Backend | Description | Hardware |
|---------|-------------|----------|
| **cuda** | NVIDIA CUDA | NVIDIA GPUs |
| **cpu** | CPU execution | x86/ARM CPUs |
| **mps** | Metal Performance Shaders | Apple Silicon (M1/M2/M3) |
| **cudnn** | CUDA Deep Neural Network library | NVIDIA GPUs |
| **mkl** | Intel Math Kernel Library | Intel CPUs |
| **mkldnn** | Intel oneDNN | Intel CPUs |
| **openmp** | OpenMP parallelization | Multi-core CPUs |
| **nki** | AWS Neuron Kernel Interface | AWS Inferentia/Trainium |
| **triton** | OpenAI Triton | NVIDIA GPUs |

## Checking Available Backends

### Python API

```python
import yirage as yr

# Get list of available backends
backends = yr.get_available_backends()
print(f"Available backends: {backends}")
# Output: Available backends: ['cuda', 'cpu']

# Check if specific backend is available
if yr.is_backend_available('cuda'):
    print("CUDA is available")

# Get default backend
default = yr.get_default_backend()
print(f"Default backend: {default}")

# Get detailed backend information
info = yr.get_backend_info('cuda')
print(f"CUDA Info: {info}")

# List all backends with details
yr.list_backends(verbose=True)
```

### C++ API

```cpp
#include "yirage/backend/backends.h"

using namespace yirage::backend;

// Get available backends
auto& registry = BackendRegistry::get_instance();
auto available = registry.get_available_backends();

for (auto type : available) {
    auto* backend = registry.get_backend(type);
    std::cout << "Backend: " << backend->get_name() << std::endl;
}

// Get specific backend
auto* cuda_backend = registry.get_backend("cuda");
if (cuda_backend && cuda_backend->is_available()) {
    std::cout << "CUDA is available" << std::endl;
}

// Get default backend
auto* default_backend = get_default_backend();
```

## Selecting a Backend

### Method 1: Using PersistentKernel

```python
import yirage as yr

# Specify backend explicitly
ypk = yr.PersistentKernel(
    mode="decode",
    backend="cuda",  # Specify backend
    world_size=1,
    mpi_rank=0,
    # ... other parameters
)
```

### Method 2: Using Fallback Backends

```python
import yirage as yr

# Try multiple backends in order of preference
ypk = yr.PersistentKernel(
    mode="decode",
    backend="cuda",
    fallback_backends=["mps", "cpu"],  # Try these if cuda fails
    world_size=1,
    mpi_rank=0,
    # ... other parameters
)
```

### Method 3: Setting Global Default

```python
import yirage as yr

# Set default backend for all subsequent operations
yr.set_default_backend("cpu")

# Now all operations will use CPU by default
ypk = yr.PersistentKernel(
    mode="decode",
    # backend will default to "cpu"
    world_size=1,
    mpi_rank=0,
    # ... other parameters
)
```

## Backend-Specific Features

### CUDA Backend

```python
# Check CUDA compute capability
if yr.is_backend_available('cuda'):
    info = yr.get_backend_info('cuda')
    compute_capability = info.get('compute_capability', 0)
    print(f"CUDA Compute Capability: {compute_capability // 10}.{compute_capability % 10}")
    
    # Use CUDA-specific features
    ypk = yr.PersistentKernel(
        mode="decode",
        backend="cuda",
        use_cutlass_kernel=True,  # CUDA-specific option
        # ... other parameters
    )
```

### CPU Backend

```python
# Check CPU cores
if yr.is_backend_available('cpu'):
    info = yr.get_backend_info('cpu')
    num_cores = info.get('compute_units', 0)
    print(f"CPU cores: {num_cores}")
    
    # Use CPU backend
    ypk = yr.PersistentKernel(
        mode="decode",
        backend="cpu",
        num_workers=num_cores,  # Use all cores
        # ... other parameters
    )
```

### MPS Backend (Apple Silicon)

```python
# Check for Apple Silicon
if yr.is_backend_available('mps'):
    print("Running on Apple Silicon")
    
    ypk = yr.PersistentKernel(
        mode="decode",
        backend="mps",
        # ... other parameters
    )
```

## Compilation Options

### Enabling Backends at Compile Time

Edit `config.cmake`:

```cmake
# Enable multiple backends
set(USE_CUDA ON)
set(USE_CPU ON)
set(USE_MPS OFF)
set(USE_TRITON ON)
# ... other backends
```

Then rebuild:

```bash
cd yirage
pip install -e . -v
```

### Checking Compiled Backends

```python
import yirage as yr

# This will show only the backends compiled into your installation
backends = yr.get_available_backends()
print(f"Compiled backends: {backends}")
```

## Performance Considerations

### Backend Selection Guidelines

1. **CUDA**: Best for NVIDIA GPUs, highest performance for large models
2. **CPU**: Universal compatibility, good for smaller models or debugging
3. **MPS**: Best for Apple Silicon, native macOS acceleration
4. **Triton**: Alternative to CUDA, may offer better performance in some cases

### Benchmarking Backends

```python
import yirage as yr
import time

def benchmark_backend(backend_name):
    if not yr.is_backend_available(backend_name):
        return None
    
    # Create and compile kernel
    ypk = yr.PersistentKernel(
        mode="decode",
        backend=backend_name,
        # ... parameters
    )
    ypk.compile()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        ypk()
    end = time.time()
    
    return (end - start) / 100

# Compare backends
for backend in yr.get_available_backends():
    latency = benchmark_backend(backend)
    if latency:
        print(f"{backend}: {latency*1000:.2f} ms/iteration")
```

## Error Handling

```python
import yirage as yr

try:
    ypk = yr.PersistentKernel(
        mode="decode",
        backend="cuda",  # Request CUDA
        fallback_backends=["cpu"],  # Fallback to CPU
        # ... parameters
    )
except ValueError as e:
    print(f"Backend initialization failed: {e}")
    
    # Check what went wrong
    if not yr.is_backend_available('cuda'):
        print("CUDA not available")
    if not yr.is_backend_available('cpu'):
        print("CPU not available")
```

## Advanced Usage

### Custom Backend Configuration

```python
import yirage as yr

# Get backend-specific configuration
cuda_info = yr.get_backend_info('cuda')
max_memory = cuda_info.get('max_memory', 0)
max_shared_mem = cuda_info.get('max_shared_memory', 0)

# Configure kernel based on backend capabilities
ypk = yr.PersistentKernel(
    mode="decode",
    backend="cuda",
    # Adjust parameters based on available memory
    max_num_pages=max_memory // (page_size * element_size),
    # ... other parameters
)
```

### Multi-Device Support

```python
import yirage as yr

# Check number of devices
info = yr.get_backend_info('cuda')
num_devices = info.get('device_count', 1)

# Create kernels for each device
kernels = []
for device_id in range(num_devices):
    ypk = yr.PersistentKernel(
        mode="decode",
        backend="cuda",
        world_size=num_devices,
        mpi_rank=device_id,
        # ... parameters
    )
    kernels.append(ypk)
```

## Troubleshooting

### Backend Not Available

**Problem**: Backend shows as unavailable even though hardware is present.

**Solutions**:
1. Check if backend was compiled: `yr.get_available_backends()`
2. Check hardware drivers (CUDA toolkit, etc.)
3. Rebuild YiRage with backend enabled in `config.cmake`

### Performance Issues

**Problem**: Unexpected low performance with certain backend.

**Solutions**:
1. Compare with other backends: use benchmarking script above
2. Check backend-specific settings (num_workers, memory limits, etc.)
3. Verify hardware is being utilized (nvidia-smi for CUDA, Activity Monitor for MPS, etc.)

### Compilation Errors

**Problem**: Errors when compiling with multiple backends.

**Solutions**:
1. Ensure all required libraries are installed
2. Check CMake output for missing dependencies
3. Try enabling backends one at a time to isolate issues

## See Also

- [Multi-Backend Design Document](multi_backend_design.md)
- [Installation Guide](../../INSTALL.md)
- [Performance Optimization Guide](performance.md)

