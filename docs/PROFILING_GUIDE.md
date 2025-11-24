# YiRage Multi-Backend Profiling Guide

## Overview

YiRage supports performance profiling for all backends using appropriate tools for each platform.

## Quick Start

```bash
# CUDA backend (NVIDIA GPU)
./analyze_multibackend.sh cuda output.ncu-rep python benchmark/gated_mlp.py --backend cuda

# MPS backend (Apple Silicon)
./analyze_multibackend.sh mps output.trace python benchmark/gated_mlp.py --backend mps

# CPU backend
./analyze_multibackend.sh cpu output.perf python benchmark/gated_mlp.py --backend cpu
```

## Backend-Specific Tools

### CUDA Backend

**Tool**: NVIDIA Nsight Compute (ncu)

**Metrics Captured**:
- Memory workload analysis
- L1/L2 cache utilization
- Shared memory usage
- DRAM bandwidth
- Kernel occupancy

**View Results**:
```bash
ncu-ui output.ncu-rep
```

**Installation**:
- Download from https://developer.nvidia.com/nsight-compute
- Or: `sudo apt-get install nvidia-nsight-compute` (Linux)

---

### MPS Backend (Apple Silicon)

**Tool**: Xcode Instruments

**Metrics Captured**:
- Metal GPU utilization
- Threadgroup execution
- Memory bandwidth
- Shader compilation
- Frame time

**View Results**:
```bash
open output.trace  # Opens in Instruments.app
```

**Installation**:
```bash
xcode-select --install
```

**Manual Profiling**:
1. Open Instruments.app
2. Choose "Metal Application" template
3. Select your Python process
4. Record and analyze

---

### CPU Backend

**Tool**: perf (Linux) or Instruments (macOS)

**Metrics Captured**:
- CPU cycles
- Cache misses (L1/L2/L3)
- Branch predictions
- SIMD utilization
- Call graph

**View Results**:

Linux:
```bash
perf report -i output.perf
```

macOS:
```bash
open output.trace  # Opens in Instruments.app
```

**Installation**:

Linux:
```bash
sudo apt-get install linux-tools-common linux-tools-generic
```

macOS:
```bash
xcode-select --install
```

---

## Advanced Profiling

### Custom Kernel Profiling (CUDA)

For the original `analyze.sh` script (CUDA kernel-specific):

```bash
./analyze.sh <report.ncu-rep> <kernel_name> <command...>

# Example: Profile specific matmul kernel
./analyze.sh matmul.ncu-rep matmul_kernel python benchmark/gated_mlp.py --backend cuda
```

### Python-Level Profiling

For application-level profiling:

```python
import cProfile
import pstats

# Profile YiRage code
cProfile.run('your_yirage_code()', 'output.prof')

# View results
stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```python
import tracemalloc

# Start memory tracking
tracemalloc.start()

# Run YiRage code
import yirage as yr
# ... your code ...

# Get memory snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("Top 10 memory consumers:")
for stat in top_stats[:10]:
    print(stat)
```

---

## Benchmarking Tips

### 1. Warm-up Phase

Always include warm-up iterations:
```python
# Warm-up
for _ in range(10):
    model()

# Actual benchmark
import time
start = time.perf_counter()
for _ in range(100):
    model()
end = time.perf_counter()
print(f"Average: {(end - start) / 100 * 1000:.2f} ms")
```

### 2. Backend Comparison

```bash
# Profile all backends
for backend in cuda mps cpu; do
    if [[ "$backend" == "cuda" ]] && command -v ncu &> /dev/null; then
        ./analyze_multibackend.sh $backend profile_$backend python benchmark/gated_mlp.py --backend $backend
    elif [[ "$backend" == "mps" ]] && [[ "$(uname)" == "Darwin" ]]; then
        ./analyze_multibackend.sh $backend profile_$backend.trace python benchmark/gated_mlp.py --backend $backend
    elif [[ "$backend" == "cpu" ]]; then
        ./analyze_multibackend.sh $backend profile_$backend.perf python benchmark/gated_mlp.py --backend $backend
    fi
done
```

### 3. Automated Benchmarking

Use the baseline scripts for quick performance comparison:

```bash
# Run baseline benchmarks on different backends
python benchmark/baselines/pytorch/gated_mlp.py -b 8 --backend mps
python benchmark/baselines/pytorch/gated_mlp.py -b 8 --backend cpu
python benchmark/baselines/pytorch/gated_mlp.py -b 8 --backend cuda
```

---

## Interpreting Results

### CUDA (Nsight Compute)

Key metrics to watch:
- **Achieved Occupancy**: Target > 75%
- **Memory Throughput**: Target > 80% of peak
- **Bank Conflicts**: Target = 0
- **Warp Execution Efficiency**: Target > 90%

### MPS (Metal)

Key metrics to watch:
- **GPU Utilization**: Target > 70%
- **Memory Bandwidth**: Check against M1/M2/M3 specs
- **Threadgroup Occupancy**: Maximize parallel threadgroups
- **Shader Execution Time**: Minimize per-operation time

### CPU (perf)

Key metrics to watch:
- **Cache Miss Rate**: L1 < 5%, L2 < 10%, L3 < 20%
- **Branch Mispredictions**: < 5%
- **IPC (Instructions Per Cycle)**: Target > 2.0
- **SIMD Utilization**: Check vectorization efficiency

---

## Troubleshooting

### "ncu not found"
Install NVIDIA Nsight Compute from https://developer.nvidia.com/nsight-compute

### "xctrace not found"
```bash
xcode-select --install
```

### "perf not found" (Linux)
```bash
sudo apt-get install linux-tools-common linux-tools-generic
```

### Permission denied (CUDA)
NCU requires sudo. Run with:
```bash
sudo -E ./analyze_multibackend.sh cuda ...
```

---

## See Also

- [analyze.sh](../analyze.sh) - Original CUDA-only profiling script
- [Backend Usage Guide](docs/ypk/backend_usage.md)
- [Performance Optimization](docs/performance.md)

---

**YiRage** - Performance profiling across all hardware platforms

Copyright 2025 Chen Xingqiang | Apache License 2.0

