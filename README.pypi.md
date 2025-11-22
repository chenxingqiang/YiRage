# YiRage - Yield Revolutionary AGile Engine

Multi-Backend LLM Inference Optimization

## Quick Start

```bash
pip install yirage
```

```python
import yirage as yr

# Query available backends
backends = yr.get_available_backends()
print(f"Available backends: {backends}")

# Use MPS backend (Apple Silicon)
# Use CUDA backend (NVIDIA GPU)
# Use CPU backend
```

## Features

- **7 Complete Backend Implementations**: CUDA, CPU, MPS, Triton, NKI, cuDNN, MKL
- **Hardware-Aware Optimizers**: 42+ optimization methods
- **Search Strategies**: 5 independent strategies with auto-tuning
- **Production Ready**: 17,000+ lines of tested code

## Documentation

Full documentation: https://github.com/chenxingqiang/YiRage

## License

Apache License 2.0

Based on Mirage by CMU. Copyright 2025 Chen Xingqiang.

