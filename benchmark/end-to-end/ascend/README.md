# Ascend End-to-End Benchmarks

This folder contains end-to-end benchmarks for Huawei Ascend NPU backend.

## Requirements

- **Hardware**: Ascend 910/910B/310P NPU
- **Software**:
  - CANN toolkit (Compute Architecture for Neural Networks)
  - torch_npu (PyTorch for Ascend)
  - YiRage compiled with Ascend backend (`USE_ASCEND=ON`)

## Supported Devices

| Device | AI Cores | L1 Buffer | HBM |
|--------|----------|-----------|-----|
| Ascend 910 | 32 | 256 KB | 32 GB |
| Ascend 910B | 32 | 512 KB | 64 GB |
| Ascend 310P | 8 | 128 KB | 8 GB |

## Benchmarks

| File | Model | Description |
|------|-------|-------------|
| `llama_ascend.py` | LLaMA | LLaMA 70B style with GQA and SwiGLU |

## Usage

```bash
# Set up CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Run benchmarks
python llama_ascend.py --batch-size 8

# Skip search (test only)
python llama_ascend.py --skip-search
```

## Ascend-Specific Optimizations

1. **AI Core Scheduling**: Block sizes optimized for AI Core count
2. **Cube Operations**: Tile sizes aligned to 16x16 for Cube acceleration
3. **L1 Buffer**: Memory layout optimized for L1 buffer utilization
4. **BiSheng Compiler**: Uses Triton-Ascend for code generation

## Notes

- First run includes optimization search (may take several minutes)
- Without NPU hardware, benchmarks run on CPU with simulation
- Use `npu-smi info` to verify NPU availability

