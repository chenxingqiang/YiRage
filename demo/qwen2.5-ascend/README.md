# Qwen2.5 Demo for Huawei Ascend NPU

This demo demonstrates running Qwen2.5/Qwen3 models on Huawei Ascend NPU with YiRage kernel optimization.

## Requirements

### Hardware
- Huawei Ascend 910/910B/310P NPU

### Software
- CANN toolkit (Compute Architecture for Neural Networks)
- torch_npu (PyTorch for Ascend NPU)
- YiRage compiled with Ascend support

## Installation

### 1. Install CANN Toolkit
Download from https://www.hiascend.com/software/cann and install:
```bash
./Ascend-cann-toolkit_<version>.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. Install torch_npu
```bash
pip install torch_npu
```

### 3. Build YiRage with Ascend Support
```bash
cd /path/to/YiRage
cp config.ascend.cmake config.cmake
pip install -e .
```

### 4. Install Demo Dependencies
```bash
pip install transformers accelerate
```

## Usage

### Basic Usage
```bash
python demo.py --model /path/to/local/model
```

### With YiRage Optimization (Default)
YiRage kernel optimization uses PyTorch operations accelerated by torch_npu on the Ascend NPU.
```bash
python demo.py --max-tokens 64 --model /path/to/model
```

### Without YiRage (Pure PyTorch)
```bash
python demo.py --disable-yirage --model /path/to/model
```

### Specify Different Model
```bash
python demo.py --model Qwen/Qwen2.5-7B-Instruct
python demo.py --model Qwen/Qwen3-8B
```

### CPU Fallback (for testing)
```bash
python demo.py --cpu-fallback
```

### Full Options
```bash
python demo.py --help
```

Options:
- `--disable-yirage`: Run without YiRage kernel optimization
- `--model`: Model name/path (default: Qwen/Qwen3-8B)
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--warmup`: Warmup iterations before timing (default: 16)
- `--device`: NPU device ID (default: 0)
- `--cpu-fallback`: Use CPU if NPU is not available

## Architecture

```
qwen2.5-ascend/
├── demo.py                  # Main demo script
├── models/
│   ├── __init__.py
│   ├── configuration_qwen2.py    # Qwen2 config
│   └── modeling_qwen2_ascend.py  # Ascend-optimized model
└── README.md
```

## Key Differences from CUDA Version

1. **Device Management**: Uses `torch.npu` instead of `torch.cuda`
2. **Attention**: Uses native PyTorch SDPA instead of flashinfer
3. **YiRage Backend**: Automatically uses Ascend backend for kernel optimization
4. **No CUDA Graph**: Ascend uses different graph capture mechanism

## Performance

Example on Ascend 910B2 with Qwen2.5-0.5B-Instruct:
- Per-token latency: ~25-35 ms
- Throughput: ~30-40 tokens/sec

The demo reports:
- Prompt length (tokens)
- Generated tokens count
- Per-token latency (ms)
- Throughput (tokens/sec)

## Troubleshooting

### NPU Not Detected
```bash
# Check if CANN is properly sourced
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Check NPU status
npu-smi info
```

### Missing Libraries
```bash
# Set library paths
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
```

### torch_npu Import Error
```bash
# Ensure CANN and driver versions match
pip install torch_npu --upgrade
```

## License

This demo is part of YiRage project, licensed under Apache 2.0.
