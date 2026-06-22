# MACA End-to-End Benchmarks

This folder contains end-to-end benchmarks for MetaX MACA GPU backend.

## Benchmarks

| File | Model | Description |
|------|-------|-------------|
| `chameleon_maca.py` | Chameleon | Multi-modal transformer with RMS norm fusion |
| `llama_maca.py` | LLaMA | LLaMA 70B style with GQA and SwiGLU |
| `lora_maca.py` | LoRA | Low-Rank Adaptation fused with base weights |
| `ngpt_maca.py` | nGPT | Normalized GPT with post-linear normalization |

## Requirements

- MetaX C500 GPU (or compatible)
- mcPytorch (PyTorch with MACA support)
- YiRage compiled with MACA backend (`USE_MACA=ON`)

## Usage

```bash
# Set up environment
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH

# Activate mcPytorch environment (if using conda)
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# Run benchmarks
python chameleon_maca.py --batch-size 8
python llama_maca.py --batch-size 8
python lora_maca.py --batch-size 8 --lora-rank 16
python ngpt_maca.py --batch-size 8
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size` | Batch size | 8 |
| `--warmup` | Warmup iterations | 16 |
| `--repeat` | Benchmark repetitions | 1000 |
| `--skip-search` | Skip optimization search | False |
| `--lora-rank` | LoRA rank (lora_maca.py only) | 16 |

## MACA-Specific Optimizations

These benchmarks are optimized for MACA's unique characteristics:

1. **64-thread Warps**: Block dimensions are multiples of 64 (vs NVIDIA's 32)
2. **CUDA Compatibility**: Uses mcPytorch's CUDA-compatible API
3. **Checkpoint Support**: Saves/loads optimization results for faster subsequent runs

## Kernel Fusions

| Benchmark | Fused Operations |
|-----------|------------------|
| Chameleon | RMSNorm + MatMul (QKV), RMSNorm + MatMul (FFN) |
| LLaMA | RMSNorm + MatMul (QKV), RMSNorm + MatMul (FFN gate/up) |
| LoRA | RMSNorm + MatMul + LoRA (fused), MatMul + LoRA (fused) |
| nGPT | MatMul + RMSNorm, MatMul + RMSNorm + Scale + RMSNorm |

## Performance Notes

- First run includes optimization search (may take several minutes)
- Subsequent runs use cached checkpoints for faster startup
- Use `--skip-search` to test without optimization
- Actual performance depends on input dimensions and batch size

