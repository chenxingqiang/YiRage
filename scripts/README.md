# YiRage Scripts

Utility scripts for development, training, and debugging.

## Visualization & Analysis

| Script | Description |
|--------|-------------|
| `display_task_graph.py` | Visualize task graphs from JSON (requires graphviz) |
| `partition_graph.py` | Partition PyTorch computational graphs for optimization |

## Verification

| Script | Description |
|--------|-------------|
| `verify_gpu.sh` | One-shot harness that automates the [GPU verification plan](VERIFY_GPU.md): environment check, build, CPU smoke tests, GPU end-to-end demo, persistent-kernel/MoE tests, and result collection. |
| `bench_fused_vs_mkl_baseline.py` | CPU: compare `superoptimize` µGraph winners vs unfused MKL (`torch`) baselines; reports speedup table (fusion value). `--quick` (default) skips `matmul_chain` / `concat_matmul`; `--full` adds chain + larger shapes. LoRA-style blocked GEMM: `--workloads concat_matmul` (`config=lora`, 4-input TB search caps). Applies tractable `YIRAGE_CPU_MAX_*` caps and `YIRAGE_CPU_BENCH_MINIMAL_EXPLORE=1` (not full production search). `--workloads plain_matmul` runs a subset. |
| `eval_optimization_value.py` | CPU: architecture-aware search + same-backend correctness/latency report. |

## RL Training

| Script | Description |
|--------|-------------|
| `train_rl.py` | Basic RL-guided kernel search training |
| `train_rl_hierarchical.py` | Hierarchical RL with curriculum learning |
| `train_rl_distributed.py` | Distributed training using Ray |

## Model Management

| Script | Description |
|--------|-------------|
| `manage_rl_models.py` | List, inspect, export, and compare RL models |

## Usage Examples

### Visualize Task Graph
```bash
python scripts/display_task_graph.py path/to/task_graph.json
```

### Train RL Policy
```bash
# Basic training
python scripts/train_rl.py --backend mps --max-iterations 100

# Hierarchical training with curriculum
python scripts/train_rl_hierarchical.py --stages 4 --episodes-per-stage 1000

# Distributed training (requires Ray)
python scripts/train_rl_distributed.py --num-workers 8
```

### Manage RL Models
```bash
# List saved models
python scripts/manage_rl_models.py list --dir checkpoints/

# Inspect a model
python scripts/manage_rl_models.py inspect checkpoints/model.pt

# Export to ONNX
python scripts/manage_rl_models.py export checkpoints/model.pt --format onnx
```

### Partition Graph
```python
from partition_graph import partition_graph

# After running a forward/backward pass
subgraphs, operators = partition_graph(loss)
```
