# YiRage × AccelForge Quick Start

AccelForge is integrated as YiRage's virtual-hardware modeling and pre-screening oracle. Use it to estimate latency, energy, area, and power before sending promising kernel candidates to a real backend profiler.

## Install

```bash
pip install "yirage[accelforge]"
```

YiRage currently tests against `accelforge>=1.0.355,<2.0.0`. If AccelForge is missing or incompatible, YiRage falls back to its analytical model and exposes diagnostics through `get_accelforge_availability()`.

## Availability diagnostics

```python
from yirage.rl.hardware import get_accelforge_availability

print(get_accelforge_availability())
```

## Evaluate a design point

```python
from yirage.rl.hardware import AccelForgeBridge, AccelForgeDesignPoint

bridge = AccelForgeBridge()
design = AccelForgeDesignPoint(
    pe_array_rows=32,
    pe_array_cols=32,
    l1_buffer_kb=128,
    dataflow="weight_stationary",
    data_precision="fp16",
)

workload = {"m_dim": 1024, "k_dim": 4096, "n_dim": 4096}
metrics = bridge.evaluate(design, workload)
print(metrics.to_dict())
```

## Convert a µGraph workload

```python
from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload

workload = mugraph_to_workload(kernel_graph_json)
```

`mugraph_to_workload()` accepts RL search JSON (`operators` + `tensors`), native `cy_to_json` operator lists (`kn_matmul_op`, `kn_customized_op`, …), and MuGraph cache entries (lists of graph variants). It extracts matrix dimensions for matmul, batch matmul, attention, convolution, reduction, and elementwise operators. Multi-op graphs preserve total estimated FLOPs and operator counts while selecting the dominant operator for AccelForge's single-einsum mapping path.

For a live optimized graph object, use `kngraph_to_workload(kn_graph)` which serializes via `cy_to_json` internally.

## Pre-screen before real profiling

```python
from yirage.rl.verifier import AccelForgeVerifier

verifier = AccelForgeVerifier(design.to_dict())
result = verifier.prescreen_kernel(
    kernel_graph_json,
    target_graph_json,
    latency_budget_ms=1.0,
    area_budget_mm2=100.0,
    power_budget_mw=5000.0,
)

if result["accepted"]:
    # Continue to CUDA/CPU/MACA/Ascend physical profiling.
    pass
```

## Pareto-front exploration

```python
from yirage.rl.search import ParetoFrontTracker, ParetoPoint

front = ParetoFrontTracker()
front.add(ParetoPoint(
    design=design.to_dict(),
    latency_ms=metrics.latency_ms,
    energy_pj=metrics.energy_per_op_pj,
    area_mm2=metrics.area_mm2,
    power_mw=metrics.total_power_mw,
))
print(front.to_dict_list())
```

## Recommended workflow

1. Use AccelForge to cheaply reject candidates that violate hardware budgets.
2. Keep accepted candidates for physical backend verification and profiling.
3. Calibrate AccelForge or surrogate-model estimates with measured CUDA/CPU/MACA/Ascend profiler results.
4. Feed latency, energy, area, and power into YiRage's RL reward for hardware-software co-design.
