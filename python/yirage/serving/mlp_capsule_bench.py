# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S30: isolated MLP FusionCapsule micro-bench — parity + RF.step latency.

G7 chain A (paragraph closure): tensor in → RF.step → MlpFusionCapsule → tensor out,
with numerical parity vs eager ``mlp_torch`` before timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .mlp_capsule import MlpFusionCapsule
from .exec_backend import BACKEND_TORCH
from .runtime_fusion import RuntimeFusion
from .torch_exec import bench_forward, default_device, mlp_torch, require_torch


@dataclass(frozen=True)
class MlpCapsuleBenchRow:
    name: str
    mean_ms: float
    iters: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean_ms": round(self.mean_ms, 6),
            "iters": self.iters,
            "device": self.device,
        }


@dataclass
class MlpCapsuleBenchReport:
    """Isolated MLP capsule: eager torch vs RF.step (G7 chain A segment)."""

    version: str
    device: str
    hidden_size: int
    intermediate_size: int
    batch: int
    parity_ok: bool
    functional_chain: str = "chain_a_mlp_capsule_min"
    rows: List[MlpCapsuleBenchRow] = field(default_factory=list)
    speedup_rf_vs_eager: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serving_mlp_capsule_bench": True,
            "version": self.version,
            "device": self.device,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "batch": self.batch,
            "parity_ok": self.parity_ok,
            "functional_chain": self.functional_chain,
            "speedup_rf_vs_eager": round(self.speedup_rf_vs_eager, 4),
            "rows": [r.to_dict() for r in self.rows],
        }


def run_mlp_capsule_bench(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 8,
    seed: int = 0,
    warmup: int = 3,
    iters: int = 15,
    quick: bool = False,
    version: str = "s30",
) -> MlpCapsuleBenchReport:
    """Benchmark isolated MLP: eager ``mlp_torch`` vs ``RF.step`` on one capsule."""
    require_torch()
    import torch

    if quick:
        warmup = min(warmup, 2)
        iters = min(iters, 8)
        batch = min(batch, 4)

    cap = MlpFusionCapsule.from_random(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
        backend=BACKEND_TORCH,
    )
    rf = RuntimeFusion([cap])
    device = default_device()
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=device)
    rms_w, w_g, w_u, w_d = cap.weights()
    # Ensure torch tensors on bench device when weights are numpy.
    rms_w = torch.as_tensor(rms_w, device=device, dtype=torch.float32)
    w_g = torch.as_tensor(w_g, device=device, dtype=torch.float32)
    w_u = torch.as_tensor(w_u, device=device, dtype=torch.float32)
    w_d = torch.as_tensor(w_d, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref = mlp_torch(x, rms_weight=rms_w, w_gate=w_g, w_up=w_u, w_down=w_d)
        step = rf.step({"hidden": x}, meta={"enabled": {cap.name}})
        parity_ok = bool(
            torch.allclose(step.outputs["hidden"], ref, rtol=1e-5, atol=1e-5)
            and step.ran == [cap.name]
        )

    def _eager():
        with torch.no_grad():
            mlp_torch(x, rms_weight=rms_w, w_gate=w_g, w_up=w_u, w_down=w_d)

    def _rf_step():
        with torch.no_grad():
            rf.step({"hidden": x}, meta={"enabled": {cap.name}})

    eager_bench = bench_forward(
        _eager, name="eager_mlp_torch", warmup=warmup, iters=iters, device=str(device)
    )
    rf_bench = bench_forward(
        _rf_step, name="rf_step_mlp_capsule", warmup=warmup, iters=iters, device=str(device)
    )
    speedup = eager_bench.mean_ms / max(rf_bench.mean_ms, 1e-9)

    return MlpCapsuleBenchReport(
        version=version,
        device=str(device),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        parity_ok=parity_ok,
        rows=[
            MlpCapsuleBenchRow(
                name=eager_bench.name,
                mean_ms=eager_bench.mean_ms,
                iters=eager_bench.iters,
                device=eager_bench.device,
            ),
            MlpCapsuleBenchRow(
                name=rf_bench.name,
                mean_ms=rf_bench.mean_ms,
                iters=rf_bench.iters,
                device=rf_bench.device,
            ),
        ],
        speedup_rf_vs_eager=speedup,
    )
