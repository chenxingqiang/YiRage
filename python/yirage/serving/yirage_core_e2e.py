# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S14: yirage.core MLP FusionCapsule full-layer hybrid e2e (real execution).

Full path = every decoder layer routes MLP through RuntimeFusion with
``backend=yirage_cpu`` capsules (gate_up via ``yirage.core`` seed graph +
superoptimized down matmul). Attention stays on the torch engine.

Decode constraint: ``YirageServingMlpRunner`` gate_up requires ``batch=1`` today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .exec_backend import BACKEND_YIRAGE_CPU
from .hybrid_model import HybridModelOverride
from .mlp_capsule import MlpFusionCapsule
from .torch_engine import TorchEngineModel
from .torch_exec import bench_forward, require_torch
from .yirage_exec import is_yirage_core_available, require_yirage_core


@dataclass(frozen=True)
class YirageCoreFullLayerE2EReport:
    """Full-layer hybrid e2e with yirage.core MLP capsules on every RF layer."""

    parity_ok: bool
    all_layers_rf: bool
    yirage_core_used: bool
    rf_layer_ids: List[int]
    device: str
    num_layers: int
    batch: int
    hidden_size: int
    intermediate_size: int
    superopt_elapsed_s_total: float
    plugin: str
    hybrid_mean_ms: Optional[float] = None
    engine_mean_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "all_layers_rf": self.all_layers_rf,
            "yirage_core_used": self.yirage_core_used,
            "rf_layer_ids": list(self.rf_layer_ids),
            "device": self.device,
            "num_layers": self.num_layers,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "superopt_elapsed_s_total": round(self.superopt_elapsed_s_total, 4),
            "plugin": self.plugin,
            "hybrid_mean_ms": self.hybrid_mean_ms,
            "engine_mean_ms": self.engine_mean_ms,
        }


def _aggregate_superopt_elapsed(capsules) -> float:
    total = 0.0
    for cap in capsules:
        if not isinstance(cap, MlpFusionCapsule):
            continue
        if cap.plan.backend != BACKEND_YIRAGE_CPU:
            continue
        runner = getattr(cap, "_yirage_runner", None)
        if runner is not None:
            total += float(getattr(runner, "superopt_elapsed_s", 0.0))
    return total


def run_yirage_core_full_layer_e2e(
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    batch: int = 1,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> YirageCoreFullLayerE2EReport:
    """All layers use yirage_cpu RF MLP capsules; parity vs torch engine reference."""
    require_yirage_core()
    require_torch()
    import torch

    if batch != 1:
        raise ValueError(
            f"yirage_cpu full-layer e2e requires batch=1 decode shape, got batch={batch}"
        )

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(
        model,
        max_rf_mlp_layers=num_layers,
        mlp_backend=BACKEND_YIRAGE_CPU,
    )
    expected_rf = list(range(num_layers))
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        ref = model.forward_engine_full(x)
        got = hybrid.forward(x)

    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=0.05, atol=0.05))
    parity_ok = parity_ok and list(got.rf_layer_ids) == expected_rf
    superopt_total = _aggregate_superopt_elapsed(hybrid.rf.capsules)
    yirage_used = all(
        isinstance(c, MlpFusionCapsule) and c.plan.backend == BACKEND_YIRAGE_CPU
        for c in hybrid.rf.capsules
    )

    hybrid_ms = eng_ms = None
    if bench:
        with torch.no_grad():
            eng_b = bench_forward(
                lambda: model.forward_engine_full(x),
                name="engine_full",
                warmup=warmup,
                iters=iters,
                device=model.device,
            )
            hyb_b = bench_forward(
                lambda: hybrid.forward(x),
                name="yirage_hybrid_full",
                warmup=warmup,
                iters=iters,
                device=model.device,
            )
        eng_ms = eng_b.mean_ms
        hybrid_ms = hyb_b.mean_ms

    return YirageCoreFullLayerE2EReport(
        parity_ok=parity_ok,
        all_layers_rf=list(got.rf_layer_ids) == expected_rf,
        yirage_core_used=yirage_used and is_yirage_core_available(),
        rf_layer_ids=list(got.rf_layer_ids),
        device=str(model.device),
        num_layers=int(num_layers),
        batch=int(batch),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        superopt_elapsed_s_total=superopt_total,
        plugin="HybridModelOverride+YirageCoreMlpCapsule",
        hybrid_mean_ms=hybrid_ms,
        engine_mean_ms=eng_ms,
    )


def run_yirage_core_full_layer_e2e_auto(
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    bench: bool = True,
) -> YirageCoreFullLayerE2EReport:
    """Cert/demo entry when ``yirage.core`` is built."""
    return run_yirage_core_full_layer_e2e(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=1,
        bench=bench,
    )
