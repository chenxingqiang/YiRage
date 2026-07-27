# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S16: yirage_maca MLP FusionCapsule full-layer hybrid e2e (MetaX VM).

Full path = every decoder layer routes MLP through RuntimeFusion with
``backend=yirage_maca`` capsules (gate_up seed + MACA superoptimized down).
``MacaServingRfSpec`` meta is bridged on each RF step.

Decode constraint: gate_up requires ``batch=1`` today. Skips on CPU CI when
``yirage_maca`` is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .exec_backend import BACKEND_YIRAGE_MACA
from .hybrid_model import HybridModelOverride
from .maca_exec import is_yirage_maca_available, require_yirage_maca
from .maca_serving_meta import MacaServingRfSpec, maca_serving_present
from .mlp_capsule import MlpFusionCapsule
from .torch_engine import TorchEngineModel
from .torch_exec import bench_forward, require_torch


@dataclass(frozen=True)
class YirageMacaFullLayerE2EReport:
    """Full-layer hybrid e2e with yirage_maca capsules + MACA serving meta."""

    parity_ok: bool
    all_layers_rf: bool
    yirage_maca_used: bool
    maca_meta_bridged: bool
    rf_layer_ids: List[int]
    device: str
    num_layers: int
    batch: int
    hidden_size: int
    intermediate_size: int
    superopt_elapsed_s_total: float
    warp_size: int
    plugin: str
    hybrid_mean_ms: Optional[float] = None
    engine_mean_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "all_layers_rf": self.all_layers_rf,
            "yirage_maca_used": self.yirage_maca_used,
            "maca_meta_bridged": self.maca_meta_bridged,
            "rf_layer_ids": list(self.rf_layer_ids),
            "device": self.device,
            "num_layers": self.num_layers,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "superopt_elapsed_s_total": round(self.superopt_elapsed_s_total, 4),
            "warp_size": self.warp_size,
            "plugin": self.plugin,
            "hybrid_mean_ms": self.hybrid_mean_ms,
            "engine_mean_ms": self.engine_mean_ms,
        }


def _aggregate_maca_superopt_elapsed(capsules) -> float:
    total = 0.0
    for cap in capsules:
        if not isinstance(cap, MlpFusionCapsule):
            continue
        if cap.plan.backend != BACKEND_YIRAGE_MACA:
            continue
        runner = getattr(cap, "_maca_runner", None)
        if runner is not None:
            total += float(getattr(runner, "superopt_elapsed_s", 0.0))
    return total


def _layer_step_meta_has_maca_serving(layer_results) -> bool:
    for layer in layer_results:
        rf = getattr(layer, "rf", None)
        if rf is None or rf.meta is None:
            continue
        if maca_serving_present({"extras": rf.meta.extras}):
            return True
    return False


def run_yirage_maca_full_layer_e2e(
    *,
    maca_spec: Optional[MacaServingRfSpec] = None,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    batch: int = 1,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> YirageMacaFullLayerE2EReport:
    """All layers use yirage_maca RF capsules with MACA serving meta on each step."""
    require_yirage_maca()
    require_torch()
    import torch

    if batch != 1:
        raise ValueError(
            f"yirage_maca full-layer e2e requires batch=1 decode shape, got batch={batch}"
        )

    spec = maca_spec or MacaServingRfSpec()
    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(
        model,
        max_rf_mlp_layers=num_layers,
        mlp_backend=BACKEND_YIRAGE_MACA,
    )
    meta = spec.as_rf_meta(sm_budget=spec.sm_count - spec.reserved_aux_sms)
    expected_rf = list(range(num_layers))
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        ref = model.forward_engine_full(x)
        got = hybrid.forward(x, rf_meta=meta)

    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=0.05, atol=0.05))
    parity_ok = parity_ok and list(got.rf_layer_ids) == expected_rf
    superopt_total = _aggregate_maca_superopt_elapsed(hybrid.rf.capsules)
    maca_used = all(
        isinstance(c, MlpFusionCapsule) and c.plan.backend == BACKEND_YIRAGE_MACA
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
                lambda: hybrid.forward(x, rf_meta=meta),
                name="yirage_maca_hybrid_full",
                warmup=warmup,
                iters=iters,
                device=model.device,
            )
        eng_ms = eng_b.mean_ms
        hybrid_ms = hyb_b.mean_ms

    return YirageMacaFullLayerE2EReport(
        parity_ok=parity_ok,
        all_layers_rf=list(got.rf_layer_ids) == expected_rf,
        yirage_maca_used=maca_used and is_yirage_maca_available(),
        maca_meta_bridged=maca_serving_present(meta)
        and _layer_step_meta_has_maca_serving(got.layer_results),
        rf_layer_ids=list(got.rf_layer_ids),
        device=str(model.device),
        num_layers=int(num_layers),
        batch=int(batch),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        superopt_elapsed_s_total=superopt_total,
        warp_size=int(spec.warp_size),
        plugin="HybridModelOverride+YirageMacaMlpCapsule+MacaServingMeta",
        hybrid_mean_ms=hybrid_ms,
        engine_mean_ms=eng_ms,
    )


def run_yirage_maca_full_layer_e2e_auto(
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    bench: bool = True,
) -> YirageMacaFullLayerE2EReport:
    """Cert/demo entry when ``YIRAGE_BACKEND=maca`` and yirage.core are built."""
    return run_yirage_maca_full_layer_e2e(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=1,
        bench=bench,
    )
