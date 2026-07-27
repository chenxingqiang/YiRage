# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S15: MACA serving meta + full-layer MLP RF hook e2e (real torch).

Full path = every decoder layer routes MLP through RuntimeFusion while
``MacaServingRfSpec`` bridges 64-warp / C500 SM constraints into each ``RF.step``.
Execution uses torch on CPU CI; MetaX hosts may swap ``mlp_backend=yirage_maca`` later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .hybrid_model import HybridModelOverride
from .maca_serving_meta import (
    MacaServingRfSpec,
    inspect_maca_serving_meta,
    maca_serving_present,
)
from .torch_engine import TorchEngineModel
from .torch_exec import bench_forward, require_torch


@dataclass(frozen=True)
class MacaServingFullLayerE2EReport:
    """Full-layer hybrid e2e with MACA serving meta on RF steps."""

    parity_ok: bool
    all_layers_rf: bool
    maca_meta_bridged: bool
    rf_layer_ids: List[int]
    device: str
    num_layers: int
    batch: int
    hidden_size: int
    plugin: str
    warp_size: int
    hybrid_mean_ms: Optional[float] = None
    engine_mean_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "all_layers_rf": self.all_layers_rf,
            "maca_meta_bridged": self.maca_meta_bridged,
            "rf_layer_ids": list(self.rf_layer_ids),
            "device": self.device,
            "num_layers": self.num_layers,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "plugin": self.plugin,
            "warp_size": self.warp_size,
            "hybrid_mean_ms": self.hybrid_mean_ms,
            "engine_mean_ms": self.engine_mean_ms,
        }


def _layer_step_meta_has_maca_serving(layer_results) -> bool:
    for layer in layer_results:
        rf = getattr(layer, "rf", None)
        if rf is None or rf.meta is None:
            continue
        if maca_serving_present({"extras": rf.meta.extras}):
            payload = inspect_maca_serving_meta({"extras": rf.meta.extras})
            if payload and int(payload.get("warp_size", 0)) == MacaServingRfSpec().warp_size:
                return True
    return False


def run_torch_maca_serving_full_layer_e2e(
    *,
    maca_spec: Optional[MacaServingRfSpec] = None,
    num_layers: int = 4,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> MacaServingFullLayerE2EReport:
    """All layers use RF MLP hook; each step carries bridged MACA serving meta."""
    require_torch()
    import torch

    spec = maca_spec or MacaServingRfSpec()
    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(model, max_rf_mlp_layers=num_layers)
    meta = spec.as_rf_meta(sm_budget=spec.sm_count - spec.reserved_aux_sms)
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=model.device)
    expected_rf = list(range(num_layers))

    with torch.no_grad():
        ref = model.forward_engine_full(x)
        got = hybrid.forward(x, rf_meta=meta)

    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
    parity_ok = parity_ok and list(got.rf_layer_ids) == expected_rf
    maca_ok = maca_serving_present(meta) and _layer_step_meta_has_maca_serving(got.layer_results)

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
                name="maca_hybrid_full",
                warmup=warmup,
                iters=iters,
                device=model.device,
            )
        eng_ms = eng_b.mean_ms
        hybrid_ms = hyb_b.mean_ms

    return MacaServingFullLayerE2EReport(
        parity_ok=parity_ok,
        all_layers_rf=list(got.rf_layer_ids) == expected_rf,
        maca_meta_bridged=maca_ok,
        rf_layer_ids=list(got.rf_layer_ids),
        device=str(model.device),
        num_layers=int(num_layers),
        batch=int(batch),
        hidden_size=int(hidden_size),
        plugin="HybridModelOverride+MacaServingMeta+TorchDecoderMlpRfHook",
        warp_size=int(spec.warp_size),
        hybrid_mean_ms=hybrid_ms,
        engine_mean_ms=eng_ms,
    )


def run_maca_serving_full_layer_e2e_auto(
    *,
    num_layers: int = 4,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    bench: bool = True,
) -> MacaServingFullLayerE2EReport:
    """Cert/demo entry: torch full-layer MACA meta path (vLLM-metax optional at S15 tier)."""
    return run_torch_maca_serving_full_layer_e2e(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=bench,
    )
