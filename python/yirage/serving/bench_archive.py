# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S8: torch-native segment hybrid forward + latency bench archive."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .capsule_orchestration import pipeline_meta_for_layer
from .exec_backend import BACKEND_TORCH
from .layer_override import LayerForwardResult, capsule_name_for_layer
from .runtime_fusion import RuntimeFusion, StepMeta, StepResult
from .segment_override import resolve_segment_layer_ids
from .split_mlp_capsule import build_layer_split_mlp_capsules
from .torch_engine import TorchEngineModel
from .torch_exec import bench_forward, require_torch, to_torch


@dataclass
class TorchSegmentForwardResult:
    hidden: Any
    layer_results: List[LayerForwardResult] = field(default_factory=list)
    segment_layer_ids: List[int] = field(default_factory=list)
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "segment_layer_ids": list(self.segment_layer_ids),
            "layers": [r.to_dict() for r in self.layer_results],
        }


class _TorchSplitMlpLayerOverride:
    def __init__(self, layer):
        self.layer = layer
        gate_up, down = build_layer_split_mlp_capsules(layer, backend=BACKEND_TORCH)
        self.rf = RuntimeFusion([gate_up, down])
        self.pipeline_names = [gate_up.name, down.name]

    def forward(self, hidden, *, rf_meta=None):
        h = self.layer.attention_forward(hidden)
        meta = StepMeta.from_mapping(rf_meta)
        step_meta = pipeline_meta_for_layer(self.layer.layer_id, base=dict(rf_meta or {}))
        result = self.rf.step({"hidden": h}, meta=step_meta)
        if len(result.ran) == 2:
            return LayerForwardResult(
                hidden=result.outputs["hidden"],
                rf=result,
                used_rf_mlp=True,
                layer_id=self.layer.layer_id,
            )
        out = self.layer.mlp_forward(h)
        return LayerForwardResult(
            hidden=out, rf=result, used_rf_mlp=False, layer_id=self.layer.layer_id
        )


class _TorchSingleMlpLayerOverride:
    def __init__(self, layer):
        from .layer_override import RuntimeFusionMlpLayerOverride, build_layer_mlp_capsule

        cap = build_layer_mlp_capsule(layer, backend=BACKEND_TORCH)
        self.override = RuntimeFusionMlpLayerOverride(layer, RuntimeFusion([cap]))

    def forward(self, hidden, *, rf_meta=None):
        return self.override.forward_mlp_only(hidden, rf_meta=rf_meta)


class TorchSegmentHybridModelOverride:
    """Torch tensors end-to-end for segment + optional single-capsule RF layers."""

    def __init__(
        self,
        model: TorchEngineModel,
        *,
        segment_layer_ids: Sequence[int],
        rf_mlp_layer_ids: Optional[Sequence[int]] = None,
    ):
        self.model = model
        self.device = model.device
        self.segment_ids = set(
            resolve_segment_layer_ids(len(model.layers), layer_ids=segment_layer_ids)
        )
        seg_set = self.segment_ids
        extra = set(rf_mlp_layer_ids or []) - seg_set
        self.split_overrides = {
            lid: _TorchSplitMlpLayerOverride(model.layers[lid]) for lid in sorted(seg_set)
        }
        self.single_overrides = {
            lid: _TorchSingleMlpLayerOverride(model.layers[lid]) for lid in sorted(extra)
        }

    def inspect(self) -> Dict[str, Any]:
        return {
            "torch_segment_hybrid": "TorchSegmentHybridModelOverride",
            "device": self.device,
            "segment_layer_ids": sorted(self.segment_ids),
            "single_rf_layer_ids": sorted(self.single_overrides),
        }

    def forward(
        self,
        hidden,
        *,
        rf_meta: Optional[Mapping[str, Any]] = None,
    ) -> TorchSegmentForwardResult:
        require_torch()
        h = to_torch(hidden, device=self.device)
        results: List[LayerForwardResult] = []
        rf_layers: List[int] = []
        for layer in self.model.layers:
            lid = layer.layer_id
            if lid in self.split_overrides:
                r = self.split_overrides[lid].forward(h, rf_meta=rf_meta)
            elif lid in self.single_overrides:
                h_attn = layer.attention_forward(h)
                meta = StepMeta.from_mapping(rf_meta)
                cap_name = capsule_name_for_layer(lid)
                step_meta = {"enabled": {cap_name}} if meta.should_run(cap_name) else {"force_skip_all": True}
                r = self.single_overrides[lid].forward(h_attn, rf_meta=step_meta)
            else:
                h = layer.forward_engine_full(h)
                r = LayerForwardResult(hidden=h, rf=None, used_rf_mlp=False, layer_id=lid)
            h = r.hidden
            results.append(r)
            if r.used_rf_mlp:
                rf_layers.append(lid)
        return TorchSegmentForwardResult(
            hidden=h,
            layer_results=results,
            segment_layer_ids=rf_layers,
            device=self.device,
        )


@dataclass
class ServingBenchArchiveRow:
    name: str
    mean_ms: float
    iters: int
    device: str
    parity_ok: bool
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean_ms": self.mean_ms,
            "iters": self.iters,
            "device": self.device,
            "parity_ok": self.parity_ok,
            **self.extras,
        }


@dataclass
class ServingBenchArchive:
    version: str
    device: str
    rows: List[ServingBenchArchiveRow] = field(default_factory=list)
    created_unix: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serving_bench_archive": True,
            "version": self.version,
            "device": self.device,
            "created_unix": self.created_unix,
            "rows": [r.to_dict() for r in self.rows],
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def run_segment_torch_bench_archive(
    *,
    num_layers: int = 4,
    segment_layer_ids: Sequence[int] = (1, 2),
    rf_mlp_layer_ids: Sequence[int] = (0,),
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 8,
    warmup: int = 3,
    iters: int = 15,
    seed: int = 0,
) -> ServingBenchArchive:
    """Measured torch bench: engine full vs segment hybrid (S8 archive)."""
    require_torch()
    import torch

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = TorchSegmentHybridModelOverride(
        model,
        segment_layer_ids=segment_layer_ids,
        rf_mlp_layer_ids=rf_mlp_layer_ids,
    )
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        ref = model.forward_engine_full(x)
        got = hybrid.forward(x)
        parity = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-5))

    def _engine():
        with torch.no_grad():
            model.forward_engine_full(x)

    def _segment_hybrid():
        with torch.no_grad():
            hybrid.forward(x)

    eng_bench = bench_forward(_engine, name="engine_full", warmup=warmup, iters=iters, device=model.device)
    hyb_bench = bench_forward(
        _segment_hybrid,
        name="segment_hybrid_torch",
        warmup=warmup,
        iters=iters,
        device=model.device,
    )

    archive = ServingBenchArchive(version="s15", device=model.device)
    archive.rows.append(
        ServingBenchArchiveRow(
            name=eng_bench.name,
            mean_ms=eng_bench.mean_ms,
            iters=eng_bench.iters,
            device=eng_bench.device,
            parity_ok=True,
            extras={"path": "engine"},
        )
    )
    archive.rows.append(
        ServingBenchArchiveRow(
            name=hyb_bench.name,
            mean_ms=hyb_bench.mean_ms,
            iters=hyb_bench.iters,
            device=hyb_bench.device,
            parity_ok=parity,
            extras={
                "path": "segment_hybrid",
                "segment_layer_ids": list(segment_layer_ids),
                "rf_mlp_layer_ids": list(rf_mlp_layer_ids),
                "speedup_vs_engine": eng_bench.mean_ms / max(hyb_bench.mean_ms, 1e-9),
            },
        )
    )
    return archive
