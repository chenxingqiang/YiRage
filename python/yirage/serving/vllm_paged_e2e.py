# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S13: vLLM PagedAttention meta + full-layer MLP RF hook e2e (torch).

Full path = every decoder layer routes MLP through RuntimeFusion while
``block_tables``/``seq_lens`` bridge into ``paged_kv_*`` on each ``RF.step``.
Attention / PagedAttention remain on the engine (torch toy attn in tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .hybrid_model import HybridModelOverride
from .kv_meta import attach_paged_kv_to_step_meta
from .torch_engine import TorchEngineModel
from .torch_exec import bench_forward, require_torch


@dataclass(frozen=True)
class VllmPagedKvBatchSpec:
    """vLLM-style paged KV batch fields for RF StepMeta (no ``vllm`` import)."""

    block_tables: Sequence[Sequence[int]]
    seq_lens: Sequence[int]
    page_size: int = 16

    def as_rf_meta(
        self,
        *,
        sm_budget: Optional[int] = None,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        merged_extras = {"total_sms": 108, "reserved_aux_sms": 8}
        if extras:
            merged_extras.update(dict(extras))
        out = attach_paged_kv_to_step_meta(
            {},
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            page_size=int(self.page_size),
        )
        out["page_size"] = int(self.page_size)
        out_extras = dict(out.get("extras") or {})
        out_extras.update(merged_extras)
        out["extras"] = out_extras
        if sm_budget is not None:
            out["sm_budget"] = int(sm_budget)
        return out


@dataclass(frozen=True)
class VllmPagedFullLayerE2EReport:
    """Full-layer hybrid e2e with PagedAttention meta bridged on RF steps."""

    parity_ok: bool
    all_layers_rf: bool
    paged_kv_bridged: bool
    rf_layer_ids: List[int]
    device: str
    num_layers: int
    batch: int
    hidden_size: int
    plugin: str
    hybrid_mean_ms: Optional[float] = None
    engine_mean_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "all_layers_rf": self.all_layers_rf,
            "paged_kv_bridged": self.paged_kv_bridged,
            "rf_layer_ids": list(self.rf_layer_ids),
            "device": self.device,
            "num_layers": self.num_layers,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "plugin": self.plugin,
            "hybrid_mean_ms": self.hybrid_mean_ms,
            "engine_mean_ms": self.engine_mean_ms,
        }


def _paged_kv_present(meta: Optional[Mapping[str, Any]]) -> bool:
    if not meta:
        return False
    extras = dict(meta.get("extras") or {})
    if "paged_kv" in extras:
        return True
    return "paged_kv_indptr" in extras


def _layer_step_meta_has_paged_kv(layer_results) -> bool:
    for layer in layer_results:
        rf = getattr(layer, "rf", None)
        if rf is None or rf.meta is None:
            continue
        bridged = rf.meta.with_paged_kv_bridge()
        if _paged_kv_present(
            {
                "extras": bridged.extras,
                "block_tables": bridged.block_tables,
                "seq_lens": bridged.seq_lens,
            }
        ):
            return True
    return False


def run_torch_vllm_paged_full_layer_e2e(
    *,
    paged_batch: Optional[VllmPagedKvBatchSpec] = None,
    num_layers: int = 4,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> VllmPagedFullLayerE2EReport:
    """All layers use RF MLP hook; each step carries bridged paged KV meta."""
    require_torch()
    import torch

    if paged_batch is None:
        paged_batch = VllmPagedKvBatchSpec(
            block_tables=[
                [1, 2, -1],
                [3, 4, -1],
                [5, 6, -1],
                [7, 8, -1],
            ][:batch],
            seq_lens=[32, 18, 24, 16][:batch],
            page_size=16,
        )

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(model, max_rf_mlp_layers=num_layers)
    meta = paged_batch.as_rf_meta()
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=model.device)
    expected_rf = list(range(num_layers))

    with torch.no_grad():
        ref = model.forward_engine_full(x)
        got = hybrid.forward(x, rf_meta=meta)

    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
    parity_ok = parity_ok and list(got.rf_layer_ids) == expected_rf
    paged_ok = _paged_kv_present(meta) and _layer_step_meta_has_paged_kv(got.layer_results)

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
                name="paged_hybrid_full",
                warmup=warmup,
                iters=iters,
                device=model.device,
            )
        eng_ms = eng_b.mean_ms
        hybrid_ms = hyb_b.mean_ms

    return VllmPagedFullLayerE2EReport(
        parity_ok=parity_ok,
        all_layers_rf=list(got.rf_layer_ids) == expected_rf,
        paged_kv_bridged=paged_ok,
        rf_layer_ids=list(got.rf_layer_ids),
        device=str(model.device),
        num_layers=int(num_layers),
        batch=int(batch),
        hidden_size=int(hidden_size),
        plugin="HybridModelOverride+PagedKv+TorchDecoderMlpRfHook",
        hybrid_mean_ms=hybrid_ms,
        engine_mean_ms=eng_ms,
    )


def run_vllm_paged_full_layer_e2e_auto(
    *,
    num_layers: int = 4,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    bench: bool = True,
) -> VllmPagedFullLayerE2EReport:
    """Cert/demo entry: torch full-layer paged KV path (vLLM optional at S11 tier)."""
    return run_torch_vllm_paged_full_layer_e2e(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=bench,
    )
