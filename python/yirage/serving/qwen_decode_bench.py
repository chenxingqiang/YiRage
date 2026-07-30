# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S27: Qwen decode-step bench — native HF vs YiRage RF fused MLP."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .exec_backend import BACKEND_YIRAGE_CPU
from .hf_qwen_cpu_e2e import (
    DEFAULT_QWEN05B_MODEL,
    _load_qwen05b_cpu,
    _prefill_kv_cache,
    _prepare_model_inputs,
    _total_superopt_elapsed,
    clear_hf_qwen_rf_hook_cache,
    collect_per_layer_superopt_stats,
    qwen2_decode_step_with_rf_mlp,
    require_transformers,
)
from .torch_exec import bench_forward, require_torch
from .yirage_exec import inspect_serving_search_tier, require_yirage_core, resolve_serving_search_tier


@dataclass(frozen=True)
class QwenDecodeBenchRow:
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
class QwenDecodeBenchReport:
    """Single decode step (q_len=1) latency: native HF vs YiRage RF path."""

    version: str
    model_id: str
    device: str
    max_rf_mlp_layers: int
    parity_ok: bool
    num_layers: int = 0
    all_rf_layers: bool = False
    rows: List[QwenDecodeBenchRow] = field(default_factory=list)
    per_layer_superopt: List[Dict[str, Any]] = field(default_factory=list)
    speedup_yirage_vs_native: float = 0.0
    superopt_elapsed_s_total: float = 0.0
    serving_search_tier: str = "seed_verify"
    search_tier: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "serving_qwen_decode_bench": True,
            "version": self.version,
            "model_id": self.model_id,
            "device": self.device,
            "max_rf_mlp_layers": self.max_rf_mlp_layers,
            "num_layers": self.num_layers,
            "all_rf_layers": self.all_rf_layers,
            "parity_ok": self.parity_ok,
            "speedup_yirage_vs_native": round(self.speedup_yirage_vs_native, 4),
            "superopt_elapsed_s_total": round(self.superopt_elapsed_s_total, 6),
            "serving_search_tier": self.serving_search_tier,
            "rows": [r.to_dict() for r in self.rows],
        }
        if self.per_layer_superopt:
            payload["per_layer_superopt"] = self.per_layer_superopt
        if self.search_tier is not None:
            payload["search_tier"] = self.search_tier
        return payload


def run_qwen_decode_bench(
    *,
    model_id: str = DEFAULT_QWEN05B_MODEL,
    prompt: str = "The capital of France is",
    max_rf_mlp_layers: int = 1,
    all_rf_layers: bool = False,
    warmup: int = 3,
    iters: int = 15,
    quick: bool = False,
    version: str = "s28",
) -> QwenDecodeBenchReport:
    """Benchmark one decode step: HF native forward vs YiRage RF ``yirage_cpu`` MLP."""
    require_transformers()
    require_torch()
    require_yirage_core()
    import torch

    if quick:
        warmup = min(warmup, 2)
        iters = min(iters, 8)

    clear_hf_qwen_rf_hook_cache()
    model, tokenizer, device = _load_qwen05b_cpu(model_id=model_id)
    num_layers = int(model.config.num_hidden_layers)
    if all_rf_layers:
        max_rf_mlp_layers = num_layers
    max_rf_mlp_layers = min(int(max_rf_mlp_layers), num_layers)

    input_ids, attention_mask = _prepare_model_inputs(model, tokenizer, prompt, device=device)

    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = prefill_out.past_key_values
        next_id = torch.argmax(prefill_out.logits[:, -1:, :], dim=-1)
        attn_dec = torch.ones((1, input_ids.shape[1] + 1), device=device, dtype=torch.long)

        def native_decode():
            return model(
                input_ids=next_id,
                attention_mask=attn_dec,
                past_key_values=past,
                use_cache=False,
            )

        # Warm superoptimize + MuGraph cache before timing YiRage path.
        cache_template, seq_len_template = _prefill_kv_cache(model, input_ids, attention_mask)

        def yirage_rf_decode():
            cache_step = copy.deepcopy(cache_template)
            return qwen2_decode_step_with_rf_mlp(
                model,
                next_id=next_id,
                attn_dec=attn_dec,
                cache=cache_step,
                seq_len=seq_len_template,
                device=device,
                max_rf_mlp_layers=max_rf_mlp_layers,
                mlp_backend=BACKEND_YIRAGE_CPU,
            )

        for _ in range(max(warmup, 2)):
            yirage_rf_decode()

        ref_logits = native_decode().logits
        yirage_logits = yirage_rf_decode()
        parity_ok = bool(torch.allclose(ref_logits, yirage_logits, rtol=1e-4, atol=1e-3))

        native_bench = bench_forward(
            native_decode,
            name="native_decode_step",
            warmup=warmup,
            iters=iters,
            device=device,
        )
        yirage_bench = bench_forward(
            yirage_rf_decode,
            name="yirage_rf_decode_step",
            warmup=warmup,
            iters=iters,
            device=device,
        )

    superopt_total = _total_superopt_elapsed(max_rf_mlp_layers, BACKEND_YIRAGE_CPU)
    per_layer = collect_per_layer_superopt_stats(max_rf_mlp_layers, BACKEND_YIRAGE_CPU)
    speedup = native_bench.mean_ms / max(yirage_bench.mean_ms, 1e-9)
    tier = inspect_serving_search_tier()

    return QwenDecodeBenchReport(
        version=version,
        model_id=model_id,
        device=device,
        max_rf_mlp_layers=max_rf_mlp_layers,
        parity_ok=parity_ok,
        num_layers=num_layers,
        all_rf_layers=all_rf_layers or max_rf_mlp_layers >= num_layers,
        rows=[
            QwenDecodeBenchRow(
                name=native_bench.name,
                mean_ms=native_bench.mean_ms,
                iters=native_bench.iters,
                device=native_bench.device,
            ),
            QwenDecodeBenchRow(
                name=yirage_bench.name,
                mean_ms=yirage_bench.mean_ms,
                iters=yirage_bench.iters,
                device=yirage_bench.device,
            ),
        ],
        speedup_yirage_vs_native=speedup,
        superopt_elapsed_s_total=superopt_total,
        serving_search_tier=resolve_serving_search_tier(),
        search_tier=tier,
        per_layer_superopt=per_layer,
    )


def run_qwen_multilayer_decode_bench(
    *,
    model_id: str = DEFAULT_QWEN05B_MODEL,
    prompt: str = "The capital of France is",
    max_rf_mlp_layers: int = 2,
    all_rf_layers: bool = False,
    warmup: int = 2,
    iters: int = 8,
    quick: bool = False,
    version: str = "s28",
) -> QwenDecodeBenchReport:
    """Multi-layer decode bench (``max_rf_mlp_layers`` or ``all_rf_layers``)."""
    if quick and not all_rf_layers:
        max_rf_mlp_layers = min(max_rf_mlp_layers, 2)
    return run_qwen_decode_bench(
        model_id=model_id,
        prompt=prompt,
        max_rf_mlp_layers=max_rf_mlp_layers,
        all_rf_layers=all_rf_layers,
        warmup=warmup,
        iters=iters,
        quick=quick,
        version=version,
    )


def qwen_decode_bench_per_layer_superopt(
    report: QwenDecodeBenchReport,
) -> List[Dict[str, Any]]:
    """Per-layer superopt stats for the decode bench RF layers."""
    return collect_per_layer_superopt_stats(
        report.max_rf_mlp_layers,
        BACKEND_YIRAGE_CPU,
    )
