# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S33: HF multi-step greedy generation — native vs RF MLP (G7 chain B extension).

Extends S27 single-step decode to ``max_new_tokens`` autoregressive steps:
prefill KV → repeated decode with ``greedy_decode_with_rf_mlp`` vs ``model.generate``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .exec_backend import BACKEND_TORCH, BACKEND_YIRAGE_CPU
from .hf_qwen_cpu_e2e import (
    DEFAULT_QWEN05B_MODEL,
    _total_superopt_elapsed,
    clear_hf_qwen_rf_hook_cache,
    collect_per_layer_superopt_stats,
    greedy_decode_with_rf_mlp,
    require_transformers,
    resolve_hf_qwen_mlp_backend,
)
from .yirage_exec import inspect_serving_search_tier, require_yirage_core, resolve_serving_search_tier


@dataclass
class QwenMultistepGenerationReport:
    """Multi-step HF greedy generation: native ``generate`` vs RF MLP decode loop."""

    version: str
    model_id: str
    device: str
    max_new_tokens: int
    max_rf_mlp_layers: int
    parity_ok: bool
    token_match_ok: bool
    mlp_backend: str
    yirage_core_used: bool
    functional_chain: str = "chain_b_multistep_generation"
    native_generate_ms: float = 0.0
    rf_generate_ms: float = 0.0
    superopt_elapsed_s_total: float = 0.0
    serving_search_tier: str = "seed_verify"
    native_token_ids: List[int] = field(default_factory=list)
    rf_token_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serving_qwen_multistep_generation_bench": True,
            "version": self.version,
            "model_id": self.model_id,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "max_rf_mlp_layers": self.max_rf_mlp_layers,
            "parity_ok": self.parity_ok,
            "token_match_ok": self.token_match_ok,
            "functional_chain": self.functional_chain,
            "mlp_backend": self.mlp_backend,
            "yirage_core_used": self.yirage_core_used,
            "native_generate_ms": round(self.native_generate_ms, 4),
            "rf_generate_ms": round(self.rf_generate_ms, 4),
            "superopt_elapsed_s_total": round(self.superopt_elapsed_s_total, 4),
            "serving_search_tier": self.serving_search_tier,
            "native_token_ids": list(self.native_token_ids),
            "rf_token_ids": list(self.rf_token_ids),
        }


def validate_serving_qwen_multistep_generation_bench(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not payload.get("serving_qwen_multistep_generation_bench"):
        errors.append("missing serving_qwen_multistep_generation_bench=true marker")
    if payload.get("parity_ok") is not True:
        errors.append("parity_ok must be true (token_match_ok required)")
    if payload.get("token_match_ok") is not True:
        errors.append("token_match_ok must be true for multistep generation")
    max_new = payload.get("max_new_tokens")
    if not isinstance(max_new, int) or max_new < 1:
        errors.append("max_new_tokens must be >= 1")
    native = payload.get("native_token_ids")
    rf = payload.get("rf_token_ids")
    if not isinstance(native, list) or not isinstance(rf, list) or len(native) != len(rf):
        errors.append("native_token_ids and rf_token_ids must be same-length lists")
    return errors


def run_qwen_multistep_generation_bench(
    *,
    model_id: str = DEFAULT_QWEN05B_MODEL,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 8,
    max_rf_mlp_layers: int = 1,
    mlp_backend: Optional[str] = None,
    quick: bool = False,
    all_rf_layers: bool = False,
    version: str = "s33",
) -> QwenMultistepGenerationReport:
    """Run multi-step greedy generation parity: HF native vs RF MLP decode loop."""
    require_transformers()
    from .hf_qwen_cpu_e2e import _load_qwen05b_cpu, _prepare_model_inputs
    from .torch_exec import require_torch

    require_torch()
    import torch

    decode_backend = resolve_hf_qwen_mlp_backend(mlp_backend)
    yirage_core_used = decode_backend == BACKEND_YIRAGE_CPU
    if yirage_core_used:
        require_yirage_core()

    if quick:
        max_new_tokens = min(int(max_new_tokens), 4)
        if not all_rf_layers:
            max_rf_mlp_layers = min(int(max_rf_mlp_layers), 1)

    clear_hf_qwen_rf_hook_cache()
    model, tokenizer, device = _load_qwen05b_cpu(model_id=model_id)
    num_layers = int(model.config.num_hidden_layers)
    if all_rf_layers:
        max_rf_mlp_layers = num_layers
    max_rf_mlp_layers = min(int(max_rf_mlp_layers), num_layers)

    input_ids, attention_mask = _prepare_model_inputs(model, tokenizer, prompt, device=device)

    with torch.no_grad():
        t0 = time.perf_counter()
        native_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        native_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        rf_ids = greedy_decode_with_rf_mlp(
            model,
            input_ids,
            attention_mask,
            max_new_tokens=int(max_new_tokens),
            max_rf_mlp_layers=max_rf_mlp_layers,
            mlp_backend=decode_backend,
        )
        rf_ms = (time.perf_counter() - t1) * 1000.0

    token_match_ok = bool(torch.equal(native_ids, rf_ids))
    superopt_total = _total_superopt_elapsed(max_rf_mlp_layers, decode_backend)

    return QwenMultistepGenerationReport(
        version=version,
        model_id=model_id,
        device=str(device),
        max_new_tokens=int(max_new_tokens),
        max_rf_mlp_layers=max_rf_mlp_layers,
        parity_ok=token_match_ok,
        token_match_ok=token_match_ok,
        mlp_backend=decode_backend,
        yirage_core_used=yirage_core_used,
        native_generate_ms=native_ms,
        rf_generate_ms=rf_ms,
        superopt_elapsed_s_total=superopt_total,
        serving_search_tier=resolve_serving_search_tier(),
        native_token_ids=native_ids[0].tolist(),
        rf_token_ids=rf_ids[0].tolist(),
    )


def run_serving_qwen_multistep_generation_archive(
    *,
    model_id: str = DEFAULT_QWEN05B_MODEL,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 8,
    max_rf_mlp_layers: int = 1,
    quick: bool = True,
    version: str = "s33",
) -> Dict[str, Any]:
    """Run bench and return validated archive payload."""
    payload = run_qwen_multistep_generation_bench(
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        max_rf_mlp_layers=max_rf_mlp_layers,
        quick=quick,
        version=version,
    ).to_dict()
    payload["search_tier"] = inspect_serving_search_tier()
    payload["per_layer_superopt"] = collect_per_layer_superopt_stats(
        int(payload["max_rf_mlp_layers"]),
        str(payload["mlp_backend"]),
    )
    errors = validate_serving_qwen_multistep_generation_bench(payload)
    if errors:
        raise RuntimeError(f"multistep generation archive validation failed: {errors}")
    return payload
