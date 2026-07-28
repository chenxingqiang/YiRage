# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S12: SGLang ForwardBatch MLP RF full-path e2e (torch measured + optional ``sglang``).

Full path = engine Attention → RF MLP hook driven by ForwardBatch meta
(``extend_seq_lens``, ``block_tables``, ``seq_lens``) → parity vs engine MLP.

When ``sglang`` is not installed, the torch surrogate path is the cert gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .hybrid_model import HybridModelOverride
from .radix_meta import build_sglang_rf_step_meta
from .sglang_plugin import (
    build_sglang_batch_torch_mlp_rf_hook,
    build_sglang_qwen2_mlp_rf_hook,
    is_sglang_available,
    require_sglang,
)
from .torch_engine import TorchDecoderLayer, TorchEngineModel
from .torch_exec import bench_forward, require_torch


@dataclass(frozen=True)
class SglangForwardBatchSpec:
    """Minimal ForwardBatch-like fields for e2e (no ``sglang`` import)."""

    extend_seq_lens: Sequence[int]
    seq_lens: Sequence[int]
    block_tables: Optional[Sequence[Sequence[int]]] = None
    page_size: int = 16

    def as_meta(
        self,
        *,
        enabled: Optional[Sequence[str]] = None,
        sm_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        return build_sglang_rf_step_meta(
            block_tables=self.block_tables,
            seq_lens=list(self.seq_lens),
            extend_lens=list(self.extend_seq_lens),
            page_size=int(self.page_size),
            enabled=enabled,
            sm_budget=sm_budget,
        )


@dataclass(frozen=True)
class SglangMlpRfE2EReport:
    """Single-layer SGLang-style MLP RF hook e2e result."""

    parity_ok: bool
    used_rf_mlp: bool
    plugin: str
    device: str
    batch: int
    hidden_size: int
    intermediate_size: int
    layer_id: int
    radix_all_hit: bool
    radix_partial: bool
    hook_mean_ms: Optional[float] = None
    engine_mean_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "used_rf_mlp": self.used_rf_mlp,
            "plugin": self.plugin,
            "device": self.device,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "layer_id": self.layer_id,
            "radix_all_hit": self.radix_all_hit,
            "radix_partial": self.radix_partial,
            "hook_mean_ms": self.hook_mean_ms,
            "engine_mean_ms": self.engine_mean_ms,
        }


@dataclass(frozen=True)
class SglangHybridE2EReport:
    """Multi-layer hybrid forward with SGLang ForwardBatch meta."""

    parity_ok: bool
    rf_layer_ids: List[int]
    device: str
    num_layers: int
    max_rf_mlp_layers: int
    batch: int
    hidden_size: int
    plugin: str
    extend_seq_lens: List[int]
    hybrid_mean_ms: Optional[float] = None
    engine_mean_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "rf_layer_ids": list(self.rf_layer_ids),
            "device": self.device,
            "num_layers": self.num_layers,
            "max_rf_mlp_layers": self.max_rf_mlp_layers,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "plugin": self.plugin,
            "extend_seq_lens": list(self.extend_seq_lens),
            "hybrid_mean_ms": self.hybrid_mean_ms,
            "engine_mean_ms": self.engine_mean_ms,
        }


def _radix_flags(extend_seq_lens: Sequence[int]) -> tuple[bool, bool]:
    ext = np.asarray(extend_seq_lens, dtype=np.int64).reshape(-1)
    all_hit = bool(ext.size > 0 and np.all(ext == 0))
    partial = bool(np.any(ext == 0) and np.any(ext > 0))
    return all_hit, partial


def _expected_partial_radix_mlp(layer, hidden_after_attn, extend_seq_lens):
    """Reference: skip MLP on radix-hit rows; engine MLP on miss rows."""
    require_torch()
    import torch

    ext = np.asarray(extend_seq_lens, dtype=np.int64).reshape(-1)
    expected = hidden_after_attn.clone()
    for row in range(int(ext.shape[0])):
        if ext[row] > 0:
            expected[row : row + 1] = layer.mlp_forward(hidden_after_attn[row : row + 1])
    return expected


def _reference_hybrid_sglang_forward(
    model: TorchEngineModel,
    rf_layer_ids: Sequence[int],
    hidden,
    extend_seq_lens: Sequence[int],
):
    """Reference forward: RF layers honor Radix skip/shrink; others run engine MLP."""
    require_torch()
    import torch

    rf_set = {int(i) for i in rf_layer_ids}
    all_hit, partial = _radix_flags(extend_seq_lens)
    h = hidden
    with torch.no_grad():
        for layer in model.layers:
            h = layer.attention_forward(h)
            lid = layer.layer_id
            if lid in rf_set:
                if all_hit:
                    continue
                if partial:
                    h = _expected_partial_radix_mlp(layer, h, extend_seq_lens)
                else:
                    h = layer.mlp_forward(h)
            else:
                h = layer.mlp_forward(h)
    return h


def build_minimal_sglang_qwen2_decoder_layer(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 128,
    layer_id: int = 0,
    device: Optional[str] = None,
):
    """Construct a tiny real SGLang ``Qwen2DecoderLayer`` (requires ``sglang`` + ``transformers``)."""
    require_sglang()
    require_torch()
    import torch
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Qwen2Config(
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        max_position_embeddings=int(max_position_embeddings),
        num_hidden_layers=1,
        vocab_size=128,
    )
    try:
        from sglang.srt.models.qwen2 import Qwen2DecoderLayer
    except ImportError as exc:
        raise RuntimeError("SGLang Qwen2DecoderLayer import failed") from exc
    try:
        layer = Qwen2DecoderLayer(config=cfg, layer_id=layer_id)
    except TypeError:
        try:
            layer = Qwen2DecoderLayer(cfg, layer_id=layer_id)
        except TypeError:
            layer = Qwen2DecoderLayer(cfg, prefix=f"layers.{layer_id}")
    layer = layer.to(dev)
    layer.eval()
    return layer


def _sglang_native_mlp_forward(sglang_decoder_layer, hidden_after_attn):
    require_torch()
    import torch

    with torch.no_grad():
        normed = sglang_decoder_layer.post_attention_layernorm(hidden_after_attn)
        return sglang_decoder_layer.mlp(normed)


def run_torch_sglang_mlp_rf_e2e(
    *,
    forward_batch: Optional[SglangForwardBatchSpec] = None,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    layer_id: int = 0,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> SglangMlpRfE2EReport:
    """Measured torch full path with ForwardBatch meta (partial Radix by default)."""
    require_torch()
    import torch

    if forward_batch is None:
        forward_batch = SglangForwardBatchSpec(
            extend_seq_lens=[0, 3, 0, 2][:batch],
            seq_lens=[16] * batch,
        )
    all_hit, partial = _radix_flags(forward_batch.extend_seq_lens)

    layer = TorchDecoderLayer(
        layer_id,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hook = build_sglang_batch_torch_mlp_rf_hook(layer)
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=layer.device)

    with torch.no_grad():
        h_attn = layer.attention_forward(x)
        if all_hit:
            ref = h_attn
        elif partial:
            ref = _expected_partial_radix_mlp(layer, h_attn, forward_batch.extend_seq_lens)
        else:
            ref = layer.mlp_forward(h_attn)
        got = hook.forward_mlp(h_attn, forward_batch=forward_batch)
    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-5))

    hook_ms = eng_ms = None
    if bench and not all_hit:
        with torch.no_grad():
            h_fixed = layer.attention_forward(x)
            hook_b = bench_forward(
                lambda: hook.forward_mlp(h_fixed, forward_batch=forward_batch),
                name="sglang_style_hook",
                warmup=warmup,
                iters=iters,
                device=layer.device,
            )
            eng_b = bench_forward(
                lambda: layer.mlp_forward(h_fixed),
                name="engine_mlp",
                warmup=warmup,
                iters=iters,
                device=layer.device,
            )
        hook_ms = hook_b.mean_ms
        eng_ms = eng_b.mean_ms

    return SglangMlpRfE2EReport(
        parity_ok=parity_ok,
        used_rf_mlp=bool(got.used_rf_mlp),
        plugin="SglangBatchTorchMlpRfHook",
        device=str(layer.device),
        batch=int(batch),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        layer_id=int(layer_id),
        radix_all_hit=all_hit,
        radix_partial=partial,
        hook_mean_ms=hook_ms,
        engine_mean_ms=eng_ms,
    )


def run_torch_sglang_hybrid_full_e2e(
    *,
    forward_batch: Optional[SglangForwardBatchSpec] = None,
    num_layers: int = 4,
    max_rf_mlp_layers: int = 2,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> SglangHybridE2EReport:
    """Multi-layer hybrid forward with SGLang ForwardBatch StepMeta."""
    require_torch()
    import torch

    if forward_batch is None:
        forward_batch = SglangForwardBatchSpec(
            extend_seq_lens=[0, 3, 0, 0][:batch],
            seq_lens=[32, 18, 24, 16][:batch],
            block_tables=[[1, 2, -1], [3, 4, -1], [5, 6, -1], [7, 8, -1]][:batch],
            page_size=16,
        )

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(model, max_rf_mlp_layers=max_rf_mlp_layers)
    meta = forward_batch.as_meta()
    meta.setdefault("extras", {})
    extras = dict(meta["extras"])
    extras.setdefault("total_sms", 108)
    extras.setdefault("reserved_aux_sms", 8)
    meta["extras"] = extras

    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=model.device)
    expected_rf = list(range(int(max_rf_mlp_layers)))
    all_hit, _partial = _radix_flags(forward_batch.extend_seq_lens)

    with torch.no_grad():
        ref = _reference_hybrid_sglang_forward(
            model, expected_rf, x, forward_batch.extend_seq_lens
        )
        got = hybrid.forward(x, rf_meta=meta)

    if all_hit:
        parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
        parity_ok = parity_ok and got.rf_layer_ids == []
    else:
        parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
        parity_ok = parity_ok and list(got.rf_layer_ids) == expected_rf

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
                name="sglang_hybrid_full",
                warmup=warmup,
                iters=iters,
                device=model.device,
            )
        eng_ms = eng_b.mean_ms
        hybrid_ms = hyb_b.mean_ms

    return SglangHybridE2EReport(
        parity_ok=parity_ok,
        rf_layer_ids=list(got.rf_layer_ids),
        device=str(model.device),
        num_layers=int(num_layers),
        max_rf_mlp_layers=int(max_rf_mlp_layers),
        batch=int(batch),
        hidden_size=int(hidden_size),
        plugin="HybridModelOverride+SglangForwardBatch",
        extend_seq_lens=list(forward_batch.extend_seq_lens),
        hybrid_mean_ms=hybrid_ms,
        engine_mean_ms=eng_ms,
    )


def run_sglang_qwen2_mlp_rf_e2e(
    *,
    forward_batch: Optional[SglangForwardBatchSpec] = None,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    layer_id: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> SglangMlpRfE2EReport:
    """Real SGLang Qwen2 layer with ForwardBatch-driven RF hook (requires ``sglang``)."""
    require_sglang()
    require_torch()
    import torch

    if forward_batch is None:
        forward_batch = SglangForwardBatchSpec(
            extend_seq_lens=[0, 2][:batch],
            seq_lens=[16] * batch,
        )
    all_hit, partial = _radix_flags(forward_batch.extend_seq_lens)

    layer = build_minimal_sglang_qwen2_decoder_layer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layer_id=layer_id,
    )
    hook = build_sglang_qwen2_mlp_rf_hook(layer, layer_id=layer_id)
    device = str(next(layer.parameters()).device)
    h_attn = torch.randn(batch, hidden_size, dtype=torch.float32, device=device)

    with torch.no_grad():
        if all_hit:
            ref = h_attn
        elif partial:
            adapter = hook.override.layer
            ref = _expected_partial_radix_mlp(adapter, h_attn, forward_batch.extend_seq_lens)
        else:
            ref = _sglang_native_mlp_forward(layer, h_attn)
        got = hook.forward_mlp(h_attn, forward_batch=forward_batch)
    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-4, atol=1e-4))

    hook_ms = eng_ms = None
    if bench and not all_hit:
        with torch.no_grad():
            hook_b = bench_forward(
                lambda: hook.forward_mlp(h_attn, forward_batch=forward_batch),
                name="sglang_rf_hook",
                warmup=warmup,
                iters=iters,
                device=device,
            )
            eng_b = bench_forward(
                lambda: _sglang_native_mlp_forward(layer, h_attn),
                name="sglang_native_mlp",
                warmup=warmup,
                iters=iters,
                device=device,
            )
        hook_ms = hook_b.mean_ms
        eng_ms = eng_b.mean_ms

    return SglangMlpRfE2EReport(
        parity_ok=parity_ok,
        used_rf_mlp=bool(got.used_rf_mlp) if not all_hit else False,
        plugin="SglangQwen2MlpRfHook",
        device=device,
        batch=int(batch),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        layer_id=int(layer_id),
        radix_all_hit=all_hit,
        radix_partial=partial,
        hook_mean_ms=hook_ms,
        engine_mean_ms=eng_ms,
    )


def run_sglang_mlp_rf_e2e_auto(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    bench: bool = True,
) -> Union[SglangMlpRfE2EReport, SglangHybridE2EReport]:
    """Run real SGLang e2e when installed; else torch hybrid ForwardBatch path."""
    if is_sglang_available():
        try:
            return run_sglang_qwen2_mlp_rf_e2e(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                batch=batch,
                bench=bench,
            )
        except Exception:
            pass
    return run_torch_sglang_hybrid_full_e2e(
        num_layers=2,
        max_rf_mlp_layers=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=bench,
    )
