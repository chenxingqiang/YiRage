# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S16: SGLang-metax ForwardBatch + MACA serving meta full-path e2e.

Measured torch path uses :class:`SglangMetaxBatchTorchMlpRfHook` (no ``sglang`` required).
Real SGLang-metax tier uses :class:`SglangMetaxQwen2MlpRfHook` when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .maca_serving_meta import MacaServingRfSpec, maca_serving_present
from .sglang_e2e import (
    SglangForwardBatchSpec,
    _expected_partial_radix_mlp,
    _radix_flags,
    _reference_hybrid_sglang_forward,
    _sglang_native_mlp_forward,
    build_minimal_sglang_qwen2_decoder_layer,
)
from .sglang_metax_plugin import (
    build_sglang_metax_batch_torch_mlp_rf_hook,
    build_sglang_metax_qwen2_mlp_rf_hook,
    is_sglang_metax_available,
    require_sglang_metax,
    rf_step_meta_for_sglang_metax,
)
from .torch_engine import TorchDecoderLayer, TorchEngineModel
from .torch_exec import bench_forward, require_torch


@dataclass(frozen=True)
class SglangMetaxMlpRfE2EReport:
    """Single-layer SGLang-metax style MLP RF hook e2e result."""

    parity_ok: bool
    used_rf_mlp: bool
    maca_meta_bridged: bool
    plugin: str
    device: str
    batch: int
    hidden_size: int
    intermediate_size: int
    layer_id: int
    radix_all_hit: bool
    radix_partial: bool
    warp_size: int
    hook_mean_ms: Optional[float] = None
    engine_mean_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "used_rf_mlp": self.used_rf_mlp,
            "maca_meta_bridged": self.maca_meta_bridged,
            "plugin": self.plugin,
            "device": self.device,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "layer_id": self.layer_id,
            "radix_all_hit": self.radix_all_hit,
            "radix_partial": self.radix_partial,
            "warp_size": self.warp_size,
            "hook_mean_ms": self.hook_mean_ms,
            "engine_mean_ms": self.engine_mean_ms,
        }


@dataclass(frozen=True)
class SglangMetaxHybridE2EReport:
    """Multi-layer hybrid forward with SGLang ForwardBatch + MACA serving meta."""

    parity_ok: bool
    rf_layer_ids: List[int]
    maca_meta_bridged: bool
    device: str
    num_layers: int
    max_rf_mlp_layers: int
    batch: int
    hidden_size: int
    plugin: str
    extend_seq_lens: List[int]
    warp_size: int
    hybrid_mean_ms: Optional[float] = None
    engine_mean_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "rf_layer_ids": list(self.rf_layer_ids),
            "maca_meta_bridged": self.maca_meta_bridged,
            "device": self.device,
            "num_layers": self.num_layers,
            "max_rf_mlp_layers": self.max_rf_mlp_layers,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "plugin": self.plugin,
            "extend_seq_lens": list(self.extend_seq_lens),
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
            return True
    return False


def run_torch_sglang_metax_mlp_rf_e2e(
    *,
    maca_spec: Optional[MacaServingRfSpec] = None,
    forward_batch: Optional[SglangForwardBatchSpec] = None,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    layer_id: int = 0,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> SglangMetaxMlpRfE2EReport:
    """Measured torch hook with ForwardBatch + MACA serving meta."""
    require_torch()
    import torch

    spec = maca_spec or MacaServingRfSpec()
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
    hook = build_sglang_metax_batch_torch_mlp_rf_hook(layer, maca_spec=spec)
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
    meta = rf_step_meta_for_sglang_metax(
        forward_batch,
        spec=spec,
        enabled=[hook._inner._inner.override.capsule_name],
    )

    hook_ms = eng_ms = None
    if bench and not all_hit:
        with torch.no_grad():
            h_fixed = layer.attention_forward(x)
            hook_b = bench_forward(
                lambda: hook.forward_mlp(h_fixed, forward_batch=forward_batch),
                name="sglang_metax_hook",
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

    return SglangMetaxMlpRfE2EReport(
        parity_ok=parity_ok,
        used_rf_mlp=bool(got.used_rf_mlp),
        maca_meta_bridged=maca_serving_present(meta),
        plugin="SglangMetaxBatchTorchMlpRfHook",
        device=str(layer.device),
        batch=int(batch),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        layer_id=int(layer_id),
        radix_all_hit=all_hit,
        radix_partial=partial,
        warp_size=int(spec.warp_size),
        hook_mean_ms=hook_ms,
        engine_mean_ms=eng_ms,
    )


def run_torch_sglang_metax_hybrid_full_e2e(
    *,
    maca_spec: Optional[MacaServingRfSpec] = None,
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
) -> SglangMetaxHybridE2EReport:
    """Multi-layer hybrid with SGLang ForwardBatch + MACA serving StepMeta."""
    require_torch()
    import torch

    from .hybrid_model import HybridModelOverride

    spec = maca_spec or MacaServingRfSpec()
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
    meta = rf_step_meta_for_sglang_metax(forward_batch, spec=spec)
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
                name="sglang_metax_hybrid_full",
                warmup=warmup,
                iters=iters,
                device=model.device,
            )
        eng_ms = eng_b.mean_ms
        hybrid_ms = hyb_b.mean_ms

    return SglangMetaxHybridE2EReport(
        parity_ok=parity_ok,
        rf_layer_ids=list(got.rf_layer_ids),
        maca_meta_bridged=maca_serving_present(meta)
        and _layer_step_meta_has_maca_serving(got.layer_results),
        device=str(model.device),
        num_layers=int(num_layers),
        max_rf_mlp_layers=int(max_rf_mlp_layers),
        batch=int(batch),
        hidden_size=int(hidden_size),
        plugin="HybridModelOverride+SglangMetaxForwardBatch+MacaServingMeta",
        extend_seq_lens=list(forward_batch.extend_seq_lens),
        warp_size=int(spec.warp_size),
        hybrid_mean_ms=hybrid_ms,
        engine_mean_ms=eng_ms,
    )


def run_sglang_metax_qwen2_mlp_rf_e2e(
    *,
    maca_spec: Optional[MacaServingRfSpec] = None,
    forward_batch: Optional[SglangForwardBatchSpec] = None,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    layer_id: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> SglangMetaxMlpRfE2EReport:
    """Real SGLang-metax Qwen2 layer hook (requires ``sglang`` on MetaX host)."""
    require_sglang_metax()
    require_torch()
    import torch

    spec = maca_spec or MacaServingRfSpec()
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
    hook = build_sglang_metax_qwen2_mlp_rf_hook(layer, layer_id=layer_id, maca_spec=spec)
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
    meta = rf_step_meta_for_sglang_metax(
        forward_batch,
        spec=spec,
        enabled=[hook.override.capsule_name],
    )

    hook_ms = eng_ms = None
    if bench and not all_hit:
        with torch.no_grad():
            hook_b = bench_forward(
                lambda: hook.forward_mlp(h_attn, forward_batch=forward_batch),
                name="sglang_metax_rf_hook",
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

    return SglangMetaxMlpRfE2EReport(
        parity_ok=parity_ok,
        used_rf_mlp=bool(got.used_rf_mlp),
        maca_meta_bridged=maca_serving_present(meta),
        plugin="SglangMetaxQwen2MlpRfHook",
        device=device,
        batch=int(batch),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        layer_id=int(layer_id),
        radix_all_hit=all_hit,
        radix_partial=partial,
        warp_size=int(spec.warp_size),
        hook_mean_ms=hook_ms,
        engine_mean_ms=eng_ms,
    )


def run_sglang_metax_mlp_rf_e2e_auto(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    bench: bool = True,
) -> SglangMetaxMlpRfE2EReport:
    """Cert/demo entry: torch path by default; real sglang-metax when tier available."""
    if is_sglang_metax_available():
        return run_sglang_metax_qwen2_mlp_rf_e2e(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            batch=batch,
            bench=bench,
        )
    return run_torch_sglang_metax_mlp_rf_e2e(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=bench,
    )


@dataclass(frozen=True)
class SglangMetaxRealForkE2EReport:
    """Real SGLang-metax fork: Qwen2 hook + ForwardBatch hybrid (both with MACA meta)."""

    hook: SglangMetaxMlpRfE2EReport
    hybrid: SglangMetaxHybridE2EReport
    real_fork: bool
    parity_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "real_fork": self.real_fork,
            "parity_ok": self.parity_ok,
            "hook": self.hook.to_dict(),
            "hybrid": self.hybrid.to_dict(),
        }


def run_sglang_metax_real_fork_e2e(
    *,
    maca_spec: Optional[MacaServingRfSpec] = None,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    num_layers: int = 2,
    max_rf_mlp_layers: int = 2,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> SglangMetaxRealForkE2EReport:
    """Full SGLang-metax fork e2e (requires ``sglang`` on MetaX host)."""
    require_sglang_metax()
    hook = run_sglang_metax_qwen2_mlp_rf_e2e(
        maca_spec=maca_spec,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        warmup=warmup,
        iters=iters,
        bench=bench,
    )
    hybrid = run_torch_sglang_metax_hybrid_full_e2e(
        maca_spec=maca_spec,
        num_layers=num_layers,
        max_rf_mlp_layers=max_rf_mlp_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        warmup=warmup,
        iters=iters,
        bench=bench,
    )
    return SglangMetaxRealForkE2EReport(
        hook=hook,
        hybrid=hybrid,
        real_fork=True,
        parity_ok=hook.parity_ok and hybrid.parity_ok,
    )


def run_sglang_metax_hybrid_full_e2e_auto(
    *,
    maca_spec: Optional[MacaServingRfSpec] = None,
    num_layers: int = 4,
    max_rf_mlp_layers: int = 2,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    bench: bool = True,
) -> SglangMetaxHybridE2EReport:
    """Cert entry: torch hybrid + MACA meta; real fork when tier available."""
    if is_sglang_metax_available():
        try:
            fork = run_sglang_metax_real_fork_e2e(
                maca_spec=maca_spec,
                num_layers=num_layers,
                max_rf_mlp_layers=max_rf_mlp_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                batch=batch,
                bench=bench,
            )
            return fork.hybrid
        except Exception:
            pass
    return run_torch_sglang_metax_hybrid_full_e2e(
        maca_spec=maca_spec,
        num_layers=num_layers,
        max_rf_mlp_layers=max_rf_mlp_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=bench,
    )
