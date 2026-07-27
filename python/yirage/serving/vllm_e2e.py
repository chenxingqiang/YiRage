# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S11: vLLM-style MLP RF full-path e2e (torch measured + optional real ``vllm``).

Full path = engine Attention → RF MLP hook → parity vs engine MLP, optionally
multi-layer :class:`~yirage.serving.hybrid_model.HybridModelOverride`.

When ``vllm`` is not installed, the torch surrogate path is the cert gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .hybrid_model import HybridModelOverride
from .torch_engine import TorchDecoderLayer, TorchEngineModel
from .torch_exec import bench_forward, require_torch
from .torch_plugin import build_torch_mlp_rf_hook
from .vllm_plugin import build_vllm_qwen2_mlp_rf_hook, is_vllm_available, require_vllm


@dataclass(frozen=True)
class VllmMlpRfE2EReport:
    """Single-layer MLP RF hook e2e result."""

    parity_ok: bool
    used_rf_mlp: bool
    plugin: str
    device: str
    batch: int
    hidden_size: int
    intermediate_size: int
    layer_id: int
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
            "hook_mean_ms": self.hook_mean_ms,
            "engine_mean_ms": self.engine_mean_ms,
        }


@dataclass(frozen=True)
class VllmHybridE2EReport:
    """Multi-layer hybrid forward e2e (first K layers RF MLP)."""

    parity_ok: bool
    rf_layer_ids: List[int]
    device: str
    num_layers: int
    max_rf_mlp_layers: int
    batch: int
    hidden_size: int
    plugin: str
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
            "hybrid_mean_ms": self.hybrid_mean_ms,
            "engine_mean_ms": self.engine_mean_ms,
        }


def _vllm_native_mlp_forward(vllm_decoder_layer, hidden_after_attn):
    """Reference MLP on a real vLLM Qwen2 decoder layer (post-attn hidden)."""
    require_torch()
    import torch

    with torch.no_grad():
        normed = vllm_decoder_layer.post_attention_layernorm(hidden_after_attn)
        return vllm_decoder_layer.mlp(normed)


def build_minimal_vllm_qwen2_decoder_layer(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 128,
    layer_id: int = 0,
    device: Optional[str] = None,
):
    """Construct a tiny real ``Qwen2DecoderLayer`` (requires ``vllm`` + ``transformers``)."""
    require_vllm()
    require_torch()
    import torch
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

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
        layer = Qwen2DecoderLayer(config=cfg, prefix=f"layers.{layer_id}")
    except TypeError:
        layer = Qwen2DecoderLayer(cfg, layer_id=layer_id)
    layer = layer.to(dev)
    layer.eval()
    return layer


def run_torch_vllm_mlp_rf_e2e(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    layer_id: int = 0,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> VllmMlpRfE2EReport:
    """Measured torch full path: Attention (engine) + MLP via RF hook."""
    require_torch()
    import torch

    layer = TorchDecoderLayer(
        layer_id,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hook = build_torch_mlp_rf_hook(layer)
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=layer.device)
    cap_name = hook.override.capsule_name
    rf_meta: Dict[str, Any] = {"enabled": {cap_name}}

    with torch.no_grad():
        h_attn = layer.attention_forward(x)
        ref = layer.mlp_forward(h_attn)
        got = hook.forward_mlp(h_attn, rf_meta=rf_meta)
    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-5))

    hook_ms = eng_ms = None
    if bench:
        with torch.no_grad():
            h_fixed = layer.attention_forward(x)
            hook_b = bench_forward(
                lambda: hook.forward_mlp(h_fixed, rf_meta=rf_meta),
                name="vllm_style_hook",
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

    return VllmMlpRfE2EReport(
        parity_ok=parity_ok,
        used_rf_mlp=bool(got.used_rf_mlp),
        plugin="TorchDecoderMlpRfHook",
        device=str(layer.device),
        batch=int(batch),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        layer_id=int(layer_id),
        hook_mean_ms=hook_ms,
        engine_mean_ms=eng_ms,
    )


def run_torch_vllm_hybrid_full_e2e(
    *,
    num_layers: int = 4,
    max_rf_mlp_layers: int = 2,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 8,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    rf_meta: Optional[Mapping[str, Any]] = None,
    bench: bool = True,
) -> VllmHybridE2EReport:
    """Multi-layer full forward: hybrid first-K RF MLP vs engine-only reference."""
    require_torch()
    import torch

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(model, max_rf_mlp_layers=max_rf_mlp_layers)
    meta = dict(rf_meta or {})
    meta.setdefault("extras", {"total_sms": 108, "reserved_aux_sms": 8})
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        ref = model.forward_engine_full(x)
        got = hybrid.forward(x, rf_meta=meta)
    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
    expected_rf = list(range(int(max_rf_mlp_layers)))

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
                name="hybrid_full",
                warmup=warmup,
                iters=iters,
                device=model.device,
            )
        eng_ms = eng_b.mean_ms
        hybrid_ms = hyb_b.mean_ms

    return VllmHybridE2EReport(
        parity_ok=parity_ok and got.rf_layer_ids == expected_rf,
        rf_layer_ids=list(got.rf_layer_ids),
        device=str(model.device),
        num_layers=int(num_layers),
        max_rf_mlp_layers=int(max_rf_mlp_layers),
        batch=int(batch),
        hidden_size=int(hidden_size),
        plugin="HybridModelOverride+TorchDecoderMlpRfHook",
        hybrid_mean_ms=hybrid_ms,
        engine_mean_ms=eng_ms,
    )


def run_vllm_qwen2_mlp_rf_e2e(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    layer_id: int = 0,
    warmup: int = 2,
    iters: int = 8,
    bench: bool = True,
) -> VllmMlpRfE2EReport:
    """Real vLLM Qwen2 layer: RF MLP hook vs native ``layer.mlp`` (requires ``vllm``)."""
    require_vllm()
    require_torch()
    import torch

    layer = build_minimal_vllm_qwen2_decoder_layer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layer_id=layer_id,
    )
    hook = build_vllm_qwen2_mlp_rf_hook(layer, layer_id=layer_id)
    device = str(next(layer.parameters()).device)
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=device)
    cap_name = hook.override.capsule_name
    rf_meta: Dict[str, Any] = {"enabled": {cap_name}}

    with torch.no_grad():
        h_attn = x
        ref = _vllm_native_mlp_forward(layer, h_attn)
        got = hook.forward_mlp(h_attn, rf_meta=rf_meta)
    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-4, atol=1e-4))

    hook_ms = eng_ms = None
    if bench:
        with torch.no_grad():
            hook_b = bench_forward(
                lambda: hook.forward_mlp(h_attn, rf_meta=rf_meta),
                name="vllm_rf_hook",
                warmup=warmup,
                iters=iters,
                device=device,
            )
            eng_b = bench_forward(
                lambda: _vllm_native_mlp_forward(layer, h_attn),
                name="vllm_native_mlp",
                warmup=warmup,
                iters=iters,
                device=device,
            )
        hook_ms = hook_b.mean_ms
        eng_ms = eng_b.mean_ms

    return VllmMlpRfE2EReport(
        parity_ok=parity_ok,
        used_rf_mlp=bool(got.used_rf_mlp),
        plugin="VllmQwen2MlpRfHook",
        device=device,
        batch=int(batch),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        layer_id=int(layer_id),
        hook_mean_ms=hook_ms,
        engine_mean_ms=eng_ms,
    )


def run_vllm_mlp_rf_e2e_auto(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    batch: int = 4,
    layer_id: int = 0,
    bench: bool = True,
) -> Union[VllmMlpRfE2EReport, VllmHybridE2EReport]:
    """Run real vLLM e2e when installed; else torch hybrid full path."""
    if is_vllm_available():
        try:
            return run_vllm_qwen2_mlp_rf_e2e(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                batch=batch,
                layer_id=layer_id,
                bench=bench,
            )
        except Exception:
            pass
    return run_torch_vllm_hybrid_full_e2e(
        num_layers=2,
        max_rf_mlp_layers=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=bench,
    )
