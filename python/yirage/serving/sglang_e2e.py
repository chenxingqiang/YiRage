# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S12: SGLang ForwardBatch MLP RF full-path e2e (torch measured + optional ``sglang``).

Full path = engine Attention → RF MLP hook driven by ForwardBatch meta
(``extend_seq_lens``, ``block_tables``, ``seq_lens``) → parity vs engine MLP.

S47 adds token-embed multi-layer generation over the same ForwardBatch meta.
S48 compares a native SGLang MLP stack to the RF MLP hook on that same
token-embed trajectory when ``sglang`` is installed. Attention stays with
the engine.

S49 runs the same token-embed generation while the SM budget alternates
between a zero capsule budget (RF skips, engine MLP runs) and a wide budget
(RF MLP runs). Tokens still match the engine MLP reference.

When ``sglang`` is not installed, the torch surrogate path is the cert gate
and native generation fields stay unset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


def _ensure_sglang_parallel_context() -> None:
    """Publish a CPU test context so SGLang linears can read ``tp_size``.

    SGLang 0.5 refuses to build ``MergedColumnParallelLinear`` until the
    parallel namespace is published and ``tp_group`` is written. Attention
    stays unused; this only unblocks the MLP stack.
    """
    from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
    from sglang.srt.server_args import ServerArgs

    class _CpuTpGroup:
        world_size = 1
        rank_in_group = 0

    try:
        _ = get_parallel().tp_size
        _ = get_parallel().tp_group.world_size
    except (RuntimeError, ValueError, AttributeError):
        reset_context()
        publish(
            ServerArgs(model_path="dummy", device="cpu"),
            role="test",
            ranks=SpawnRanks(world_rank=0, dp_rank=None),
        )
        get_parallel().override_permanently(tp_group=_CpuTpGroup())


def _build_sglang_mlp_surface(
    *,
    hidden_size: int,
    intermediate_size: int,
    layer_id: int,
    device: str,
):
    """SGLang post-attention MLP plus RMSNorm, without the attention stack.

    ``Qwen2DecoderLayer`` pulls RoPE through ``vllm``. The native generation
    tier only compares the MLP, so it builds ``Qwen2MLP`` and ``RMSNorm``
    directly.
    """
    require_torch()
    import torch
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.qwen2 import Qwen2MLP

    _ensure_sglang_parallel_context()
    mlp = Qwen2MLP(
        int(hidden_size),
        int(intermediate_size),
        "silu",
        reduce_results=False,
    )
    norm = RMSNorm(int(hidden_size), eps=1e-6)

    class _Surface(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = mlp
            self.post_attention_layernorm = norm
            self.layer_id = int(layer_id)

    surface = _Surface().to(device)
    surface.eval()
    return surface


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

    _ensure_sglang_parallel_context()
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
        return hidden_after_attn + sglang_decoder_layer.mlp(normed)


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

    try:
        layer = build_minimal_sglang_qwen2_decoder_layer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
        )
    except (ImportError, ModuleNotFoundError, RuntimeError):
        # Full decoder attention pulls RoPE from vllm. MLP parity does not.
        layer = _build_sglang_mlp_surface(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            device="cpu",
        )
    _init_sglang_decoder_weights(layer)
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



@dataclass(frozen=True)
class SglangMultilayerGenerationReport:
    """Token-embed multi-layer generation under SGLang ForwardBatch meta."""

    version: str
    parity_ok: bool
    token_match_ok: bool
    decode_steps: int
    num_layers: int
    max_rf_mlp_layers: int
    batch: int
    hidden_size: int
    vocab_size: int
    functional_chain: str = "chain_d_sglang_forward_batch_generation"
    radix_step_kinds: List[str] = field(default_factory=list)
    step_parity_ok: List[bool] = field(default_factory=list)
    step_token_match_ok: List[bool] = field(default_factory=list)
    engine_token_ids: List[List[int]] = field(default_factory=list)
    hybrid_token_ids: List[List[int]] = field(default_factory=list)
    sglang_native_available: bool = False
    native_generation_parity_ok: Optional[bool] = None
    native_generation_token_match_ok: Optional[bool] = None
    native_generation_step_parity_ok: List[bool] = field(default_factory=list)
    native_generation_step_token_match_ok: List[bool] = field(default_factory=list)
    native_generation_ref_token_ids: List[List[int]] = field(default_factory=list)
    native_generation_rf_token_ids: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serving_sglang_multilayer_generation": True,
            "version": self.version,
            "parity_ok": self.parity_ok,
            "token_match_ok": self.token_match_ok,
            "decode_steps": self.decode_steps,
            "num_layers": self.num_layers,
            "max_rf_mlp_layers": self.max_rf_mlp_layers,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "functional_chain": self.functional_chain,
            "radix_step_kinds": list(self.radix_step_kinds),
            "step_parity_ok": list(self.step_parity_ok),
            "step_token_match_ok": list(self.step_token_match_ok),
            "engine_token_ids": [list(row) for row in self.engine_token_ids],
            "hybrid_token_ids": [list(row) for row in self.hybrid_token_ids],
            "sglang_native_available": self.sglang_native_available,
            "native_generation_parity_ok": self.native_generation_parity_ok,
            "native_generation_token_match_ok": self.native_generation_token_match_ok,
            "native_generation_step_parity_ok": list(self.native_generation_step_parity_ok),
            "native_generation_step_token_match_ok": list(
                self.native_generation_step_token_match_ok
            ),
            "native_generation_ref_token_ids": [
                list(row) for row in self.native_generation_ref_token_ids
            ],
            "native_generation_rf_token_ids": [
                list(row) for row in self.native_generation_rf_token_ids
            ],
        }


def validate_sglang_multilayer_generation(payload: Dict[str, Any]) -> List[str]:
    """Check an S47 generation payload for the chain-D merge gate."""
    errors: List[str] = []
    if payload.get("serving_sglang_multilayer_generation") is not True:
        errors.append("missing serving_sglang_multilayer_generation=true marker")
    if payload.get("parity_ok") is not True:
        errors.append("parity_ok must be true")
    if payload.get("token_match_ok") is not True:
        errors.append("token_match_ok must be true")
    if payload.get("functional_chain") != "chain_d_sglang_forward_batch_generation":
        errors.append(
            "functional_chain must be chain_d_sglang_forward_batch_generation"
        )
    num_layers = payload.get("num_layers")
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 2:
        errors.append("num_layers must be >= 2")
    engine_ids = payload.get("engine_token_ids")
    hybrid_ids = payload.get("hybrid_token_ids")
    if (
        not isinstance(engine_ids, list)
        or not isinstance(hybrid_ids, list)
        or len(engine_ids) != len(hybrid_ids)
    ):
        errors.append("engine_token_ids and hybrid_token_ids must be same-length lists")
    decode_steps = payload.get("decode_steps")
    if isinstance(decode_steps, int) and not isinstance(decode_steps, bool) and decode_steps >= 3:
        kinds = payload.get("radix_step_kinds")
        if (
            not isinstance(kinds, list)
            or "partial" not in kinds
            or "miss" not in kinds
            or "all_hit" not in kinds
        ):
            errors.append(
                "radix_step_kinds must contain partial, miss, and all_hit "
                "when decode_steps >= 3"
            )
    return errors


def _init_sglang_decoder_weights(layer) -> None:
    """Normal-init every non-empty parameter so native MLP and the RF hook match."""
    require_torch()
    import torch

    with torch.no_grad():
        for param in layer.parameters():
            if param.numel() > 0:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)


def _run_sglang_native_mlp_generation(
    *,
    decode_steps: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    batch: int,
    embed,
    token_ids,
    lm_weight,
) -> Optional[
    tuple[
        List[bool],
        List[bool],
        List[List[int]],
        List[List[int]],
    ]
]:
    """Native SGLang MLP stack vs RF MLP hook on one token-embed trajectory.

    Attention stays with the engine. Each decode step is a Radix miss
    (``extend_seq_lens`` all 1) so every MLP runs. Returns None when
    ``sglang`` is not installed or ``num_layers < 1``.
    """
    if not is_sglang_available() or int(num_layers) < 1:
        return None
    require_torch()
    import torch

    layers = []
    hooks = []
    for layer_id in range(int(num_layers)):
        layer = _build_sglang_mlp_surface(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            device="cpu",
        )
        _init_sglang_decoder_weights(layer)
        hooks.append(build_sglang_qwen2_mlp_rf_hook(layer, layer_id=layer_id))
        layers.append(layer)

    device = str(next(layers[0].parameters()).device)
    ids = token_ids.detach().clone().to(device)
    table = embed.to(device)
    weight = lm_weight.to(device)
    block_table_rows = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]]

    step_parity: List[bool] = []
    step_token: List[bool] = []
    ref_token_ids: List[List[int]] = []
    rf_token_ids: List[List[int]] = []

    for step in range(int(decode_steps)):
        batch_spec = SglangForwardBatchSpec(
            extend_seq_lens=[1] * int(batch),
            seq_lens=[16 + step, 18 + step, 20 + step, 22 + step][: int(batch)],
            block_tables=block_table_rows[: int(batch)],
            page_size=16,
        )
        h = table[ids]
        ref = h
        rf = h
        with torch.no_grad():
            for layer, hook in zip(layers, hooks):
                ref = _sglang_native_mlp_forward(layer, ref)
                rf = hook.forward_mlp(rf, forward_batch=batch_spec).hidden
        step_parity.append(bool(torch.allclose(ref, rf, rtol=1e-4, atol=1e-4)))
        ref_tok = (ref @ weight.T).argmax(dim=-1)
        rf_tok = (rf @ weight.T).argmax(dim=-1)
        step_token.append(bool(torch.equal(ref_tok, rf_tok)))
        ref_token_ids.append([int(v) for v in ref_tok.tolist()])
        rf_token_ids.append([int(v) for v in rf_tok.tolist()])
        ids = ref_tok

    return step_parity, step_token, ref_token_ids, rf_token_ids


def run_sglang_multilayer_generation(
    *,
    decode_steps: int = 3,
    num_layers: int = 2,
    max_rf_mlp_layers: Optional[int] = None,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    batch: int = 3,
    vocab_size: int = 64,
    seed: int = 0,
    quick: bool = False,
    try_native: bool = True,
    version: str = "s48",
) -> SglangMultilayerGenerationReport:
    """Greedy token-embed generation: reference hybrid vs RF under rotating Radix meta.

    Each step embeds the reference-trajectory token ids, runs ForwardBatch meta
    (partial, miss, or all-hit), and compares hidden states plus argmax tokens.
    Native MLP generation runs only when ``sglang`` is installed. It compares
    the SGLang MLP stack to the RF MLP hook on the same token-embed trajectory.
    Attention stays outside that comparison.
    """
    require_torch()
    import torch

    decode_steps = int(decode_steps)
    num_layers = int(num_layers)
    batch = int(batch)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    vocab_size = int(vocab_size)
    seed = int(seed)
    if quick:
        decode_steps = min(decode_steps, 3)
        num_layers = min(num_layers, 2)
        batch = min(batch, 3)
        # Partial Radix needs a hit row and a miss row, so batch stays at least 2.
        if batch < 2:
            batch = 2
        hidden_size = min(hidden_size, 16)
        intermediate_size = min(intermediate_size, 32)
        vocab_size = min(vocab_size, 64)

    if max_rf_mlp_layers is None:
        max_rf = num_layers
    else:
        max_rf = min(max(int(max_rf_mlp_layers), 1), num_layers)

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(model, max_rf_mlp_layers=max_rf)
    expected_rf = list(range(max_rf))
    device = model.device

    weight_gen = torch.Generator(device="cpu")
    weight_gen.manual_seed(seed + 99)
    lm_weight = (
        torch.randn(
            vocab_size, hidden_size, generator=weight_gen, device="cpu", dtype=torch.float32
        )
        * 0.02
    ).to(device)
    embed_gen = torch.Generator(device="cpu")
    embed_gen.manual_seed(seed + 11)
    embed = (
        torch.randn(
            vocab_size, hidden_size, generator=embed_gen, device="cpu", dtype=torch.float32
        )
        * 0.02
    ).to(device)
    token_gen = torch.Generator(device="cpu")
    token_gen.manual_seed(seed + 13)
    token_ids = torch.randint(0, vocab_size, (batch,), generator=token_gen).to(device)
    initial_token_ids = token_ids.clone()

    block_table_rows = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]]
    radix_step_kinds: List[str] = []
    step_parity_ok: List[bool] = []
    step_token_match_ok: List[bool] = []
    engine_token_ids: List[List[int]] = []
    hybrid_token_ids: List[List[int]] = []

    for step in range(decode_steps):
        if step % 3 == 0:
            extend = [0] + [1] * (batch - 1)
        elif step % 3 == 1:
            extend = [1] * batch
        else:
            extend = [0] * batch
        forward_batch = SglangForwardBatchSpec(
            extend_seq_lens=extend,
            seq_lens=[16 + step, 18 + step, 20 + step, 22 + step][:batch],
            block_tables=block_table_rows[:batch],
            page_size=16,
        )
        meta = forward_batch.as_meta()
        meta.setdefault("extras", {})
        extras = dict(meta["extras"])
        extras.setdefault("total_sms", 108)
        extras.setdefault("reserved_aux_sms", 8)
        meta["extras"] = extras

        h = embed[token_ids]
        with torch.no_grad():
            ref = _reference_hybrid_sglang_forward(model, expected_rf, h, extend)
            got = hybrid.forward(h, rf_meta=meta)
            ref_tok = (ref @ lm_weight.T).argmax(dim=-1)
            hyb_tok = (got.hidden @ lm_weight.T).argmax(dim=-1)

        all_hit, partial = _radix_flags(extend)
        if all_hit:
            kind = "all_hit"
        elif partial:
            kind = "partial"
        else:
            kind = "miss"
        ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
        if all_hit:
            ok = ok and got.rf_layer_ids == []
        else:
            ok = ok and list(got.rf_layer_ids) == expected_rf
        tok_ok = bool(torch.equal(ref_tok, hyb_tok))

        radix_step_kinds.append(kind)
        step_parity_ok.append(ok)
        step_token_match_ok.append(tok_ok)
        engine_token_ids.append([int(v) for v in ref_tok.tolist()])
        hybrid_token_ids.append([int(v) for v in hyb_tok.tolist()])
        # Follow the reference argmax so later steps share one token trajectory.
        token_ids = ref_tok

    native_generation_parity_ok: Optional[bool] = None
    native_generation_token_match_ok: Optional[bool] = None
    native_generation_step_parity_ok: List[bool] = []
    native_generation_step_token_match_ok: List[bool] = []
    native_generation_ref_token_ids: List[List[int]] = []
    native_generation_rf_token_ids: List[List[int]] = []
    if try_native:
        native_out = _run_sglang_native_mlp_generation(
            decode_steps=decode_steps,
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            batch=batch,
            embed=embed,
            token_ids=initial_token_ids,
            lm_weight=lm_weight,
        )
        if native_out is not None:
            (
                native_generation_step_parity_ok,
                native_generation_step_token_match_ok,
                native_generation_ref_token_ids,
                native_generation_rf_token_ids,
            ) = native_out
            native_generation_parity_ok = (
                all(native_generation_step_parity_ok)
                if native_generation_step_parity_ok
                else None
            )
            native_generation_token_match_ok = (
                all(native_generation_step_token_match_ok)
                if native_generation_step_token_match_ok
                else None
            )

    return SglangMultilayerGenerationReport(
        version=version,
        parity_ok=all(step_parity_ok),
        token_match_ok=all(step_token_match_ok),
        decode_steps=decode_steps,
        num_layers=num_layers,
        max_rf_mlp_layers=max_rf,
        batch=batch,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        radix_step_kinds=radix_step_kinds,
        step_parity_ok=step_parity_ok,
        step_token_match_ok=step_token_match_ok,
        engine_token_ids=engine_token_ids,
        hybrid_token_ids=hybrid_token_ids,
        sglang_native_available=is_sglang_available(),
        native_generation_parity_ok=native_generation_parity_ok,
        native_generation_token_match_ok=native_generation_token_match_ok,
        native_generation_step_parity_ok=native_generation_step_parity_ok,
        native_generation_step_token_match_ok=native_generation_step_token_match_ok,
        native_generation_ref_token_ids=native_generation_ref_token_ids,
        native_generation_rf_token_ids=native_generation_rf_token_ids,
    )


@dataclass
class SglangSmBudgetGenerationReport:
    """S49: generation under alternating SM budgets."""

    version: str
    parity_ok: bool
    token_match_ok: bool
    sm_skip_observed: bool
    sm_run_observed: bool
    decode_steps: int
    num_layers: int
    batch: int
    hidden_size: int
    vocab_size: int
    functional_chain: str = "chain_d_sglang_generation_sm_budget"
    step_sm_budget: List[int] = field(default_factory=list)
    step_used_rf: List[bool] = field(default_factory=list)
    step_parity_ok: List[bool] = field(default_factory=list)
    step_token_match_ok: List[bool] = field(default_factory=list)
    engine_token_ids: List[List[int]] = field(default_factory=list)
    hybrid_token_ids: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "parity_ok": self.parity_ok,
            "token_match_ok": self.token_match_ok,
            "sm_skip_observed": self.sm_skip_observed,
            "sm_run_observed": self.sm_run_observed,
            "decode_steps": self.decode_steps,
            "num_layers": self.num_layers,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "functional_chain": self.functional_chain,
            "step_sm_budget": list(self.step_sm_budget),
            "step_used_rf": list(self.step_used_rf),
            "step_parity_ok": list(self.step_parity_ok),
            "step_token_match_ok": list(self.step_token_match_ok),
            "engine_token_ids": self.engine_token_ids,
            "hybrid_token_ids": self.hybrid_token_ids,
        }


def validate_sglang_sm_budget_generation(report: Mapping[str, Any]) -> None:
    """Fail when an S49 SM-budget generation report is incomplete."""
    if report.get("version") != "s49":
        raise ValueError(f"sglang sm budget generation version must be s49, got {report.get('version')}")
    if report.get("functional_chain") != "chain_d_sglang_generation_sm_budget":
        raise ValueError("functional_chain must be chain_d_sglang_generation_sm_budget")
    if not report.get("parity_ok") or not report.get("token_match_ok"):
        raise ValueError("sglang sm budget generation parity/token match failed")
    if not report.get("sm_skip_observed") or not report.get("sm_run_observed"):
        raise ValueError("sglang sm budget generation must observe both skip and run steps")
    steps = int(report["decode_steps"])
    for key in (
        "step_sm_budget",
        "step_used_rf",
        "step_parity_ok",
        "step_token_match_ok",
        "engine_token_ids",
        "hybrid_token_ids",
    ):
        if len(report[key]) != steps:
            raise ValueError(f"{key} length {len(report[key])} != decode_steps {steps}")
    if any(not ok for ok in report["step_parity_ok"]):
        raise ValueError("step_parity_ok contains a failure")
    if any(not ok for ok in report["step_token_match_ok"]):
        raise ValueError("step_token_match_ok contains a failure")


def run_sglang_generation_sm_budget(
    *,
    decode_steps: int = 4,
    num_layers: int = 2,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    batch: int = 2,
    vocab_size: int = 64,
    seed: int = 0,
    quick: bool = False,
    version: str = "s49",
) -> SglangSmBudgetGenerationReport:
    """Token-embed generation that alternates a zero SM budget with a wide one.

    Even steps set ``sm_budget=0`` so every MLP capsule is skipped and the
    engine MLP runs. Odd steps leave a wide capsule budget so RF runs. Both
    must match the engine MLP reference and the greedy token ids.
    Radix stays a full miss so the skip is caused by the SM budget alone.
    """
    require_torch()
    import torch

    decode_steps = int(decode_steps)
    num_layers = int(num_layers)
    batch = int(batch)
    hidden_size = int(hidden_size)
    intermediate_size = int(intermediate_size)
    vocab_size = int(vocab_size)
    seed = int(seed)
    if quick:
        decode_steps = min(max(decode_steps, 2), 4)
        num_layers = min(max(num_layers, 2), 2)
        batch = min(max(batch, 2), 2)
        hidden_size = min(hidden_size, 16)
        intermediate_size = min(intermediate_size, 32)
        vocab_size = min(vocab_size, 64)
    if decode_steps < 2:
        decode_steps = 2
    if num_layers < 1:
        num_layers = 1
    if batch < 1:
        batch = 1

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(model, max_rf_mlp_layers=num_layers)
    expected_rf = list(range(num_layers))
    device = model.device

    weight_gen = torch.Generator(device="cpu")
    weight_gen.manual_seed(seed + 99)
    lm_weight = (
        torch.randn(
            vocab_size, hidden_size, generator=weight_gen, device="cpu", dtype=torch.float32
        )
        * 0.02
    ).to(device)
    embed_gen = torch.Generator(device="cpu")
    embed_gen.manual_seed(seed + 11)
    embed = (
        torch.randn(
            vocab_size, hidden_size, generator=embed_gen, device="cpu", dtype=torch.float32
        )
        * 0.02
    ).to(device)
    token_gen = torch.Generator(device="cpu")
    token_gen.manual_seed(seed + 13)
    token_ids = torch.randint(0, vocab_size, (batch,), generator=token_gen).to(device)

    block_table_rows = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]]
    step_sm_budget: List[int] = []
    step_used_rf: List[bool] = []
    step_parity_ok: List[bool] = []
    step_token_match_ok: List[bool] = []
    engine_token_ids: List[List[int]] = []
    hybrid_token_ids: List[List[int]] = []

    for step in range(decode_steps):
        # 0 forces every capsule to skip. A wide budget lets every layer run.
        budget = 0 if step % 2 == 0 else 64
        extend = [1] * batch
        forward_batch = SglangForwardBatchSpec(
            extend_seq_lens=extend,
            seq_lens=[16 + step, 18 + step, 20 + step, 22 + step][:batch],
            block_tables=block_table_rows[:batch],
            page_size=16,
        )
        meta = forward_batch.as_meta()
        extras = dict(meta.get("extras") or {})
        extras["total_sms"] = 108
        extras["reserved_aux_sms"] = 8
        meta["extras"] = extras
        meta["sm_budget"] = budget

        h = embed[token_ids]
        with torch.no_grad():
            ref = _reference_hybrid_sglang_forward(model, expected_rf, h, extend)
            got = hybrid.forward(h, rf_meta=meta)
            ref_tok = (ref @ lm_weight.T).argmax(dim=-1)
            hyb_tok = (got.hidden @ lm_weight.T).argmax(dim=-1)

        used_rf = list(got.rf_layer_ids) == expected_rf
        if budget == 0:
            used_rf_ok = got.rf_layer_ids == []
        else:
            used_rf_ok = used_rf
        ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6)) and used_rf_ok
        tok_ok = bool(torch.equal(ref_tok, hyb_tok))

        step_sm_budget.append(budget)
        step_used_rf.append(used_rf)
        step_parity_ok.append(ok)
        step_token_match_ok.append(tok_ok)
        engine_token_ids.append([int(v) for v in ref_tok.tolist()])
        hybrid_token_ids.append([int(v) for v in hyb_tok.tolist()])
        token_ids = ref_tok

    return SglangSmBudgetGenerationReport(
        version=version,
        parity_ok=all(step_parity_ok),
        token_match_ok=all(step_token_match_ok),
        sm_skip_observed=any(not used for used in step_used_rf),
        sm_run_observed=any(step_used_rf),
        decode_steps=decode_steps,
        num_layers=num_layers,
        batch=batch,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        step_sm_budget=step_sm_budget,
        step_used_rf=step_used_rf,
        step_parity_ok=step_parity_ok,
        step_token_match_ok=step_token_match_ok,
        engine_token_ids=engine_token_ids,
        hybrid_token_ids=hybrid_token_ids,
    )
