# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S11: vLLM-style MLP RF full-path e2e (torch measured + optional ``vllm``).

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
from .vllm_plugin import (
    _VllmMlpLayerAdapter,
    build_vllm_qwen2_mlp_rf_hook,
    extract_qwen2_mlp_weights,
    is_vllm_available,
    require_vllm,
)
from .vllm_runtime import (
    allocate_vllm_test_layer_prefix,
    ensure_vllm_single_process_runtime,
    vllm_test_config_context,
)


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
    """Reference fused MLP on a vLLM Qwen2 decoder layer (post-attn hidden).

    Computes the same function as the RF MLP capsule (``mlp_torch``):
    ``hidden + mlp(rmsnorm(hidden))`` -- the post-attention norm + MLP +
    residual-add block the capsule replaces. The plain ``layer.mlp`` call
    alone is NOT the reference (it skips the norm and the residual add).
    """
    require_torch()
    import torch

    with torch.no_grad():
        try:
            normed = vllm_decoder_layer.post_attention_layernorm(hidden_after_attn)
            return hidden_after_attn + vllm_decoder_layer.mlp(normed)
        except (AttributeError, RuntimeError, NotImplementedError):
            # CPU wheel: freshly constructed layers may lack cpu_linear kernels;
            # fall back to torch reference using weights extracted from vLLM modules.
            adapter = _VllmMlpLayerAdapter(
                extract_qwen2_mlp_weights(vllm_decoder_layer)
            )
            return adapter.mlp_forward(hidden_after_attn)


def _vllm_decoder_stack_logits_hidden(
    hidden_states,
    residual,
):
    """Combine decoder layer stack outputs for greedy token projection."""
    if residual is None:
        return hidden_states
    return hidden_states + residual


def _vllm_standalone_decode_forward_context(layers, *, vcfg, batch, device):
    """Install the minimal vLLM forward context for standalone decode chains.

    vLLM >=0.26 routes every attention through the unified attention custom
    op, which reads a process-global ``ForwardContext`` (attention metadata,
    per-layer KV caches and slot mappings) normally maintained by the engine's
    model runner. The standalone parity chains run outside the engine, so we
    install a minimal context modelling each forward as a one-token-per-request
    decode over a fresh paged cache: every request writes its own K/V slot and
    attends to that slot only. The attention stays deterministic and identical
    across the reference and RF-hook chains -- exactly what the S45/S46 parity
    benches assert -- without depending on engine-side scheduler state.

    Raises ``NotImplementedError`` off the CPU wheel so the native tiers
    degrade honestly instead of computing on garbage (the warming-up path
    returns uninitialized memory when ``attn_metadata`` is None).
    """
    import torch

    try:
        from vllm.platforms import current_platform

        if not current_platform.is_cpu():
            raise NotImplementedError(
                "standalone native decode context is CPU-wheel only"
            )
        from vllm import _custom_ops as ops
        from vllm import envs as vllm_envs
        from vllm.forward_context import (
            create_forward_context,
            override_forward_context,
        )
        from vllm.v1.attention.backends.cpu_attn import (
            CPUAttentionMetadata,
            _get_attn_isa,
        )
    except ImportError as e:
        raise NotImplementedError(f"vLLM CPU wheel backend unavailable: {e}") from e

    attns = [layer.self_attn.attn for layer in layers]
    # One token per request: request i owns block i (BLOCK-aligned), its token
    # lives at slot i * BLOCK + 0. Write kernel wants int64 slot_mapping; the
    # read kernel wants int32 query_start_loc/seq_lens/block_table; cache
    # blocks must satisfy the CPU kernel's block-size alignment.
    BLOCK = 16
    slot_mapping = (
        torch.arange(batch, dtype=torch.int64, device=device) * BLOCK
    )
    block_table = torch.arange(batch, dtype=torch.int32, device=device).view(batch, 1)
    seq_lens = torch.ones(batch, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32, device=device)
    head_dim = int(attns[0].head_size)
    scheduler_metadata = ops.cpu_attn_get_scheduler_metadata(
        num_reqs=batch,
        num_heads=int(attns[0].num_heads),
        num_kv_heads=int(attns[0].num_kv_heads),
        head_dim=head_dim,
        seq_lens=seq_lens,
        dtype=torch.float32,
        query_start_loc=query_start_loc,
        causal=True,
        sliding_window_size=-1,
        isa=_get_attn_isa(torch.float32, BLOCK, head_dim, "auto"),
        enable_kv_split=vllm_envs.VLLM_CPU_ATTN_SPLIT_KV,
        dynamic_causal=None,
    )
    meta = CPUAttentionMetadata(
        num_actual_tokens=batch,
        max_query_len=1,
        query_start_loc=query_start_loc,
        max_seq_len=1,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        scheduler_metadata=scheduler_metadata,
    )
    for attn in attns:
        attn.kv_cache = torch.zeros(
            batch,
            attn.num_kv_heads,
            BLOCK,
            2 * attn.head_size,
            dtype=torch.float32,
            device=device,
        )
    ctx = create_forward_context(
        attn_metadata=meta,
        vllm_config=vcfg,
        slot_mapping={attn.layer_name: slot_mapping for attn in attns},
    )
    return override_forward_context(ctx)


def _vllm_chain_native_decoder_layers(
    layers,
    *,
    positions,
    hidden_states,
) -> Any:
    """Run full vLLM decoder layers (native attn + native MLP)."""
    require_torch()
    import torch

    residual = None
    hidden = hidden_states
    with vllm_test_config_context() as vcfg:
        with _vllm_standalone_decode_forward_context(
            layers,
            vcfg=vcfg,
            batch=int(hidden_states.shape[0]),
            device=hidden_states.device,
        ):
            with torch.no_grad():
                for layer in layers:
                    hidden, residual = layer(positions, hidden, residual)
    return _vllm_decoder_stack_logits_hidden(hidden, residual)


def _vllm_chain_native_decoder_layers_rf_mlp(
    layers,
    hooks,
    *,
    positions,
    hidden_states,
    rf_meta_base,
) -> Any:
    """Run vLLM self-attn + RF MLP hook per layer (S45 full decoder hybrid)."""
    require_torch()
    import torch

    residual = None
    hidden = hidden_states
    with vllm_test_config_context() as vcfg:
        with _vllm_standalone_decode_forward_context(
            layers,
            vcfg=vcfg,
            batch=int(hidden_states.shape[0]),
            device=hidden_states.device,
        ):
            with torch.no_grad():
                for layer, hook in zip(layers, hooks):
                    hidden, residual = _rf_hook_decoder_step(
                        layer,
                        hook,
                        positions=positions,
                        hidden=hidden,
                        residual=residual,
                        rf_meta_base=rf_meta_base,
                    )
    return _vllm_decoder_stack_logits_hidden(hidden, residual)


def _rf_hook_decoder_step(layer, hook, *, positions, hidden, residual, rf_meta_base):
    """One native-attention step whose MLP is the RF capsule.

    The capsule replaces post-attention norm + MLP + residual-add, matching
    ``_vllm_native_mlp_forward``. The hook therefore receives
    ``attn_out + residual`` (not the post-attention norm output) and the
    residual is cleared so the next layer does not add it again.
    """
    cap_name = hook.override.capsule_name
    meta = dict(rf_meta_base)
    meta["enabled"] = {cap_name}
    if residual is None:
        residual = hidden
        normed = layer.input_layernorm(hidden)
    else:
        normed, residual = layer.input_layernorm(hidden, residual)
    attn_out = layer.self_attn(positions=positions, hidden_states=normed)
    mlp_in = attn_out + residual
    hidden_out = hook.forward_mlp(mlp_in, rf_meta=meta).hidden
    return hidden_out, None


def _init_vllm_qwen2_layer_weights(layer) -> None:
    """Deterministic small init so CPU e2e parity is numerically stable."""
    require_torch()
    import torch

    with torch.no_grad():
        for p in layer.parameters():
            if p.numel() > 0:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)


def _mount_vllm_cpu_linear(layer) -> int:
    """Mount ``cpu_linear`` on vLLM CPU-wheel linear modules (cert-gate repair).

    vLLM CPU wheels route linear forwards through ``layer.cpu_linear``, which
    the engine's weight-loading pipeline normally attaches via
    ``process_weights_after_loading`` -> ``dispatch_cpu_unquantized_gemm``.
    Tiny decoder layers constructed standalone for parity benches never run
    that pipeline, so native forwards raise ``AttributeError``. Mount it here
    with ``remove_weight=False`` (the same out-of-engine pattern as
    ``vocab_parallel_embedding.py``) so weights stay in place for
    torch-reference parity checks. Returns the number of modules mounted
    (0 on non-CPU platforms or wheels without the dispatch helper).
    """
    try:
        from vllm.platforms import current_platform

        if not current_platform.is_cpu():
            return 0
        from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
    except Exception:
        return 0

    mounted = 0
    for module in _pending_cpu_linear_modules(layer):
        try:
            dispatch_cpu_unquantized_gemm(module, remove_weight=False)
            mounted += 1
        except Exception:
            continue
    return mounted


def _pending_cpu_linear_modules(layer) -> List[Any]:
    """Modules whose 2-D weight still needs a ``cpu_linear`` mount.

    vLLM CPU linears are not ``torch.nn.Linear``. Selection is by weight
    rank, so a custom module with a non-empty 2-D ``weight`` is included.
    Rank-1 norms, empty parameters, and modules that already have
    ``cpu_linear`` are skipped so the mount cannot clobber them.
    """
    require_torch()
    import torch

    pending: List[Any] = []
    for module in layer.modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        if weight.numel() == 0 or getattr(module, "cpu_linear", None) is not None:
            continue
        pending.append(module)
    return pending


def native_tier_qwen2_head_layout(hidden_size: int) -> "tuple[int, int]":
    """``(num_attention_heads, num_key_value_heads)`` with a kernel-safe head_dim.

    The vLLM CPU attention kernel only accepts head_dim sizes aligned to its
    vector kernels: the default tiny-layer layout (hidden 64 / 4 heads ->
    head_dim 16) is rejected at runtime with
    ``Unsupported CPU attention configuration``. Native-tier decoder stacks
    pick the largest supported head_dim (128/64/32) that divides
    ``hidden_size``. Raises ``NotImplementedError`` when no supported layout
    exists so the tier degrades honestly instead of crashing the bench.
    """
    for head_dim in (128, 64, 32):
        if hidden_size % head_dim == 0:
            heads = hidden_size // head_dim
            return heads, heads
    raise NotImplementedError(
        f"hidden_size={hidden_size} admits no CPU-kernel-supported "
        "head_dim (32/64/128)"
    )


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
    """Construct a tiny ``Qwen2DecoderLayer`` (requires ``vllm`` + ``transformers``)."""
    require_vllm()
    require_torch()
    ensure_vllm_single_process_runtime()
    import torch
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    prefix = allocate_vllm_test_layer_prefix()
    cfg = Qwen2Config(
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        max_position_embeddings=int(max_position_embeddings),
        num_hidden_layers=1,
        vocab_size=128,
    )
    with vllm_test_config_context():
        try:
            layer = Qwen2DecoderLayer(config=cfg, prefix=prefix)
        except TypeError:
            layer = Qwen2DecoderLayer(cfg, layer_id=layer_id)
    layer = layer.to(dev)
    _init_vllm_qwen2_layer_weights(layer)
    _mount_vllm_cpu_linear(layer)
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
    """vLLM Qwen2 layer: RF MLP hook vs native ``layer.mlp`` (requires ``vllm``)."""
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

    with vllm_test_config_context():
        with torch.no_grad():
            h_attn = x
            ref = _vllm_native_mlp_forward(layer, h_attn)
            got = hook.forward_mlp(h_attn, rf_meta=rf_meta)
    parity_ok = bool(torch.allclose(got.hidden, ref, rtol=1e-4, atol=1e-4))

    hook_ms = eng_ms = None
    if bench:
        with vllm_test_config_context():
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
    """Run vLLM e2e when installed; else torch hybrid full path."""
    if is_vllm_available():
        return run_vllm_qwen2_mlp_rf_e2e(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            batch=batch,
            layer_id=layer_id,
            bench=bench,
        )
    return run_torch_vllm_hybrid_full_e2e(
        num_layers=2,
        max_rf_mlp_layers=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=bench,
    )
