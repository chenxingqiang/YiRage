# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU full-model Qwen2 HF e2e: prefill + decode + RuntimeFusion MLP hooks.

Default checkpoint: ``Qwen/Qwen2-0.5B`` (24 layers, H=896).

**YiRage optimization path (default when ``yirage.core`` is built):**
decode steps route MLP through ``backend=yirage_cpu`` capsules — gate_up via
``yirage.core`` seed graph + down via ``superoptimize(backend=\"cpu\")``.
Prefill stays on torch (``YirageServingMlpRunner`` gate_up requires batch=1),
matching the CUDA Qwen demo decode-only kernel policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .engine_stub import QWEN2_MLP_HF_ATTACH
from .exec_backend import BACKEND_TORCH, BACKEND_YIRAGE_CPU
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
from .runtime_fusion import RuntimeFusion, StepMeta
from .torch_exec import require_torch, to_torch
from .yirage_exec import is_yirage_core_available, require_yirage_core, resolve_serving_search_tier


DEFAULT_QWEN05B_MODEL = "Qwen/Qwen2-0.5B"

# Cache RF hooks — yirage_cpu superoptimize runs once per layer at init.
_RF_HOOK_CACHE: Dict[Tuple[int, str], "HfQwen2MlpRfHook"] = {}


def is_transformers_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def require_transformers() -> None:
    if not is_transformers_available():
        raise RuntimeError(
            "Qwen2 CPU full-model e2e requires transformers. "
            "Install with: pip install transformers"
        )


def resolve_hf_qwen_mlp_backend(mlp_backend: Optional[str] = None) -> str:
    """Default to ``yirage_cpu`` when native tier is built."""
    if mlp_backend is not None:
        if mlp_backend == BACKEND_YIRAGE_CPU:
            require_yirage_core()
        return mlp_backend
    if is_yirage_core_available():
        return BACKEND_YIRAGE_CPU
    return BACKEND_TORCH


class HfQwen2MlpLayerAdapter:
    """Duck-typed decoder layer for RF MLP hooks (HF Qwen2 weights)."""

    def __init__(self, hf_decoder_layer, *, layer_id: int):
        self.hf_layer = hf_decoder_layer
        self.layer_id = int(layer_id)
        mlp = hf_decoder_layer.mlp
        self.hidden_size = int(mlp.gate_proj.in_features)
        self.intermediate_size = int(mlp.gate_proj.out_features)
        self.device = str(mlp.gate_proj.weight.device)
        self.hf_attach = {
            k: f"model.layers.{layer_id}.{v}" for k, v in QWEN2_MLP_HF_ATTACH.items()
        }

    @property
    def rms_weight(self):
        return self.hf_layer.post_attention_layernorm.weight

    @property
    def w_gate(self):
        return self.hf_layer.mlp.gate_proj.weight.T.contiguous()

    @property
    def w_up(self):
        return self.hf_layer.mlp.up_proj.weight.T.contiguous()

    @property
    def w_down(self):
        return self.hf_layer.mlp.down_proj.weight.T.contiguous()

    def mlp_forward(self, hidden_after_attn):
        require_torch()
        h = self.hf_layer.post_attention_layernorm(hidden_after_attn)
        h = self.hf_layer.mlp(h)
        return hidden_after_attn + h


class HfQwen2MlpRfHook:
    """RuntimeFusion MLP hook on a HF Qwen2 decoder layer."""

    def __init__(
        self,
        adapter: HfQwen2MlpLayerAdapter,
        *,
        backend: str = BACKEND_TORCH,
    ):
        require_torch()
        if backend == BACKEND_YIRAGE_CPU:
            require_yirage_core()
        self.adapter = adapter
        self.backend = backend
        cap = build_layer_mlp_capsule(adapter, backend=backend)
        rf = RuntimeFusion([cap])
        self.override = RuntimeFusionMlpLayerOverride(
            adapter, rf, capsule_name=capsule_name_for_layer(adapter.layer_id)
        )
        self.superopt_elapsed_s = _capsule_superopt_elapsed(cap)

    def forward_mlp(
        self,
        hidden_after_attn,
        *,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> LayerForwardResult:
        return self.override.forward_mlp_only(hidden_after_attn, rf_meta=rf_meta)


def _capsule_superopt_elapsed(cap) -> float:
    runner = getattr(cap, "_yirage_runner", None)
    if runner is not None:
        return float(getattr(runner, "superopt_elapsed_s", 0.0))
    return 0.0


def build_hf_qwen_mlp_rf_hook(
    adapter: HfQwen2MlpLayerAdapter,
    *,
    backend: str = BACKEND_TORCH,
    use_cache: bool = True,
) -> HfQwen2MlpRfHook:
    key = (adapter.layer_id, backend)
    if use_cache and key in _RF_HOOK_CACHE:
        return _RF_HOOK_CACHE[key]
    hook = HfQwen2MlpRfHook(adapter, backend=backend)
    if use_cache:
        _RF_HOOK_CACHE[key] = hook
    return hook


def clear_hf_qwen_rf_hook_cache() -> None:
    _RF_HOOK_CACHE.clear()


def _forward_rf_mlp_hook(
    hook: HfQwen2MlpRfHook,
    hidden,
    *,
    rf_meta,
    device: str,
):
    """RF MLP on [B,S,H] or [B,H] hidden states."""
    require_torch()
    import torch

    if hidden.dim() == 3:
        batch, seq, width = hidden.shape
        if hook.backend == BACKEND_YIRAGE_CPU and batch * seq != 1:
            raise ValueError(
                "yirage_cpu MLP requires decode shape batch*seq==1; "
                "use torch backend for prefill"
            )
        flat = hidden.reshape(batch * seq, width)
        got = hook.forward_mlp(flat, rf_meta=rf_meta)
        out = to_torch(got.hidden, device=device).reshape(batch, seq, width)
        return out
    if hook.backend == BACKEND_YIRAGE_CPU and hidden.shape[0] != 1:
        raise ValueError(
            f"yirage_cpu MLP requires batch=1, got shape={tuple(hidden.shape)}"
        )
    got = hook.forward_mlp(hidden, rf_meta=rf_meta)
    return to_torch(got.hidden, device=device)


def _build_causal_mask_mapping(
    model,
    *,
    inputs_embeds,
    attention_mask,
    past_key_values,
    position_ids,
):
    """Transformers 5.x causal mask (replaces ``Qwen2Model._update_causal_mask``)."""
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    mask_kwargs = {
        "config": model.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if model.model.has_sliding_layers:
        mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    return mapping


def _layer_attention_mask(model, causal_mask_mapping, layer_idx: int):
    return causal_mask_mapping[model.config.layer_types[layer_idx]]


def _load_qwen05b_cpu(*, model_id: str = DEFAULT_QWEN05B_MODEL, dtype=None):
    require_transformers()
    require_torch()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = dtype or torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dt,
    )
    model.eval()
    device = "cpu"
    model.to(device)
    return model, tok, device


def _prepare_model_inputs(model, tokenizer, prompt: str, *, device: str):
    require_torch()
    import torch

    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(ids)
    return ids, attention_mask


def _native_forward_logits(model, input_ids, attention_mask):
    require_torch()
    with __import__("torch").no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits


def _get_rf_hooks(
    model,
    *,
    max_rf_mlp_layers: int,
    mlp_backend: str,
) -> List[HfQwen2MlpRfHook]:
    hooks: List[HfQwen2MlpRfHook] = []
    for i in range(int(max_rf_mlp_layers)):
        adapter = HfQwen2MlpLayerAdapter(model.model.layers[i], layer_id=i)
        hooks.append(build_hf_qwen_mlp_rf_hook(adapter, backend=mlp_backend))
    return hooks


def qwen2_forward_with_rf_mlp(
    model,
    input_ids,
    attention_mask,
    *,
    max_rf_mlp_layers: int = 2,
    mlp_backend: str = BACKEND_TORCH,
):
    """Prefill forward: first ``max_rf_mlp_layers`` MLP blocks via RuntimeFusion."""
    require_torch()
    import torch
    from transformers.cache_utils import DynamicCache

    if mlp_backend == BACKEND_YIRAGE_CPU:
        raise ValueError(
            "yirage_cpu cannot run prefill (seq_len>1); use mlp_backend=torch for prefill"
        )

    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        batch, seq_len = input_ids.shape
        device = input_ids.device
        cache = DynamicCache(config=model.config)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
        causal_mask_mapping = _build_causal_mask_mapping(
            model,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            position_ids=position_ids,
        )

        hooks = _get_rf_hooks(
            model,
            max_rf_mlp_layers=max_rf_mlp_layers,
            mlp_backend=mlp_backend,
        )

        for layer_idx, decoder_layer in enumerate(model.model.layers):
            attn_mask = _layer_attention_mask(model, causal_mask_mapping, layer_idx)
            if layer_idx < max_rf_mlp_layers:
                residual = hidden_states
                normed = decoder_layer.input_layernorm(hidden_states)
                attn_out, _ = decoder_layer.self_attn(
                    hidden_states=normed,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                hidden_states = residual + attn_out
                hook = hooks[layer_idx]
                cap = hook.override.capsule_name
                hidden_states = _forward_rf_mlp_hook(
                    hook,
                    hidden_states,
                    rf_meta={"enabled": {cap}},
                    device=device,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )

        hidden_states = model.model.norm(hidden_states)
        return model.lm_head(hidden_states)


def qwen2_decode_step_with_rf_mlp(
    model,
    *,
    next_id,
    attn_dec,
    cache,
    seq_len: int,
    device: str,
    max_rf_mlp_layers: int,
    mlp_backend: str,
):
    """Single decode step (q_len=1) with RF MLP on first ``max_rf_mlp_layers``."""
    require_torch()
    import torch

    dec_pos = torch.tensor([[seq_len]], device=device)
    dec_emb = model.model.embed_tokens(next_id)
    dec_pe = model.model.rotary_emb(dec_emb, dec_pos)
    dec_mask_mapping = _build_causal_mask_mapping(
        model,
        inputs_embeds=dec_emb,
        attention_mask=attn_dec,
        past_key_values=cache,
        position_ids=dec_pos,
    )
    hooks = _get_rf_hooks(
        model,
        max_rf_mlp_layers=max_rf_mlp_layers,
        mlp_backend=mlp_backend,
    )
    h = dec_emb
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn_mask = _layer_attention_mask(model, dec_mask_mapping, layer_idx)
        residual = h
        normed = decoder_layer.input_layernorm(h)
        attn_out, _ = decoder_layer.self_attn(
            hidden_states=normed,
            attention_mask=attn_mask,
            position_ids=dec_pos,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=dec_pe,
        )
        h = residual + attn_out
        residual = h
        if layer_idx < max_rf_mlp_layers:
            hook = hooks[layer_idx]
            cap = hook.override.capsule_name
            h = _forward_rf_mlp_hook(
                hook,
                h,
                rf_meta={"enabled": {cap}},
                device=device,
            )
        else:
            h = decoder_layer.post_attention_layernorm(h)
            h = decoder_layer.mlp(h)
            h = residual + h
    return model.lm_head(model.model.norm(h))


def _prefill_kv_cache(model, input_ids, attention_mask):
    """Native prefill; return ``DynamicCache`` for yirage decode steps."""
    require_torch()
    import torch
    from transformers.cache_utils import DynamicCache

    hidden_states = model.model.embed_tokens(input_ids)
    device = input_ids.device
    cache = DynamicCache(config=model.config)
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    mask_mapping = _build_causal_mask_mapping(
        model,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=cache,
        position_ids=position_ids,
    )
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=_layer_attention_mask(model, mask_mapping, layer_idx),
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
    return cache, seq_len


def greedy_decode_with_rf_mlp(
    model,
    input_ids,
    attention_mask,
    *,
    max_new_tokens: int,
    max_rf_mlp_layers: int,
    mlp_backend: str,
):
    """Greedy decode: native prefill KV + yirage/torch RF MLP on each decode step."""
    require_torch()
    import torch

    with torch.no_grad():
        cache, seq_len = _prefill_kv_cache(model, input_ids, attention_mask)
        out_ids = input_ids.clone()
        attn = attention_mask.clone()

        ref_prefill_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        next_id = torch.argmax(ref_prefill_logits[:, -1:, :], dim=-1)

        for _ in range(int(max_new_tokens)):
            attn = torch.cat(
                [attn, torch.ones((1, 1), device=input_ids.device, dtype=torch.long)],
                dim=1,
            )
            out_ids = torch.cat([out_ids, next_id], dim=1)
            logits = qwen2_decode_step_with_rf_mlp(
                model,
                next_id=next_id,
                attn_dec=attn,
                cache=cache,
                seq_len=seq_len,
                device=str(input_ids.device),
                max_rf_mlp_layers=max_rf_mlp_layers,
                mlp_backend=mlp_backend,
            )
            seq_len += 1
            next_id = torch.argmax(logits[:, -1:, :], dim=-1)

        return out_ids


@dataclass(frozen=True)
class HfQwen05bCpuE2EReport:
    """Full CPU flow: HF generate + RF/yirage MLP parity."""

    model_id: str
    parity_ok: bool
    prefill_parity_ok: bool
    decode_parity_ok: bool
    yirage_decode_parity_ok: bool
    generate_token_match_ok: bool
    used_rf_mlp_layers: int
    num_layers: int
    hidden_size: int
    prompt: str
    generated_text: str
    yirage_generated_text: str
    device: str
    prefill_ms: float
    generate_ms: float
    yirage_generate_ms: float
    mlp_backend: str
    decode_mlp_backend: str
    yirage_core_used: bool
    superopt_elapsed_s_total: float
    plugin: str
    serving_search_tier: str = "seed_verify"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parity_ok": self.parity_ok,
            "prefill_parity_ok": self.prefill_parity_ok,
            "decode_parity_ok": self.decode_parity_ok,
            "yirage_decode_parity_ok": self.yirage_decode_parity_ok,
            "generate_token_match_ok": self.generate_token_match_ok,
            "used_rf_mlp_layers": self.used_rf_mlp_layers,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "yirage_generated_text": self.yirage_generated_text,
            "device": self.device,
            "prefill_ms": round(self.prefill_ms, 4),
            "generate_ms": round(self.generate_ms, 4),
            "yirage_generate_ms": round(self.yirage_generate_ms, 4),
            "mlp_backend": self.mlp_backend,
            "decode_mlp_backend": self.decode_mlp_backend,
            "yirage_core_used": self.yirage_core_used,
            "superopt_elapsed_s_total": round(self.superopt_elapsed_s_total, 4),
            "plugin": self.plugin,
            "serving_search_tier": self.serving_search_tier,
        }


def _total_superopt_elapsed(max_rf_mlp_layers: int, decode_backend: str) -> float:
    total = 0.0
    for (layer_id, backend), hook in _RF_HOOK_CACHE.items():
        if backend == decode_backend and layer_id < max_rf_mlp_layers:
            total += hook.superopt_elapsed_s
    return total


def run_hf_qwen05b_cpu_e2e(
    *,
    model_id: str = DEFAULT_QWEN05B_MODEL,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 16,
    max_rf_mlp_layers: int = 2,
    mlp_backend: Optional[str] = None,
    quick: bool = False,
) -> HfQwen05bCpuE2EReport:
    """Load Qwen2-0.5B on CPU: generate + RF/yirage MLP decode parity."""
    require_transformers()
    require_torch()
    import torch

    clear_hf_qwen_rf_hook_cache()
    decode_backend = resolve_hf_qwen_mlp_backend(mlp_backend)
    prefill_backend = BACKEND_TORCH
    yirage_core_used = decode_backend == BACKEND_YIRAGE_CPU

    if quick:
        max_new_tokens = min(max_new_tokens, 8)
        max_rf_mlp_layers = min(max_rf_mlp_layers, 1)

    model, tokenizer, device = _load_qwen05b_cpu(model_id=model_id)
    cfg = model.config
    input_ids, attention_mask = _prepare_model_inputs(model, tokenizer, prompt, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generate_ms = (time.perf_counter() - t0) * 1000.0
    generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    t1 = time.perf_counter()
    ref_logits = _native_forward_logits(model, input_ids, attention_mask)
    rf_logits = qwen2_forward_with_rf_mlp(
        model,
        input_ids,
        attention_mask,
        max_rf_mlp_layers=max_rf_mlp_layers,
        mlp_backend=prefill_backend,
    )
    prefill_ms = (time.perf_counter() - t1) * 1000.0
    prefill_parity_ok = bool(
        torch.allclose(ref_logits, rf_logits, rtol=1e-4, atol=1e-3)
    )

    decode_parity_ok = True
    yirage_decode_parity_ok = True
    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = prefill_out.past_key_values
        next_id = torch.argmax(prefill_out.logits[:, -1:, :], dim=-1)
        attn_dec = torch.ones((1, input_ids.shape[1] + 1), device=device, dtype=torch.long)
        ref_dec = model(
            input_ids=next_id,
            attention_mask=attn_dec,
            past_key_values=past,
            use_cache=False,
        ).logits

        # Torch RF decode (orchestration baseline)
        cache, seq_len = _prefill_kv_cache(model, input_ids, attention_mask)
        torch_dec = qwen2_decode_step_with_rf_mlp(
            model,
            next_id=next_id,
            attn_dec=attn_dec,
            cache=cache,
            seq_len=seq_len,
            device=device,
            max_rf_mlp_layers=max_rf_mlp_layers,
            mlp_backend=prefill_backend,
        )
        decode_parity_ok = bool(
            torch.allclose(ref_dec, torch_dec, rtol=1e-4, atol=1e-3)
        )

        # YiRage optimized decode (gate_up core + superopt down)
        if yirage_core_used:
            cache_y, seq_len_y = _prefill_kv_cache(model, input_ids, attention_mask)
            yirage_dec = qwen2_decode_step_with_rf_mlp(
                model,
                next_id=next_id,
                attn_dec=attn_dec,
                cache=cache_y,
                seq_len=seq_len_y,
                device=device,
                max_rf_mlp_layers=max_rf_mlp_layers,
                mlp_backend=decode_backend,
            )
            yirage_decode_parity_ok = bool(
                torch.allclose(ref_dec, yirage_dec, rtol=1e-4, atol=1e-3)
            )

    t2 = time.perf_counter()
    generate_token_match_ok = True
    yirage_generated_text = generated_text
    if yirage_core_used and max_new_tokens > 0:
        yirage_ids = greedy_decode_with_rf_mlp(
            model,
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            max_rf_mlp_layers=max_rf_mlp_layers,
            mlp_backend=decode_backend,
        )
        yirage_generated_text = tokenizer.decode(yirage_ids[0], skip_special_tokens=True)
        generate_token_match_ok = bool(torch.equal(gen_ids, yirage_ids))
    yirage_generate_ms = (time.perf_counter() - t2) * 1000.0

    superopt_total = _total_superopt_elapsed(max_rf_mlp_layers, decode_backend)
    plugin = (
        "HfQwen2MlpRfHook+yirage_cpu+transformers"
        if yirage_core_used
        else "HfQwen2MlpRfHook+transformers"
    )

    if yirage_core_used:
        parity_ok = (
            prefill_parity_ok
            and decode_parity_ok
            and yirage_decode_parity_ok
            and generate_token_match_ok
        )
    else:
        parity_ok = prefill_parity_ok and decode_parity_ok
        yirage_decode_parity_ok = True
        generate_token_match_ok = True

    return HfQwen05bCpuE2EReport(
        model_id=model_id,
        parity_ok=parity_ok,
        prefill_parity_ok=prefill_parity_ok,
        decode_parity_ok=decode_parity_ok,
        yirage_decode_parity_ok=yirage_decode_parity_ok,
        generate_token_match_ok=generate_token_match_ok,
        used_rf_mlp_layers=int(max_rf_mlp_layers),
        num_layers=int(cfg.num_hidden_layers),
        hidden_size=int(cfg.hidden_size),
        prompt=prompt,
        generated_text=generated_text,
        yirage_generated_text=yirage_generated_text,
        device=device,
        prefill_ms=prefill_ms,
        generate_ms=generate_ms,
        yirage_generate_ms=yirage_generate_ms,
        mlp_backend=decode_backend,
        decode_mlp_backend=decode_backend,
        yirage_core_used=yirage_core_used,
        superopt_elapsed_s_total=superopt_total,
        plugin=plugin,
        serving_search_tier=resolve_serving_search_tier(),
    )
