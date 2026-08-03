# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S37/S40: vLLM PagedAttention multistep decode bench (G7 chain C paged extension).

N decode steps with evolving ``VllmPagedKvBatchSpec`` (seq_lens grow per step).
Torch path: full-layer hybrid vs engine with greedy token match (cert gate).
Optional native ``vllm`` single-layer MLP + paged meta loop when installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .hybrid_model import HybridModelOverride
from .torch_engine import TorchEngineModel
from .torch_exec import require_torch
from .vllm_e2e import (
    _vllm_chain_native_decoder_layers,
    _vllm_chain_native_decoder_layers_rf_mlp,
    _vllm_native_mlp_forward,
    build_minimal_vllm_qwen2_decoder_layer,
)
from .vllm_paged_e2e import VllmPagedKvBatchSpec, _layer_step_meta_has_paged_kv, _paged_kv_present
from .vllm_plugin import build_vllm_qwen2_mlp_rf_hook, is_vllm_available
from .vllm_runtime import vllm_test_config_context


@dataclass
class VllmPagedMultistepReport:
    """Multistep paged-KV full-layer hybrid parity + greedy token match."""

    version: str
    parity_ok: bool
    token_match_ok: bool
    decode_steps: int
    paged_kv_bridged: bool
    functional_chain: str = "chain_c_vllm_paged_multistep"
    device: str = "cpu"
    num_layers: int = 0
    batch: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    step_parity_ok: List[bool] = field(default_factory=list)
    step_token_match_ok: List[bool] = field(default_factory=list)
    engine_token_ids: List[List[int]] = field(default_factory=list)
    hybrid_token_ids: List[List[int]] = field(default_factory=list)
    vllm_native_available: bool = False
    native_parity_ok: Optional[bool] = None
    native_step_parity_ok: List[bool] = field(default_factory=list)
    native_full_layer_parity_ok: Optional[bool] = None
    native_full_layer_step_parity_ok: List[bool] = field(default_factory=list)
    native_decoder_parity_ok: Optional[bool] = None
    native_decoder_token_match_ok: Optional[bool] = None
    native_decoder_step_parity_ok: List[bool] = field(default_factory=list)
    native_decoder_step_token_match_ok: List[bool] = field(default_factory=list)
    native_decoder_ref_token_ids: List[List[int]] = field(default_factory=list)
    native_decoder_rf_token_ids: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serving_vllm_paged_multistep_bench": True,
            "version": self.version,
            "parity_ok": self.parity_ok,
            "token_match_ok": self.token_match_ok,
            "decode_steps": self.decode_steps,
            "paged_kv_bridged": self.paged_kv_bridged,
            "functional_chain": self.functional_chain,
            "device": self.device,
            "num_layers": self.num_layers,
            "batch": self.batch,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "step_parity_ok": list(self.step_parity_ok),
            "step_token_match_ok": list(self.step_token_match_ok),
            "engine_token_ids": [list(row) for row in self.engine_token_ids],
            "hybrid_token_ids": [list(row) for row in self.hybrid_token_ids],
            "vllm_native_available": self.vllm_native_available,
            "native_parity_ok": self.native_parity_ok,
            "native_step_parity_ok": list(self.native_step_parity_ok),
            "native_full_layer_parity_ok": self.native_full_layer_parity_ok,
            "native_full_layer_step_parity_ok": list(self.native_full_layer_step_parity_ok),
            "native_decoder_parity_ok": self.native_decoder_parity_ok,
            "native_decoder_token_match_ok": self.native_decoder_token_match_ok,
            "native_decoder_step_parity_ok": list(self.native_decoder_step_parity_ok),
            "native_decoder_step_token_match_ok": list(self.native_decoder_step_token_match_ok),
            "native_decoder_ref_token_ids": [list(row) for row in self.native_decoder_ref_token_ids],
            "native_decoder_rf_token_ids": [list(row) for row in self.native_decoder_rf_token_ids],
        }


def validate_serving_vllm_paged_multistep_bench(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not payload.get("serving_vllm_paged_multistep_bench"):
        errors.append("missing serving_vllm_paged_multistep_bench=true marker")
    if payload.get("parity_ok") is not True:
        errors.append("parity_ok must be true")
    if payload.get("token_match_ok") is not True:
        errors.append("token_match_ok must be true")
    if payload.get("paged_kv_bridged") is not True:
        errors.append("paged_kv_bridged must be true")
    steps = payload.get("decode_steps")
    if not isinstance(steps, int) or steps < 1:
        errors.append("decode_steps must be >= 1")
    eng = payload.get("engine_token_ids")
    hyb = payload.get("hybrid_token_ids")
    if not isinstance(eng, list) or not isinstance(hyb, list) or len(eng) != len(hyb):
        errors.append("engine_token_ids and hybrid_token_ids must be same-length lists")
    return errors


def _default_block_tables(batch: int) -> List[List[int]]:
    return [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]][:batch]


def _paged_batch_for_step(
    *,
    step: int,
    batch: int,
    base_seq: Sequence[int],
    block_tables: Sequence[Sequence[int]],
    page_size: int,
) -> VllmPagedKvBatchSpec:
    return VllmPagedKvBatchSpec(
        block_tables=block_tables,
        seq_lens=[int(s) + step for s in base_seq],
        page_size=page_size,
    )


def _run_vllm_native_paged_multistep(
    *,
    decode_steps: int,
    hidden_size: int,
    intermediate_size: int,
    batch: int,
    base_seq: Sequence[int],
    block_tables: Sequence[Sequence[int]],
    page_size: int,
    layer_id: int = 0,
) -> Optional[List[bool]]:
    """Native vLLM MLP hook loop with evolving paged KV meta (optional tier)."""
    if not is_vllm_available():
        return None
    require_torch()
    import torch

    layer = build_minimal_vllm_qwen2_decoder_layer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layer_id=layer_id,
    )
    hook = build_vllm_qwen2_mlp_rf_hook(layer, layer_id=layer_id)
    device = str(next(layer.parameters()).device)
    h = torch.randn(batch, hidden_size, dtype=torch.float32, device=device)
    cap_name = hook.override.capsule_name
    step_ok: List[bool] = []

    for step in range(decode_steps):
        paged_batch = _paged_batch_for_step(
            step=step,
            batch=batch,
            base_seq=base_seq,
            block_tables=block_tables,
            page_size=page_size,
        )
        meta = dict(paged_batch.as_rf_meta())
        meta["enabled"] = {cap_name}
        with vllm_test_config_context():
            with torch.no_grad():
                ref = _vllm_native_mlp_forward(layer, h)
                got = hook.forward_mlp(h, rf_meta=meta)
        ok = bool(torch.allclose(got.hidden, ref, rtol=1e-4, atol=1e-4))
        ok = ok and _paged_kv_present(meta)
        step_ok.append(ok)
        h = ref.detach()

    return step_ok


def _run_vllm_native_paged_multistep_full_layer(
    *,
    decode_steps: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    batch: int,
    base_seq: Sequence[int],
    block_tables: Sequence[Sequence[int]],
    page_size: int,
) -> Optional[List[bool]]:
    """Native vLLM MLP hooks on every layer with evolving paged KV meta (S43)."""
    if not is_vllm_available() or num_layers < 1:
        return None
    require_torch()
    import torch

    layers = [
        build_minimal_vllm_qwen2_decoder_layer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
        )
        for layer_id in range(num_layers)
    ]
    hooks = [
        build_vllm_qwen2_mlp_rf_hook(layer, layer_id=layer_id)
        for layer_id, layer in enumerate(layers)
    ]
    device = str(next(layers[0].parameters()).device)
    step_ok: List[bool] = []

    for step in range(decode_steps):
        paged_batch = _paged_batch_for_step(
            step=step,
            batch=batch,
            base_seq=base_seq,
            block_tables=block_tables,
            page_size=page_size,
        )
        meta_base = dict(paged_batch.as_rf_meta())
        h = torch.randn(batch, hidden_size, dtype=torch.float32, device=device)
        layers_ok = True
        with vllm_test_config_context():
            with torch.no_grad():
                for layer, hook in zip(layers, hooks):
                    cap_name = hook.override.capsule_name
                    meta = dict(meta_base)
                    meta["enabled"] = {cap_name}
                    ref = _vllm_native_mlp_forward(layer, h)
                    got = hook.forward_mlp(h, rf_meta=meta)
                    ok = bool(torch.allclose(got.hidden, ref, rtol=1e-4, atol=1e-4))
                    ok = ok and _paged_kv_present(meta)
                    if not ok:
                        layers_ok = False
                        break
                    h = ref.detach()
        step_ok.append(layers_ok)
    return step_ok


def _run_vllm_native_decoder_paged_multistep(
    *,
    decode_steps: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    batch: int,
    base_seq: Sequence[int],
    block_tables: Sequence[Sequence[int]],
    page_size: int,
    vocab_size: int,
    seed: int,
) -> Optional[
    tuple[
        List[bool],
        List[bool],
        List[List[int]],
        List[List[int]],
    ]
]:
    """Native vLLM full decoder stack vs RF-on-MLP hybrid with paged KV (S45)."""
    if not is_vllm_available() or num_layers < 1:
        return None
    require_torch()
    import torch

    layers = [
        build_minimal_vllm_qwen2_decoder_layer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
        )
        for layer_id in range(num_layers)
    ]
    hooks = [
        build_vllm_qwen2_mlp_rf_hook(layer, layer_id=layer_id)
        for layer_id, layer in enumerate(layers)
    ]
    device = str(next(layers[0].parameters()).device)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 99)
    lm_weight = (
        torch.randn(vocab_size, hidden_size, generator=gen, device="cpu", dtype=torch.float32)
        * 0.02
    ).to(device)

    h = torch.randn(batch, hidden_size, dtype=torch.float32, device=device)
    step_parity: List[bool] = []
    step_token: List[bool] = []
    ref_tokens: List[List[int]] = []
    rf_tokens: List[List[int]] = []

    for step in range(decode_steps):
        paged_batch = _paged_batch_for_step(
            step=step,
            batch=batch,
            base_seq=base_seq,
            block_tables=block_tables,
            page_size=page_size,
        )
        meta_base = dict(paged_batch.as_rf_meta())
        positions = torch.tensor(
            [int(s) - 1 for s in paged_batch.seq_lens],
            dtype=torch.long,
            device=device,
        )
        with vllm_test_config_context():
            with torch.no_grad():
                ref_hidden = _vllm_chain_native_decoder_layers(
                    layers,
                    positions=positions,
                    hidden_states=h,
                )
                rf_hidden = _vllm_chain_native_decoder_layers_rf_mlp(
                    layers,
                    hooks,
                    positions=positions,
                    hidden_states=h,
                    rf_meta_base=meta_base,
                )
        ok = bool(torch.allclose(ref_hidden, rf_hidden, rtol=1e-4, atol=1e-4))
        ok = ok and _paged_kv_present(meta_base)
        step_parity.append(ok)

        ref_logits = ref_hidden @ lm_weight.T
        rf_logits = rf_hidden @ lm_weight.T
        ref_tok = ref_logits.argmax(dim=-1)
        rf_tok = rf_logits.argmax(dim=-1)
        step_token.append(bool(torch.equal(ref_tok, rf_tok)))
        ref_tokens.append(ref_tok.tolist())
        rf_tokens.append(rf_tok.tolist())
        h = ref_hidden.detach()

    return step_parity, step_token, ref_tokens, rf_tokens


def run_vllm_paged_multistep_bench(
    *,
    decode_steps: int = 4,
    num_layers: int = 3,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    batch: int = 3,
    vocab_size: int = 128,
    page_size: int = 16,
    seed: int = 0,
    quick: bool = False,
    try_native: bool = True,
    try_native_full_layer: bool = True,
    try_native_decoder: bool = True,
    version: str = "s45",
) -> VllmPagedMultistepReport:
    """Run multistep paged-KV hybrid vs engine with per-step token parity."""
    require_torch()
    import torch

    if quick:
        decode_steps = min(int(decode_steps), 3)
        num_layers = min(num_layers, 2)
        batch = min(batch, 3)
        hidden_size = min(hidden_size, 16)
        intermediate_size = min(intermediate_size, 32)
        vocab_size = min(vocab_size, 64)

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(model, max_rf_mlp_layers=num_layers)
    device = model.device

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 99)
    lm_weight = (
        torch.randn(vocab_size, hidden_size, generator=gen, device="cpu", dtype=torch.float32)
        * 0.02
    ).to(device)

    h = torch.randn(batch, hidden_size, dtype=torch.float32, device=device)
    base_seq = [16, 18, 20][:batch]
    block_tables = _default_block_tables(batch)
    expected_rf = list(range(num_layers))

    step_parity: List[bool] = []
    step_token: List[bool] = []
    engine_tokens: List[List[int]] = []
    hybrid_tokens: List[List[int]] = []
    any_paged = False

    for step in range(decode_steps):
        paged_batch = _paged_batch_for_step(
            step=step,
            batch=batch,
            base_seq=base_seq,
            block_tables=block_tables,
            page_size=page_size,
        )
        meta = paged_batch.as_rf_meta()
        any_paged = any_paged or _paged_kv_present(meta)

        with torch.no_grad():
            ref = model.forward_engine_full(h)
            got = hybrid.forward(h, rf_meta=meta)

        ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
        ok = ok and list(got.rf_layer_ids) == expected_rf
        ok = ok and _layer_step_meta_has_paged_kv(got.layer_results)
        step_parity.append(ok)

        ref_logits = ref @ lm_weight.T
        hyb_logits = got.hidden @ lm_weight.T
        ref_tok = ref_logits.argmax(dim=-1)
        hyb_tok = hyb_logits.argmax(dim=-1)
        tok_ok = bool(torch.equal(ref_tok, hyb_tok))
        step_token.append(tok_ok)

        engine_tokens.append(ref_tok.tolist())
        hybrid_tokens.append(hyb_tok.tolist())
        h = ref.detach()

    parity_ok = all(step_parity)
    token_match_ok = all(step_token)

    vllm_avail = is_vllm_available()
    native_steps: Optional[List[bool]] = None
    if try_native and vllm_avail:
        native_steps = _run_vllm_native_paged_multistep(
            decode_steps=decode_steps,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            batch=batch,
            base_seq=base_seq,
            block_tables=block_tables,
            page_size=page_size,
        )
    native_parity_ok = all(native_steps) if native_steps else None

    native_full_steps: Optional[List[bool]] = None
    if try_native_full_layer and vllm_avail and num_layers > 0:
        native_full_steps = _run_vllm_native_paged_multistep_full_layer(
            decode_steps=decode_steps,
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            batch=batch,
            base_seq=base_seq,
            block_tables=block_tables,
            page_size=page_size,
        )
    native_full_layer_parity_ok = all(native_full_steps) if native_full_steps else None

    native_decoder_steps: Optional[List[bool]] = None
    native_decoder_token_steps: Optional[List[bool]] = None
    native_decoder_ref_tokens: List[List[int]] = []
    native_decoder_rf_tokens: List[List[int]] = []
    if try_native_decoder and vllm_avail and num_layers > 0:
        decoder_out = _run_vllm_native_decoder_paged_multistep(
            decode_steps=decode_steps,
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            batch=batch,
            base_seq=base_seq,
            block_tables=block_tables,
            page_size=page_size,
            vocab_size=vocab_size,
            seed=seed,
        )
        if decoder_out is not None:
            native_decoder_steps, native_decoder_token_steps, native_decoder_ref_tokens, native_decoder_rf_tokens = (
                decoder_out
            )
    native_decoder_parity_ok = all(native_decoder_steps) if native_decoder_steps else None
    native_decoder_token_match_ok = (
        all(native_decoder_token_steps) if native_decoder_token_steps else None
    )

    return VllmPagedMultistepReport(
        version=version,
        parity_ok=parity_ok,
        token_match_ok=token_match_ok,
        decode_steps=decode_steps,
        paged_kv_bridged=any_paged and all(step_parity),
        device=str(device),
        num_layers=int(num_layers),
        batch=int(batch),
        hidden_size=int(hidden_size),
        vocab_size=int(vocab_size),
        step_parity_ok=step_parity,
        step_token_match_ok=step_token,
        engine_token_ids=engine_tokens,
        hybrid_token_ids=hybrid_tokens,
        vllm_native_available=vllm_avail,
        native_parity_ok=native_parity_ok,
        native_step_parity_ok=list(native_steps or []),
        native_full_layer_parity_ok=native_full_layer_parity_ok,
        native_full_layer_step_parity_ok=list(native_full_steps or []),
        native_decoder_parity_ok=native_decoder_parity_ok,
        native_decoder_token_match_ok=native_decoder_token_match_ok,
        native_decoder_step_parity_ok=list(native_decoder_steps or []),
        native_decoder_step_token_match_ok=list(native_decoder_token_steps or []),
        native_decoder_ref_token_ids=native_decoder_ref_tokens,
        native_decoder_rf_token_ids=native_decoder_rf_tokens,
    )


def run_serving_vllm_paged_multistep_archive(
    *,
    decode_steps: int = 4,
    quick: bool = True,
    try_native: bool = True,
    try_native_full_layer: bool = True,
    try_native_decoder: bool = True,
    version: str = "s45",
) -> Dict[str, Any]:
    payload = run_vllm_paged_multistep_bench(
        decode_steps=decode_steps,
        quick=quick,
        try_native=try_native,
        try_native_full_layer=try_native_full_layer,
        try_native_decoder=try_native_decoder,
        version=version,
    ).to_dict()
    errors = validate_serving_vllm_paged_multistep_bench(payload)
    if errors:
        raise RuntimeError(f"vllm paged multistep archive validation failed: {errors}")
    return payload
