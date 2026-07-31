# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S35: Engine-native multistep MLP bench (G7 chains C/D multistep extension).

Runs N decode-step iterations on engine-cooperative paths:
- Torch surrogate vLLM-style hybrid + paged KV meta evolution
- Torch surrogate SGLang ForwardBatch meta evolution
- Optional native ``vllm`` / ``sglang`` single-layer MLP hook loops when installed

Cert gate: torch chains must pass ``parity_ok``; native tiers are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .hybrid_model import HybridModelOverride
from .sglang_e2e import (
    SglangForwardBatchSpec,
    _expected_partial_radix_mlp,
    _reference_hybrid_sglang_forward,
    _radix_flags,
    _sglang_native_mlp_forward,
    build_minimal_sglang_qwen2_decoder_layer,
)
from .sglang_plugin import build_sglang_qwen2_mlp_rf_hook, is_sglang_available
from .torch_engine import TorchEngineModel
from .torch_exec import require_torch
from .vllm_e2e import (
    _vllm_native_mlp_forward,
    build_minimal_vllm_qwen2_decoder_layer,
)
from .vllm_plugin import build_vllm_qwen2_mlp_rf_hook, is_vllm_available
from .vllm_runtime import vllm_test_config_context


@dataclass(frozen=True)
class EngineMultistepChainRow:
    chain_id: str
    functional_chain: str
    engine: str
    decode_steps: int
    step_parity_ok: List[bool]
    parity_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "functional_chain": self.functional_chain,
            "engine": self.engine,
            "decode_steps": self.decode_steps,
            "step_parity_ok": list(self.step_parity_ok),
            "parity_ok": self.parity_ok,
        }


@dataclass
class EngineNativeMultistepReport:
    """Multistep engine MLP parity across vLLM- and SGLang-style paths."""

    version: str
    parity_ok: bool
    decode_steps: int
    chains: List[EngineMultistepChainRow] = field(default_factory=list)
    vllm_native_available: bool = False
    sglang_native_available: bool = False
    native_parity_ok: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serving_engine_native_multistep_bench": True,
            "version": self.version,
            "parity_ok": self.parity_ok,
            "decode_steps": self.decode_steps,
            "functional_chain": "chain_c_d_engine_multistep",
            "vllm_native_available": self.vllm_native_available,
            "sglang_native_available": self.sglang_native_available,
            "native_parity_ok": self.native_parity_ok,
            "chains": [c.to_dict() for c in self.chains],
        }


def validate_serving_engine_native_multistep_bench(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not payload.get("serving_engine_native_multistep_bench"):
        errors.append("missing serving_engine_native_multistep_bench=true marker")
    if payload.get("parity_ok") is not True:
        errors.append("parity_ok must be true (torch multistep chains required)")
    steps = payload.get("decode_steps")
    if not isinstance(steps, int) or steps < 1:
        errors.append("decode_steps must be >= 1")
    chains = payload.get("chains")
    if not isinstance(chains, list) or len(chains) < 2:
        errors.append("chains must include vLLM and SGLang torch multistep paths")
    else:
        ids = {c.get("chain_id") for c in chains if isinstance(c, dict)}
        if "chain_c_vllm_torch_multistep" not in ids:
            errors.append("missing chain_c_vllm_torch_multistep")
        if "chain_d_sglang_torch_multistep" not in ids:
            errors.append("missing chain_d_sglang_torch_multistep")
    return errors


def _run_torch_vllm_multistep(
    *,
    decode_steps: int,
    num_layers: int,
    max_rf_mlp_layers: int,
    hidden_size: int,
    intermediate_size: int,
    batch: int,
    seed: int,
) -> EngineMultistepChainRow:
    require_torch()
    import torch

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(model, max_rf_mlp_layers=max_rf_mlp_layers)
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=model.device)
    base_seq = [16, 18, 20][:batch]
    block_tables = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]][:batch]
    step_ok: List[bool] = []

    for step in range(decode_steps):
        meta: Dict[str, Any] = {
            "block_tables": block_tables,
            "seq_lens": [s + step for s in base_seq],
            "page_size": 16,
            "extras": {"total_sms": 108, "reserved_aux_sms": 8},
        }
        with torch.no_grad():
            ref = model.forward_engine_full(x)
            got = hybrid.forward(x, rf_meta=meta)
        step_ok.append(
            bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
            and list(got.rf_layer_ids) == list(range(int(max_rf_mlp_layers)))
        )
        x = got.hidden.detach()

    return EngineMultistepChainRow(
        chain_id="chain_c_vllm_torch_multistep",
        functional_chain="chain_c_vllm_plugin",
        engine="torch_surrogate",
        decode_steps=decode_steps,
        step_parity_ok=step_ok,
        parity_ok=all(step_ok),
    )


def _run_torch_sglang_multistep(
    *,
    decode_steps: int,
    num_layers: int,
    max_rf_mlp_layers: int,
    hidden_size: int,
    intermediate_size: int,
    batch: int,
    seed: int,
) -> EngineMultistepChainRow:
    require_torch()
    import torch

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(model, max_rf_mlp_layers=max_rf_mlp_layers)
    x = torch.randn(batch, hidden_size, dtype=torch.float32, device=model.device)
    base_extend = [0, 3, 0, 0][:batch]
    base_seq = [16, 18, 20][:batch]
    block_tables = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]][:batch]
    expected_rf = list(range(int(max_rf_mlp_layers)))
    step_ok: List[bool] = []

    for step in range(decode_steps):
        extend = [
            (1 if step > 0 and e == 0 else e) for e in base_extend
        ]
        if step > 0:
            extend = [1 if e == 0 and i < 2 else e for i, e in enumerate(base_extend)]
        forward_batch = SglangForwardBatchSpec(
            extend_seq_lens=extend,
            seq_lens=[s + step for s in base_seq],
            block_tables=block_tables,
            page_size=16,
        )
        meta = forward_batch.as_meta()
        meta.setdefault("extras", {})
        extras = dict(meta["extras"])
        extras.setdefault("total_sms", 108)
        extras.setdefault("reserved_aux_sms", 8)
        meta["extras"] = extras
        all_hit, _partial = _radix_flags(forward_batch.extend_seq_lens)

        with torch.no_grad():
            ref = _reference_hybrid_sglang_forward(
                model, expected_rf, x, forward_batch.extend_seq_lens
            )
            got = hybrid.forward(x, rf_meta=meta)

        if all_hit:
            ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
            ok = ok and got.rf_layer_ids == []
        else:
            ok = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6))
            ok = ok and list(got.rf_layer_ids) == expected_rf
        step_ok.append(ok)
        x = got.hidden.detach()

    return EngineMultistepChainRow(
        chain_id="chain_d_sglang_torch_multistep",
        functional_chain="chain_d_sglang_forward_batch",
        engine="torch_surrogate",
        decode_steps=decode_steps,
        step_parity_ok=step_ok,
        parity_ok=all(step_ok),
    )


def _run_vllm_native_multistep(
    *,
    decode_steps: int,
    hidden_size: int,
    intermediate_size: int,
    batch: int,
    layer_id: int,
) -> Optional[EngineMultistepChainRow]:
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
    rf_meta: Dict[str, Any] = {"enabled": {cap_name}}
    step_ok: List[bool] = []

    for _step in range(decode_steps):
        with vllm_test_config_context():
            with torch.no_grad():
                ref = _vllm_native_mlp_forward(layer, h)
                got = hook.forward_mlp(h, rf_meta=rf_meta)
        step_ok.append(bool(torch.allclose(got.hidden, ref, rtol=1e-4, atol=1e-4)))
        h = ref.detach()

    return EngineMultistepChainRow(
        chain_id="chain_c_vllm_native_multistep",
        functional_chain="chain_c_vllm_plugin",
        engine="vllm_native",
        decode_steps=decode_steps,
        step_parity_ok=step_ok,
        parity_ok=all(step_ok),
    )


def _run_sglang_native_multistep(
    *,
    decode_steps: int,
    hidden_size: int,
    intermediate_size: int,
    batch: int,
    layer_id: int,
) -> Optional[EngineMultistepChainRow]:
    if not is_sglang_available():
        return None
    require_torch()
    import torch

    layer = build_minimal_sglang_qwen2_decoder_layer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layer_id=layer_id,
    )
    hook = build_sglang_qwen2_mlp_rf_hook(layer, layer_id=layer_id)
    device = str(next(layer.parameters()).device)
    h = torch.randn(batch, hidden_size, dtype=torch.float32, device=device)
    base_extend = [0, 2][:batch]
    base_seq = [16] * batch
    step_ok: List[bool] = []

    for step in range(decode_steps):
        forward_batch = SglangForwardBatchSpec(
            extend_seq_lens=[1 if step > 0 else e for e in base_extend],
            seq_lens=[s + step for s in base_seq],
        )
        all_hit, partial = _radix_flags(forward_batch.extend_seq_lens)
        with torch.no_grad():
            if all_hit:
                ref = h
            elif partial:
                adapter = hook.override.layer
                ref = _expected_partial_radix_mlp(adapter, h, forward_batch.extend_seq_lens)
            else:
                ref = _sglang_native_mlp_forward(layer, h)
            got = hook.forward_mlp(h, forward_batch=forward_batch)
        step_ok.append(bool(torch.allclose(got.hidden, ref, rtol=1e-4, atol=1e-4)))
        h = ref.detach()

    return EngineMultistepChainRow(
        chain_id="chain_d_sglang_native_multistep",
        functional_chain="chain_d_sglang_forward_batch",
        engine="sglang_native",
        decode_steps=decode_steps,
        step_parity_ok=step_ok,
        parity_ok=all(step_ok),
    )


def run_engine_native_multistep_bench(
    *,
    decode_steps: int = 4,
    num_layers: int = 2,
    max_rf_mlp_layers: int = 1,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    batch: int = 3,
    quick: bool = False,
    try_native: bool = True,
    version: str = "s35",
) -> EngineNativeMultistepReport:
    """Run multistep engine MLP parity (torch gate + optional native tiers)."""
    require_torch()

    if quick:
        decode_steps = min(int(decode_steps), 3)
        num_layers = min(num_layers, 2)
        max_rf_mlp_layers = min(max_rf_mlp_layers, 1)
        batch = min(batch, 3)
        hidden_size = min(hidden_size, 16)
        intermediate_size = min(intermediate_size, 32)

    chains: List[EngineMultistepChainRow] = [
        _run_torch_vllm_multistep(
            decode_steps=decode_steps,
            num_layers=num_layers,
            max_rf_mlp_layers=max_rf_mlp_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            batch=batch,
            seed=0,
        ),
        _run_torch_sglang_multistep(
            decode_steps=decode_steps,
            num_layers=num_layers,
            max_rf_mlp_layers=max_rf_mlp_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            batch=batch,
            seed=1,
        ),
    ]

    vllm_avail = is_vllm_available()
    sglang_avail = is_sglang_available()
    native_rows: List[EngineMultistepChainRow] = []

    if try_native and vllm_avail:
        row = _run_vllm_native_multistep(
            decode_steps=decode_steps,
            hidden_size=max(hidden_size, 64),
            intermediate_size=max(intermediate_size, 128),
            batch=min(batch, 2),
            layer_id=0,
        )
        if row is not None:
            native_rows.append(row)

    if try_native and sglang_avail:
        row = _run_sglang_native_multistep(
            decode_steps=decode_steps,
            hidden_size=max(hidden_size, 64),
            intermediate_size=max(intermediate_size, 128),
            batch=min(batch, 2),
            layer_id=0,
        )
        if row is not None:
            native_rows.append(row)

    chains.extend(native_rows)
    torch_chains = [c for c in chains if c.engine == "torch_surrogate"]
    parity_ok = all(c.parity_ok for c in torch_chains)
    native_parity_ok = (
        all(c.parity_ok for c in native_rows) if native_rows else None
    )

    return EngineNativeMultistepReport(
        version=version,
        parity_ok=parity_ok,
        decode_steps=decode_steps,
        chains=chains,
        vllm_native_available=vllm_avail,
        sglang_native_available=sglang_avail,
        native_parity_ok=native_parity_ok,
    )


def run_serving_engine_native_multistep_archive(
    *,
    decode_steps: int = 4,
    quick: bool = True,
    try_native: bool = True,
    version: str = "s35",
) -> Dict[str, Any]:
    """Run bench and return validated archive payload."""
    payload = run_engine_native_multistep_bench(
        decode_steps=decode_steps,
        quick=quick,
        try_native=try_native,
        version=version,
    ).to_dict()
    errors = validate_serving_engine_native_multistep_bench(payload)
    if errors:
        raise RuntimeError(f"engine native multistep archive validation failed: {errors}")
    return payload
