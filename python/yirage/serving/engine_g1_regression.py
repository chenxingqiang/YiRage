# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S32: vLLM + SGLang G1 engine-cooperative regression (G7 chains C/D).

Runs torch-surrogate full paths that mirror S11/S12 e2e:
- Chain C: engine Attention → RF MLP hook (vLLM-style hybrid + paged KV meta)
- Chain D: ForwardBatch meta → RF MLP hook (SGLang-style hybrid + Radix/KV)

``parity_ok`` requires both chains; optional real ``vllm``/``sglang`` tiers are
reported separately when installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .sglang_e2e import SglangForwardBatchSpec, run_torch_sglang_hybrid_full_e2e
from .sglang_plugin import is_sglang_available
from .torch_exec import require_torch
from .vllm_e2e import run_torch_vllm_hybrid_full_e2e
from .vllm_plugin import is_vllm_available


@dataclass(frozen=True)
class EngineG1ChainRow:
    chain_id: str
    functional_chain: str
    parity_ok: bool
    plugin: str
    engine: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "functional_chain": self.functional_chain,
            "parity_ok": self.parity_ok,
            "plugin": self.plugin,
            "engine": self.engine,
        }


@dataclass
class EngineG1RegressionReport:
    """Unified G1 regression across vLLM- and SGLang-style engine paths."""

    version: str
    parity_ok: bool
    device: str
    chains: List[EngineG1ChainRow] = field(default_factory=list)
    vllm_hybrid: Optional[Dict[str, Any]] = None
    sglang_hybrid: Optional[Dict[str, Any]] = None
    vllm_native_available: bool = False
    sglang_native_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serving_engine_g1_regression": True,
            "version": self.version,
            "parity_ok": self.parity_ok,
            "device": self.device,
            "vllm_native_available": self.vllm_native_available,
            "sglang_native_available": self.sglang_native_available,
            "chains": [c.to_dict() for c in self.chains],
            "vllm_hybrid": self.vllm_hybrid,
            "sglang_hybrid": self.sglang_hybrid,
        }


def validate_serving_engine_g1_regression(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not payload.get("serving_engine_g1_regression"):
        errors.append("missing serving_engine_g1_regression=true marker")
    if payload.get("parity_ok") is not True:
        errors.append("parity_ok must be true for G1 regression archive")
    chains = payload.get("chains")
    if not isinstance(chains, list) or len(chains) < 2:
        errors.append("chains must include vLLM and SGLang torch paths")
    else:
        ids = {c.get("chain_id") for c in chains if isinstance(c, dict)}
        if "chain_c_vllm_torch" not in ids:
            errors.append("missing chain_c_vllm_torch")
        if "chain_d_sglang_torch" not in ids:
            errors.append("missing chain_d_sglang_torch")
    return errors


def run_engine_g1_regression(
    *,
    num_layers: int = 3,
    max_rf_mlp_layers: int = 2,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    batch: int = 4,
    quick: bool = False,
    version: str = "s32",
) -> EngineG1RegressionReport:
    """Run G7 chain C + D torch regression (G1 engine-cooperative gate)."""
    require_torch()

    if quick:
        num_layers = min(num_layers, 2)
        max_rf_mlp_layers = min(max_rf_mlp_layers, 1)
        batch = min(batch, 3)
        hidden_size = min(hidden_size, 16)
        intermediate_size = min(intermediate_size, 32)

    vllm_report = run_torch_vllm_hybrid_full_e2e(
        num_layers=num_layers,
        max_rf_mlp_layers=max_rf_mlp_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=False,
        rf_meta={
            "block_tables": [[1, 2, -1], [3, 4, -1], [5, 6, -1], [7, 8, -1]][:batch],
            "seq_lens": [32, 18, 24, 16][:batch],
            "page_size": 16,
        },
    )

    sglang_batch = SglangForwardBatchSpec(
        extend_seq_lens=[0, 3, 0, 0][:batch],
        seq_lens=[32, 18, 24, 16][:batch],
        block_tables=[[1, 2, -1], [3, 4, -1], [5, 6, -1], [7, 8, -1]][:batch],
        page_size=16,
    )
    sglang_report = run_torch_sglang_hybrid_full_e2e(
        forward_batch=sglang_batch,
        num_layers=num_layers,
        max_rf_mlp_layers=max_rf_mlp_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=False,
    )

    chains = [
        EngineG1ChainRow(
            chain_id="chain_c_vllm_torch",
            functional_chain="chain_c_vllm_plugin",
            parity_ok=bool(vllm_report.parity_ok),
            plugin=vllm_report.plugin,
            engine="torch_surrogate",
        ),
        EngineG1ChainRow(
            chain_id="chain_d_sglang_torch",
            functional_chain="chain_d_sglang_forward_batch",
            parity_ok=bool(sglang_report.parity_ok),
            plugin=sglang_report.plugin,
            engine="torch_surrogate",
        ),
    ]
    parity_ok = all(c.parity_ok for c in chains)

    return EngineG1RegressionReport(
        version=version,
        parity_ok=parity_ok,
        device=vllm_report.device,
        chains=chains,
        vllm_hybrid=vllm_report.to_dict(),
        sglang_hybrid=sglang_report.to_dict(),
        vllm_native_available=is_vllm_available(),
        sglang_native_available=is_sglang_available(),
    )
