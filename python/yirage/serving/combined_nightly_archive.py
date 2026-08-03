# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S34/S35/S38: Combined Serving nightly archive — decode + G1 + multistep + engine multistep + paged multistep.

Bundles S31 decode bench, S32 engine G1 regression, S33 HF multistep generation,
S35 engine-native multistep, and S37 vLLM paged multistep into one validated JSON
artifact for nightly CI.
``parity_ok`` requires all sub-reports (torch gates).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .decode_bench_archive import (
    run_serving_qwen_decode_bench_archive,
    validate_serving_qwen_decode_bench_archive,
)
from .engine_g1_regression import (
    run_engine_g1_regression,
    validate_serving_engine_g1_regression,
)
from .engine_native_multistep_bench import (
    run_serving_engine_native_multistep_archive,
    validate_serving_engine_native_multistep_bench,
)
from .vllm_paged_multistep_bench import (
    run_serving_vllm_paged_multistep_archive,
    validate_serving_vllm_paged_multistep_bench,
)
from .hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL
from .qwen_multistep_generation_bench import (
    run_serving_qwen_multistep_generation_archive,
    validate_serving_qwen_multistep_generation_bench,
)


@dataclass
class ServingCombinedNightlyArchiveReport:
    """Unified nightly archive spanning decode, engine G1, and multistep generation."""

    version: str
    parity_ok: bool
    quick: bool
    functional_chains: List[str] = field(default_factory=list)
    decode: Optional[Dict[str, Any]] = None
    engine_g1: Optional[Dict[str, Any]] = None
    multistep: Optional[Dict[str, Any]] = None
    engine_multistep: Optional[Dict[str, Any]] = None
    paged_multistep: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serving_combined_nightly_archive": True,
            "version": self.version,
            "parity_ok": self.parity_ok,
            "quick": self.quick,
            "functional_chains": list(self.functional_chains),
            "decode": self.decode,
            "engine_g1": self.engine_g1,
            "multistep": self.multistep,
            "engine_multistep": self.engine_multistep,
            "paged_multistep": self.paged_multistep,
        }


def validate_serving_combined_nightly_archive(payload: Mapping[str, Any]) -> List[str]:
    """Return validation errors (empty list means OK)."""
    errors: List[str] = []
    if not payload.get("serving_combined_nightly_archive"):
        errors.append("missing serving_combined_nightly_archive=true marker")
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        errors.append("missing or empty version string")
    if payload.get("parity_ok") is not True:
        errors.append("parity_ok must be true for combined nightly merge")

    chains = payload.get("functional_chains")
    if not isinstance(chains, list) or len(chains) < 7:
        errors.append(
            "functional_chains must list decode + G1 + HF multistep + engine multistep + paged multistep"
        )

    decode = payload.get("decode")
    if not isinstance(decode, dict):
        errors.append("decode subsection must be a dict")
    else:
        errors.extend(f"decode.{e}" for e in validate_serving_qwen_decode_bench_archive(decode))

    engine_g1 = payload.get("engine_g1")
    if not isinstance(engine_g1, dict):
        errors.append("engine_g1 subsection must be a dict")
    else:
        errors.extend(f"engine_g1.{e}" for e in validate_serving_engine_g1_regression(engine_g1))

    multistep = payload.get("multistep")
    if not isinstance(multistep, dict):
        errors.append("multistep subsection must be a dict")
    else:
        errors.extend(
            f"multistep.{e}" for e in validate_serving_qwen_multistep_generation_bench(multistep)
        )

    engine_multistep = payload.get("engine_multistep")
    if not isinstance(engine_multistep, dict):
        errors.append("engine_multistep subsection must be a dict")
    else:
        errors.extend(
            f"engine_multistep.{e}"
            for e in validate_serving_engine_native_multistep_bench(engine_multistep)
        )

    paged_multistep = payload.get("paged_multistep")
    if not isinstance(paged_multistep, dict):
        errors.append("paged_multistep subsection must be a dict")
    else:
        errors.extend(
            f"paged_multistep.{e}"
            for e in validate_serving_vllm_paged_multistep_bench(paged_multistep)
        )

    return errors


def serving_combined_nightly_archive_metadata(
    payload: Mapping[str, Any],
    *,
    archive_path: str,
    validation_ok: bool,
    quick: bool = False,
) -> Dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, default=str)
    decode = payload.get("decode") if isinstance(payload.get("decode"), dict) else {}
    engine_g1 = payload.get("engine_g1") if isinstance(payload.get("engine_g1"), dict) else {}
    multistep = payload.get("multistep") if isinstance(payload.get("multistep"), dict) else {}
    engine_multistep = (
        payload.get("engine_multistep") if isinstance(payload.get("engine_multistep"), dict) else {}
    )
    paged_multistep = (
        payload.get("paged_multistep") if isinstance(payload.get("paged_multistep"), dict) else {}
    )
    return {
        "serving_combined_nightly_archive_metadata": True,
        "archive_path": archive_path,
        "validation_ok": validation_ok,
        "quick": quick,
        "version": payload.get("version"),
        "parity_ok": payload.get("parity_ok"),
        "functional_chains": payload.get("functional_chains"),
        "decode_parity_ok": decode.get("parity_ok"),
        "engine_g1_parity_ok": engine_g1.get("parity_ok"),
        "multistep_parity_ok": multistep.get("parity_ok"),
        "multistep_token_match_ok": multistep.get("token_match_ok"),
        "engine_multistep_parity_ok": engine_multistep.get("parity_ok"),
        "engine_multistep_native_parity_ok": engine_multistep.get("native_parity_ok"),
        "paged_multistep_parity_ok": paged_multistep.get("parity_ok"),
        "paged_multistep_token_match_ok": paged_multistep.get("token_match_ok"),
        "paged_multistep_native_parity_ok": paged_multistep.get("native_parity_ok"),
        "paged_multistep_native_full_layer_parity_ok": paged_multistep.get(
            "native_full_layer_parity_ok"
        ),
        "paged_multistep_native_decoder_parity_ok": paged_multistep.get(
            "native_decoder_parity_ok"
        ),
        "paged_multistep_native_decoder_token_match_ok": paged_multistep.get(
            "native_decoder_token_match_ok"
        ),
        "vllm_native_available": engine_g1.get("vllm_native_available"),
        "sglang_native_available": engine_g1.get("sglang_native_available"),
        "archive_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "created_unix": time.time(),
    }


def run_serving_combined_nightly_archive(
    *,
    model_id: str = DEFAULT_QWEN05B_MODEL,
    prompt: str = "The capital of France is",
    max_rf_mlp_layers: int = 1,
    all_rf_layers: bool = False,
    max_new_tokens: int = 8,
    quick: bool = True,
    version: str = "s38",
    skip_decode: bool = False,
    skip_multistep: bool = False,
    skip_engine_multistep: bool = False,
    skip_paged_multistep: bool = False,
) -> Dict[str, Any]:
    """Run decode + G1 + multistep + engine multistep + paged multistep archives."""
    functional_chains = [
        "chain_b_decode_step",
        "chain_c_vllm_torch",
        "chain_d_sglang_torch",
        "chain_b_multistep_generation",
        "chain_c_vllm_torch_multistep",
        "chain_d_sglang_torch_multistep",
        "chain_c_vllm_paged_multistep",
    ]

    decode_payload: Optional[Dict[str, Any]] = None
    if not skip_decode:
        decode_payload = run_serving_qwen_decode_bench_archive(
            model_id=model_id,
            prompt=prompt,
            max_rf_mlp_layers=max_rf_mlp_layers,
            all_rf_layers=all_rf_layers,
            quick=quick,
            version=version,
        )

    g1_report = run_engine_g1_regression(quick=quick, version=version)
    engine_g1_payload = g1_report.to_dict()

    multistep_payload: Optional[Dict[str, Any]] = None
    if not skip_multistep:
        multistep_payload = run_serving_qwen_multistep_generation_archive(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            max_rf_mlp_layers=max_rf_mlp_layers,
            quick=quick,
            version=version,
        )

    parity_ok = bool(engine_g1_payload.get("parity_ok"))
    if decode_payload is not None:
        parity_ok = parity_ok and bool(decode_payload.get("parity_ok"))
    if multistep_payload is not None:
        parity_ok = parity_ok and bool(multistep_payload.get("parity_ok"))

    engine_multistep_payload: Optional[Dict[str, Any]] = None
    if not skip_engine_multistep:
        engine_multistep_payload = run_serving_engine_native_multistep_archive(
            quick=quick,
            try_native=True,
            version=version,
        )
        parity_ok = parity_ok and bool(engine_multistep_payload.get("parity_ok"))

    paged_multistep_payload: Optional[Dict[str, Any]] = None
    if not skip_paged_multistep:
        paged_multistep_payload = run_serving_vllm_paged_multistep_archive(
            quick=quick,
            try_native=True,
            version=version,
        )
        parity_ok = parity_ok and bool(paged_multistep_payload.get("parity_ok"))

    report = ServingCombinedNightlyArchiveReport(
        version=version,
        parity_ok=parity_ok,
        quick=quick,
        functional_chains=functional_chains,
        decode=decode_payload,
        engine_g1=engine_g1_payload,
        multistep=multistep_payload,
        engine_multistep=engine_multistep_payload,
        paged_multistep=paged_multistep_payload,
    )
    payload = report.to_dict()
    errors = validate_serving_combined_nightly_archive(payload)
    if errors:
        raise RuntimeError(f"combined nightly archive validation failed: {errors}")
    return payload
