# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S34: Combined Serving nightly archive — decode + G1 + multistep."""

from __future__ import annotations

import json
import os

import pytest

from serving_test_utils import serving


def _synthetic_decode_subsection(*, version: str = "s34") -> dict:
    return {
        "serving_qwen_decode_bench": True,
        "version": version,
        "model_id": "Qwen/Qwen2-0.5B",
        "device": "cpu",
        "max_rf_mlp_layers": 1,
        "parity_ok": True,
        "speedup_yirage_vs_native": 1.05,
        "rows": [
            {"name": "native_decode_step", "mean_ms": 10.0, "iters": 8, "device": "cpu"},
            {"name": "yirage_rf_decode_step", "mean_ms": 9.5, "iters": 8, "device": "cpu"},
        ],
    }


def _synthetic_g1_subsection(*, version: str = "s34") -> dict:
    chain = {
        "chain_id": "chain_c_vllm_torch",
        "functional_chain": "chain_c_vllm_plugin",
        "parity_ok": True,
        "plugin": "torch_surrogate",
        "engine": "torch_surrogate",
    }
    chain_d = {
        **chain,
        "chain_id": "chain_d_sglang_torch",
        "functional_chain": "chain_d_sglang_forward_batch",
    }
    return {
        "serving_engine_g1_regression": True,
        "version": version,
        "parity_ok": True,
        "device": "cpu",
        "vllm_native_available": False,
        "sglang_native_available": False,
        "chains": [chain, chain_d],
        "vllm_hybrid": {"parity_ok": True, "rf_layer_ids": [0]},
        "sglang_hybrid": {"parity_ok": True, "rf_layer_ids": [0]},
    }


def _synthetic_multistep_subsection(*, version: str = "s34") -> dict:
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return {
        "serving_qwen_multistep_generation_bench": True,
        "version": version,
        "model_id": "Qwen/Qwen2-0.5B",
        "device": "cpu",
        "max_new_tokens": 4,
        "max_rf_mlp_layers": 1,
        "parity_ok": True,
        "token_match_ok": True,
        "functional_chain": "chain_b_multistep_generation",
        "native_token_ids": ids,
        "rf_token_ids": ids,
    }


def _synthetic_combined_archive(*, version: str = "s34") -> dict:
    return {
        "serving_combined_nightly_archive": True,
        "version": version,
        "parity_ok": True,
        "quick": True,
        "functional_chains": [
            "chain_b_decode_step",
            "chain_c_vllm_torch",
            "chain_d_sglang_torch",
            "chain_b_multistep_generation",
            "chain_c_vllm_torch_multistep",
            "chain_d_sglang_torch_multistep",
            "chain_c_vllm_paged_multistep",
        ],
        "decode": _synthetic_decode_subsection(version=version),
        "engine_g1": _synthetic_g1_subsection(version=version),
        "multistep": _synthetic_multistep_subsection(version=version),
        "engine_multistep": {
            "serving_engine_native_multistep_bench": True,
            "version": version,
            "parity_ok": True,
            "decode_steps": 3,
            "chains": [
                {
                    "chain_id": "chain_c_vllm_torch_multistep",
                    "parity_ok": True,
                    "step_parity_ok": [True, True, True],
                },
                {
                    "chain_id": "chain_d_sglang_torch_multistep",
                    "parity_ok": True,
                    "step_parity_ok": [True, True, True],
                },
            ],
        },
        "paged_multistep": {
            "serving_vllm_paged_multistep_bench": True,
            "version": version,
            "parity_ok": True,
            "token_match_ok": True,
            "decode_steps": 3,
            "paged_kv_bridged": True,
            "functional_chain": "chain_c_vllm_paged_multistep",
            "engine_token_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "hybrid_token_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "vllm_native_available": False,
            "native_parity_ok": None,
            "native_step_parity_ok": [],
            "native_full_layer_parity_ok": None,
            "native_full_layer_step_parity_ok": [],
            "native_decoder_parity_ok": None,
            "native_decoder_token_match_ok": None,
            "native_decoder_step_parity_ok": [],
            "native_decoder_step_token_match_ok": [],
        },
    }


def test_runtime_fusion_version_s34(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s45"


def test_validate_combined_nightly_archive_synthetic(serving):
    from yirage.serving.combined_nightly_archive import (
        validate_serving_combined_nightly_archive,
    )

    payload = _synthetic_combined_archive()
    assert validate_serving_combined_nightly_archive(payload) == []


def test_validate_combined_rejects_bad_parity(serving):
    from yirage.serving.combined_nightly_archive import (
        validate_serving_combined_nightly_archive,
    )

    bad = _synthetic_combined_archive()
    bad["parity_ok"] = False
    errors = validate_serving_combined_nightly_archive(bad)
    assert any("parity_ok" in e for e in errors)


def test_combined_nightly_archive_metadata(serving):
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )

    payload = _synthetic_combined_archive()
    meta = serving_combined_nightly_archive_metadata(
        payload, archive_path="artifacts/combined.json", validation_ok=True, quick=True
    )
    assert meta["serving_combined_nightly_archive_metadata"] is True
    assert meta["decode_parity_ok"] is True
    assert meta["engine_g1_parity_ok"] is True
    assert meta["multistep_token_match_ok"] is True
    assert meta["paged_multistep_token_match_ok"] is True
    json.dumps(meta)


def test_combined_g1_subsection_live(serving):
    serving.require_torch()
    report = serving.run_engine_g1_regression(quick=True, version="s34")
    assert report.parity_ok
    payload = report.to_dict()
    errors = serving.validate_serving_engine_g1_regression(payload)
    assert errors == []


def test_combined_bench_cli_g1_only_smoke(serving):
    """Torch-only G1 subsection matches combined archive shape."""
    serving.require_torch()
    g1 = serving.run_engine_g1_regression(quick=True, version="s34").to_dict()
    partial = {
        "serving_combined_nightly_archive": True,
        "version": "s34",
        "parity_ok": g1["parity_ok"],
        "quick": True,
        "functional_chains": ["chain_c_vllm_torch", "chain_d_sglang_torch"],
        "decode": None,
        "engine_g1": g1,
        "multistep": None,
    }
    assert partial["parity_ok"] is True
    json.dumps(partial)


def test_cpu_cert_manifest_s34_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s34_contract" in names


@pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)
@pytest.mark.slow
def test_run_combined_nightly_archive_quick(serving):
    from yirage.serving.combined_nightly_archive import run_serving_combined_nightly_archive
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available
    from yirage.serving.yirage_exec import is_yirage_core_available, require_yirage_core

    if not is_yirage_core_available():
        pytest.skip("yirage.core not available")
    if not is_transformers_available():
        pytest.skip("transformers not installed")
    require_yirage_core()

    payload = run_serving_combined_nightly_archive(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
        version="s34",
    )
    assert payload["parity_ok"] is True
    assert payload["decode"]["parity_ok"] is True
    assert payload["engine_g1"]["parity_ok"] is True
    assert payload["multistep"]["token_match_ok"] is True
    assert payload["paged_multistep"]["token_match_ok"] is True
