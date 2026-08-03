# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S36: Unified Serving dashboard from combined nightly archive."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving


def _synthetic_combined_for_dashboard(*, version: str = "s38") -> dict:
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
        "decode": {
            "serving_qwen_decode_bench": True,
            "version": version,
            "parity_ok": True,
            "speedup_yirage_vs_native": 1.05,
            "serving_search_tier": "seed_verify",
            "max_rf_mlp_layers": 1,
            "rows": [
                {"name": "native_decode_step", "mean_ms": 10.0, "iters": 8, "device": "cpu"},
                {"name": "yirage_rf_decode_step", "mean_ms": 9.5, "iters": 8, "device": "cpu"},
            ],
        },
        "engine_g1": {
            "serving_engine_g1_regression": True,
            "version": version,
            "parity_ok": True,
            "native_parity_ok": None,
            "vllm_native_available": False,
            "sglang_native_available": False,
            "chains": [
                {"chain_id": "chain_c_vllm_torch", "parity_ok": True},
                {"chain_id": "chain_d_sglang_torch", "parity_ok": True},
            ],
            "vllm_hybrid": {"parity_ok": True, "rf_layer_ids": [0]},
            "sglang_hybrid": {"parity_ok": True, "rf_layer_ids": [0]},
        },
        "multistep": {
            "serving_qwen_multistep_generation_bench": True,
            "version": version,
            "parity_ok": True,
            "token_match_ok": True,
            "max_new_tokens": 4,
            "mlp_backend": "yirage_cpu",
            "yirage_core_used": True,
            "native_token_ids": ids,
            "rf_token_ids": ids,
        },
        "engine_multistep": {
            "serving_engine_native_multistep_bench": True,
            "version": version,
            "parity_ok": True,
            "decode_steps": 3,
            "native_parity_ok": None,
            "chains": [
                {"chain_id": "chain_c_vllm_torch_multistep", "parity_ok": True},
                {"chain_id": "chain_d_sglang_torch_multistep", "parity_ok": True},
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
        },
    }


def test_runtime_fusion_version_s36(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s40"


def test_build_dashboard_from_synthetic_archive(serving):
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        validate_serving_dashboard,
    )

    archive = _synthetic_combined_for_dashboard()
    report = build_serving_dashboard_from_combined_archive(archive)
    assert report.merge_gate_ok
    assert report.parity_ok
    assert len(report.rows) == 5
    payload = report.to_dict()
    assert validate_serving_dashboard(payload) == []


def test_dashboard_markdown_renders(serving):
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        render_serving_dashboard_markdown,
    )

    report = build_serving_dashboard_from_combined_archive(_synthetic_combined_for_dashboard())
    md = render_serving_dashboard_markdown(report)
    assert "# Serving Loop Dashboard" in md
    assert "chain_b_decode_step" in md
    assert "PASS" in md


def test_dashboard_g1_only_partial(serving):
    from yirage.serving.serving_dashboard import build_serving_dashboard_from_combined_archive

    serving.require_torch()
    g1 = serving.run_engine_g1_regression(quick=True, try_native=False, version="s36").to_dict()
    partial = {
        "serving_combined_nightly_archive": True,
        "version": "s36",
        "parity_ok": g1["parity_ok"],
        "quick": True,
        "functional_chains": ["chain_c_vllm_torch", "chain_d_sglang_torch"],
        "decode": None,
        "engine_g1": g1,
        "multistep": None,
        "engine_multistep": None,
    }
    report = build_serving_dashboard_from_combined_archive(partial, allow_partial=True)
    assert report.merge_gate_ok
    assert len(report.rows) == 1
    assert report.rows[0].section == "engine_g1"
    assert report.rows[0].parity_ok is True


def test_dashboard_json_contract(serving):
    from yirage.serving.serving_dashboard import build_serving_dashboard_from_combined_archive

    payload = build_serving_dashboard_from_combined_archive(
        _synthetic_combined_for_dashboard()
    ).to_dict()
    assert payload["serving_dashboard"] is True
    assert payload["version"] == "s40"
    json.dumps(payload)


def test_cpu_cert_manifest_s36_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s36_contract" in names
