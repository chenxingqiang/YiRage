# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S33: HF multi-step greedy generation — native vs RF MLP (G7 chain B extension)."""

from __future__ import annotations

import json
import os

import pytest

from serving_test_utils import serving


pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


def _synthetic_multistep_payload(*, version: str = "s33") -> dict:
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
        "mlp_backend": "torch",
        "yirage_core_used": False,
        "native_token_ids": ids,
        "rf_token_ids": ids,
    }


@pytest.fixture(scope="module")
def hf_serving(serving):
    from yirage.serving.hf_qwen_cpu_e2e import is_transformers_available

    if not is_transformers_available():
        pytest.skip("transformers not installed")
    return serving


def test_runtime_fusion_version_s33(hf_serving):
    assert hf_serving.RuntimeFusion([]).inspect()["version"] == "s33"


def test_validate_multistep_generation_synthetic(hf_serving):
    from yirage.serving.qwen_multistep_generation_bench import (
        validate_serving_qwen_multistep_generation_bench,
    )

    payload = _synthetic_multistep_payload()
    assert validate_serving_qwen_multistep_generation_bench(payload) == []


def test_validate_multistep_rejects_token_mismatch(hf_serving):
    from yirage.serving.qwen_multistep_generation_bench import (
        validate_serving_qwen_multistep_generation_bench,
    )

    bad = _synthetic_multistep_payload()
    bad["token_match_ok"] = False
    bad["parity_ok"] = False
    errors = validate_serving_qwen_multistep_generation_bench(bad)
    assert any("token_match_ok" in e for e in errors)


def test_cpu_cert_manifest_s33_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s33_contract" in names


@pytest.mark.slow
def test_multistep_generation_torch_token_match(hf_serving):
    from yirage.serving.exec_backend import BACKEND_TORCH
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL
    from yirage.serving.qwen_multistep_generation_bench import run_qwen_multistep_generation_bench

    report = run_qwen_multistep_generation_bench(
        model_id=DEFAULT_QWEN05B_MODEL,
        max_new_tokens=4,
        max_rf_mlp_layers=1,
        mlp_backend=BACKEND_TORCH,
        quick=True,
        version="s33",
    )
    assert report.token_match_ok is True
    assert report.parity_ok is True
    assert len(report.native_token_ids) == len(report.rf_token_ids)
    json.dumps(report.to_dict())


@pytest.mark.slow
def test_multistep_generation_yirage_cpu_token_match(hf_serving):
    from yirage.serving.exec_backend import BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL
    from yirage.serving.qwen_multistep_generation_bench import run_qwen_multistep_generation_bench
    from yirage.serving.yirage_exec import is_yirage_core_available, require_yirage_core

    if not is_yirage_core_available():
        pytest.skip("yirage.core not available")
    require_yirage_core()
    report = run_qwen_multistep_generation_bench(
        model_id=DEFAULT_QWEN05B_MODEL,
        max_new_tokens=4,
        max_rf_mlp_layers=1,
        mlp_backend=BACKEND_YIRAGE_CPU,
        quick=True,
        version="s33",
    )
    assert report.yirage_core_used is True
    assert report.token_match_ok is True
    assert report.superopt_elapsed_s_total > 0.0
