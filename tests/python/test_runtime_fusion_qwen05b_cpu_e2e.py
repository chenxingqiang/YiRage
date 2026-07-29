# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU full-model Qwen2-0.5B e2e: HF generate + YiRage superoptimize decode."""

from __future__ import annotations

import pytest

from serving_test_utils import import_serving


@pytest.fixture(scope="module")
def hf_qwen_e2e():
    serving = import_serving()
    from yirage.serving.exec_backend import BACKEND_TORCH, BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        is_transformers_available,
        run_hf_qwen05b_cpu_e2e,
    )
    from yirage.serving.yirage_exec import is_yirage_core_available, require_yirage_core

    if not is_transformers_available():
        pytest.skip("transformers not installed")
    return serving, DEFAULT_QWEN05B_MODEL, run_hf_qwen05b_cpu_e2e, BACKEND_TORCH, BACKEND_YIRAGE_CPU, require_yirage_core, is_yirage_core_available


def test_hf_qwen05b_cpu_e2e_torch_prefill(hf_qwen_e2e):
    """Prefill RF orchestration (torch backend; prefill cannot use yirage batch>1)."""
    _, default_model, run_e2e, BACKEND_TORCH, _, _, _ = hf_qwen_e2e
    report = run_e2e(quick=True, mlp_backend=BACKEND_TORCH)
    assert report.model_id == default_model
    assert report.prefill_parity_ok is True
    assert report.decode_parity_ok is True
    assert report.yirage_core_used is False


def test_hf_qwen05b_cpu_e2e_yirage_decode(hf_qwen_e2e):
    """Full YiRage path: yirage.core gate_up + CPU superoptimize down on decode."""
    _, default_model, run_e2e, _, BACKEND_YIRAGE_CPU, require_yirage_core, is_yirage_core_available = hf_qwen_e2e
    if not is_yirage_core_available():
        pytest.skip("yirage.core not built — run scripts/setup_serving_yirage_core.sh")
    require_yirage_core()
    report = run_e2e(quick=True, mlp_backend=BACKEND_YIRAGE_CPU)
    assert report.model_id == default_model
    assert report.num_layers == 24
    assert report.hidden_size == 896
    assert report.yirage_core_used is True
    assert report.yirage_decode_parity_ok is True
    assert report.generate_token_match_ok is True
    assert report.parity_ok is True
    assert "yirage_cpu" in report.plugin
    assert report.superopt_elapsed_s_total > 0.0


def test_hf_qwen05b_down_superoptimize_no_fallback(hf_qwen_e2e):
    """Down matmul superoptimize must succeed (no seed fallback)."""
    _, _, _, _, BACKEND_YIRAGE_CPU, require_yirage_core, is_yirage_core_available = hf_qwen_e2e
    if not is_yirage_core_available():
        pytest.skip("yirage.core not built")
    require_yirage_core()
    from yirage.serving.yirage_exec import (
        apply_serving_kn_down_matmul_tractability,
        superoptimize_down_matmul_cpu,
    )

    apply_serving_kn_down_matmul_tractability()
    opt = superoptimize_down_matmul_cpu(896, 4864, quick=True)
    assert opt is not None
    assert opt.backend == "cpu"


def test_hf_qwen05b_cpu_e2e_contract(hf_qwen_e2e):
    _, _, run_e2e, BACKEND_TORCH, BACKEND_YIRAGE_CPU, _, is_yirage_core_available = hf_qwen_e2e
    backend = BACKEND_YIRAGE_CPU if is_yirage_core_available() else BACKEND_TORCH
    report = run_e2e(quick=True, mlp_backend=backend)
    payload = report.to_dict()
    assert payload["device"] == "cpu"
    assert payload["used_rf_mlp_layers"] >= 1
    assert "transformers" in payload["plugin"]
