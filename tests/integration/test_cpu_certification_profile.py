# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU certification profile helpers (Loop R66)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.cpu_cert_utils import (  # noqa: E402
    cert_profile_from_stages,
    parse_pytest_summary,
)
from tests.integration.cpu_inventory import planned_value_verify_count  # noqa: E402


def test_parse_pytest_summary_passed_and_skipped():
    stdout = "....\n111 passed, 1 skipped in 1.26s\n"
    stats = parse_pytest_summary(stdout)
    assert stats["passed"] == 111
    assert stats["skipped"] == 1
    assert stats["failed"] == 0


def test_parse_pytest_summary_missing_returns_none():
    assert parse_pytest_summary("no summary here")["passed"] is None


def test_planned_value_verify_still_402():
    assert planned_value_verify_count() == 402


def test_rms_matmul_tractability_sets_env(monkeypatch):
    monkeypatch.delenv("YIRAGE_CPU_MAX_KN_GRAPH_OP", raising=False)
    monkeypatch.delenv("YIRAGE_CPU_MAX_TB_GRAPH_OP", raising=False)
    monkeypatch.delenv("YIRAGE_CPU_BENCH_MINIMAL_EXPLORE", raising=False)
    from scripts.cpu_cert_utils import apply_rms_matmul_search_tractability

    apply_rms_matmul_search_tractability()
    import os

    assert os.environ["YIRAGE_CPU_MAX_KN_GRAPH_OP"] == "4"
    assert os.environ["YIRAGE_CPU_MAX_TB_GRAPH_OP"] == "6"
    assert os.environ["YIRAGE_CPU_BENCH_MINIMAL_EXPLORE"] == "1"


def test_cert_profile_from_stages_includes_walkthrough_substage_elapsed():
    stages = {
        "walkthrough": {
            "ok": True,
            "elapsed_s": 12.5,
            "walkthrough_substage_elapsed_s": {
                "1. YiRage graph build": 0.1,
                "2. Ray µGraph search": 8.0,
            },
        },
    }
    profile = cert_profile_from_stages(stages, planned_value_verify=402)
    assert profile["walkthrough_substage_elapsed_s"]["2. Ray µGraph search"] == 8.0
    assert profile["walkthrough_elapsed_s"] == 12.5


def test_cert_profile_from_stages_stage_elapsed_and_alignment():
    stages = {
        "value_verify_all": {
            "ok": True,
            "elapsed_s": 2.5,
            "pytest": {"passed": 402, "skipped": 1, "failed": 0},
        },
        "op_contract": {"ok": True, "elapsed_s": 1.2},
        "native_gemm": {"ok": True, "elapsed_s": 25.0},
        "superoptimize_smoke": {"ok": True, "elapsed_s": 1.5},
    }
    profile = cert_profile_from_stages(stages, planned_value_verify=402)
    assert profile["value_verify_aligned"] is True
    assert profile["stages_run"] == 4
    assert profile["stages_ok"] == 4
    assert profile["stage_elapsed_s"]["native_gemm"] == 25.0
    assert profile["total_elapsed_s"] == 30.2


def test_cert_profile_from_stages_includes_mlir_bench_profile():
    stages = {
        "mlir_bench_profile": {
            "ok": True,
            "elapsed_s": 3.0,
            "profile": {
                "profile_ok": True,
                "concat_matmul": {"contract_ok": True},
            },
        },
    }
    profile = cert_profile_from_stages(stages, planned_value_verify=402)
    assert profile["mlir_bench_profile_ok"] is True
    assert profile["mlir_bench_elapsed_s"] == 3.0
    assert profile["mlir_bench_profile"]["profile_ok"] is True


def test_loop_profile_from_stages_aggregates_demos_and_mlir():
    from scripts.cpu_cert_utils import loop_profile_from_stages

    stages = {
        "demos": {
            "ok": True,
            "elapsed_s": 30.0,
            "pytest": {"passed": 19, "skipped": 0, "failed": 0},
        },
        "mlir_bench_profile": {
            "ok": True,
            "elapsed_s": 3.0,
            "profile": {"profile_ok": True},
        },
    }
    profile = loop_profile_from_stages(stages)
    assert profile["demos_passed"] == 19
    assert profile["mlir_bench_profile_ok"] is True
    assert profile["stages_ok"] == 2
    assert profile["total_elapsed_s"] == 33.0
