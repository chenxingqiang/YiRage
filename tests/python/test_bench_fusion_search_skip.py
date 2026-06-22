# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bench fusion search skip predicates (quick and full)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.bench_fused_vs_mkl_baseline import (  # noqa: E402
    CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON,
    MLIR_JIT_BENCH_TIMING_KEYS,
    MLIR_JIT_EMIT_PATH_VALUES,
    MUGRAPH_SOURCE_VALUES,
    _bench_skip_fusion_search,
    _cap_bench_search_explore,
    _workloads,
    concat_matmul_mlir_jit_deferred_contract,
    mlir_jit_bench_json_field_guide,
    mlir_jit_bench_json_timing_contract,
    validate_concat_matmul_bench_row,
    validate_mlir_jit_bench_row,
)
from scripts.cpu_mlir_bench_utils import (  # noqa: E402
    mlir_bench_profile_from_rows,
    parse_bench_json,
    run_mlir_bench_profile,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.mark.parametrize(
    "workload_name",
    ["plain_matmul", "rms_norm_matmul", "matmul_chain", "concat_matmul"],
)
def test_quick_fusion_search_skipped_for_p0_seed_workloads(workload_name: str):
    wl = next(w for w in _workloads(quick=True) if w.name == workload_name)
    g = wl.build()
    assert _bench_skip_fusion_search(wl, g.cygraph)


@pytest.mark.parametrize(
    "workload_name",
    ["plain_matmul", "rms_norm_matmul", "matmul_chain", "concat_matmul"],
)
def test_full_fusion_search_skipped_for_p0_seed_workloads(workload_name: str):
    wl = next(w for w in _workloads(quick=False) if w.name == workload_name)
    g = wl.build()
    assert _bench_skip_fusion_search(wl, g.cygraph)


def test_cap_bench_search_explore_single_grid_point():
    grids, blocks, franges = _cap_bench_search_explore(
        [(2, 1, 1), (4, 1, 1)],
        [(32, 1, 1), (64, 1, 1)],
        [4, 8],
    )
    assert grids == [(1, 1, 1)]
    assert blocks == [(32, 1, 1)]
    assert franges == [1]


def test_fusion_search_not_skipped_when_env_disabled():
    wl = next(w for w in _workloads(quick=True) if w.name == "plain_matmul")
    g = wl.build()
    prev = os.environ.get("YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH")
    os.environ["YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH"] = "0"
    try:
        assert not _bench_skip_fusion_search(wl, g.cygraph)
    finally:
        if prev is None:
            os.environ.pop("YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH", None)
        else:
            os.environ["YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH"] = prev


def test_mlir_jit_bench_json_field_guide_contract():
    """R51: documented mugraph_source × mlir_jit_emit_path pairs are well-formed."""
    guide = mlir_jit_bench_json_field_guide()
    assert len(guide) >= 4
    for row in guide:
        for src in row["mugraph_source"]:
            assert src in MUGRAPH_SOURCE_VALUES
        for path in row["mlir_jit_emit_path"]:
            assert path in MLIR_JIT_EMIT_PATH_VALUES
        assert isinstance(row["fusion_search_skipped"], bool)
        assert isinstance(row["mlir_jit_fused_seed"], bool)


def test_mlir_jit_bench_json_timing_contract_keys():
    contract = mlir_jit_bench_json_timing_contract()
    assert set(contract["required_keys"]) == MLIR_JIT_BENCH_TIMING_KEYS


def test_validate_mlir_jit_bench_row_accepts_smoke_row():
    row = {
        "workload": "rms_norm_matmul",
        "mlir_jit": True,
        "hand_mlir_jit_ms": 0.5,
        "dialect_lowered_jit_ms": 0.4,
        "speedup_hand_over_dialect_lowered": 0.8,
        "mlir_hand_dialect_aligned": True,
        "mlir_jit_emit_path": "dialect_lowered",
        "mugraph_source": "mlir_jit",
    }
    assert validate_mlir_jit_bench_row(row) == []


def test_validate_mlir_jit_bench_row_rejects_missing_timing():
    row = {"workload": "rms_norm_matmul", "mlir_jit": True}
    errs = validate_mlir_jit_bench_row(row)
    assert any("hand_mlir_jit_ms" in e for e in errs)


def test_concat_matmul_mlir_jit_deferred_contract_shape():
    contract = concat_matmul_mlir_jit_deferred_contract()
    assert contract["workload"] == "concat_matmul"
    assert contract["mlir_jit_applicable"] is False
    assert contract["expected_fast_path_key"] == "concat_matmul_fast_path"


def test_validate_concat_matmul_bench_row_accepts_deferred_row():
    row = {
        "workload": "concat_matmul",
        "mlir_jit_applicable": False,
        "mlir_jit": False,
        "mlir_jit_deferred_reason": CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON,
        "concat_matmul_fast_path": True,
    }
    assert validate_concat_matmul_bench_row(row) == []


def test_validate_concat_matmul_bench_row_rejects_mlir_timing_keys():
    row = {
        "workload": "concat_matmul",
        "mlir_jit_applicable": False,
        "mlir_jit": True,
        "hand_mlir_jit_ms": 1.0,
    }
    errs = validate_concat_matmul_bench_row(row)
    assert any("mlir_jit" in e for e in errs)
    assert any("hand_mlir_jit_ms" in e for e in errs)


def test_parse_bench_json_extracts_payload():
    payload = '[{"workload": "concat_matmul", "ok": true}]'
    text = f"noise\nYIRAGE_BENCH_JSON_BEGIN\n{payload}\nYIRAGE_BENCH_JSON_END\n"
    rows = parse_bench_json(text)
    assert len(rows) == 1
    assert rows[0]["workload"] == "concat_matmul"


def test_mlir_bench_profile_from_rows_concat_only():
    from scripts.cpu_bench_shapes import bench_shape_label

    rows = [
        {
            "workload": "concat_matmul",
            "ok": True,
            "shapes": bench_shape_label("concat_matmul", quick=True),
            "mlir_jit_applicable": False,
            "mlir_jit": False,
            "mlir_jit_deferred_reason": CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON,
        }
    ]
    profile = mlir_bench_profile_from_rows(rows, mlir_jit_requested=True, elapsed_s=3.5)
    assert profile["concat_matmul"]["contract_ok"] is True
    assert profile["profile_ok"] is True


def test_run_mlir_bench_profile_smoke():
    """E2e archive: concat contract must pass on CPU VM (rms skipped when no MLIR)."""
    stage = run_mlir_bench_profile()
    assert stage["ok"] is True
    assert stage["profile"]["concat_matmul"]["contract_ok"] is True


def test_run_mlir_bench_profile_rms_contract_when_mlir_available():
    """When MLIR JIT is built, rms timing contract must pass (not skipped)."""
    try:
        from yirage.kernel.cpu_mlir_jit import is_mlir_jit_available
    except Exception:
        pytest.skip("yirage MLIR JIT module unavailable")
    if not is_mlir_jit_available():
        pytest.skip("USE_MLIR=0 / MLIR JIT not available on this VM")
    stage = run_mlir_bench_profile()
    rms = stage["profile"]["rms_norm_matmul"]
    assert rms.get("contract_skipped") is not True
    assert rms["contract_ok"] is True, rms.get("validation_errors")
