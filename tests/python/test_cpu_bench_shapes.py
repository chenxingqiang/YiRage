# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared bench/reference quick shapes (Loop R68)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.cpu_bench_shapes import (  # noqa: E402
    BENCH_FULL_SHAPES,
    BENCH_QUICK_SHAPES,
    REFERENCE_DEMO_WORKLOADS,
    REFERENCE_QUICK_SHAPES,
    bench_quick_dims,
    bench_shape_label,
    bench_shape_tuple,
    reference_quick_dims,
    shape_contract,
    validate_bench_json_row_shapes,
)
from scripts.cpu_cert_utils import cpu_bench_reference_shape_contract  # noqa: E402


def test_shape_contract_matches_cert_utils_wrapper():
    assert cpu_bench_reference_shape_contract() == shape_contract()


def test_reference_quick_dims_plain_matmul():
    dims = reference_quick_dims("plain_matmul")
    assert dims == {"m": 8, "k": 128, "n": 256}
    assert REFERENCE_QUICK_SHAPES["plain_matmul"] == (8, 128, 256)
    assert BENCH_QUICK_SHAPES["plain_matmul"] == (64, 128, 256)


def test_bench_quick_dims_matmul_chain():
    dims = bench_quick_dims("matmul_chain")
    assert dims["cm"] == 32
    assert dims["cn"] == 256


def test_reference_demo_workloads_match_bench_reference_map():
    from scripts.cpu_cert_utils import cpu_bench_workload_reference_map

    assert set(REFERENCE_DEMO_WORKLOADS.values()) == set(
        cpu_bench_workload_reference_map().keys()
    )


def test_bench_shape_tuple_quick_and_full():
    assert bench_shape_tuple("plain_matmul", quick=True) == (64, 128, 256)
    assert bench_shape_tuple("concat_matmul", quick=False) == (64, 128, 128, 256)


def test_bench_workloads_use_shared_quick_shapes():
    from scripts.bench_fused_vs_mkl_baseline import _workloads

    for workload in BENCH_QUICK_SHAPES:
        wl = next(w for w in _workloads(quick=True) if w.name == workload)
        expected = "×".join(str(x) for x in BENCH_QUICK_SHAPES[workload])
        if workload == "plain_matmul":
            m, k, n = BENCH_QUICK_SHAPES[workload]
            assert wl.shapes == f"{m}×{k} @ {k}×{n}"
        elif workload == "rms_norm_matmul":
            m, k, n = BENCH_QUICK_SHAPES[workload]
            assert f"rms({m}×{k})" in wl.shapes
        elif workload == "matmul_chain":
            cm, ck, ck2, cn = BENCH_QUICK_SHAPES[workload]
            assert wl.shapes == f"({cm}×{ck}@{ck}×{ck2})@{ck2}×{cn}"
        elif workload == "concat_matmul":
            m, k1, k2, n = BENCH_QUICK_SHAPES[workload]
            assert f"cat({m}×{k1}+{k2})" in wl.shapes


def test_bench_workloads_use_shared_full_shapes():
    from scripts.bench_fused_vs_mkl_baseline import _workloads

    for workload in BENCH_FULL_SHAPES:
        wl = next(w for w in _workloads(quick=False) if w.name == workload)
        if workload == "plain_matmul":
            m, k, n = BENCH_FULL_SHAPES[workload]
            assert wl.shapes == f"{m}×{k} @ {k}×{n}"
        elif workload == "rms_norm_matmul":
            m, k, n = BENCH_FULL_SHAPES[workload]
            assert f"rms({m}×{k})" in wl.shapes
        elif workload == "matmul_chain":
            cm, ck, ck2, cn = BENCH_FULL_SHAPES[workload]
            assert wl.shapes == f"({cm}×{ck}@{ck}×{ck2})@{ck2}×{cn}"
        elif workload == "concat_matmul":
            m, k1, k2, n = BENCH_FULL_SHAPES[workload]
            assert f"cat({m}×{k1}+{k2})" in wl.shapes


def test_shape_contract_includes_bench_full():
    contract = shape_contract()
    for workload in BENCH_FULL_SHAPES:
        assert contract[workload]["bench_full"] == BENCH_FULL_SHAPES[workload]


def test_bench_shape_label_matches_workload_shapes():
    from scripts.bench_fused_vs_mkl_baseline import _workloads

    for quick in (True, False):
        for wl in _workloads(quick=quick):
            assert wl.shapes == bench_shape_label(wl.name, quick=quick)


def test_validate_bench_json_row_shapes_accepts_matching_rows():
    rows = [
        {
            "workload": "plain_matmul",
            "shapes": bench_shape_label("plain_matmul", quick=True),
        },
        {
            "workload": "concat_matmul",
            "shapes": bench_shape_label("concat_matmul", quick=True),
        },
    ]
    assert validate_bench_json_row_shapes(rows, quick=True) == []


def test_validate_bench_json_row_shapes_rejects_mismatch():
    rows = [{"workload": "plain_matmul", "shapes": "wrong"}]
    errs = validate_bench_json_row_shapes(rows, quick=True)
    assert len(errs) == 1
    assert "plain_matmul" in errs[0]


def test_shape_contract_includes_bench_labels():
    contract = shape_contract()
    for workload in BENCH_QUICK_SHAPES:
        assert contract[workload]["bench_quick_label"] == bench_shape_label(
            workload, quick=True
        )
        assert contract[workload]["bench_full_label"] == bench_shape_label(
            workload, quick=False
        )
