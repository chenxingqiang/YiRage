# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Shared quick shapes for bench vs reference_mugraphs (Loop R68)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

BENCH_WORKLOADS = (
    "plain_matmul",
    "rms_norm_matmul",
    "matmul_chain",
    "concat_matmul",
)

DIM_FIELDS: Dict[str, Tuple[str, ...]] = {
    "plain_matmul": ("m", "k", "n"),
    "rms_norm_matmul": ("m", "k", "n"),
    "matmul_chain": ("cm", "ck", "ck2", "cn"),
    "concat_matmul": ("m", "k1", "k2", "n"),
}

BENCH_QUICK_SHAPES: Dict[str, Tuple[int, ...]] = {
    "plain_matmul": (64, 128, 256),
    "rms_norm_matmul": (64, 128, 256),
    "matmul_chain": (32, 64, 128, 256),
    "concat_matmul": (32, 64, 64, 128),
}

REFERENCE_QUICK_SHAPES: Dict[str, Tuple[int, ...]] = {
    "plain_matmul": (8, 128, 256),
    "rms_norm_matmul": (8, 128, 256),
    "matmul_chain": (8, 32, 64, 128),
    "concat_matmul": (8, 32, 32, 128),
}

BENCH_FULL_SHAPES: Dict[str, Tuple[int, ...]] = {
    "plain_matmul": (128, 256, 512),
    "rms_norm_matmul": (128, 256, 512),
    "matmul_chain": (64, 128, 256, 512),
    "concat_matmul": (64, 128, 128, 256),
}


def bench_shape_tuple(workload: str, *, quick: bool) -> Tuple[int, ...]:
    """Return shape tuple for bench workload (quick or full)."""
    table = BENCH_QUICK_SHAPES if quick else BENCH_FULL_SHAPES
    return table[workload]


def bench_shape_label(workload: str, *, quick: bool) -> str:
    """Human-readable shapes label used in bench JSON ``shapes`` field."""
    t = bench_shape_tuple(workload, quick=quick)
    if workload == "plain_matmul":
        m, k, n = t
        return f"{m}×{k} @ {k}×{n}"
    if workload == "rms_norm_matmul":
        m, k, n = t
        return f"rms({m}×{k}) @ {k}×{n}"
    if workload == "matmul_chain":
        cm, ck, ck2, cn = t
        return f"({cm}×{ck}@{ck}×{ck2})@{ck2}×{cn}"
    if workload == "concat_matmul":
        m, k1, k2, n = t
        return f"cat({m}×{k1}+{k2}) @ cat({k1}+{k2}×{n})"
    raise KeyError(workload)


def validate_bench_json_row_shapes(
    rows: List[Dict[str, Any]],
    *,
    quick: bool,
) -> List[str]:
    """Validate bench JSON row ``shapes`` labels match ``cpu_bench_shapes`` (R71)."""
    errors: List[str] = []
    for row in rows:
        workload = row.get("workload")
        if workload not in BENCH_WORKLOADS:
            continue
        expected = bench_shape_label(workload, quick=quick)
        actual = row.get("shapes")
        if actual != expected:
            errors.append(
                f"{workload}: shapes {actual!r} != expected {expected!r}"
            )
    return errors


REFERENCE_DEMO_WORKLOADS: Dict[str, str] = {
    "plain_matmul.py": "plain_matmul",
    "rms_norm.py": "rms_norm_matmul",
    "matmul_chain.py": "matmul_chain",
    "concat_matmul.py": "concat_matmul",
}


def reference_quick_dims(workload: str) -> Dict[str, int]:
    """Return named quick dimensions for a bench workload reference demo."""
    fields = DIM_FIELDS[workload]
    values = REFERENCE_QUICK_SHAPES[workload]
    return dict(zip(fields, values))


def bench_quick_dims(workload: str) -> Dict[str, int]:
    """Return named quick dimensions for bench ``--quick`` workloads."""
    fields = DIM_FIELDS[workload]
    values = BENCH_QUICK_SHAPES[workload]
    return dict(zip(fields, values))


def shape_contract() -> Dict[str, Dict[str, Any]]:
    """Bench vs reference quick shape contract (single source of truth)."""
    out: Dict[str, Dict[str, Any]] = {}
    for workload in BENCH_WORKLOADS:
        out[workload] = {
            "dim_fields": DIM_FIELDS[workload],
            "bench_quick": BENCH_QUICK_SHAPES[workload],
            "reference_quick": REFERENCE_QUICK_SHAPES[workload],
            "bench_full": BENCH_FULL_SHAPES[workload],
            "bench_quick_label": bench_shape_label(workload, quick=True),
            "bench_full_label": bench_shape_label(workload, quick=False),
        }
    return out
