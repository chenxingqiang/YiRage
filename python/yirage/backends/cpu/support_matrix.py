# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Load CPU op support contract from docs/cpu_support_matrix.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_CHUNK_DIM_OPS: Tuple[Tuple[str, str, int], ...] = (
    ("kn_chunk_0_op", "tb_chunk_0_op", 0),
    ("kn_chunk_1_op", "tb_chunk_1_op", 1),
    ("kn_chunk_2_op", "tb_chunk_2_op", 2),
)

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MATRIX_PATH = _REPO_ROOT / "docs" / "cpu_support_matrix.yaml"


@dataclass(frozen=True)
class OpContract:
    op_type: str
    tier: str
    layer: str = "kn"
    reason: Optional[str] = None
    note: Optional[str] = None


@lru_cache(maxsize=1)
def load_cpu_support_matrix() -> Dict[str, Any]:
    if not _MATRIX_PATH.is_file():
        raise FileNotFoundError(f"CPU support matrix not found: {_MATRIX_PATH}")
    with _MATRIX_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def kn_op_contracts() -> Dict[str, OpContract]:
    data = load_cpu_support_matrix()
    out: Dict[str, OpContract] = {}
    for op_type, spec in (data.get("kn_ops") or {}).items():
        out[op_type] = OpContract(
            op_type=op_type,
            tier=spec.get("tier", "unsupported"),
            layer=spec.get("layer", "kn"),
            reason=spec.get("reason"),
            note=spec.get("note"),
        )
    return out


def tb_op_contracts() -> Dict[str, OpContract]:
    data = load_cpu_support_matrix()
    out: Dict[str, OpContract] = {}
    for op_type, spec in (data.get("tb_ops") or {}).items():
        out[op_type] = OpContract(
            op_type=op_type,
            tier=spec.get("tier", "unsupported"),
            layer=spec.get("layer", "tb"),
            reason=spec.get("reason"),
            note=spec.get("note"),
        )
    return out


def ops_by_tier(tier: str, *, layer: str = "kn") -> List[str]:
    contracts = kn_op_contracts() if layer == "kn" else tb_op_contracts()
    return sorted(op for op, c in contracts.items() if c.tier == tier)


def cpu_supported_kn_ops() -> List[str]:
    return ops_by_tier("supported", layer="kn")


def cpu_unsupported_kn_ops() -> List[str]:
    contracts = kn_op_contracts()
    return sorted(
        op
        for op, c in contracts.items()
        if c.tier == "unsupported" and c.layer not in ("io",)
    )


def cpu_search_yaml_explore(*, layer: str = "kn") -> List[str]:
    """Ops listed in search_cpu_default_explore (docs/cpu_support_matrix.yaml)."""
    data = load_cpu_support_matrix()
    explore = data.get("search_cpu_default_explore") or {}
    key = "kn" if layer == "kn" else "tb"
    return sorted(explore.get(key) or [])


def cpu_layout_explore_gap_meta() -> Dict[str, Any]:
    """YAML-documented layout explore asymmetry (e.g. TB chunk deferred until R54)."""
    data = load_cpu_support_matrix()
    return dict(data.get("layout_explore_gaps") or {})


def cpu_layout_explore_gap_table() -> List[Dict[str, Any]]:
    """Per-dim KN/TB chunk: matrix tier vs CPU search explore inclusion."""
    kn_explore = set(cpu_search_yaml_explore(layer="kn"))
    tb_explore = set(cpu_search_yaml_explore(layer="tb"))
    kn_contracts = kn_op_contracts()
    tb_contracts = tb_op_contracts()
    rows: List[Dict[str, Any]] = []
    for kn_op, tb_op, dim in _CHUNK_DIM_OPS:
        kn_c = kn_contracts[kn_op]
        tb_c = tb_contracts[tb_op]
        tb_in_explore = tb_op in tb_explore
        rows.append(
            {
                "dim": dim,
                "kn_op": kn_op,
                "tb_op": tb_op,
                "kn_matrix_tier": kn_c.tier,
                "tb_matrix_tier": tb_c.tier,
                "kn_in_search_explore": kn_op in kn_explore,
                "tb_in_search_explore": tb_in_explore,
                "gap_kind": "none" if tb_in_explore else "tb_chunk_deferred",
                "rationale": tb_c.note,
            }
        )
    return rows


def cpu_search_explore_not_supported() -> List[str]:
    """Ops in default CPU search explore list marked unsupported in the matrix."""
    data = load_cpu_support_matrix()
    explore = data.get("search_cpu_default_explore") or {}
    kn_contracts = kn_op_contracts()
    tb_contracts = tb_op_contracts()
    gaps: List[str] = []
    for op in sorted(explore.get("kn") or []):
        c = kn_contracts.get(op)
        if c is None or c.tier == "unsupported":
            gaps.append(f"kn:{op}")
    for op in sorted(explore.get("tb") or []):
        c = tb_op_contracts().get(op)
        if c is None or c.tier == "unsupported":
            gaps.append(f"tb:{op}")
    return gaps


def is_cpu_supported_kn_op(op_type: str) -> bool:
    c = kn_op_contracts().get(op_type)
    return c is not None and c.tier in ("supported", "fast_path")


def is_cpu_unsupported_kn_op(op_type: str) -> bool:
    c = kn_op_contracts().get(op_type)
    return c is not None and c.tier == "unsupported"


def supported_tb_forloop_ops() -> Set[str]:
    contracts = tb_op_contracts()
    return {
        op
        for op, c in contracts.items()
        if c.tier in ("supported", "experimental", "io")
    }


def supported_tb_post_ops() -> Set[str]:
    return supported_tb_forloop_ops() | {"tb_output_op"}


def cpu_verifiable_kn_ops() -> List[str]:
    """KN ops that must have torch reference verifiers (excludes io)."""
    contracts = kn_op_contracts()
    return sorted(
        op
        for op, c in contracts.items()
        if c.tier in ("supported", "fast_path") and c.layer == "kn"
    )


def cpu_verifiable_tb_ops() -> List[str]:
    """TB ops exercised via kn_customized_op interpreter."""
    contracts = tb_op_contracts()
    return sorted(
        op
        for op, c in contracts.items()
        if c.tier in ("supported", "experimental") and c.layer == "tb"
    )


def cpu_unsupported_tb_ops() -> List[str]:
    contracts = tb_op_contracts()
    return sorted(
        op
        for op, c in contracts.items()
        if c.tier == "unsupported" and c.layer not in ("io", "layout")
    )
