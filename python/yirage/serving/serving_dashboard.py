# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S36: Unified Serving dashboard — render combined nightly archive (G7 closure).

Consumes ``serving_combined_nightly_archive`` JSON and produces a compact dashboard
JSON + markdown summary for CI artifacts and human triage. Does not replace archive
validation; surfaces merge-gate status and per-chain metrics in one view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .combined_nightly_archive import validate_serving_combined_nightly_archive


@dataclass(frozen=True)
class ServingDashboardRow:
    section: str
    functional_chain: str
    parity_ok: bool
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section": self.section,
            "functional_chain": self.functional_chain,
            "parity_ok": self.parity_ok,
            "metrics": dict(self.metrics),
        }


@dataclass
class ServingDashboardReport:
    version: str
    archive_version: str
    parity_ok: bool
    merge_gate_ok: bool
    quick: bool
    functional_chains: List[str] = field(default_factory=list)
    rows: List[ServingDashboardRow] = field(default_factory=list)
    native_availability: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serving_dashboard": True,
            "version": self.version,
            "archive_version": self.archive_version,
            "parity_ok": self.parity_ok,
            "merge_gate_ok": self.merge_gate_ok,
            "quick": self.quick,
            "functional_chains": list(self.functional_chains),
            "native_availability": dict(self.native_availability),
            "rows": [r.to_dict() for r in self.rows],
        }


def validate_serving_dashboard(payload: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if not payload.get("serving_dashboard"):
        errors.append("missing serving_dashboard=true marker")
    if not isinstance(payload.get("version"), str) or not payload.get("version"):
        errors.append("missing dashboard version")
    if not isinstance(payload.get("archive_version"), str):
        errors.append("missing archive_version")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < 1:
        errors.append("rows must be a non-empty list")
    if payload.get("merge_gate_ok") is not True:
        errors.append("merge_gate_ok must be true when dashboard built from valid archive")
    return errors


def _row_from_subsection(
    section: str,
    functional_chain: str,
    subsection: Optional[Mapping[str, Any]],
    *,
    parity_keys: tuple[str, ...] = ("parity_ok",),
    metric_keys: tuple[str, ...] = (),
) -> ServingDashboardRow:
    if not isinstance(subsection, dict):
        return ServingDashboardRow(
            section=section,
            functional_chain=functional_chain,
            parity_ok=False,
            metrics={"present": False},
        )
    parity_ok = all(subsection.get(k) is True for k in parity_keys)
    metrics: Dict[str, Any] = {"present": True}
    for key in metric_keys:
        if key in subsection:
            metrics[key] = subsection[key]
    return ServingDashboardRow(
        section=section,
        functional_chain=functional_chain,
        parity_ok=parity_ok,
        metrics=metrics,
    )


def build_serving_dashboard_from_combined_archive(
    archive: Mapping[str, Any],
    *,
    version: str = "s36",
    allow_partial: bool = False,
) -> ServingDashboardReport:
    """Build dashboard from combined nightly archive payload."""
    errors = validate_serving_combined_nightly_archive(dict(archive))
    if allow_partial:
        errors = [
            e
            for e in errors
            if not e.endswith("subsection must be a dict")
            and "functional_chains must" not in e
        ]
    merge_gate_ok = len(errors) == 0

    decode = archive.get("decode") if isinstance(archive.get("decode"), dict) else None
    engine_g1 = archive.get("engine_g1") if isinstance(archive.get("engine_g1"), dict) else None
    multistep = archive.get("multistep") if isinstance(archive.get("multistep"), dict) else None
    engine_multistep = (
        archive.get("engine_multistep") if isinstance(archive.get("engine_multistep"), dict) else None
    )

    rows: List[ServingDashboardRow] = []
    if decode is not None:
        rows.append(
            _row_from_subsection(
                "decode",
                "chain_b_decode_step",
                decode,
                metric_keys=("speedup_yirage_vs_native", "serving_search_tier", "max_rf_mlp_layers"),
            )
        )
    if engine_g1 is not None:
        rows.append(
            _row_from_subsection(
                "engine_g1",
                "chain_c_d_engine_g1",
                engine_g1,
                metric_keys=("native_parity_ok", "vllm_native_available", "sglang_native_available"),
            )
        )
    if multistep is not None:
        rows.append(
            _row_from_subsection(
                "multistep",
                "chain_b_multistep_generation",
                multistep,
                parity_keys=("parity_ok", "token_match_ok"),
                metric_keys=("max_new_tokens", "mlp_backend", "yirage_core_used"),
            )
        )
    if engine_multistep is not None:
        rows.append(
            _row_from_subsection(
                "engine_multistep",
                "chain_c_d_engine_multistep",
                engine_multistep,
                metric_keys=("decode_steps", "native_parity_ok"),
            )
        )

    native_availability = {
        "vllm": bool((engine_g1 or {}).get("vllm_native_available")),
        "sglang": bool((engine_g1 or {}).get("sglang_native_available")),
    }

    parity_ok = bool(archive.get("parity_ok")) and all(r.parity_ok for r in rows)
    if not merge_gate_ok:
        parity_ok = False

    chains = archive.get("functional_chains")
    functional_chains = list(chains) if isinstance(chains, list) else []

    return ServingDashboardReport(
        version=version,
        archive_version=str(archive.get("version") or ""),
        parity_ok=parity_ok,
        merge_gate_ok=merge_gate_ok,
        quick=bool(archive.get("quick")),
        functional_chains=functional_chains,
        rows=rows,
        native_availability=native_availability,
    )


def render_serving_dashboard_markdown(report: ServingDashboardReport) -> str:
    """Render dashboard as markdown for CI logs / artifact README."""
    lines = [
        "# Serving Loop Dashboard",
        "",
        f"- **Dashboard version**: `{report.version}`",
        f"- **Archive version**: `{report.archive_version}`",
        f"- **Merge gate**: {'PASS' if report.merge_gate_ok else 'FAIL'}",
        f"- **Overall parity**: {'PASS' if report.parity_ok else 'FAIL'}",
        f"- **Quick mode**: `{report.quick}`",
        "",
        "## Native availability",
        "",
        f"- vLLM: `{report.native_availability.get('vllm', False)}`",
        f"- SGLang: `{report.native_availability.get('sglang', False)}`",
        "",
        "## Subsections",
        "",
        "| Section | Chain | Parity | Key metrics |",
        "|---------|-------|--------|-------------|",
    ]
    for row in report.rows:
        metrics_str = ", ".join(f"{k}={v}" for k, v in row.metrics.items() if k != "present")
        status = "PASS" if row.parity_ok else "FAIL"
        lines.append(
            f"| `{row.section}` | `{row.functional_chain}` | {status} | {metrics_str or '—'} |"
        )
    if report.functional_chains:
        lines.extend(["", "## Functional chains", ""])
        for chain in report.functional_chains:
            lines.append(f"- `{chain}`")
    lines.append("")
    return "\n".join(lines)


def load_combined_archive(path: str) -> Dict[str, Any]:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("combined archive root must be a JSON object")
    return payload
