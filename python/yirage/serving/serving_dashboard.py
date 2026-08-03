# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S36/S39: Unified Serving dashboard — render combined nightly archive (G7 closure).

Consumes ``serving_combined_nightly_archive`` JSON and produces a compact dashboard
JSON + markdown + HTML summary for CI artifacts and human triage. Does not replace
archive validation; surfaces merge-gate status and per-chain metrics in one view.
"""

from __future__ import annotations

import html
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
    version: str = "s41",
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
    paged_multistep = (
        archive.get("paged_multistep") if isinstance(archive.get("paged_multistep"), dict) else None
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
    if paged_multistep is not None:
        rows.append(
            _row_from_subsection(
                "paged_multistep",
                "chain_c_vllm_paged_multistep",
                paged_multistep,
                parity_keys=("parity_ok", "token_match_ok"),
                metric_keys=("decode_steps", "paged_kv_bridged", "native_parity_ok", "vllm_native_available"),
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


def _dashboard_status_badge(ok: bool) -> str:
    label = "PASS" if ok else "FAIL"
    css = "pass" if ok else "fail"
    return f'<span class="badge {css}">{label}</span>'


def _dashboard_avail_badge(available: bool) -> str:
    label = "yes" if available else "no"
    css = "pass" if available else "neutral"
    return f'<span class="badge {css}">{label}</span>'


def render_serving_dashboard_html(report: ServingDashboardReport) -> str:
    """Render dashboard as self-contained HTML for CI artifact browsing."""
    merge_badge = _dashboard_status_badge(report.merge_gate_ok)
    parity_badge = _dashboard_status_badge(report.parity_ok)
    vllm_avail = report.native_availability.get("vllm", False)
    sglang_avail = report.native_availability.get("sglang", False)

    row_html: List[str] = []
    for row in report.rows:
        metrics_str = ", ".join(
            f"{html.escape(str(k))}={html.escape(str(v))}"
            for k, v in row.metrics.items()
            if k != "present"
        )
        row_html.append(
            "<tr>"
            f"<td><code>{html.escape(row.section)}</code></td>"
            f"<td><code>{html.escape(row.functional_chain)}</code></td>"
            f"<td>{_dashboard_status_badge(row.parity_ok)}</td>"
            f"<td>{metrics_str if metrics_str else '—'}</td>"
            "</tr>"
        )

    chains_html = "".join(
        f"<li><code>{html.escape(str(chain))}</code></li>"
        for chain in report.functional_chains
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Serving Loop Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .meta {{ color: #555; margin-bottom: 1.5rem; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; min-width: 10rem; }}
    .card h2 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; margin: 0 0 0.5rem; color: #666; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }}
    .badge.pass {{ background: #d4edda; color: #155724; }}
    .badge.fail {{ background: #f8d7da; color: #721c24; }}
    .badge.neutral {{ background: #e9ecef; color: #495057; }}
    code {{ font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Serving Loop Dashboard</h1>
  <p class="meta">Dashboard <code>{html.escape(report.version)}</code> · Archive <code>{html.escape(report.archive_version)}</code> · Quick mode <code>{html.escape(str(report.quick))}</code></p>
  <div class="cards">
    <div class="card"><h2>Merge gate</h2>{merge_badge}</div>
    <div class="card"><h2>Overall parity</h2>{parity_badge}</div>
    <div class="card"><h2>vLLM native</h2>{_dashboard_avail_badge(vllm_avail)}</div>
    <div class="card"><h2>SGLang native</h2>{_dashboard_avail_badge(sglang_avail)}</div>
  </div>
  <h2>Subsections</h2>
  <table>
    <thead><tr><th>Section</th><th>Chain</th><th>Parity</th><th>Key metrics</th></tr></thead>
    <tbody>
      {''.join(row_html)}
    </tbody>
  </table>
  <h2>Functional chains</h2>
  <ul>{chains_html}</ul>
</body>
</html>
"""


def validate_serving_dashboard_markdown(document: str) -> List[str]:
    """Lightweight contract checks for rendered markdown artifacts."""
    errors: List[str] = []
    if not document.strip():
        errors.append("markdown document is empty")
        return errors
    required = ("# Serving Loop Dashboard", "## Subsections", "| Section |")
    for marker in required:
        if marker not in document:
            errors.append(f"markdown missing marker: {marker!r}")
    return errors


def validate_serving_dashboard_artifact_bundle(
    *,
    json_payload: Mapping[str, Any],
    html_document: Optional[str] = None,
    markdown_document: Optional[str] = None,
) -> List[str]:
    """Validate dashboard JSON plus optional HTML/markdown sibling artifacts."""
    errors: List[str] = list(validate_serving_dashboard(json_payload))
    version = json_payload.get("version")
    archive_version = json_payload.get("archive_version")
    if html_document is not None:
        errors.extend(f"html.{e}" for e in validate_serving_dashboard_html(html_document))
        if isinstance(version, str) and version and html.escape(version) not in html_document:
            errors.append("html version mismatch with dashboard json")
        if (
            isinstance(archive_version, str)
            and archive_version
            and html.escape(archive_version) not in html_document
        ):
            errors.append("html archive_version mismatch with dashboard json")
    if markdown_document is not None:
        errors.extend(f"markdown.{e}" for e in validate_serving_dashboard_markdown(markdown_document))
        if isinstance(version, str) and version and f"`{version}`" not in markdown_document:
            errors.append("markdown version mismatch with dashboard json")
    return errors


def serving_dashboard_artifact_metadata(
    payload: Mapping[str, Any],
    *,
    json_path: str,
    validation_ok: bool,
    html_path: str = "",
    markdown_path: str = "",
    html_ok: Optional[bool] = None,
    markdown_ok: Optional[bool] = None,
) -> Dict[str, Any]:
    import hashlib
    import json
    import time

    raw = json.dumps(payload, sort_keys=True, default=str)
    return {
        "serving_dashboard_artifact_metadata": True,
        "json_path": json_path,
        "html_path": html_path or None,
        "markdown_path": markdown_path or None,
        "validation_ok": validation_ok,
        "html_ok": html_ok,
        "markdown_ok": markdown_ok,
        "version": payload.get("version"),
        "archive_version": payload.get("archive_version"),
        "parity_ok": payload.get("parity_ok"),
        "merge_gate_ok": payload.get("merge_gate_ok"),
        "quick": payload.get("quick"),
        "row_count": len(payload.get("rows") or []),
        "json_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "created_unix": time.time(),
    }


def validate_serving_dashboard_html(document: str) -> List[str]:
    """Lightweight contract checks for rendered HTML artifacts."""
    errors: List[str] = []
    if not document.strip():
        errors.append("html document is empty")
        return errors
    required = (
        "<!DOCTYPE html>",
        "Serving Loop Dashboard",
        "<table>",
        'class="badge pass"',
        "Functional chains",
    )
    for marker in required:
        if marker not in document:
            errors.append(f"html missing marker: {marker!r}")
    return errors


def load_combined_archive(path: str) -> Dict[str, Any]:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("combined archive root must be a JSON object")
    return payload
