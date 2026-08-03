#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Validate Serving dashboard artifact bundle (JSON + optional HTML/markdown) — S41."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))


def _install_yirage_stub() -> None:
    yirage_dir = _REPO / "python" / "yirage"
    if "yirage" not in sys.modules or not hasattr(sys.modules.get("yirage"), "core"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="Path to serving-dashboard.json")
    parser.add_argument("--html", default="", help="Optional serving-dashboard.html path")
    parser.add_argument("--markdown", default="", help="Optional serving-dashboard.md path")
    parser.add_argument(
        "--metadata-output",
        metavar="PATH",
        help="Write artifact sidecar metadata JSON after successful validation",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow merge_gate_ok=false (partial/G1-only archives)",
    )
    args = parser.parse_args()

    _install_yirage_stub()
    dashboard = importlib.import_module("yirage.serving.serving_dashboard")

    json_path = Path(args.json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    html_doc = Path(args.html).read_text(encoding="utf-8") if args.html else None
    md_doc = Path(args.markdown).read_text(encoding="utf-8") if args.markdown else None

    errors = dashboard.validate_serving_dashboard_artifact_bundle(
        json_payload=payload,
        html_document=html_doc,
        markdown_document=md_doc,
    )
    if args.allow_partial:
        errors = [e for e in errors if e != "merge_gate_ok must be true when dashboard built from valid archive"]

    html_ok = None if html_doc is None else not any(e.startswith("html.") for e in errors)
    markdown_ok = None if md_doc is None else not any(e.startswith("markdown.") for e in errors)

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    summary: dict = {
        "ok": True,
        "parity_ok": payload.get("parity_ok"),
        "merge_gate_ok": payload.get("merge_gate_ok"),
    }
    if args.metadata_output:
        metadata = dashboard.serving_dashboard_artifact_metadata(
            payload,
            json_path=str(json_path),
            validation_ok=True,
            html_path=args.html,
            markdown_path=args.markdown,
            html_ok=html_ok,
            markdown_ok=markdown_ok,
        )
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summary["metadata_output"] = args.metadata_output
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
