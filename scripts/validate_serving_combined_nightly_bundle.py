#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Validate combined nightly archive + dashboard artifact bundle (S42)."""

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
    parser.add_argument("archive", help="Path to serving-combined-nightly.json")
    parser.add_argument("--archive-meta", required=True, help="Combined archive metadata JSON")
    parser.add_argument("--dashboard", required=True, help="Dashboard JSON path")
    parser.add_argument("--dashboard-meta", required=True, help="Dashboard metadata JSON")
    parser.add_argument("--html", default="", help="Optional dashboard HTML path")
    parser.add_argument("--markdown", default="", help="Optional dashboard markdown path")
    parser.add_argument("--metadata-output", default="", help="Write bundle metadata JSON")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow G1-only / partial combined archives",
    )
    args = parser.parse_args()

    _install_yirage_stub()
    bundle = importlib.import_module("yirage.serving.serving_nightly_bundle")

    archive_payload = json.loads(Path(args.archive).read_text(encoding="utf-8"))
    archive_metadata = json.loads(Path(args.archive_meta).read_text(encoding="utf-8"))
    dashboard_payload = json.loads(Path(args.dashboard).read_text(encoding="utf-8"))
    dashboard_metadata = json.loads(Path(args.dashboard_meta).read_text(encoding="utf-8"))
    html_doc = Path(args.html).read_text(encoding="utf-8") if args.html else None
    md_doc = Path(args.markdown).read_text(encoding="utf-8") if args.markdown else None

    errors = bundle.validate_serving_combined_nightly_bundle(
        archive_payload=archive_payload,
        archive_metadata=archive_metadata,
        dashboard_payload=dashboard_payload,
        dashboard_metadata=dashboard_metadata,
        html_document=html_doc,
        markdown_document=md_doc,
        allow_partial=args.allow_partial,
    )
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    summary: dict = {
        "ok": True,
        "parity_ok": archive_payload.get("parity_ok"),
        "version": archive_payload.get("version"),
    }
    if args.metadata_output:
        metadata = bundle.serving_combined_nightly_bundle_metadata(
            archive_payload=archive_payload,
            archive_metadata=archive_metadata,
            dashboard_metadata=dashboard_metadata,
            validation_ok=True,
            archive_path=args.archive,
            archive_meta_path=args.archive_meta,
            dashboard_json_path=args.dashboard,
            dashboard_meta_path=args.dashboard_meta,
            dashboard_html_path=args.html,
            dashboard_markdown_path=args.markdown,
        )
        meta_errors = bundle.validate_serving_combined_nightly_bundle_metadata(metadata)
        if meta_errors:
            print(json.dumps({"ok": False, "errors": meta_errors}, indent=2))
            return 1
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summary["metadata_output"] = args.metadata_output
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
