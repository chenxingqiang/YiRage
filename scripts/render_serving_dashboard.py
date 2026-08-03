#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Render unified Serving dashboard from combined nightly archive JSON (S36/S39).

Usage::

    python3 scripts/render_serving_dashboard.py artifacts/serving-combined-nightly.json \\
        --json --markdown-output artifacts/serving-dashboard.md \\
        --html-output artifacts/serving-dashboard.html \\
        --output artifacts/serving-dashboard.json
"""

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
        for key in list(sys.modules):
            if key == "yirage.serving" or key.startswith("yirage.serving."):
                del sys.modules[key]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="Path to combined nightly archive JSON")
    parser.add_argument("--output", default="", help="Write dashboard JSON to path")
    parser.add_argument("--markdown-output", default="", help="Write markdown summary to path")
    parser.add_argument("--html-output", default="", help="Write HTML summary to path")
    parser.add_argument("--json", action="store_true", help="Print dashboard JSON to stdout")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow G1-only / partial combined archives",
    )
    args = parser.parse_args()

    _install_yirage_stub()
    dashboard = importlib.import_module("yirage.serving.serving_dashboard")

    archive = dashboard.load_combined_archive(args.archive)
    report = dashboard.build_serving_dashboard_from_combined_archive(
        archive, allow_partial=args.allow_partial
    )
    payload = report.to_dict()
    errors = dashboard.validate_serving_dashboard(payload)
    if errors and not args.allow_partial:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        md = dashboard.render_serving_dashboard_markdown(report)
        Path(args.markdown_output).write_text(md, encoding="utf-8")
    if args.html_output:
        doc = dashboard.render_serving_dashboard_html(report)
        html_errors = dashboard.validate_serving_dashboard_html(doc)
        if html_errors:
            print(json.dumps({"ok": False, "html_errors": html_errors}, indent=2))
            return 1
        Path(args.html_output).write_text(doc, encoding="utf-8")
    if args.json or (not args.output and not args.markdown_output and not args.html_output):
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
