#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Render unified Serving dashboard from combined nightly archive JSON (S36).

Usage::

    python3 scripts/render_serving_dashboard.py artifacts/serving-combined-nightly.json \\
        --json --markdown-output artifacts/serving-dashboard.md \\
        --output artifacts/serving-dashboard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="Path to combined nightly archive JSON")
    parser.add_argument("--output", default="", help="Write dashboard JSON to path")
    parser.add_argument("--markdown-output", default="", help="Write markdown summary to path")
    parser.add_argument("--json", action="store_true", help="Print dashboard JSON to stdout")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow G1-only / partial combined archives",
    )
    args = parser.parse_args()

    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        load_combined_archive,
        render_serving_dashboard_markdown,
        validate_serving_dashboard,
    )

    archive = load_combined_archive(args.archive)
    report = build_serving_dashboard_from_combined_archive(
        archive, allow_partial=args.allow_partial
    )
    payload = report.to_dict()
    errors = validate_serving_dashboard(payload)
    if errors and not args.allow_partial:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        md = render_serving_dashboard_markdown(report)
        Path(args.markdown_output).write_text(md, encoding="utf-8")
    if args.json or (not args.output and not args.markdown_output):
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
