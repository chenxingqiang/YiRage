#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S41: Validate serving dashboard artifact bundle CLI smoke."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT / "python", _REPO_ROOT / "tests" / "python", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", default="", help="Write validation summary JSON")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        render_serving_dashboard_html,
        render_serving_dashboard_markdown,
    )

    report = build_serving_dashboard_from_combined_archive(_synthetic_combined_for_dashboard())
    tmp = Path("/tmp/serving-dashboard-s41-smoke")
    tmp.mkdir(parents=True, exist_ok=True)
    json_path = tmp / "dashboard.json"
    html_path = tmp / "dashboard.html"
    md_path = tmp / "dashboard.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_serving_dashboard_html(report), encoding="utf-8")
    md_path.write_text(render_serving_dashboard_markdown(report), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "validate_serving_dashboard.py"),
            str(json_path),
            "--html",
            str(html_path),
            "--markdown",
            str(md_path),
        ],
        cwd=str(_REPO_ROOT),
        env={"PYTHONPATH": "python", "YIRAGE_BACKEND": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    summary = json.loads(proc.stdout) if proc.stdout.strip() else {"ok": False, "stderr": proc.stderr}
    if proc.returncode != 0:
        print(json.dumps(summary, indent=2))
        return proc.returncode

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
