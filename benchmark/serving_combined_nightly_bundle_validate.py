#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S42: Validate combined nightly archive + dashboard bundle CLI smoke."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT / "python", _REPO_ROOT / "tests" / "python", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _install_yirage_stub() -> None:
    yirage_dir = _REPO_ROOT / "python" / "yirage"
    if "yirage" not in sys.modules or not hasattr(sys.modules.get("yirage"), "core"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", default="", help="Write validation summary JSON")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    _install_yirage_stub()

    from test_runtime_fusion_s42_nightly_bundle_validate import _synthetic_bundle_artifacts

    bundle = _synthetic_bundle_artifacts()
    tmp = Path("/tmp/serving-nightly-bundle-s42-smoke")
    tmp.mkdir(parents=True, exist_ok=True)
    paths = {
        "archive": tmp / "archive.json",
        "archive_meta": tmp / "archive.meta.json",
        "dashboard": tmp / "dashboard.json",
        "dashboard_meta": tmp / "dashboard.meta.json",
        "html": tmp / "dashboard.html",
        "markdown": tmp / "dashboard.md",
        "bundle_meta": tmp / "bundle.meta.json",
    }
    paths["archive"].write_text(json.dumps(bundle["archive"], indent=2) + "\n", encoding="utf-8")
    paths["archive_meta"].write_text(json.dumps(bundle["archive_meta"], indent=2) + "\n", encoding="utf-8")
    paths["dashboard"].write_text(json.dumps(bundle["dashboard"], indent=2) + "\n", encoding="utf-8")
    paths["dashboard_meta"].write_text(json.dumps(bundle["dashboard_meta"], indent=2) + "\n", encoding="utf-8")
    paths["html"].write_text(bundle["html"], encoding="utf-8")
    paths["markdown"].write_text(bundle["markdown"], encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "validate_serving_combined_nightly_bundle.py"),
            str(paths["archive"]),
            "--archive-meta",
            str(paths["archive_meta"]),
            "--dashboard",
            str(paths["dashboard"]),
            "--dashboard-meta",
            str(paths["dashboard_meta"]),
            "--html",
            str(paths["html"]),
            "--markdown",
            str(paths["markdown"]),
            "--metadata-output",
            str(paths["bundle_meta"]),
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
