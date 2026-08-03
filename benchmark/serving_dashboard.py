#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S36/S39: Unified Serving dashboard CLI (from combined archive or G1-only smoke)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT / "python", _REPO_ROOT / "tests" / "python", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _g1_only_archive() -> dict:
    from yirage.serving.engine_g1_regression import run_engine_g1_regression
    from yirage.serving.torch_exec import require_torch

    require_torch()
    g1 = run_engine_g1_regression(quick=True, try_native=False, version="s38").to_dict()
    return {
        "serving_combined_nightly_archive": True,
        "version": "s38",
        "parity_ok": g1.get("parity_ok"),
        "quick": True,
        "functional_chains": ["chain_c_vllm_torch", "chain_d_sglang_torch"],
        "decode": None,
        "engine_g1": g1,
        "multistep": None,
        "engine_multistep": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="", help="Combined nightly archive JSON path")
    parser.add_argument("--g1-only", action="store_true", help="Build smoke archive from G1 regression")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        load_combined_archive,
        render_serving_dashboard_html,
        render_serving_dashboard_markdown,
    )

    if args.g1_only:
        archive = _g1_only_archive()
        report = build_serving_dashboard_from_combined_archive(archive, allow_partial=True)
    elif args.archive:
        archive = load_combined_archive(args.archive)
        report = build_serving_dashboard_from_combined_archive(archive)
    else:
        parser.error("Provide --archive PATH or --g1-only")

    payload = report.to_dict()
    if args.markdown:
        print(render_serving_dashboard_markdown(report))
    if args.html:
        print(render_serving_dashboard_html(report))
    if args.json or (not args.markdown and not args.html):
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
