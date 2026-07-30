#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Generate Serving search-tier bench archives for CI/nightly compare (S26).

Usage::

    export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
    export YIRAGE_BACKEND=cpu PYTHONPATH=python:tests/python
    python3 scripts/serving_search_tier_archive.py --json --quick --output artifacts/serving-tier-seed_verify.json
    python3 scripts/serving_search_tier_archive.py --json --quick --tier seed_verify --output artifacts/tier.json
    python3 scripts/serving_search_tier_archive.py --compare baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[1]
    pkg_root = root / "python"
    tests_root = root / "tests" / "python"
    yirage_dir = pkg_root / "yirage"
    for path in (str(pkg_root), str(tests_root), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    if "yirage" not in sys.modules or not hasattr(sys.modules.get("yirage"), "core"):
        try:
            import yirage as yr  # noqa: F401
        except ImportError:
            stub = types.ModuleType("yirage")
            stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
            sys.modules["yirage"] = stub
    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for sub in ("build/abstract_subexpr/release", "build/formal_verifier/release"):
        path = root / sub
        if path.exists() and str(path) not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{ld}"
            ld = os.environ["LD_LIBRARY_PATH"]


def main(argv: list[str] | None = None) -> int:
    _bootstrap()
    from yirage.serving.search_tier_archive import (
        compare_serving_search_tier_archives,
        load_serving_bench_archive,
        run_serving_multi_tier_bench_archive,
        run_serving_search_tier_bench_archive_for_preset,
        serving_search_tier_preset_names,
        validate_serving_search_tier_archive,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    parser.add_argument("--quick", action="store_true", help="Quick shapes (1 RF layer)")
    parser.add_argument(
        "--tier",
        choices=serving_search_tier_preset_names(),
        default="seed_verify",
        help="Single-tier preset (default seed_verify)",
    )
    parser.add_argument(
        "--multi-tier",
        action="store_true",
        help="Run all tier presets and emit combined multi-tier archive",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write archive JSON to this path",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="Compare two archive JSON files",
    )
    args = parser.parse_args(argv)

    if args.compare:
        baseline = load_serving_bench_archive(args.compare[0])
        candidate = load_serving_bench_archive(args.compare[1])
        report = compare_serving_search_tier_archives(baseline, candidate)
        if args.json or not args.output:
            print(json.dumps(report, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0 if report.get("ok") else 1

    run_kwargs = dict(quick=args.quick, archive_version="s29")
    if args.multi_tier:
        multi = run_serving_multi_tier_bench_archive(
            tier_names=serving_search_tier_preset_names(),
            **run_kwargs,
        )
        payload = multi.to_dict()
        errors = validate_serving_search_tier_archive(payload)
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, indent=2), file=sys.stderr)
            return 1
    else:
        _report, archive = run_serving_search_tier_bench_archive_for_preset(
            args.tier,
            **run_kwargs,
        )
        payload = archive.to_dict()
        errors = validate_serving_search_tier_archive(payload)
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, indent=2), file=sys.stderr)
            return 1

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    elif not args.output:
        print(json.dumps({"ok": True, "tier": payload.get("search_tier", {}).get("tier")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
