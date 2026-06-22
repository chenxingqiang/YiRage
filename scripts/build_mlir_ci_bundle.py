#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Build MLIR CI artifact bundle (shared by PR and nightly workflows, Loop R82).

Usage:
  PYTHONPATH=. python3 scripts/build_mlir_ci_bundle.py \
    --bundle-dir artifacts/mlir-ci-RUN --workflow cpu-mlir-jit-contract \
    --run-id RUN --sha SHA --with-dialect-smoke
  PYTHONPATH=. python3 scripts/build_mlir_ci_bundle.py \
    --bundle-dir DIR --workflow cpu-mlir-ci-nightly --manifest-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _run_dialect_smoke_log(bundle_dir: str) -> None:
    log_path = os.path.join(bundle_dir, "mlir-dialect-smoke.log")
    os.makedirs(bundle_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            ["make", "test-cpu-mlir-dialect-smoke"],
            cwd=_REPO,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _run_bench_profile(bundle_dir: str) -> str:
    os.makedirs(bundle_dir, exist_ok=True)
    profile_path = os.path.join(bundle_dir, "mlir-bench-profile.json")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/cpu_mlir_bench_profile.py",
            "--json",
            "--output",
            profile_path,
        ],
        cwd=_REPO,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return profile_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, help="Output bundle directory")
    parser.add_argument(
        "--workflow",
        required=True,
        choices=["cpu-mlir-jit-contract", "cpu-mlir-ci-nightly"],
    )
    parser.add_argument("--run-id", help="GitHub run id for manifest")
    parser.add_argument("--sha", help="Git commit sha for manifest")
    parser.add_argument(
        "--with-dialect-smoke",
        action="store_true",
        help="Run dialect smoke and write mlir-dialect-smoke.log",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write bundle-manifest.json (profile + log must exist)",
    )
    args = parser.parse_args()

    from scripts.cpu_mlir_bench_utils import write_mlir_ci_bundle_manifest

    if not args.manifest_only:
        if args.with_dialect_smoke:
            _run_dialect_smoke_log(args.bundle_dir)
        _run_bench_profile(args.bundle_dir)

    manifest_path = write_mlir_ci_bundle_manifest(
        args.bundle_dir,
        workflow=args.workflow,
        run_id=args.run_id,
        sha=args.sha,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "bundle_dir": args.bundle_dir,
                "manifest_path": manifest_path,
                "workflow": args.workflow,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
