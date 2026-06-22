#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Simulate downloaded MLIR CI bundle regression validate (Loop R82).

Usage:
  PYTHONPATH=. python3 scripts/regression_validate_mlir_ci_bundle.py \
    --source artifacts/mlir-ci-RUN --dest artifacts/downloaded-regression-mlir
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Built MLIR CI bundle directory")
    parser.add_argument(
        "--dest",
        required=True,
        help="Simulated download destination directory",
    )
    args = parser.parse_args()

    from scripts.cpu_mlir_bench_utils import simulate_downloaded_mlir_ci_bundle_validate

    errors = simulate_downloaded_mlir_ci_bundle_validate(args.source, args.dest)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"ok": True, "dest": args.dest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
