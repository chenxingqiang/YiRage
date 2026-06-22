#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Validate an MLIR CI artifact bundle directory (Loop R72).

Usage:
  PYTHONPATH=. python3 scripts/validate_mlir_ci_bundle.py artifacts/mlir-ci-<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", help="Path to mlir-ci bundle directory")
    args = parser.parse_args()

    from scripts.cpu_mlir_bench_utils import validate_mlir_ci_bundle

    errors = validate_mlir_ci_bundle(args.bundle_dir)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"ok": True, "bundle_dir": args.bundle_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
