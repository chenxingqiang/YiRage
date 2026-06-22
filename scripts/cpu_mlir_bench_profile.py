#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Archive MLIR JIT bench JSON with contract validation (Loop R64/R65).

Usage:
  export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
  export YIRAGE_BACKEND=cpu
  PYTHONPATH=. python3 scripts/cpu_mlir_bench_profile.py --json
  PYTHONPATH=. python3 scripts/cpu_mlir_bench_profile.py --json --output artifacts/mlir-bench-profile.json
  PYTHONPATH=. python3 scripts/cpu_mlir_bench_profile.py --json --skip-rms-mlir
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
    parser.add_argument("--json", action="store_true", help="Emit JSON archive")
    parser.add_argument(
        "--skip-rms-mlir",
        action="store_true",
        help="Archive concat_matmul deferred contract only (no --mlir-jit on rms)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write JSON report to file (in addition to stdout markers when --json)",
    )
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    from scripts.cpu_mlir_bench_utils import run_mlir_bench_profile

    stage = run_mlir_bench_profile(skip_rms_mlir=args.skip_rms_mlir)
    report = {
        "backend": "cpu",
        "mode": "mlir_bench_profile",
        "stage": {
            "ok": stage["ok"],
            "elapsed_s": stage["elapsed_s"],
            "returncode": stage["returncode"],
            "rows": stage["rows"],
        },
        "profile": stage["profile"],
        "ok": stage["ok"],
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")

    if args.json:
        print("YIRAGE_MLIR_BENCH_PROFILE_JSON_BEGIN")
        print(json.dumps(report, indent=2))
        print("YIRAGE_MLIR_BENCH_PROFILE_JSON_END", flush=True)
    else:
        print(json.dumps(stage["profile"], indent=2))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
