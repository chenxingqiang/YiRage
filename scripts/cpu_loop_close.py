#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Archive CPU infinite-loop close stages as JSON (Loop R67).

Usage:
  export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
  export YIRAGE_BACKEND=cpu
  PYTHONPATH=. python3 scripts/cpu_loop_close.py --json --quick
  PYTHONPATH=. python3 scripts/cpu_loop_close.py --json
  PYTHONPATH=. python3 scripts/cpu_loop_close.py --json --output artifacts/cpu-loop-close.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _run_pytest(markers: List[str], *, quiet: bool = True) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", *markers, "--tb=no"]
    if quiet:
        cmd.append("-q")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=_REPO,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    from scripts.cpu_cert_utils import parse_pytest_summary

    stats = parse_pytest_summary(proc.stdout + proc.stderr)
    return {
        "ok": proc.returncode == 0,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "returncode": proc.returncode,
        "pytest": stats,
        "stdout_tail": proc.stdout[-1500:],
    }


def _run_cert_e2e() -> Dict[str, Any]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "scripts/cpu_certification.py", "--json"],
        cwd=_REPO,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    from scripts.cpu_cert_utils import parse_json_marker

    cert = parse_json_marker(
        proc.stdout + proc.stderr,
        "YIRAGE_CPU_CERT_JSON_BEGIN",
        "YIRAGE_CPU_CERT_JSON_END",
    )
    elapsed_s = round(time.perf_counter() - t0, 2)
    ok = proc.returncode == 0 and cert is not None and cert.get("ok", False)
    return {
        "ok": ok,
        "elapsed_s": elapsed_s,
        "returncode": proc.returncode,
        "profile": (cert or {}).get("profile"),
        "inventory": (cert or {}).get("inventory"),
        "stdout_tail": (proc.stdout + proc.stderr)[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON archive")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Archive demos + mlir profile + contract pytest (skip cert e2e)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write JSON report to file (in addition to stdout markers when --json)",
    )
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    from scripts.cpu_cert_utils import (
        cert_inventory_summary,
        cpu_loop_close_manifest,
        loop_profile_from_stages,
    )
    from scripts.cpu_mlir_bench_utils import run_mlir_bench_profile

    mode = "quick" if args.quick else "full"
    report: Dict[str, Any] = {
        "backend": "cpu",
        "mode": mode,
        "manifest": cpu_loop_close_manifest(),
        "inventory": cert_inventory_summary(),
        "stages": {},
    }

    report["stages"]["demos"] = _run_pytest(
        [
            "tests/integration/test_cpu_demos.py",
            "tests/integration/test_cpu_demo_loop.py",
            "tests/integration/test_cpu_loop_close.py",
        ],
        quiet=False,
    )
    report["stages"]["mlir_bench_profile"] = run_mlir_bench_profile(quick=args.quick)
    report["stages"]["mlir_bench_contract"] = _run_pytest(
        [
            "tests/python/test_bench_fusion_search_skip.py",
            "-k",
            "mlir_jit or concat_matmul or parse_bench or mlir_bench_profile or run_mlir_bench",
        ],
    )

    if not args.quick:
        report["stages"]["cert_e2e"] = _run_cert_e2e()

    report["ok"] = all(s.get("ok") for s in report["stages"].values())
    report["profile"] = loop_profile_from_stages(report["stages"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")

    if args.json:
        print("YIRAGE_CPU_LOOP_CLOSE_JSON_BEGIN")
        print(json.dumps(report, indent=2))
        print("YIRAGE_CPU_LOOP_CLOSE_JSON_END", flush=True)
    else:
        print(json.dumps(report["profile"], indent=2))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
