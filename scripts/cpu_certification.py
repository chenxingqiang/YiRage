#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
CPU certification entrypoint: op contract tests + capability walkthrough summary.

Usage:
  export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
  export YIRAGE_BACKEND=cpu
  PYTHONPATH=. python3 scripts/cpu_certification.py
  PYTHONPATH=. python3 scripts/cpu_certification.py --json
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
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *markers,
        "--tb=no",
    ]
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
        "stdout_tail": proc.stdout[-1500:],
        "stderr_tail": proc.stderr[-1500:],
        "returncode": proc.returncode,
        "pytest": stats,
    }


def _matrix_summary() -> Dict[str, Any]:
    from yirage.backends.cpu.support_matrix import (
        cpu_layout_explore_gap_meta,
        cpu_layout_explore_gap_table,
        cpu_search_explore_not_supported,
        cpu_supported_kn_ops,
        cpu_unsupported_kn_ops,
        kn_op_contracts,
    )

    contracts = kn_op_contracts()
    tiers: Dict[str, int] = {}
    for c in contracts.values():
        tiers[c.tier] = tiers.get(c.tier, 0) + 1
    return {
        "kn_tiers": tiers,
        "kn_supported_count": len(cpu_supported_kn_ops()),
        "kn_unsupported_count": len(cpu_unsupported_kn_ops()),
        "search_explore_gaps": cpu_search_explore_not_supported(),
        "layout_explore_chunk_gaps": cpu_layout_explore_gap_table(),
        "layout_explore_gap_meta": cpu_layout_explore_gap_meta(),
    }


def _run_walkthrough_stage(*, quick: bool) -> Dict[str, Any]:
    from scripts.business_capability_walkthrough import (
        build_walkthrough_report,
        walkthrough_report_to_dict,
    )

    t0 = time.perf_counter()
    report = build_walkthrough_report(quick=quick)
    payload = walkthrough_report_to_dict(report)
    return {
        "ok": payload["ok"],
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "returncode": 0 if payload["ok"] else 1,
        "walkthrough_substage_elapsed_s": payload["walkthrough_substage_elapsed_s"],
        "walkthrough": {
            "stages": payload["stages"],
            "business_scores": payload["business_scores"],
            "verdict": payload["verdict"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument(
        "--skip-walkthrough",
        action="store_true",
        help="Skip business_capability_walkthrough (faster)",
    )
    parser.add_argument(
        "--walkthrough-profile",
        action="store_true",
        help="Run walkthrough only (quick tractability) and emit profile JSON",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Skip walkthrough and superoptimize smoke "
            "(contract + value verify + native_gemm + profile)"
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    from scripts.cpu_cert_utils import cert_inventory_summary, cert_profile_from_stages

    if args.walkthrough_profile:
        report: Dict[str, Any] = {
            "backend": "cpu",
            "mode": "walkthrough_profile",
            "inventory": cert_inventory_summary(),
            "stages": {"walkthrough": _run_walkthrough_stage(quick=True)},
        }
        planned = report["inventory"]["planned_value_verify_count"]
        report["profile"] = cert_profile_from_stages(
            report["stages"],
            planned_value_verify=planned,
        )
        report["ok"] = report["stages"]["walkthrough"]["ok"]
        if args.json:
            print("YIRAGE_CPU_CERT_JSON_BEGIN")
            print(json.dumps(report, indent=2))
            print("YIRAGE_CPU_CERT_JSON_END", flush=True)
        else:
            print(json.dumps(report["profile"], indent=2))
        return 0 if report["ok"] else 1

    mode = "quick" if args.quick else "full"
    report: Dict[str, Any] = {
        "backend": "cpu",
        "mode": mode,
        "inventory": cert_inventory_summary(),
        "matrix": _matrix_summary(),
        "stages": {},
    }

    report["stages"]["value_verify_all"] = _run_pytest(
        ["tests/integration/test_cpu_full_value_verify.py"],
        quiet=False,
    )
    report["stages"]["op_contract"] = _run_pytest(
        ["tests/integration/test_cpu_op_contract.py"]
    )
    report["stages"]["native_gemm"] = _run_pytest(
        ["tests/integration/test_cpu_native_gemm.py", "-k", "not fused_rms_matmul_near"]
    )
    if not args.quick:
        report["stages"]["superoptimize_smoke"] = _run_pytest(
            ["tests/integration/test_cpu_superoptimize_value.py"]
        )
        from scripts.cpu_mlir_bench_utils import run_mlir_bench_profile

        report["stages"]["mlir_bench_profile"] = run_mlir_bench_profile()

    skip_walkthrough = args.skip_walkthrough or args.quick
    if not skip_walkthrough:
        report["stages"]["walkthrough"] = _run_walkthrough_stage(quick=True)

    report["ok"] = all(s.get("ok") for s in report["stages"].values())

    planned = report["inventory"]["planned_value_verify_count"]
    report["profile"] = cert_profile_from_stages(
        report["stages"],
        planned_value_verify=planned,
    )

    if args.json:
        print("YIRAGE_CPU_CERT_JSON_BEGIN")
        print(json.dumps(report, indent=2))
        print("YIRAGE_CPU_CERT_JSON_END", flush=True)
    else:
        print("=" * 60)
        print("YiRage CPU Certification")
        print("=" * 60)
        print(json.dumps(report["matrix"], indent=2))
        print(json.dumps(report.get("profile", {}), indent=2))
        for name, stage in report["stages"].items():
            status = "OK" if stage.get("ok") else "FAIL"
            print(f"[{status}] {name} ({stage.get('elapsed_s', '?')}s)")
        print("=" * 60)
        print("OVERALL:", "PASS" if report["ok"] else "FAIL")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
