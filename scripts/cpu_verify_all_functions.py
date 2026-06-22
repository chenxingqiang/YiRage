#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Verify every CPU-backend function value against torch references.

Runs matrix-driven KN/TB op tests, native primitives, fast paths, and coverage
audit. For full certification (demos, walkthrough), use cpu_certification.py.

Usage:
  export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
  export YIRAGE_BACKEND=cpu
  PYTHONPATH=. python3 scripts/cpu_verify_all_functions.py
  PYTHONPATH=. python3 scripts/cpu_verify_all_functions.py --json
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


def _inventory() -> Dict[str, Any]:
    from tests.integration.cpu_op_builders import (
        CUSTOMIZED_OP_BUILDERS,
        FAST_PATH_BUILDERS,
        KN_OP_BUILDERS,
        LAYOUT_EXPLORE_BUILDERS,
    )
    from tests.integration.cpu_tb_op_builders import (
        TB_LAYOUT_CHUNK_DEFERRED_PATTERNS,
        TB_LAYOUT_EXPLORE_BUILDERS,
        TB_OP_BUILDERS,
        TB_UNSUPPORTED_BUILDERS,
    )
    from yirage.backends.cpu.support_matrix import (
        cpu_layout_explore_gap_meta,
        cpu_layout_explore_gap_table,
        cpu_unsupported_kn_ops,
        cpu_unsupported_tb_ops,
        cpu_verifiable_kn_ops,
        cpu_verifiable_tb_ops,
        kn_op_contracts,
        tb_op_contracts,
    )

    from tests.integration.cpu_inventory import planned_value_verify_count, registry_sizes

    kn_verify = sorted(KN_OP_BUILDERS) + sorted(FAST_PATH_BUILDERS)
    sizes = registry_sizes()
    planned = planned_value_verify_count()
    return {
        "kn_verifiable_matrix": cpu_verifiable_kn_ops(),
        "tb_verifiable_matrix": cpu_verifiable_tb_ops(),
        "kn_builders": sorted(KN_OP_BUILDERS),
        "tb_builders": sorted(TB_OP_BUILDERS),
        "fast_path_builders": sorted(FAST_PATH_BUILDERS),
        "customized_patterns": sorted(CUSTOMIZED_OP_BUILDERS),
        "tb_unsupported_builders": sorted(TB_UNSUPPORTED_BUILDERS),
        "kn_unsupported_matrix": cpu_unsupported_kn_ops(),
        "tb_unsupported_matrix": cpu_unsupported_tb_ops(),
        "kn_tiers": {
            op: kn_op_contracts()[op].tier for op in sorted(kn_op_contracts())
        },
        "tb_tiers": {
            op: tb_op_contracts()[op].tier for op in sorted(tb_op_contracts())
        },
        "layout_explore_builders": sorted(LAYOUT_EXPLORE_BUILDERS),
        "tb_layout_explore_builders": sorted(TB_LAYOUT_EXPLORE_BUILDERS),
        "tb_layout_chunk_deferred_patterns": sorted(TB_LAYOUT_CHUNK_DEFERRED_PATTERNS),
        "layout_explore_chunk_gaps": cpu_layout_explore_gap_table(),
        "layout_explore_gap_meta": cpu_layout_explore_gap_meta(),
        "registry_sizes": sizes,
        "planned_value_verify_count": planned,
        "total_value_checks": planned,
        "kn_verify_ids": kn_verify,
    }


def _run_pytest() -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/test_cpu_full_value_verify.py",
        "-q",
        "--tb=no",
    ]
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
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "returncode": proc.returncode,
        "pytest": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    report: Dict[str, Any] = {
        "backend": "cpu",
        "inventory": _inventory(),
        "value_verify": _run_pytest(),
    }
    inv = report["inventory"]
    passed = (report["value_verify"].get("pytest") or {}).get("passed")
    report["profile"] = {
        "planned_value_verify_count": inv["planned_value_verify_count"],
        "value_verify_passed": passed,
        "value_verify_aligned": passed == inv["planned_value_verify_count"]
        if passed is not None
        else None,
    }
    report["ok"] = report["value_verify"]["ok"]

    if args.json:
        print("YIRAGE_CPU_VALUE_VERIFY_JSON_BEGIN")
        print(json.dumps(report, indent=2))
        print("YIRAGE_CPU_VALUE_VERIFY_JSON_END", flush=True)
    else:
        inv = report["inventory"]
        print("=" * 60)
        print("YiRage CPU — verify every function value")
        print("=" * 60)
        print(f"KN builders: {len(inv['kn_builders'])}  TB builders: {len(inv['tb_builders'])}")
        print(f"Customized patterns: {len(inv['customized_patterns'])}")
        print(f"Fast-path checks: {len(inv['fast_path_builders'])} + blas toggle")
        print(f"Planned value checks: {inv['total_value_checks']}")
        vv = report["value_verify"]
        status = "PASS" if vv["ok"] else "FAIL"
        print(f"[{status}] test_cpu_full_value_verify ({vv['elapsed_s']}s)")
        if not vv["ok"]:
            print(vv["stdout_tail"])
            print(vv["stderr_tail"], file=sys.stderr)
        print("=" * 60)
        print("OVERALL:", "PASS" if report["ok"] else "FAIL")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
