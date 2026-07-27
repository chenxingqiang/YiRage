# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU certification runner for RuntimeFusion Serving Loops (S1–Sn).

Runs contract pytest modules and demo smokes without ``yirage.core``, torch, or vLLM.
Use from Cloud Agent merge gates and local dev::

    PYTHONPATH=python python3 scripts/serving_cpu_cert.py --quick
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .bootstrap import import_serving, repo_root, require_numpy


@dataclass(frozen=True)
class CertStage:
    name: str
    kind: str  # "pytest" | "smoke"
    target: str
    quick: bool = True


def serving_cpu_cert_manifest(*, quick: bool = True) -> List[CertStage]:
    """Ordered stages for Serving Loop CPU verification."""
    stages: List[CertStage] = [
        CertStage("s1_contract", "pytest", "tests/python/test_runtime_fusion_s1.py"),
        CertStage("s2_s3_contract", "pytest", "tests/python/test_runtime_fusion_s2_s3.py"),
        CertStage("s4_contract", "pytest", "tests/python/test_runtime_fusion_s4_kv.py"),
        CertStage("s5_contract", "pytest", "tests/python/test_runtime_fusion_s5_sm.py"),
        CertStage("mlp_capsule_smoke", "smoke", "demo/serving/mlp_capsule_smoke.py"),
        CertStage("vllm_mlp_override_smoke", "smoke", "demo/serving/vllm_mlp_override_smoke.py"),
        CertStage("hybrid_first_k_smoke", "smoke", "demo/serving/hybrid_first_k_smoke.py --k 2"),
        CertStage("kv_meta_bridge_smoke", "smoke", "demo/serving/kv_meta_bridge_smoke.py"),
        CertStage("sm_budget_coresidence_smoke", "smoke", "demo/serving/sm_budget_coresidence_smoke.py"),
    ]
    if not quick:
        stages.append(
            CertStage(
                "hybrid_k_sweep",
                "smoke",
                "demo/serving/hybrid_first_k_smoke.py --k 1 && "
                "demo/serving/hybrid_first_k_smoke.py --k 4",
                quick=False,
            )
        )
    return stages


@dataclass
class StageResult:
    name: str
    kind: str
    ok: bool
    elapsed_s: float
    returncode: int
    command: str
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class CertReport:
    ok: bool
    quick: bool
    stages: List[StageResult] = field(default_factory=list)
    bootstrap_ok: bool = False
    serving_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "quick": self.quick,
            "bootstrap_ok": self.bootstrap_ok,
            "serving_version": self.serving_version,
            "stages": [asdict(s) for s in self.stages],
        }


def _tail(text: str, max_lines: int = 40) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _run_shell(command: str, *, cwd: Path, env: Dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )


def run_serving_cpu_cert(*, quick: bool = True) -> CertReport:
    """Execute all manifest stages; return structured report."""
    require_numpy()
    root = repo_root()
    report = CertReport(ok=True, quick=quick)

    try:
        serving = import_serving(force_reload=True)
        info = serving.RuntimeFusion([]).inspect()
        report.bootstrap_ok = True
        report.serving_version = str(info.get("version"))
    except Exception as e:
        report.ok = False
        report.bootstrap_ok = False
        report.stages.append(
            StageResult(
                name="bootstrap",
                kind="import",
                ok=False,
                elapsed_s=0.0,
                returncode=1,
                command="import_serving()",
                stderr_tail=str(e),
            )
        )
        return report

    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(root / "python")
    py = sys.executable

    for stage in serving_cpu_cert_manifest(quick=quick):
        t0 = time.monotonic()
        if stage.kind == "pytest":
            cmd = f"{py} -m pytest {stage.target} -q --tb=short"
        elif stage.kind == "smoke":
            cmd = f"{py} {stage.target}"
        else:
            cmd = stage.target

        proc = _run_shell(cmd, cwd=root, env=env)
        elapsed = time.monotonic() - t0
        ok = proc.returncode == 0
        report.stages.append(
            StageResult(
                name=stage.name,
                kind=stage.kind,
                ok=ok,
                elapsed_s=round(elapsed, 3),
                returncode=int(proc.returncode),
                command=cmd,
                stdout_tail=_tail(proc.stdout),
                stderr_tail=_tail(proc.stderr),
            )
        )
        if not ok:
            report.ok = False

    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", default=True)
    p.add_argument("--full", action="store_true", help="include extra smoke sweeps")
    p.add_argument("--json", action="store_true")
    p.add_argument("--manifest", action="store_true", help="print stage manifest and exit")
    args = p.parse_args(list(argv) if argv is not None else None)

    quick = not args.full
    if args.manifest:
        manifest = [
            {"name": s.name, "kind": s.kind, "target": s.target}
            for s in serving_cpu_cert_manifest(quick=quick)
        ]
        print(json.dumps({"manifest": manifest}, indent=2))
        return 0

    report = run_serving_cpu_cert(quick=quick)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("Serving CPU cert (RuntimeFusion S1–S5)")
        print(f"  bootstrap_ok={report.bootstrap_ok} rf_version={report.serving_version}")
        for s in report.stages:
            mark = "PASS" if s.ok else "FAIL"
            print(f"  [{mark}] {s.name} ({s.elapsed_s:.2f}s)")
            if not s.ok:
                if s.stderr_tail:
                    print(s.stderr_tail)
                if s.stdout_tail:
                    print(s.stdout_tail)
        print("PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
