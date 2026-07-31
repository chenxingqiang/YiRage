# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU certification runner for RuntimeFusion Serving Loops (S1–Sn).

All stages use **PyTorch** tensor execution.

**Policy (permanent):** Do NOT add ``demo/serving/*smoke*.py``, ``--contract-only``,
or NumPy stub (``EngineModelStub`` / ``BACKEND_NUMPY_REF``) cert paths.
See AGENTS.md § Serving 验证禁令.

Use from Cloud Agent merge gates::

    PYTHONPATH=python python3 scripts/serving_cpu_cert.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .bootstrap import import_serving, repo_root, require_numpy


@dataclass(frozen=True)
class CertStage:
    name: str
    kind: str  # "pytest" | "smoke"
    target: str
    quick: bool = True
    yirage_core: bool = False  # requires built yirage.core


def serving_cpu_cert_manifest(
    *, quick: bool = True, yirage_core: bool = False
) -> List[CertStage]:
    """Ordered stages for Serving Loop verification (torch only)."""
    stages: List[CertStage] = [
        CertStage("s1_contract", "pytest", "tests/python/test_runtime_fusion_s1.py"),
        CertStage("s2_s3_contract", "pytest", "tests/python/test_runtime_fusion_s2_s3.py"),
        CertStage("s4_contract", "pytest", "tests/python/test_runtime_fusion_s4_kv.py"),
        CertStage("s5_contract", "pytest", "tests/python/test_runtime_fusion_s5_sm.py"),
        CertStage("s6_contract", "pytest", "tests/python/test_runtime_fusion_s6_radix.py"),
        CertStage("s7_contract", "pytest", "tests/python/test_runtime_fusion_s7_multi_capsule.py"),
        CertStage(
            "s8_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s8_vllm_bench.py",
        ),
        CertStage(
            "s9_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s9_sglang_meta.py",
        ),
        CertStage(
            "s10_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s10_sglang_plugin.py",
        ),
        CertStage(
            "s11_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s11_vllm_e2e.py",
        ),
        CertStage(
            "s12_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s12_sglang_e2e.py",
        ),
        CertStage(
            "s13_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s13_vllm_paged_e2e.py",
        ),
        CertStage(
            "s14_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s14_yirage_core_e2e.py",
        ),
        CertStage(
            "s15_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s15_maca_serving.py",
        ),
        CertStage(
            "s16_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s16_metax_tiers.py",
        ),
        CertStage(
            "s17_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s17_maca_generation.py",
        ),
        CertStage(
            "s18_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s18_maca_baseline.py",
        ),
        CertStage(
            "torch_contract",
            "pytest",
            "tests/python/test_runtime_fusion_torch.py",
        ),
        CertStage(
            "qwen05b_contract",
            "pytest",
            "tests/python/test_runtime_fusion_qwen05b_cpu_e2e.py",
        ),
        CertStage(
            "s19_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s19_yirage_cpu_search.py",
        ),
        CertStage(
            "s20_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s20_coordinator_ray.py",
        ),
        CertStage(
            "s21_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s21_full_tb_search.py",
        ),
        CertStage(
            "s22_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s22_full_tb_ray.py",
        ),
        CertStage(
            "s23_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s23_full_tb_ray_e2e.py",
        ),
        CertStage(
            "s24_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s24_multilayer_prescreen.py",
        ),
        CertStage(
            "s25_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s25_all_layer_archive.py",
        ),
        CertStage(
            "s26_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s26_multi_tier_archive.py",
        ),
        CertStage(
            "s27_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s27_qwen_decode_bench.py",
        ),
        CertStage(
            "s28_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s28_multilayer_decode_bench.py",
        ),
        CertStage(
            "s29_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s29_full_tb_ray_nightly.py",
        ),
        CertStage(
            "s30_contract",
            "pytest",
            "tests/python/test_runtime_fusion_s30_mlp_capsule_bench.py",
        ),
        CertStage("torch_e2e", "smoke", "demo/serving/torch_e2e.py"),
        CertStage("segment_torch_bench", "smoke", "demo/serving/segment_torch_bench.py"),
        CertStage("vllm_mlp_e2e", "smoke", "demo/serving/vllm_mlp_e2e.py"),
        CertStage("qwen05b_cpu_e2e", "smoke", "demo/serving/qwen05b_cpu_e2e.py --quick"),
    ]
    if yirage_core:
        stages.append(
            CertStage(
                "qwen05b_yirage_e2e",
                "smoke",
                "demo/serving/qwen05b_cpu_e2e.py --quick --mlp-backend yirage_cpu",
                yirage_core=True,
            )
        )
    if yirage_core:
        stages.extend(
            [
                CertStage(
                    "serving_ray_contract",
                    "pytest",
                    "tests/python/test_serving_ray_search.py",
                    yirage_core=True,
                ),
                CertStage(
                    "yirage_core_contract",
                    "pytest",
                    "tests/python/test_runtime_fusion_yirage_core.py",
                    yirage_core=True,
                ),
                CertStage(
                    "yirage_superopt_e2e",
                    "smoke",
                    "demo/serving/yirage_superopt_e2e.py --quick",
                    yirage_core=True,
                ),
                CertStage(
                    "yirage_core_full_e2e",
                    "smoke",
                    "demo/serving/yirage_core_full_e2e.py --quick",
                    yirage_core=True,
                ),
            ]
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
    yirage_core: bool = False
    stages: List[StageResult] = field(default_factory=list)
    bootstrap_ok: bool = False
    serving_version: Optional[str] = None
    torch_device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "quick": self.quick,
            "yirage_core": self.yirage_core,
            "bootstrap_ok": self.bootstrap_ok,
            "serving_version": self.serving_version,
            "torch_device": self.torch_device,
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


def run_serving_cpu_cert(*, quick: bool = True, yirage_core: bool = False) -> CertReport:
    """Execute manifest stages; return structured report."""
    require_numpy()
    root = repo_root()
    report = CertReport(ok=True, quick=quick, yirage_core=yirage_core)

    try:
        serving = import_serving(force_reload=True)
        info = serving.RuntimeFusion([]).inspect()
        report.bootstrap_ok = True
        report.serving_version = str(info.get("version"))
        serving.require_torch()
        report.torch_device = serving.default_device()
        serving.require_vllm_cpu_serving()
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
    env["PYTHONPATH"] = f"{root / 'python'}:{root / 'tests' / 'python'}:{root}"
    env.setdefault("YIRAGE_BACKEND", "cpu")
    for sub in ("build/abstract_subexpr/release", "build/formal_verifier/release"):
        p = root / sub
        if p.exists():
            env["LD_LIBRARY_PATH"] = f"{p}:{env.get('LD_LIBRARY_PATH', '')}"
    py = sys.executable

    for stage in serving_cpu_cert_manifest(quick=quick, yirage_core=yirage_core):
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
    p.add_argument(
        "--yirage-core",
        action="store_true",
        help="include yirage.core superoptimize tier (requires native build)",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--manifest", action="store_true", help="print stage manifest and exit")
    args = p.parse_args(list(argv) if argv is not None else None)

    quick = not args.full
    yirage_core = bool(args.yirage_core)
    if args.manifest:
        manifest = [
            {
                "name": s.name,
                "kind": s.kind,
                "target": s.target,
                "yirage_core": s.yirage_core,
            }
            for s in serving_cpu_cert_manifest(quick=quick, yirage_core=yirage_core)
        ]
        print(json.dumps({"manifest": manifest, "yirage_core": yirage_core}, indent=2))
        return 0

    report = run_serving_cpu_cert(quick=quick, yirage_core=yirage_core)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        mode = "torch"
        if yirage_core:
            mode += "+yirage-core"
        print(f"Serving cert (RuntimeFusion S1–S18, {mode})")
        print(
            f"  bootstrap_ok={report.bootstrap_ok} rf_version={report.serving_version} "
            f"device={report.torch_device}"
        )
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
