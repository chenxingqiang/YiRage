#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end business capability walkthrough (same-backend hardware optimization):
  YiRage µGraph search on the installed backend → profile/select on that backend
  → Ray parallelism → AccelForge prescreen → RL reward loop.

YiRage does not search on one device and execute on another (e.g. CPU search → CUDA run).
On this host, all stages use ``backend=cpu`` and CPU-native profiling/execution.

Run:
  export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
  export YIRAGE_BACKEND=cpu
  PYTHONPATH=. python3 scripts/business_capability_walkthrough.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure repo root on path
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@dataclass
class StageResult:
    name: str
    ok: bool
    elapsed_s: float
    details: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class WalkthroughReport:
    stages: List[StageResult] = field(default_factory=list)
    business_scores: Dict[str, float] = field(default_factory=dict)
    verdict: str = ""

    def add(self, stage: StageResult) -> None:
        self.stages.append(stage)

    def print_report(self) -> None:
        print("\n" + "=" * 72)
        print("YiRage × Ray × AccelForge × RL — Same-Backend Capability Walkthrough")
        print("=" * 72)
        for s in self.stages:
            status = "OK" if s.ok else "FAIL"
            print(f"\n[{status}] {s.name} ({s.elapsed_s:.3f}s)")
            for k, v in s.details.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6g}")
                elif isinstance(v, dict) and len(json.dumps(v)) > 200:
                    print(f"    {k}: <dict {len(v)} keys>")
                else:
                    print(f"    {k}: {v}")
            for note in s.notes:
                print(f"    · {note}")

        print("\n" + "-" * 72)
        print("Business value scores (0–5, higher = stronger capability)")
        print("-" * 72)
        for dim, score in self.business_scores.items():
            bar = "█" * int(round(score)) + "░" * (5 - int(round(score)))
            print(f"  {dim:28s} {score:.1f}/5  {bar}")

        print("\n" + "-" * 72)
        print("Verdict")
        print("-" * 72)
        print(self.verdict)
        print("=" * 72 + "\n")


@contextmanager
def _isolated_mugraph_store():
    prev_home = os.environ.get("HOME")
    with tempfile.TemporaryDirectory(prefix="yirage_walkthrough_") as tmp:
        os.environ["HOME"] = tmp
        try:
            import yirage.storage.mugraph_store as ms

            ms._default_store = None
            yield
        finally:
            if prev_home is not None:
                os.environ["HOME"] = prev_home
            else:
                os.environ.pop("HOME", None)


def _timed(name: str, fn) -> StageResult:
    t0 = time.perf_counter()
    try:
        details = fn()
        elapsed = time.perf_counter() - t0
        return StageResult(name=name, ok=True, elapsed_s=elapsed, details=details or {})
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return StageResult(
            name=name,
            ok=False,
            elapsed_s=elapsed,
            details={"error": str(exc)},
            notes=[type(exc).__name__],
        )


def stage_yirage_graph():
    import torch
    import yirage as yr

    from yirage.backends.cpu.config import get_cpu_info, get_cpu_runtime_config

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(8, 32), dtype=yr.float16)
    b = g.new_input(dims=(32, 64), dtype=yr.float16)
    out = g.matmul(a, b)
    g.mark_output(out)
    cpu_info = get_cpu_info()
    rt = get_cpu_runtime_config()
    return {
        "backend": os.environ.get("YIRAGE_BACKEND", "cpu"),
        "design": "same-backend (search/profile/execute on installed hardware)",
        "input_shapes": "8×32 @ 32×64",
        "kn_ops_before_search": len(g.cygraph.get_graph_structure()),
        "cpu_cores": cpu_info["num_cores"],
        "cpu_simd": cpu_info["simd_type"],
        "cpu_torch_threads": rt["torch_num_threads"],
    }, g


def _walkthrough_quick_enabled() -> bool:
    return os.environ.get("YIRAGE_CPU_WALKTHROUGH_QUICK") == "1"


def stage_cpu_homomorphic_value(graph) -> Dict[str, Any]:
    """Measure correctness and latency on CPU after superoptimize (same backend)."""
    import torch

    from scripts.cpu_cert_utils import apply_plain_matmul_search_tractability

    quick = _walkthrough_quick_enabled()
    if quick:
        apply_plain_matmul_search_tractability()
        griddims = [(1, 1, 1)]
        warmup_iters, profile_iters = 1, 5
        bench_iters = 10
    else:
        griddims = [(1, 1, 1), (2, 1, 1)]
        warmup_iters, profile_iters = 4, 80
        bench_iters = 60
    with _isolated_mugraph_store():
        so_kwargs: Dict[str, Any] = {
            "backend": "cpu",
            "griddims": griddims,
            "blockdims": [(64, 1, 1)],
            "use_graph_dataset": False,
            "use_cached_graphs": False,
            "use_persistent_cache": False,
            "use_ray": False,
            "warmup_iters": warmup_iters,
            "profile_iters": profile_iters,
            "verbose": False,
        }
        if quick:
            so_kwargs["franges"] = [1]
        optimized = graph.superoptimize(**so_kwargs)

    assert optimized is not None
    assert optimized.backend == "cpu"

    ref_a = torch.randn(8, 32, dtype=torch.float16)
    ref_b = torch.randn(32, 64, dtype=torch.float16)
    ref_out = torch.matmul(ref_a, ref_b)

    out = optimized(inputs=[ref_a, ref_b])
    max_err = (out[0].float() - ref_out.float()).abs().max().item()
    correct = max_err < 0.05

    iters = bench_iters
    for _ in range(8):
        torch.matmul(ref_a, ref_b)
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(ref_a, ref_b)
    baseline_ms = (time.perf_counter() - t0) / iters * 1000

    for _ in range(8):
        optimized(inputs=[ref_a, ref_b])
    t0 = time.perf_counter()
    for _ in range(iters):
        optimized(inputs=[ref_a, ref_b])
    optimized_ms = (time.perf_counter() - t0) / iters * 1000

    return {
        "optimized_backend": optimized.backend,
        "numerically_correct": correct,
        "max_abs_error": max_err,
        "baseline_torch_matmul_ms": baseline_ms,
        "optimized_mugraph_ms": optimized_ms,
        "mugraph_vs_torch_ratio": optimized_ms / max(baseline_ms, 1e-6),
        "candidates_profiled": len(griddims),
    }


def _search_once(yr, griddims, use_ray: bool) -> float:
    import ray

    from scripts.cpu_cert_utils import apply_plain_matmul_search_tractability

    quick = _walkthrough_quick_enabled()
    if quick:
        apply_plain_matmul_search_tractability()
        griddims = [(1, 1, 1)]
    if ray.is_initialized():
        ray.shutdown()
    with _isolated_mugraph_store():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 64), dtype=yr.float16)
        g.mark_output(g.matmul(a, b))
        t0 = time.perf_counter()
        so_kwargs: Dict[str, Any] = {
            "backend": "cpu",
            "griddims": griddims,
            "blockdims": [(32, 1, 1)] if quick else [(128, 1, 1)],
            "use_graph_dataset": False,
            "use_cached_graphs": False,
            "use_persistent_cache": False,
            "use_ray": use_ray,
            "num_workers": 2,
            "verbose": False,
        }
        if quick:
            so_kwargs["franges"] = [1]
        g.superoptimize(**so_kwargs)
        return time.perf_counter() - t0


def stage_ray_search(graph) -> Dict[str, Any]:
    import yirage as yr

    from scripts.cpu_cert_utils import apply_plain_matmul_search_tractability

    quick = _walkthrough_quick_enabled()
    griddims = (
        [(1, 1, 1)]
        if quick
        else [(1, 1, 1), (2, 1, 1), (4, 1, 1), (8, 1, 1), (16, 1, 1), (32, 1, 1)]
    )
    prod_griddims = griddims[:1] if quick else griddims[:3]

    if quick:
        apply_plain_matmul_search_tractability()
        seq_elapsed = None
        ray_elapsed = None
        speedup = None
    else:
        seq_elapsed = _search_once(yr, griddims, use_ray=False)
        ray_elapsed = _search_once(yr, griddims, use_ray=True)
        speedup = seq_elapsed / max(ray_elapsed, 1e-6)

    with _isolated_mugraph_store():
        t_ray = time.perf_counter()
        so_kwargs: Dict[str, Any] = {
            "backend": "cpu",
            "griddims": prod_griddims,
            "blockdims": [(32, 1, 1)] if quick else [(128, 1, 1)],
            "use_graph_dataset": False,
            "use_cached_graphs": False,
            "use_persistent_cache": False,
            "use_ray": True,
            "num_workers": 2,
            "verbose": False,
        }
        if quick:
            so_kwargs["franges"] = [1]
        optimized = graph.superoptimize(**so_kwargs)
        prod_ray_s = time.perf_counter() - t_ray

    from yirage.storage.graph_serde import serialize_optimized_graph

    payload = serialize_optimized_graph(optimized)
    note = (
        "quick mode: seq/ray benchmark skipped; single-grid Ray search only"
        if quick
        else "speedup>1 means Ray finished faster (isolated cache)"
    )
    return {
        "griddims_benchmarked": len(griddims),
        "sequential_search_s": seq_elapsed,
        "ray_search_s": ray_elapsed,
        "speedup_seq_over_ray": speedup,
        "demo_ray_search_s": prod_ray_s,
        "graph_json_bytes": len(payload or ""),
        "optimized_ops": len(optimized.cygraph.get_graph_structure()),
        "note": note,
    }, optimized, payload


def stage_accelforge(payload: str) -> Dict[str, Any]:
    from yirage.rl.hardware.accelforge_bridge import (
        get_accelforge_availability,
        mugraph_to_workload,
    )
    from yirage.rl.verifier.accelforge_verifier import AccelForgeVerifier

    avail = get_accelforge_availability()
    workload_from_cy = mugraph_to_workload(payload)
    prescreen = AccelForgeVerifier(
        design_point={"pe_array_rows": 16, "pe_array_cols": 16, "data_precision": "fp16"}
    ).prescreen_kernel(
        payload,
        latency_budget_ms=5000.0,
        area_budget_mm2=500.0,
        power_budget_mw=50000.0,
    )

    return {
        "accelforge_available": avail["available"],
        "accelforge_version": avail.get("version"),
        "workload_m_k_n": (
            workload_from_cy.get("m_dim"),
            workload_from_cy.get("k_dim"),
            workload_from_cy.get("n_dim"),
        ),
        "estimated_flops": workload_from_cy.get("estimated_flops"),
        "prescreen_accepted": prescreen["accepted"],
        "prescreen_rejections": prescreen.get("rejections", []),
        "prescreen_verified": prescreen.get("verified"),
        "prescreen_latency_ms": prescreen["metrics"].get("latency_ms"),
        "prescreen_energy_pj": prescreen["metrics"].get("energy_per_op_pj"),
        "prescreen_area_mm2": prescreen["metrics"].get("area_mm2"),
        "prescreen_power_mw": prescreen["metrics"].get("total_power_mw"),
    }


def stage_rl_accelforge_loop(payload: str) -> Dict[str, Any]:
    import numpy as np

    from yirage.rl.search.config_space import HardwareConfig, SearchSpaceConstraints
    from yirage.rl.search.graph_space import GraphAction
    from yirage.rl.search.hierarchical_env import (
        ConstrainedGraphEnv,
        HierarchicalEnvConfig,
        HierarchicalSearchEnv,
    )

    cfg = HierarchicalEnvConfig(
        target_graph_json=payload,
        backend="accelforge",
        accelforge_design={
            "pe_array_rows": 16,
            "pe_array_cols": 16,
            "data_precision": "fp16",
        },
        max_graph_steps=12,
    )

    hier = HierarchicalSearchEnv(vars(cfg))
    hier.reset()
    _, h_reward, _, _, h_info = hier.step(hier.action_space.sample())

    constraints = SearchSpaceConstraints(HardwareConfig())
    graph_env = ConstrainedGraphEnv(constraints, vars(cfg))
    graph_env.reset()  # seeds kernel_graph_json from target_graph_json

    finish_info: Dict[str, Any] = {}
    scripted = [
        GraphAction.ADD_KN_OP,
        GraphAction.CREATE_TB,
        GraphAction.ADD_TB_OP,
        GraphAction.FINISH,
    ]
    for action_type in scripted:
        action = np.zeros(8, dtype=int)
        action[0] = action_type
        _, _, terminated, truncated, step_info = graph_env.step(action)
        if terminated or truncated:
            finish_info = step_info
            break

    return {
        "hierarchical_level1_reward": float(h_reward),
        "kernel_json_seeded_on_reset": graph_env.kernel_graph_json != "{}",
        "finish_verified": finish_info.get("verified"),
        "finish_has_accelforge_metrics": "accelforge_metrics" in finish_info,
        "finish_latency_ms_af": finish_info.get("latency_ms_af"),
        "finish_energy_pj": finish_info.get("energy_pj"),
        "level2_accelforge_in_result": "accelforge_metrics"
        in (h_info.get("level2_result") or {}),
    }


def compute_business_scores(report: WalkthroughReport) -> None:
    """Score 0–5 on pragmatic business dimensions."""
    by_name = {s.name: s for s in report.stages}

    # 1. Pipeline integration
    integration = 5.0 if all(s.ok for s in report.stages) else 2.0
    if all(s.ok for s in report.stages):
        integration = min(5.0, integration)

    # 2. Search acceleration (Ray)
    ray = by_name.get("2. Ray µGraph search")
    if ray and ray.ok:
        speedup_raw = ray.details.get(
            "speedup_seq_over_ray", ray.details.get("speedup_vs_sequential")
        )
        if speedup_raw is None:
            ray_score = 2.5  # quick: seq/ray benchmark skipped
        else:
            speedup = float(speedup_raw)
            if speedup >= 1.3:
                ray_score = 4.0
            elif speedup >= 1.05:
                ray_score = 3.0
            elif speedup >= 0.95:
                ray_score = 2.5  # parity on small CPU jobs
            else:
                ray_score = 2.0
    else:
        ray_score = 0.0

    # 3. Virtual HW oracle (AccelForge)
    af = by_name.get("3. AccelForge prescreen")
    if af and af.ok and af.details.get("accelforge_available"):
        if af.details.get("prescreen_accepted"):
            af_score = 4.5
        else:
            af_score = 3.5
    elif af and af.ok:
        af_score = 2.0  # analytical fallback only
    else:
        af_score = 0.0

    # 4. RL closed-loop (reward + AccelForge coupling)
    rl = by_name.get("4. RL × AccelForge env loop")
    if rl and rl.ok:
        rl_score = (
            4.5
            if rl.details.get("finish_has_accelforge_metrics")
            else 2.5
        )
    else:
        rl_score = 0.0

    # 5. Production readiness (honest gaps)
    gaps = []
    if ray and ray.ok and ray.details.get("speedup_seq_over_ray") is not None:
        if ray.details.get("speedup_seq_over_ray", 1) < 1.1:
            gaps.append("Ray speedup modest on small CPU search")
    rl_stage = by_name.get("4. RL × AccelForge env loop")
    if rl_stage and rl_stage.ok and not rl_stage.details.get("finish_has_accelforge_metrics"):
        gaps.append("RL FINISH step missing AccelForge metrics")
    prod = 3.5 - 0.5 * len(gaps)
    prod = max(2.0, min(4.0, prod))

    report.business_scores = {
        "E2E integration": integration,
        "Ray search acceleration": ray_score,
        "AccelForge virtual oracle": af_score,
        "RL multi-objective loop": rl_score,
        "Production readiness": prod,
    }

    overall = sum(report.business_scores.values()) / len(report.business_scores)
    if overall >= 4.0:
        verdict = (
            "Strong same-backend pipeline: µGraph search, profile, and execute align on "
            "the installed hardware; Ray, AccelForge prescreen, and RL rewards stack on top. "
            "Best ROI when search space is large or target accelerator co-design is in scope."
        )
    elif overall >= 3.0:
        verdict = (
            "Capable integrated stack: optimization is scoped to the active backend "
            "(not cross-device). Ray/RL gains are workload-dependent; AccelForge is a "
            "fast pre-filter—confirm winners via same-backend profiling on target hardware."
        )
    else:
        verdict = (
            "Components exist but end-to-end business value is limited in this "
            "environment—check failed stages and missing dependencies."
        )
    report.verdict = verdict


def walkthrough_report_to_dict(report: WalkthroughReport) -> Dict[str, Any]:
    """Serialize walkthrough report for cert JSON / ``--json`` CLI."""
    stages = [
        {
            "name": s.name,
            "ok": s.ok,
            "elapsed_s": round(s.elapsed_s, 3),
            "details": s.details,
        }
        for s in report.stages
    ]
    substage_elapsed = {
        s["name"]: s["elapsed_s"] for s in stages
    }
    return {
        "ok": all(s.ok for s in report.stages),
        "stages": stages,
        "walkthrough_substage_elapsed_s": substage_elapsed,
        "business_scores": report.business_scores,
        "verdict": report.verdict,
    }


def build_walkthrough_report(*, quick: bool = False) -> WalkthroughReport:
    """Run all walkthrough stages and return structured report."""
    if quick:
        os.environ["YIRAGE_CPU_WALKTHROUGH_QUICK"] = "1"
    else:
        os.environ.pop("YIRAGE_CPU_WALKTHROUGH_QUICK", None)

    report = WalkthroughReport()
    optimized = None
    payload: Optional[str] = None

    def s1():
        nonlocal optimized, payload
        details, optimized = stage_yirage_graph()
        return details

    report.add(_timed("1. YiRage graph build", s1))

    def s1b():
        if optimized is None:
            raise RuntimeError("graph not built")
        return stage_cpu_homomorphic_value(optimized)

    report.add(_timed("1b. CPU same-backend value", s1b))

    def s2():
        nonlocal optimized, payload
        if optimized is None:
            raise RuntimeError("graph not built")
        details, optimized, payload = stage_ray_search(optimized)
        return details

    report.add(_timed("2. Ray µGraph search", s2))

    def s3():
        if not payload:
            raise RuntimeError("no graph json")
        return stage_accelforge(payload)

    report.add(_timed("3. AccelForge prescreen", s3))

    def s4():
        if not payload:
            raise RuntimeError("no graph json")
        return stage_rl_accelforge_loop(payload)

    report.add(_timed("4. RL × AccelForge env loop", s4))

    compute_business_scores(report)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tractable search grids for cert/CI (YIRAGE_CPU_WALKTHROUGH_QUICK)",
    )
    args = parser.parse_args()

    report = build_walkthrough_report(quick=args.quick)
    if args.json:
        print("YIRAGE_WALKTHROUGH_JSON_BEGIN")
        print(json.dumps(walkthrough_report_to_dict(report), indent=2))
        print("YIRAGE_WALKTHROUGH_JSON_END", flush=True)
    else:
        report.print_report()
    return 0 if all(s.ok for s in report.stages) else 1


if __name__ == "__main__":
    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    raise SystemExit(main())
