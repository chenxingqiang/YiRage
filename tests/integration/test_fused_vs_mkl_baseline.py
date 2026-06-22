# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for fused µGraph vs MKL baseline benchmark script."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


def test_bench_fused_vs_mkl_script_runs_quick():
    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "cpu")
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ld = env.get("LD_LIBRARY_PATH", "")
    for p in (
        "build/abstract_subexpr/release",
        "build/formal_verifier/release",
    ):
        if os.path.isdir(p) and p not in ld:
            ld = f"{p}:{ld}" if ld else p
    env["LD_LIBRARY_PATH"] = ld

    proc = subprocess.run(
        [sys.executable, "scripts/bench_fused_vs_mkl_baseline.py", "--quick", "--json"],
        cwd=env["PYTHONPATH"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:] + proc.stdout[-2000:]
    assert "plain_matmul" in proc.stdout
    assert "speedup_mkl_over_mugraph" in proc.stdout

    import json
    import re

    match = re.search(
        r"YIRAGE_BENCH_JSON_BEGIN\s*(\[.*?\])\s*YIRAGE_BENCH_JSON_END",
        proc.stdout,
        re.DOTALL,
    )
    assert match, "benchmark JSON sentinel missing from stdout"
    rows = json.loads(match.group(1))
    for workload_name in ("plain_matmul", "rms_norm_matmul"):
        row = next(r for r in rows if r["workload"] == workload_name)
        assert row["ok"]
        assert row.get("fusion_search_skipped") is True
        assert row["search_time_s"] == 0.0
        assert row["runtime_verified"] is True

    plain = next(r for r in rows if r["workload"] == "plain_matmul")
    assert 0.2 < plain["speedup_mkl_over_mugraph"] < 2.5
    assert plain.get("plain_matmul_fast_path") is True


def test_bench_plain_matmul_quick_skips_search():
    """Quick plain_matmul bench uses P0 cpu_matmul without ~7s superoptimize (Loop R35)."""
    import json
    import re
    import tempfile

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "cpu")
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ld = env.get("LD_LIBRARY_PATH", "")
    for p in (
        "build/abstract_subexpr/release",
        "build/formal_verifier/release",
    ):
        if os.path.isdir(p) and p not in ld:
            ld = f"{p}:{ld}" if ld else p
    env["LD_LIBRARY_PATH"] = ld

    with tempfile.TemporaryDirectory(prefix="yirage_plain_bench_") as tmp:
        log_path = os.path.join(tmp, "bench.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/bench_fused_vs_mkl_baseline.py",
                    "--quick",
                    "--workloads",
                    "plain_matmul",
                    "--json",
                ],
                cwd=env["PYTHONPATH"],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
        with open(log_path) as logf:
            bench_out = logf.read()

    assert proc.returncode == 0, bench_out[-4000:]
    match = re.search(
        r"YIRAGE_BENCH_JSON_BEGIN\s*(\[.*?\])\s*YIRAGE_BENCH_JSON_END",
        bench_out,
        re.DOTALL,
    )
    assert match, "benchmark JSON sentinel missing from stdout"
    rows = json.loads(match.group(1))
    row = next(r for r in rows if r["workload"] == "plain_matmul")
    assert row["ok"]
    assert row.get("fusion_search_skipped") is True
    assert row["search_time_s"] == 0.0
    assert row.get("plain_matmul_fast_path") is True
    assert row["runtime_verified"] is True
    assert row["mugraph_source"] == "interpreter_unfused"


def test_bench_plain_matmul_full_skips_search():
    """Full plain_matmul bench skips superoptimize on P0 seed (Loop R39)."""
    import json
    import re
    import tempfile

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "cpu")
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ld = env.get("LD_LIBRARY_PATH", "")
    for p in (
        "build/abstract_subexpr/release",
        "build/formal_verifier/release",
    ):
        if os.path.isdir(p) and p not in ld:
            ld = f"{p}:{ld}" if ld else p
    env["LD_LIBRARY_PATH"] = ld

    with tempfile.TemporaryDirectory(prefix="yirage_plain_full_bench_") as tmp:
        log_path = os.path.join(tmp, "bench.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/bench_fused_vs_mkl_baseline.py",
                    "--full",
                    "--workloads",
                    "plain_matmul",
                    "--json",
                ],
                cwd=env["PYTHONPATH"],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        with open(log_path) as logf:
            bench_out = logf.read()

    assert proc.returncode == 0, bench_out[-4000:]
    match = re.search(
        r"YIRAGE_BENCH_JSON_BEGIN\s*(\[.*?\])\s*YIRAGE_BENCH_JSON_END",
        bench_out,
        re.DOTALL,
    )
    assert match, "benchmark JSON sentinel missing from stdout"
    rows = json.loads(match.group(1))
    row = next(r for r in rows if r["workload"] == "plain_matmul")
    assert row["ok"]
    assert row.get("fusion_search_skipped") is True
    assert row["search_time_s"] == 0.0
    assert row.get("plain_matmul_fast_path") is True
    assert row["runtime_verified"] is True


def test_bench_rms_matmul_forced_search_tractable():
    """Forced fusion search uses capped explore grid and finishes in CI (Loop R41)."""
    import json
    import re
    import tempfile

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "cpu")
    env["YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH"] = "0"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ld = env.get("LD_LIBRARY_PATH", "")
    for p in (
        "build/abstract_subexpr/release",
        "build/formal_verifier/release",
    ):
        if os.path.isdir(p) and p not in ld:
            ld = f"{p}:{ld}" if ld else p
    env["LD_LIBRARY_PATH"] = ld

    with tempfile.TemporaryDirectory(prefix="yirage_rms_full_bench_") as tmp:
        log_path = os.path.join(tmp, "bench.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/bench_fused_vs_mkl_baseline.py",
                    "--quick",
                    "--workloads",
                    "rms_norm_matmul",
                    "--json",
                ],
                cwd=env["PYTHONPATH"],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        with open(log_path) as logf:
            bench_out = logf.read()

    assert proc.returncode == 0, bench_out[-4000:]
    match = re.search(
        r"YIRAGE_BENCH_JSON_BEGIN\s*(\[.*?\])\s*YIRAGE_BENCH_JSON_END",
        bench_out,
        re.DOTALL,
    )
    assert match, "benchmark JSON sentinel missing from stdout"
    rows = json.loads(match.group(1))
    row = next(r for r in rows if r["workload"] == "rms_norm_matmul")
    assert row["ok"]
    assert row.get("fusion_search_skipped") is False
    assert row["search_time_s"] > 0.0
    assert row["search_time_s"] < 90.0
    assert row["runtime_verified"] is True


def test_bench_matmul_chain_quick_skips_search():
    """Quick matmul_chain bench uses P0 fast path without ~90s superoptimize (Loop R30)."""
    import json
    import re
    import tempfile

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "cpu")
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ld = env.get("LD_LIBRARY_PATH", "")
    for p in (
        "build/abstract_subexpr/release",
        "build/formal_verifier/release",
    ):
        if os.path.isdir(p) and p not in ld:
            ld = f"{p}:{ld}" if ld else p
    env["LD_LIBRARY_PATH"] = ld

    with tempfile.TemporaryDirectory(prefix="yirage_chain_bench_") as tmp:
        log_path = os.path.join(tmp, "bench.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/bench_fused_vs_mkl_baseline.py",
                    "--quick",
                    "--workloads",
                    "matmul_chain",
                    "--json",
                ],
                cwd=env["PYTHONPATH"],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        with open(log_path) as logf:
            bench_out = logf.read()

    assert proc.returncode == 0, bench_out[-4000:]
    match = re.search(
        r"YIRAGE_BENCH_JSON_BEGIN\s*(\[.*?\])\s*YIRAGE_BENCH_JSON_END",
        bench_out,
        re.DOTALL,
    )
    assert match, "benchmark JSON sentinel missing from stdout"
    rows = json.loads(match.group(1))
    row = next(r for r in rows if r["workload"] == "matmul_chain")
    assert row["ok"]
    assert row.get("fusion_search_skipped") is True
    assert row["search_time_s"] == 0.0
    assert row.get("matmul_chain_fast_path") is True
    assert row["runtime_verified"] is True
    assert row["mugraph_source"] == "interpreter_unfused"


def test_matmul_chain_fast_path_smoke():
    """Smoke: 3-input matmul_chain uses host-BLAS cpu_call (Loop R29; no full search)."""
    import torch
    import yirage as yr
    from yirage.kernel.cpu_mlir_jit import is_production_matmul_chain_mugraph
    from yirage.kernel.graph import KNGraph
    from yirage.search.verifier_config import runtime_verify_mugraph

    m, k, k2, n = 16, 32, 32, 64
    inputs = [
        torch.randn(m, k, dtype=torch.float16),
        torch.randn(k, k2, dtype=torch.float16),
        torch.randn(k2, n, dtype=torch.float16),
    ]
    ref = torch.matmul(torch.matmul(inputs[0], inputs[1]), inputs[2])

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(m, k), dtype=yr.float16)
    b = g.new_input(dims=(k, k2), dtype=yr.float16)
    c = g.new_input(dims=(k2, n), dtype=yr.float16)
    t = g.matmul(a, b)
    g.mark_output(g.matmul(t, c))
    kg = KNGraph(g.cygraph, backend="cpu")

    assert is_production_matmul_chain_mugraph(g.cygraph)
    ok, err, _ = runtime_verify_mugraph(kg, inputs, ref)
    assert ok, err


def test_bench_concat_matmul_workload_runs():
    """Quick LoRA concat_matmul bench uses P0 fast path without fusion search (Loop R31)."""
    import json
    import re
    import tempfile

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "cpu")
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ld = env.get("LD_LIBRARY_PATH", "")
    for p in (
        "build/abstract_subexpr/release",
        "build/formal_verifier/release",
    ):
        if os.path.isdir(p) and p not in ld:
            ld = f"{p}:{ld}" if ld else p
    env["LD_LIBRARY_PATH"] = ld

    with tempfile.TemporaryDirectory(prefix="yirage_concat_bench_") as tmp:
        log_path = os.path.join(tmp, "bench.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/bench_fused_vs_mkl_baseline.py",
                    "--quick",
                    "--workloads",
                    "concat_matmul",
                    "--json",
                ],
                cwd=env["PYTHONPATH"],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        with open(log_path) as logf:
            bench_out = logf.read()

    assert proc.returncode == 0, bench_out[-4000:]
    match = re.search(
        r"YIRAGE_BENCH_JSON_BEGIN\s*(\[.*?\])\s*YIRAGE_BENCH_JSON_END",
        bench_out,
        re.DOTALL,
    )
    assert match, "benchmark JSON sentinel missing from stdout"
    rows = json.loads(match.group(1))
    row = next(r for r in rows if r["workload"] == "concat_matmul")
    assert row["ok"]
    assert row["runtime_verified"] is True
    assert row.get("fusion_search_skipped") is True
    assert row["search_time_s"] == 0.0
    assert row.get("concat_matmul_fast_path") is True
    assert row["mugraph_source"] == "interpreter_unfused"


def test_bench_rms_matmul_quick_skips_search():
    """Quick rms_norm_matmul bench uses P0 cpu_rms_matmul without ~12s superoptimize (Loop R34)."""
    import json
    import re
    import tempfile

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "cpu")
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ld = env.get("LD_LIBRARY_PATH", "")
    for p in (
        "build/abstract_subexpr/release",
        "build/formal_verifier/release",
    ):
        if os.path.isdir(p) and p not in ld:
            ld = f"{p}:{ld}" if ld else p
    env["LD_LIBRARY_PATH"] = ld

    with tempfile.TemporaryDirectory(prefix="yirage_rms_bench_") as tmp:
        log_path = os.path.join(tmp, "bench.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/bench_fused_vs_mkl_baseline.py",
                    "--quick",
                    "--workloads",
                    "rms_norm_matmul",
                    "--json",
                ],
                cwd=env["PYTHONPATH"],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
        with open(log_path) as logf:
            bench_out = logf.read()

    assert proc.returncode == 0, bench_out[-4000:]
    match = re.search(
        r"YIRAGE_BENCH_JSON_BEGIN\s*(\[.*?\])\s*YIRAGE_BENCH_JSON_END",
        bench_out,
        re.DOTALL,
    )
    assert match, "benchmark JSON sentinel missing from stdout"
    rows = json.loads(match.group(1))
    row = next(r for r in rows if r["workload"] == "rms_norm_matmul")
    assert row["ok"]
    assert row.get("fusion_search_skipped") is True
    assert row["search_time_s"] == 0.0
    assert row.get("rms_matmul_fast_path") is True
    assert row["runtime_verified"] is True
    assert row["mugraph_source"] == "interpreter_unfused"


def _mlir_bench_env(extra_env: dict | None = None) -> dict:
    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "cpu")
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ld = env.get("LD_LIBRARY_PATH", "")
    for p in (
        "build/abstract_subexpr/release",
        "build/formal_verifier/release",
        "/usr/lib/llvm-17/lib",
    ):
        if os.path.isdir(p) and p not in ld:
            ld = f"{p}:{ld}" if ld else p
    env["LD_LIBRARY_PATH"] = ld
    if extra_env:
        env.update(extra_env)
    return env


def _run_rms_matmul_mlir_bench_json(
    extra_env: dict | None = None,
    *,
    extra_args: list[str] | None = None,
) -> tuple[dict, str]:
    import json
    import re
    import tempfile

    env = _mlir_bench_env(extra_env)
    cmd = [
        sys.executable,
        "scripts/bench_fused_vs_mkl_baseline.py",
        "--quick",
        "--workloads",
        "rms_norm_matmul",
        "--mlir-jit",
        "--json",
    ]
    if extra_args:
        cmd.extend(extra_args)
    with tempfile.TemporaryDirectory(prefix="yirage_mlir_bench_") as tmp:
        log_path = os.path.join(tmp, "bench.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd,
                cwd=env["PYTHONPATH"],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        with open(log_path) as logf:
            bench_out = logf.read()

    assert proc.returncode == 0, bench_out[-4000:]
    match = re.search(
        r"YIRAGE_BENCH_JSON_BEGIN\s*(\[.*?\])\s*YIRAGE_BENCH_JSON_END",
        bench_out,
        re.DOTALL,
    )
    assert match, "benchmark JSON sentinel missing from stdout"
    rows = json.loads(match.group(1))
    return next(r for r in rows if r["workload"] == "rms_norm_matmul"), bench_out


def test_bench_rms_matmul_mlir_jit_quick_smoke():
    """Quick rms_norm_matmul bench with --mlir-jit reports JIT timing (Loop R33)."""
    from yirage.kernel.cpu_mlir_jit import is_mlir_jit_available

    if not is_mlir_jit_available():
        pytest.skip("yirage._yirage_mlir not built (USE_MLIR=1 required)")

    row, _ = _run_rms_matmul_mlir_bench_json()
    assert row["ok"]
    assert row["runtime_verified"] is True
    assert row.get("mlir_jit_ms") is not None
    assert row["mlir_jit_ms"] > 0
    assert row.get("interpreter_ms") is not None
    assert row.get("speedup_interp_over_mlir_jit", 0) > 0
    assert row.get("hand_mlir_jit_ms") is not None
    assert row["hand_mlir_jit_ms"] > 0
    assert row.get("dialect_lowered_jit_ms") is not None
    assert row["dialect_lowered_jit_ms"] > 0
    assert row.get("speedup_hand_over_dialect_lowered", 0) > 0
    assert row.get("mlir_hand_dialect_aligned") is True
    assert row.get("mlir_jit_emit_path") in {
        "dialect_lowered",
        "dialect_raw",
        "hand_bgrid_tiled",
        "hand_tiled",
        "hand_flat",
    }


def test_bench_rms_matmul_mlir_jit_dialect_emit_path_smoke():
    """With YIRAGE_CPU_MLIR_JIT_DIALECT=1, bench reports dialect_lowered emit (R46)."""
    from yirage.kernel.cpu_mlir_jit import (
        _yirage_cpu_opt_path,
        is_mlir_jit_available,
    )

    if not is_mlir_jit_available():
        pytest.skip("yirage._yirage_mlir not built (USE_MLIR=1 required)")
    if _yirage_cpu_opt_path() is None:
        pytest.skip("yirage-cpu-opt not built")

    row, _ = _run_rms_matmul_mlir_bench_json(
        {"YIRAGE_CPU_MLIR_JIT_DIALECT": "1"}
    )
    assert row["ok"]
    assert row.get("mlir_jit_emit_path") == "dialect_lowered"


def test_bench_rms_matmul_mlir_jit_fused_emit_path_smoke():
    """--mlir-jit-fused superoptimizes seed and reports hand_bgrid_tiled emit (R49)."""
    from yirage.kernel.cpu_mlir_jit import is_mlir_jit_available

    if not is_mlir_jit_available():
        pytest.skip("yirage._yirage_mlir not built (USE_MLIR=1 required)")

    row, _ = _run_rms_matmul_mlir_bench_json(extra_args=["--mlir-jit-fused"])
    assert row["ok"]
    assert row.get("mlir_jit_fused_seed") is True
    assert row.get("fusion_search_skipped") is False
    assert row.get("mlir_jit_emit_path") == "hand_bgrid_tiled"
    graph = row.get("graph") or {}
    assert graph.get("kn_customized", 0) >= 1


def test_bench_rms_matmul_mlir_jit_fused_dialect_emit_path_smoke():
    """DIALECT=1 + --mlir-jit-fused reports dialect_lowered on fused seed (R50)."""
    from yirage.kernel.cpu_mlir_jit import (
        _yirage_cpu_opt_path,
        is_mlir_jit_available,
    )

    if not is_mlir_jit_available():
        pytest.skip("yirage._yirage_mlir not built (USE_MLIR=1 required)")
    if _yirage_cpu_opt_path() is None:
        pytest.skip("yirage-cpu-opt not built")

    row, _ = _run_rms_matmul_mlir_bench_json(
        {"YIRAGE_CPU_MLIR_JIT_DIALECT": "1"},
        extra_args=["--mlir-jit-fused"],
    )
    assert row["ok"]
    assert row.get("mlir_jit_fused_seed") is True
    assert row.get("fusion_search_skipped") is False
    assert row.get("mlir_jit_emit_path") == "dialect_lowered"
