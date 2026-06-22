#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.python._yirage_test_support import ensure_native_library_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _demo_env() -> dict[str, str]:
    """Environment for subprocess demos (native libs + unbuffered stdout)."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    ensure_native_library_path()
    env["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
    root = str(PROJECT_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not existing else f"{root}{os.pathsep}{existing}"
    return env


def _run_demo(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_demo_env(),
    )
    assert result.returncode == 0, (
        f"command failed: {' '.join(args)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


@pytest.mark.cpu
@pytest.mark.integration
def test_backend_selection_demo_runs_on_cpu():
    result = _run_demo("demo/backend_selection_demo.py")
    assert "Demo completed successfully!" in result.stdout
    assert "cpu" in result.stdout.lower()
    assert "Graph executed on cpu" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_demo_jit_runs_on_cpu():
    result = _run_demo("demo/demo_jit.py", "--device", "cpu", "--quiet")
    assert "Correctness of output[0]: True" in result.stdout
    assert "Correctness of output[1]: True" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
def test_submission_validate_runs_on_cpu():
    result = _run_demo("examples/submission.py", "--validate")
    assert "All validation steps completed" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_demo_lora_smoke_on_cpu():
    """LoRA-style blocked GEMM executes on CPU without superoptimize."""
    result = _run_demo("demo/demo_lora.py")
    assert "LoRA muGraph run time (ms):" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_reference_mugraph_lora_smoke_on_cpu():
    """Reference LoRA blocked GEMM µGraph executes on CPU."""
    result = _run_demo("demo/reference_mugraphs/lora.py", "--quick")
    assert "reference_muGraph run time (ms):" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_reference_mugraph_rms_norm_smoke_on_cpu():
    """Reference fused customized RMS+matmul µGraph executes on CPU."""
    result = _run_demo("demo/reference_mugraphs/rms_norm.py", "--quick")
    assert "reference_muGraph run time (ms):" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_reference_mugraph_gated_mlp_smoke_on_cpu():
    """Reference gated MLP µGraph executes on CPU."""
    result = _run_demo("demo/reference_mugraphs/gated_mlp.py", "--quick")
    assert "reference_muGraph run time (ms):" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_reference_mugraph_plain_matmul_smoke_on_cpu():
    """Reference plain matmul µGraph executes on CPU."""
    result = _run_demo("demo/reference_mugraphs/plain_matmul.py", "--quick")
    assert "reference_muGraph run time (ms):" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_reference_mugraph_matmul_chain_smoke_on_cpu():
    """Reference matmul chain µGraph executes on CPU."""
    result = _run_demo("demo/reference_mugraphs/matmul_chain.py", "--quick")
    assert "reference_muGraph run time (ms):" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_reference_mugraph_concat_matmul_smoke_on_cpu():
    """Reference dual-concat matmul µGraph executes on CPU."""
    result = _run_demo("demo/reference_mugraphs/concat_matmul.py", "--quick")
    assert "reference_muGraph run time (ms):" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_demo_rms_norm_smoke_on_cpu():
    """Smoke-run RMSNorm demo (auto device → cpu on Linux CI)."""
    result = _run_demo("demo/demo_rms_norm.py")
    assert "Best muGraph run time" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_demo_jit_subprocess_without_inherited_ld_library_path(monkeypatch):
    """Demos must set LD_LIBRARY_PATH themselves when the parent env omits it."""
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    result = subprocess.run(
        [sys.executable, "demo/demo_jit.py", "--device", "cpu", "--quiet"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        env=_demo_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "Correctness of output[0]: True" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_llama3b_moe_demo_runs_on_cpu():
    result = _run_demo(
        "demo/llama3b_moe/demo.py",
        "--pytorch-only",
        "--batch-size",
        "1",
        "--seq-len",
        "2",
        "--warmup",
        "1",
        "--repeats",
        "1",
    )
    assert "mean forward latency" in result.stdout
    assert "Sample generation" in result.stdout


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.torch
def test_llama3b_moe_benchmark_runs_on_cpu():
    result = _run_demo(
        "benchmark/end-to-end/llama3b_moe_cpu.py",
        "--skip-search",
        "--batch-size",
        "1",
        "--seq-len",
        "2",
        "--warmup",
        "1",
        "--repeat",
        "1",
    )
    assert "Results" in result.stdout
    assert "PyTorch-only" in result.stdout
