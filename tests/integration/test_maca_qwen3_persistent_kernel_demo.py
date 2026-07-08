"""Subprocess smoke for MACA Qwen3 PersistentKernel scaffold demo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "demo" / "maca" / "qwen3_persistent_kernel_demo.py"


def _maca_vm_available() -> bool:
    if os.environ.get("YIRAGE_MACA_INTEGRATION", "") == "1":
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return "MetaX" in torch.cuda.get_device_name(0)
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_compile_plan():
    """Cloud-safe contract: compile-plan validates minimal embed PK prerequisites."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--compile-plan", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_plan"
    assert payload["compile_plan"]["variant"] == "embed_only"
    assert payload["compile_plan"]["minimal_task_graph"] == "embed_layer"


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_compile_plan_one_layer():
    """Cloud-safe contract: one-layer PK compile plan prerequisites."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [
        sys.executable,
        str(_DEMO),
        "--compile-plan",
        "--compile-plan-variant",
        "one_layer",
        "--json",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_plan"
    assert payload["compile_plan"]["variant"] == "one_layer"
    assert "layer[0]" in payload["compile_plan"]["minimal_task_graph"]


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_compile_plan_stack():
    """Cloud-safe contract: stack PK compile plan prerequisites."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [
        sys.executable,
        str(_DEMO),
        "--compile-plan",
        "--compile-plan-variant",
        "stack",
        "--pk-compile-layers",
        "2",
        "--json",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_plan"
    assert payload["compile_plan"]["variant"] == "stack"
    assert payload["compile_plan"]["pk_compile_layers"] == 2
    assert "argmax_reduce" in payload["compile_plan"]["tasks"]


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_runtime_plan():
    """Cloud-safe contract: runtime-plan validates launch prerequisites."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [
        sys.executable,
        str(_DEMO),
        "--runtime-plan",
        "--runtime-plan-variant",
        "stack",
        "--pk-compile-layers",
        "1",
        "--json",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "runtime_plan"
    assert payload["runtime_plan"]["runtime_plan_ready"] is True
    assert payload["runtime_plan"]["pk_runtime_layers"] == 1


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_hf_runtime_plan():
    """Cloud-safe contract: HF runtime plan."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--hf-runtime-plan", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "hf_runtime_plan"
    assert payload["hf_runtime_plan"]["hf_runtime_ready"] is True
    assert payload["hf_runtime_plan"]["weight_injection_status"] == "implemented"


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_hf_weight_plan():
    """Cloud-safe contract: HF weight attach mapping plan."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--hf-weight-plan", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "hf_weight_plan"
    assert payload["hf_weight_plan"]["weight_plan_ready"] is True


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_hf_generation_plan():
    """Cloud-safe contract: HF generation loop scaffold plan."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--hf-generation-plan", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "hf_generation_plan"
    assert payload["hf_generation_plan"]["generation_plan_ready"] is True
    assert payload["hf_generation_plan"]["generation_ready"] is False
    assert payload["hf_generation_plan"]["multi_step_decode_ready"] is True


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_hf_decode_step_plan():
    """Cloud-safe contract: multi-step decode tensor semantics."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--hf-decode-step-plan", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "hf_decode_step_plan"
    assert payload["hf_decode_step_plan"]["decode_step_contract_ready"] is True


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_hf_padded_plan():
    """Cloud-safe contract: padded lm_head 153600 plan."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--hf-padded-plan", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "hf_padded_plan"
    assert payload["hf_padded_plan"]["padded_lm_head_plan_ready"] is True
    assert payload["hf_padded_plan"]["pad_vocab_size"] == 153600


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_hf_weight_plan_multi_layer():
    """Cloud-safe contract: 2-layer HF weight attach plan."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [
        sys.executable,
        str(_DEMO),
        "--hf-weight-plan",
        "--pk-compile-layers",
        "2",
        "--json",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["hf_weight_plan"]["max_layers"] == 2
    assert payload["hf_weight_plan"]["weight_plan_ready"] is True


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_compile_inspect():
    """Cloud-safe contract: compile-inspect validates mxcc PK flags without GPU."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--compile-inspect", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_inspect"
    assert payload["compile_contract"]["compile_ready"] is True


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_inspect_only():
    """Cloud-safe contract: inspect-only exits 0 without MetaX GPU."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--inspect-only", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "inspect_only"
    assert payload["scaffold"]["cuda_reference"] == "demo/qwen3/demo.py --use-yirage"


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_qwen3_persistent_kernel_demo_compile_only():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    env.setdefault("YIRAGE_HOME", str(_REPO))

    cmd = [sys.executable, str(_DEMO), "--compile-only", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_only"
    assert payload["compile"]["compiled"] is True
    assert payload["compile"]["compiler"] == "mxcc"


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_qwen3_persistent_kernel_demo_compile_one_layer():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    env.setdefault("YIRAGE_HOME", str(_REPO))

    cmd = [sys.executable, str(_DEMO), "--compile-one-layer", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_one_layer"
    assert payload["compile"]["compiled"] is True
    assert payload["compile"]["compiler"] == "mxcc"
    assert "paged_attention" in payload["compile"]["tasks"]


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_qwen3_persistent_kernel_demo_compile_stack():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    env.setdefault("YIRAGE_HOME", str(_REPO))

    cmd = [
        sys.executable,
        str(_DEMO),
        "--compile-stack",
        "--pk-compile-layers",
        "2",
        "--json",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_stack"
    assert payload["compile"]["compiled"] is True
    assert payload["compile"]["compiler"] == "mxcc"
    assert payload["compile"]["pk_compile_layers"] == 2
    assert "argmax_reduce" in payload["compile"]["tasks"]


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_qwen3_persistent_kernel_demo_runtime_stack():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    env.setdefault("YIRAGE_HOME", str(_REPO))

    cmd = [
        sys.executable,
        str(_DEMO),
        "--runtime-stack",
        "--pk-compile-layers",
        "1",
        "--json",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "runtime_stack"
    assert payload["runtime"]["compiled"] is True
    assert payload["runtime"]["launched"] is True


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_qwen3_persistent_kernel_demo_quick():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")

    cmd = [sys.executable, str(_DEMO), "--quick", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["pk_runtime"]["backend"] == "maca"


def test_maca_qwen3_persistent_kernel_demo_script_exists():
    assert _DEMO.is_file()
    text = _DEMO.read_text(encoding="utf-8")
    assert "qwen3_pk_utils" in text
    assert "--inspect-only" in text
    assert "--compile-one-layer" in text
    assert "--compile-stack" in text
    assert "--runtime-stack" in text
    assert "--runtime-plan" in text
    assert "--hf-weight-plan" in text
    assert "--hf-runtime-stack" in text
    assert "--hf-padded-plan" in text
    assert "--hf-generation-plan" in text
    assert "--hf-decode-step-plan" in text
    assert "--hf-generation-smoke" in text
