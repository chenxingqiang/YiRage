"""Contract tests for MACA HF Qwen scaffold and MetaX rebuild script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "python"


def _load_module(name: str, path: Path):
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_qwen_hf_utils_default_dims_match_qwen3_8b():
    hf = _load_module("qwen_hf_utils", _REPO / "demo" / "maca" / "qwen_hf_utils.py")
    dims = hf.default_qwen_dims()
    assert dims.hidden_size == 4096
    assert dims.intermediate_size == 12288
    assert dims.num_heads == 32
    assert dims.num_kv_heads == 8
    assert dims.fused_qkv_outdim == (32 + 2 * 8) * 128


def test_qwen_dims_from_hf_config_object():
    hf = _load_module("qwen_hf_utils", _REPO / "demo" / "maca" / "qwen_hf_utils.py")

    class _Cfg:
        hidden_size = 4096
        intermediate_size = 12288
        num_attention_heads = 32
        num_key_value_heads = 8

    dims = hf.qwen_dims_from_hf_config(_Cfg())
    assert dims == hf.default_qwen_dims()


def test_qwen_inference_demo_exposes_hf_flags():
    demo = _REPO / "demo" / "maca" / "qwen_inference_demo.py"
    text = demo.read_text(encoding="utf-8")
    assert "--model" in text
    assert "--config-only" in text
    assert "--from-pretrained" in text
    assert "qwen_hf_utils" in text


def test_maca_rebuild_core_script_exists_and_documents_smem():
    script = _REPO / "scripts" / "maca_rebuild_core.sh"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "YIRAGE_BACKEND=maca" in text
    assert "get_shared_memory_capacity" in text
    assert "65536" in text
    assert text.startswith("#!/usr/bin/env bash")


def test_qwen_from_pretrained_demo_exists_and_uses_maca_modeling():
    demo = _REPO / "demo" / "maca" / "qwen_from_pretrained_demo.py"
    assert demo.is_file()
    text = demo.read_text(encoding="utf-8")
    assert "modeling_qwen2_maca" in text
    assert "from_pretrained" in text
    assert "superoptimize_kernels" in text
    assert "demo/qwen2.5/demo.py" in text
    assert "--max-layers" in text


def test_modeling_qwen2_maca_superoptimize_uses_maca_backend():
    modeling = _REPO / "demo" / "maca" / "models" / "modeling_qwen2_maca.py"
    assert modeling.is_file()
    text = modeling.read_text(encoding="utf-8")
    assert "superoptimize_mlp_gate_up" in text
    assert "superoptimize_attn_qkv" in text
    assert "qwen_kernel_utils" in text
    assert "import flashinfer" not in text


def test_qwen_kernel_utils_maca_search_contract():
    utils_path = _REPO / "demo" / "maca" / "qwen_kernel_utils.py"
    text = utils_path.read_text(encoding="utf-8")
    assert 'backend": "maca"' in text or "backend='maca'" in text
    assert "maca_search_kwargs" in text
    assert "superoptimize_mlp_gate_up" in text


@pytest.mark.skipif(
    not (_REPO / "demo" / "maca" / "qwen_hf_utils.py").is_file(),
    reason="qwen_hf_utils missing",
)
def test_resolve_qwen_dims_without_model_uses_builtin():
    hf = _load_module("qwen_hf_utils", _REPO / "demo" / "maca" / "qwen_hf_utils.py")
    assert hf.resolve_qwen_dims(None) == hf.default_qwen_dims()
