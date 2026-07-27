# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S8 contracts: vLLM plugin duck-type + torch segment bench archive."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _import_serving():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules or not hasattr(sys.modules["yirage"], "__path__"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    for key in list(sys.modules):
        if key == "yirage.serving" or key.startswith("yirage.serving."):
            del sys.modules[key]
    return importlib.import_module("yirage.serving")


@pytest.fixture(scope="module")
def serving():
    return _import_serving()


class _Lin:
    def __init__(self, w):
        self.weight = w


class _Norm:
    def __init__(self, w):
        self.weight = w


class _Mlp:
    def __init__(self, g, u, d):
        self.gate_proj = _Lin(g)
        self.up_proj = _Lin(u)
        self.down_proj = _Lin(d)

    def __call__(self, hidden):
        import torch.nn.functional as F

        gate = hidden @ self.gate_proj.weight.t()
        up = hidden @ self.up_proj.weight.t()
        mid = F.silu(gate) * up
        return mid @ self.down_proj.weight.t()


class _MockVllmLayer:
    def __init__(self):
        import torch

        h, i = 16, 32
        self.post_attention_layernorm = _Norm(torch.ones(h))
        self.mlp = _Mlp(
            torch.randn(i, h) * 0.02,
            torch.randn(i, h) * 0.02,
            torch.randn(h, i) * 0.02,
        )


def test_is_vllm_available_bool(serving):
    assert isinstance(serving.is_vllm_available(), bool)


def test_extract_qwen2_mlp_weights_mock(serving):
    mock = _MockVllmLayer()
    view = serving.extract_qwen2_mlp_weights(mock, layer_id=1)
    assert view.hidden_size == 16
    assert view.intermediate_size == 32
    assert view.w_gate.shape == (16, 32)


def test_vllm_hook_forward_mlp_only(serving):
    serving.require_torch()
    import torch

    mock = _MockVllmLayer()
    hook = serving.build_vllm_qwen2_mlp_rf_hook(mock, layer_id=0)
    x = torch.randn(2, 16)
    ref = serving.mlp_torch(
        x,
        rms_weight=mock.post_attention_layernorm.weight,
        w_gate=mock.mlp.gate_proj.weight.t(),
        w_up=mock.mlp.up_proj.weight.t(),
        w_down=mock.mlp.down_proj.weight.t(),
    )
    got = hook.forward_mlp(x, rf_meta={"enabled": {hook.override.capsule_name}})
    assert got.used_rf_mlp
    assert torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-5)


def test_segment_torch_bench_archive_parity(serving):
    serving.require_torch()
    archive = serving.run_segment_torch_bench_archive(
        num_layers=3,
        segment_layer_ids=(1,),
        rf_mlp_layer_ids=(0,),
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        warmup=1,
        iters=5,
    )
    assert archive.version == "s8"
    hybrid = next(r for r in archive.rows if r.name == "segment_hybrid_torch")
    assert hybrid.parity_ok
    assert hybrid.mean_ms > 0


def test_forward_mlp_only_skip_fallback(serving):
    serving.require_torch()
    import torch

    mock = _MockVllmLayer()
    hook = serving.build_vllm_qwen2_mlp_rf_hook(mock, layer_id=0)
    x = torch.randn(2, 16)
    got = hook.forward_mlp(x, rf_meta={"force_skip_all": True})
    assert not got.used_rf_mlp


def test_rf_inspect_version_s8(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s8"
