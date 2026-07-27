# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S8 contracts: real torch MLP RF hook + segment bench archive (no mock)."""

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


def test_is_vllm_available_bool(serving):
    assert isinstance(serving.is_vllm_available(), bool)


def test_torch_mlp_rf_hook_real_forward(serving):
    serving.require_torch()
    import torch

    layer = serving.TorchDecoderLayer(0, hidden_size=16, intermediate_size=32, seed=1)
    hook = serving.build_torch_mlp_rf_hook(layer)
    x = torch.randn(2, 16, device=layer.device)
    with torch.no_grad():
        h_attn = layer.attention_forward(x)
        ref = layer.mlp_forward(h_attn)
        got = hook.forward_mlp(h_attn, rf_meta={"enabled": {hook.override.capsule_name}})
    assert got.used_rf_mlp
    assert torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-5)


def test_torch_mlp_rf_hook_skip_uses_engine_mlp(serving):
    serving.require_torch()
    import torch

    layer = serving.TorchDecoderLayer(1, hidden_size=16, intermediate_size=32, seed=2)
    hook = serving.build_torch_mlp_rf_hook(layer)
    x = torch.randn(2, 16, device=layer.device)
    with torch.no_grad():
        h_attn = layer.attention_forward(x)
        ref = layer.mlp_forward(h_attn)
        got = hook.forward_mlp(h_attn, rf_meta={"force_skip_all": True})
    assert not got.used_rf_mlp
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
    assert archive.version == "s10"
    hybrid = next(r for r in archive.rows if r.name == "segment_hybrid_torch")
    assert hybrid.parity_ok
    assert hybrid.mean_ms > 0


@pytest.mark.skipif(
    not __import__("yirage.serving.vllm_plugin", fromlist=["is_vllm_available"]).is_vllm_available(),
    reason="requires installed vllm",
)
def test_vllm_plugin_requires_real_package(serving):
    serving.require_vllm()
    assert serving.is_vllm_available()


def test_vllm_hook_raises_without_package(serving):
    if serving.is_vllm_available():
        pytest.skip("vllm installed; skip negative test")
    with pytest.raises(RuntimeError, match="vllm"):
        serving.build_vllm_qwen2_mlp_rf_hook(object())


def test_rf_inspect_version_s8(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s10"
