# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S11: vLLM full-path MLP RF e2e (torch; vllm when installed)."""

from __future__ import annotations

import sys
import types

import pytest

from serving_test_utils import require_vllm_installed, serving, torch  # noqa: F401


def test_torch_vllm_mlp_rf_e2e_parity(serving):
    report = serving.run_torch_vllm_mlp_rf_e2e(
        hidden_size=16,
        intermediate_size=32,
        batch=3,
        bench=False,
    )
    assert report.parity_ok
    assert report.used_rf_mlp
    assert report.plugin == "TorchDecoderMlpRfHook"


def test_torch_vllm_hybrid_full_e2e_parity(serving):
    report = serving.run_torch_vllm_hybrid_full_e2e(
        num_layers=3,
        max_rf_mlp_layers=2,
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        bench=False,
    )
    assert report.parity_ok
    assert report.rf_layer_ids == [0, 1]
    assert report.plugin == "HybridModelOverride+TorchDecoderMlpRfHook"


def test_torch_vllm_hybrid_with_paged_kv_meta(serving, torch):
    report = serving.run_torch_vllm_hybrid_full_e2e(
        num_layers=2,
        max_rf_mlp_layers=1,
        hidden_size=8,
        intermediate_size=16,
        batch=2,
        bench=False,
        rf_meta={
            "block_tables": [[1, 2, -1], [3, 4, -1]],
            "seq_lens": [32, 18],
            "page_size": 16,
        },
    )
    assert report.parity_ok


def test_vllm_e2e_auto_returns_report(serving):
    report = serving.run_vllm_mlp_rf_e2e_auto(
        hidden_size=16,
        intermediate_size=32,
        batch=2,
        bench=False,
    )
    assert hasattr(report, "parity_ok")
    assert report.parity_ok


def test_vllm_qwen2_mlp_rf_e2e(serving):
    require_vllm_installed()
    pytest.importorskip("transformers")
    report = serving.run_vllm_qwen2_mlp_rf_e2e(
        hidden_size=64,
        intermediate_size=128,
        batch=2,
        bench=False,
    )
    assert report.plugin == "VllmQwen2MlpRfHook"
    assert report.parity_ok
    assert report.used_rf_mlp


def test_rf_inspect_version_s11(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s50"


def test_torch_vllm_hybrid_tiny_parity(serving):
    """Tiny torch hybrid e2e: RF MLP vs engine hidden parity (no vllm)."""
    import math

    report = serving.run_torch_vllm_hybrid_full_e2e(
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        batch=2,
        bench=False,
    )
    assert report.parity_ok is True
    payload = report.to_dict() if hasattr(report, "to_dict") else {}
    for key, value in payload.items():
        lowered = str(key).lower()
        if "hidden" not in lowered and "token" not in lowered:
            continue
        if not any(
            token in lowered
            for token in ("diff", "abs", "err", "delta", "max", "match", "cmp", "compare")
        ):
            continue
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            assert math.isfinite(float(value))
            continue
        isfinite = getattr(value, "isfinite", None)
        if callable(isfinite):
            assert bool(isfinite().all())


def _bind_fake_vllm(monkeypatch, *, cpu: bool, dispatch):
    """Install a throwaway vLLM package so mount/context branches run without a wheel."""

    def _pkg(name):
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__package__ = name
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    vllm = _pkg("vllm")
    platforms = _pkg("vllm.platforms")
    vllm.platforms = platforms

    class _Platform:
        def is_cpu(self):
            return cpu

    platforms.current_platform = _Platform()
    executor = _pkg("vllm.model_executor")
    layers = _pkg("vllm.model_executor.layers")
    utils = _pkg("vllm.model_executor.layers.utils")
    vllm.model_executor = executor
    executor.layers = layers
    layers.utils = utils
    utils.dispatch_cpu_unquantized_gemm = dispatch
    return utils


def test_native_mlp_forward_falls_back_to_residual_adapter(serving, torch, monkeypatch):
    """S51: a missing cpu_linear must use residual MLP, not the bare norm output.

    The native branch is ``hidden + mlp(norm(hidden))``. When that call raises,
    the torch adapter is the reference. Returning the partial norm (or letting
    ``ValueError`` disappear into the same handler) would make native parity
    match a capsule that dropped the residual.
    """
    from yirage.serving import vllm_e2e
    from yirage.serving.torch_exec import mlp_torch
    from yirage.serving.vllm_plugin import VllmMlpWeightView

    hidden = torch.tensor([[1.0, -2.0], [0.5, 4.0]])
    rms = torch.ones(2)
    eye = torch.eye(2)
    view = VllmMlpWeightView(
        layer_id=1,
        hidden_size=2,
        intermediate_size=2,
        rms_weight=rms,
        w_gate=eye,
        w_up=eye.clone(),
        w_down=eye.clone(),
    )
    extracted = []

    def _extract(layer, *, layer_id=None):
        extracted.append(layer)
        return view

    monkeypatch.setattr(vllm_e2e, "extract_qwen2_mlp_weights", _extract)
    expected = mlp_torch(hidden, rms_weight=rms, w_gate=eye, w_up=eye, w_down=eye)
    bare = expected - hidden

    class _Raise(torch.nn.Module):
        def __init__(self, exc_type):
            super().__init__()
            self.exc_type = exc_type

        def forward(self, hidden):
            raise self.exc_type("cpu_linear missing")

    class _Layer(torch.nn.Module):
        def __init__(self, exc_type):
            super().__init__()
            self.post_attention_layernorm = _Raise(exc_type)
            self.mlp = _Raise(exc_type)

    for exc_type in (AttributeError, RuntimeError, NotImplementedError):
        out = vllm_e2e._vllm_native_mlp_forward(_Layer(exc_type), hidden)
        assert torch.allclose(out, expected)
        assert not torch.allclose(out, bare)

    class _Norm(torch.nn.Module):
        def forward(self, hidden):
            return hidden * 0.25

    class _Mlp(torch.nn.Module):
        def forward(self, hidden):
            raise RuntimeError("dispatch failed after norm")

    class _Partial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = _Norm()
            self.mlp = _Mlp()

    partial = _Partial()
    out = vllm_e2e._vllm_native_mlp_forward(partial, hidden)
    assert torch.allclose(out, expected)
    assert not torch.allclose(out, partial.post_attention_layernorm(hidden))
    assert extracted[-1] is partial

    class _Bug(torch.nn.Module):
        def forward(self, hidden):
            raise ValueError("shape bug")

    class _Bad(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = _Bug()
            self.mlp = _Bug()

    with pytest.raises(ValueError, match="shape bug"):
        vllm_e2e._vllm_native_mlp_forward(_Bad(), hidden)


def test_decoder_stack_logits_add_residual_only_when_present(serving, torch):
    """Greedy projection must keep a live residual and must not invent one."""
    from yirage.serving.vllm_e2e import _vllm_decoder_stack_logits_hidden

    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(_vllm_decoder_stack_logits_hidden(hidden, None), hidden)
    residual = torch.tensor([[0.5, -1.0], [1.0, 0.0]])
    assert torch.equal(
        _vllm_decoder_stack_logits_hidden(hidden, residual),
        hidden + residual,
    )


def test_mount_cpu_linear_skips_dispatch_off_cpu(serving, torch, monkeypatch):
    """A CUDA platform must not attach cpu_linear or drop in-place weights."""
    from yirage.serving.vllm_e2e import _mount_vllm_cpu_linear

    calls = []

    def _dispatch(module, remove_weight=True):
        calls.append((module, remove_weight))

    _bind_fake_vllm(monkeypatch, cpu=False, dispatch=_dispatch)

    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))

    module = _Linear()
    before = module.weight.detach().clone()
    assert _mount_vllm_cpu_linear(module) == 0
    assert calls == []
    assert not hasattr(module, "cpu_linear")
    assert torch.equal(module.weight, before)


def test_mount_cpu_linear_continues_when_one_dispatch_fails(serving, torch, monkeypatch):
    """One rejected linear must not abort the rest or delete its weight."""
    from yirage.serving.vllm_e2e import _mount_vllm_cpu_linear

    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

    class _Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))

    class _Empty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(0, 2))

    class _Done(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.cpu_linear = "already"

    class _Tree(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bad = _Linear()
            self.good = _Linear()
            self.norm = _Norm()
            self.empty = _Empty()
            self.done = _Done()

    tree = _Tree()
    seen = []

    def _dispatch(module, remove_weight=True):
        seen.append((module, remove_weight))
        if module is tree.bad:
            raise RuntimeError("no isa for this shape")
        module.cpu_linear = "mounted"

    _bind_fake_vllm(monkeypatch, cpu=True, dispatch=_dispatch)
    before_bad = tree.bad.weight.detach().clone()
    before_good = tree.good.weight.detach().clone()
    assert _mount_vllm_cpu_linear(tree) == 1
    assert [item[0] for item in seen] == [tree.bad, tree.good]
    assert all(remove_weight is False for _, remove_weight in seen)
    assert not hasattr(tree.bad, "cpu_linear")
    assert tree.good.cpu_linear == "mounted"
    assert tree.done.cpu_linear == "already"
    assert torch.equal(tree.bad.weight, before_bad)
    assert torch.equal(tree.good.weight, before_good)


def test_standalone_decode_context_refuses_non_cpu_platform(serving, monkeypatch):
    """Non-CPU standalone attention would read uninitialized metadata.

    The refusal has to be ``NotImplementedError`` so ``_native_tier_or_none``
    can degrade the tier. ImportError is a different branch.
    """
    from yirage.serving.vllm_e2e import _vllm_standalone_decode_forward_context

    def _dispatch(module, remove_weight=True):
        raise AssertionError("dispatch must not run while building attention context")

    _bind_fake_vllm(monkeypatch, cpu=False, dispatch=_dispatch)
    with pytest.raises(NotImplementedError, match="CPU-wheel only"):
        _vllm_standalone_decode_forward_context([], vcfg=None, batch=1, device="cpu")
