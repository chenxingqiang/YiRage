# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S11: vLLM full-path MLP RF e2e (torch; vllm when installed)."""

from __future__ import annotations

import pytest

from serving_test_utils import require_vllm_installed, serving, torch, vllm_available  # noqa: F401


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


def test_pending_cpu_linear_modules_keep_custom_2d_weights_only(serving, torch):
    """S51 mount selection: custom 2-D weights in, norms / empty / already mounted out.

    vLLM CPU linears are not ``nn.Linear``. Filtering on type would skip the
    modules ``dispatch_cpu_unquantized_gemm`` is supposed to attach, and the
    native forward would raise ``AttributeError`` again.
    """
    from yirage.serving.vllm_e2e import _pending_cpu_linear_modules

    class _CustomLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 3))

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _CustomLinear()

    class _Empty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(0, 3))

    class _Already(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.cpu_linear = object()

    class _Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))

    class _NoWeight(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = "not-a-tensor"

    class _Tree(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrap = _Wrap()
            self.empty = _Empty()
            self.done = _Already()
            self.norm = _Norm()
            self.other = _NoWeight()

    tree = _Tree()
    assert _pending_cpu_linear_modules(tree) == [tree.wrap.inner]


@pytest.mark.skipif(vllm_available(), reason="CPU wheel may attach cpu_linear")
def test_mount_cpu_linear_leaves_weights_untouched_without_wheel(serving, torch):
    """A missing wheel must not mount or drop weights. Parity reads them in place."""
    from yirage.serving.vllm_e2e import _mount_vllm_cpu_linear

    class _CustomLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))

    module = _CustomLinear()
    before = module.weight.detach().clone()
    assert _mount_vllm_cpu_linear(module) == 0
    assert not hasattr(module, "cpu_linear")
    assert torch.equal(module.weight, before)


def test_rf_hook_decoder_step_feeds_attn_plus_residual(serving, torch):
    """S51: the capsule owns post-attn norm + MLP + residual, so the hook sees the sum.

    Feeding ``post_attention_layernorm`` output double-applies the norm. The
    reference side (``_vllm_native_mlp_forward``) can still match a capsule
    that is wrong in the same way; this locks the hook input itself.
    """
    from yirage.serving.vllm_e2e import _rf_hook_decoder_step

    class _Norm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.one_arg = 0
            self.two_arg = 0

        def forward(self, hidden, residual=None):
            if residual is None:
                self.one_arg += 1
                return hidden * 2
            self.two_arg += 1
            return hidden * 2, residual + 1

    class _Attn(torch.nn.Module):
        def forward(self, positions=None, hidden_states=None):
            return hidden_states + 3

    class _Post(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("post_attention_layernorm must stay outside the hook")

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = _Norm()
            self.self_attn = _Attn()
            self.post_attention_layernorm = _Post()

    class _Out:
        def __init__(self, hidden):
            self.hidden = hidden

    class _Hook:
        def __init__(self):
            self.seen = []
            self.override = type("Override", (), {"capsule_name": "cap.mlp"})()

        def forward_mlp(self, mlp_in, rf_meta):
            self.seen.append((mlp_in.detach().clone(), dict(rf_meta)))
            return _Out(mlp_in + 5)

    layer = _Layer()
    hook = _Hook()
    hidden = torch.tensor([[1.0, 2.0]])
    positions = torch.zeros(1, dtype=torch.long)

    out, residual = _rf_hook_decoder_step(
        layer,
        hook,
        positions=positions,
        hidden=hidden,
        residual=None,
        rf_meta_base={"layer": 0},
    )
    expected = hidden * 2 + 3 + hidden
    assert torch.equal(hook.seen[0][0], expected)
    assert hook.seen[0][1]["enabled"] == {"cap.mlp"}
    assert hook.seen[0][1]["layer"] == 0
    assert residual is None
    assert torch.equal(out, expected + 5)
    assert layer.input_layernorm.one_arg == 1
    assert layer.input_layernorm.two_arg == 0

    hidden2 = out
    out2, residual2 = _rf_hook_decoder_step(
        layer,
        hook,
        positions=positions,
        hidden=hidden2,
        residual=residual,
        rf_meta_base={},
    )
    assert residual2 is None
    assert layer.input_layernorm.one_arg == 2
    assert layer.input_layernorm.two_arg == 0
    expected2 = hidden2 * 2 + 3 + hidden2
    assert torch.equal(hook.seen[1][0], expected2)
    assert torch.equal(out2, expected2 + 5)

    live = torch.tensor([[10.0, 20.0]])
    _out3, residual3 = _rf_hook_decoder_step(
        layer,
        hook,
        positions=positions,
        hidden=hidden,
        residual=live,
        rf_meta_base={},
    )
    assert residual3 is None
    assert layer.input_layernorm.two_arg == 1
    assert torch.equal(hook.seen[2][0], (hidden * 2 + 3) + (live + 1))
