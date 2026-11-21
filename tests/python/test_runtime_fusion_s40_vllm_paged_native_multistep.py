# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S40: Native vLLM tier for paged multistep bench (G7 chain C paged native)."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving, vllm_available


def _synthetic_paged_multistep_native(*, version: str = "s40") -> dict:
    steps = 3
    ok = [True] * steps
    tok = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return {
        "serving_vllm_paged_multistep_bench": True,
        "version": version,
        "parity_ok": True,
        "token_match_ok": True,
        "decode_steps": steps,
        "paged_kv_bridged": True,
        "functional_chain": "chain_c_vllm_paged_multistep",
        "step_parity_ok": ok,
        "step_token_match_ok": ok,
        "engine_token_ids": tok,
        "hybrid_token_ids": tok,
        "vllm_native_available": False,
        "native_parity_ok": None,
        "native_step_parity_ok": [],
    }


def test_runtime_fusion_version_s40(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s50"


def test_paged_multistep_native_fields_contract(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        quick=True,
        try_native=False,
        version="s40",
    )
    payload = report.to_dict()
    assert "vllm_native_available" in payload
    assert "native_parity_ok" in payload
    assert "native_step_parity_ok" in payload
    assert payload["vllm_native_available"] is False
    assert payload["native_parity_ok"] is None
    # try_native is the master gate: child tiers stay unset even though their
    # own flags default to True. Quick geometry must stay on a kernel-legal
    # head_dim (vLLM 0.26 CPU attention rejects head_dim < 32).
    assert payload["native_full_layer_parity_ok"] is None
    assert payload["native_decoder_parity_ok"] is None
    assert payload["native_decoder_generation_parity_ok"] is None
    assert payload["hidden_size"] >= 32
    from yirage.serving.vllm_e2e import native_tier_qwen2_head_layout

    heads, kv_heads = native_tier_qwen2_head_layout(payload["hidden_size"])
    assert heads == kv_heads >= 1
    assert payload["hidden_size"] // heads in (32, 64, 128)
    json.dumps(payload)


def test_paged_multistep_torch_gate_unchanged(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=False,
        version="s40",
    )
    assert report.parity_ok
    assert report.token_match_ok
    assert report.paged_kv_bridged


@pytest.mark.skipif(not vllm_available(), reason="vllm not installed")
def test_paged_multistep_native_tier_when_vllm(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=True,
        version="s40",
    )
    assert report.vllm_native_available is True
    assert report.native_parity_ok is True
    assert len(report.native_step_parity_ok) == 3
    assert all(report.native_step_parity_ok)


def test_combined_metadata_native_paged_fields(serving):
    from yirage.serving.combined_nightly_archive import serving_combined_nightly_archive_metadata
    from test_runtime_fusion_s34_combined_nightly import _synthetic_combined_archive

    payload = _synthetic_combined_archive(version="s40")
    payload["paged_multistep"]["vllm_native_available"] = False
    payload["paged_multistep"]["native_parity_ok"] = None
    meta = serving_combined_nightly_archive_metadata(
        payload, archive_path="artifacts/combined.json", validation_ok=True, quick=True
    )
    assert "paged_multistep_native_parity_ok" in meta


@pytest.mark.parametrize(
    ("hidden_size", "heads", "head_dim"),
    [
        (32, 1, 32),
        (64, 1, 64),
        (96, 3, 32),
        (128, 1, 128),
        (192, 3, 64),
        (256, 2, 128),
        (320, 5, 64),
    ],
)
def test_native_tier_head_layout_picks_largest_legal_head_dim(hidden_size, heads, head_dim):
    """S51: CPU attention rejects head_dim 16. Layout must take the largest of 32/64/128."""
    from yirage.serving.vllm_e2e import native_tier_qwen2_head_layout

    num_heads, num_kv_heads = native_tier_qwen2_head_layout(hidden_size)
    assert (num_heads, num_kv_heads) == (heads, heads)
    assert hidden_size // num_heads == head_dim


def test_native_tier_head_layout_does_not_revive_rejected_head_dim_16():
    """Pre-S51 tiny default was hidden 64 / 4 heads (head_dim 16), which the kernel rejects."""
    from yirage.serving.vllm_e2e import native_tier_qwen2_head_layout

    num_heads, num_kv_heads = native_tier_qwen2_head_layout(64)
    assert (num_heads, num_kv_heads) == (1, 1)
    assert 64 // num_heads == 64


@pytest.mark.parametrize("hidden_size", [0, -32, -1, 1, 16, 31, 48])
def test_native_tier_head_layout_rejects_unsupported_hidden_size(hidden_size):
    from yirage.serving.vllm_e2e import native_tier_qwen2_head_layout

    with pytest.raises(NotImplementedError, match="head_dim"):
        native_tier_qwen2_head_layout(hidden_size)


def test_native_mlp_reference_includes_residual(serving):
    """S51 reference is hidden + mlp(norm(hidden)), not the bare MLP output.

    The pre-fix native branch omitted the residual add and only matched the
    capsule because the native call crashed into the torch fallback.
    """
    serving.require_torch()
    import torch

    from yirage.serving.vllm_e2e import _vllm_native_mlp_forward

    class _Norm(torch.nn.Module):
        def forward(self, hidden):
            return hidden * 0.5

    class _Mlp(torch.nn.Module):
        def forward(self, hidden):
            return hidden + 1.0

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = _Norm()
            self.mlp = _Mlp()

    hidden = torch.tensor([[2.0, 4.0], [0.0, -2.0]])
    out = _vllm_native_mlp_forward(_Block(), hidden)
    expected = hidden + (hidden * 0.5 + 1.0)
    assert torch.equal(out, expected)
    assert not torch.equal(out, hidden * 0.5 + 1.0)


def test_native_tier_or_none_degrades_only_wheel_failures():
    """Wheel gaps become None. Programming errors must still escape the torch gate."""
    from yirage.serving.vllm_paged_multistep_bench import _native_tier_or_none

    assert _native_tier_or_none(lambda value: value + 1, value=2) == 3

    def _raise(exc_type):
        def _fn(**_kwargs):
            raise exc_type("native tier unavailable")

        return _fn

    for exc_type in (AttributeError, RuntimeError, NotImplementedError):
        assert _native_tier_or_none(_raise(exc_type)) is None

    def _value_error(**_kwargs):
        raise ValueError("caller bug")

    with pytest.raises(ValueError, match="caller bug"):
        _native_tier_or_none(_value_error)


@pytest.mark.skipif(vllm_available(), reason="CPU wheel path is covered by native tier benches")
def test_standalone_decode_context_degrades_without_cpu_backend():
    """Import failure must be NotImplementedError so _native_tier_or_none can degrade it.

    ImportError is not in that wrapper's allowlist; leaking it would fail the torch gate.
    """
    from yirage.serving.vllm_e2e import _vllm_standalone_decode_forward_context

    with pytest.raises(NotImplementedError, match="CPU wheel"):
        _vllm_standalone_decode_forward_context([], vcfg=None, batch=1, device="cpu")


@pytest.mark.skipif(not vllm_available(), reason="vllm not installed")
def test_native_child_tiers_stay_off_when_flags_disabled(serving):
    """Master gate on does not force the full-layer or decoder tiers."""
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=2,
        quick=True,
        try_native=True,
        try_native_full_layer=False,
        try_native_decoder=False,
        version="s40",
    )
    assert report.vllm_native_available is True
    assert report.native_parity_ok is True
    assert report.native_full_layer_parity_ok is None
    assert report.native_decoder_parity_ok is None
    assert report.native_decoder_generation_parity_ok is None
    assert report.native_full_layer_step_parity_ok == []
    assert report.native_decoder_step_parity_ok == []


def test_cpu_cert_manifest_s40_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s40_contract" in names
