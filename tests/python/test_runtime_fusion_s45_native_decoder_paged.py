# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S45: Native vLLM full decoder paged multistep tier (G7 chain C paged native decoder)."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving, vllm_available
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard
from test_runtime_fusion_s42_nightly_bundle_validate import _synthetic_bundle_artifacts


def _synthetic_paged_multistep_native_decoder(*, version: str = "s45") -> dict:
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
        "num_layers": 2,
        "step_parity_ok": ok,
        "step_token_match_ok": ok,
        "engine_token_ids": tok,
        "hybrid_token_ids": tok,
        "generation_parity_ok": True,
        "generation_token_match_ok": True,
        "generation_engine_token_ids": tok,
        "generation_hybrid_token_ids": tok,
        "vllm_native_available": False,
        "native_parity_ok": None,
        "native_step_parity_ok": [],
        "native_full_layer_parity_ok": None,
        "native_full_layer_step_parity_ok": [],
        "native_decoder_parity_ok": None,
        "native_decoder_token_match_ok": None,
        "native_decoder_step_parity_ok": [],
        "native_decoder_step_token_match_ok": [],
        "native_decoder_ref_token_ids": [],
        "native_decoder_rf_token_ids": [],
    }


def test_runtime_fusion_version_s45(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s50"


def test_paged_multistep_native_decoder_fields_contract(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        quick=True,
        try_native=False,
        try_native_full_layer=False,
        try_native_decoder=False,
        version="s45",
    )
    payload = report.to_dict()
    assert "native_decoder_parity_ok" in payload
    assert "native_decoder_token_match_ok" in payload
    assert "native_decoder_step_parity_ok" in payload
    assert "native_decoder_step_token_match_ok" in payload
    assert payload["native_decoder_parity_ok"] is None
    assert payload["native_decoder_token_match_ok"] is None
    json.dumps(payload)


def test_paged_multistep_torch_gate_unchanged_s45(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=False,
        try_native_full_layer=False,
        try_native_decoder=False,
        version="s45",
    )
    assert report.parity_ok
    assert report.token_match_ok
    assert report.paged_kv_bridged


@pytest.mark.skipif(not vllm_available(), reason="vllm not installed")
def test_paged_multistep_native_decoder_when_vllm(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=True,
        try_native_full_layer=True,
        try_native_decoder=True,
        version="s45",
    )
    assert report.vllm_native_available is True
    assert report.native_decoder_parity_ok is True
    assert report.native_decoder_token_match_ok is True
    assert len(report.native_decoder_step_parity_ok) == 3
    assert len(report.native_decoder_step_token_match_ok) == 3
    assert report.native_decoder_ref_token_ids == report.native_decoder_rf_token_ids


def test_combined_archive_metadata_native_decoder_fields(serving):
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )

    payload = {
        "serving_combined_nightly_archive": True,
        "version": "s45",
        "parity_ok": True,
        "quick": True,
        "decode": {"parity_ok": True},
        "engine_g1": {"parity_ok": True},
        "multistep": {"parity_ok": True, "token_match_ok": True},
        "engine_multistep": {"parity_ok": True},
        "paged_multistep": _synthetic_paged_multistep_native_decoder(),
    }
    meta = serving_combined_nightly_archive_metadata(
        payload, archive_path="artifacts/combined.json", validation_ok=True, quick=True
    )
    assert "paged_multistep_native_decoder_parity_ok" in meta
    assert "paged_multistep_native_decoder_token_match_ok" in meta
    assert meta["paged_multistep_native_decoder_parity_ok"] is None
    assert meta["paged_multistep_native_decoder_token_match_ok"] is None


def test_bundle_metadata_includes_native_decoder_fields(serving):
    from yirage.serving.serving_nightly_bundle import (
        serving_combined_nightly_bundle_metadata,
        validate_serving_combined_nightly_bundle_metadata,
    )

    archive = _synthetic_combined_for_dashboard(version="s45")
    archive["paged_multistep"] = _synthetic_paged_multistep_native_decoder()
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        serving_dashboard_artifact_metadata,
    )

    archive_meta = serving_combined_nightly_archive_metadata(
        archive,
        archive_path="artifacts/serving-combined-nightly.json",
        validation_ok=True,
        quick=True,
    )
    dashboard = build_serving_dashboard_from_combined_archive(archive).to_dict()
    dashboard_meta = serving_dashboard_artifact_metadata(
        dashboard,
        json_path="artifacts/serving-dashboard.json",
        validation_ok=True,
    )
    meta = serving_combined_nightly_bundle_metadata(
        archive_payload=archive,
        archive_metadata=archive_meta,
        dashboard_metadata=dashboard_meta,
        validation_ok=True,
        archive_path="artifacts/serving-combined-nightly.json",
        archive_meta_path="artifacts/serving-combined-nightly.meta.json",
        dashboard_json_path="artifacts/serving-dashboard.json",
        dashboard_meta_path="artifacts/serving-dashboard.meta.json",
    )
    assert "paged_multistep_native_decoder_parity_ok" in meta
    assert "paged_multistep_native_decoder_token_match_ok" in meta
    assert validate_serving_combined_nightly_bundle_metadata(meta) == []


def test_bundle_validate_rejects_native_decoder_meta_mismatch(serving):
    from yirage.serving.serving_nightly_bundle import validate_serving_combined_nightly_bundle

    bundle = _synthetic_bundle_artifacts(version="s45")
    bundle["archive"]["paged_multistep"] = _synthetic_paged_multistep_native_decoder()
    bundle["archive_meta"] = serving_combined_nightly_archive_metadata_from_bundle(bundle)
    bad_meta = dict(bundle["archive_meta"])
    bad_meta["paged_multistep_native_decoder_parity_ok"] = True
    errors = validate_serving_combined_nightly_bundle(
        archive_payload=bundle["archive"],
        archive_metadata=bad_meta,
        dashboard_payload=bundle["dashboard"],
        dashboard_metadata=bundle["dashboard_meta"],
        html_document=bundle["html"],
        markdown_document=bundle["markdown"],
    )
    assert any("native_decoder_parity_ok mismatch" in e for e in errors)


def serving_combined_nightly_archive_metadata_from_bundle(bundle: dict) -> dict:
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )

    return serving_combined_nightly_archive_metadata(
        bundle["archive"],
        archive_path="artifacts/serving-combined-nightly.json",
        validation_ok=True,
        quick=True,
    )


def test_dashboard_paged_multistep_native_decoder_metric(serving):
    from yirage.serving.serving_dashboard import build_serving_dashboard_from_combined_archive

    archive = _synthetic_combined_for_dashboard(version="s45")
    archive["paged_multistep"] = _synthetic_paged_multistep_native_decoder()
    archive["paged_multistep"]["native_decoder_parity_ok"] = True
    archive["paged_multistep"]["native_decoder_token_match_ok"] = True
    report = build_serving_dashboard_from_combined_archive(archive)
    paged = [r for r in report.rows if r.section == "paged_multistep"]
    assert len(paged) == 1
    assert paged[0].metrics.get("native_decoder_parity_ok") is True
    assert paged[0].metrics.get("native_decoder_token_match_ok") is True


def test_cpu_cert_manifest_s45_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s45_contract" in names


def test_paged_multistep_torch_gate_when_native_disabled(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        quick=True,
        try_native=False,
        try_native_full_layer=False,
        try_native_decoder=False,
    )
    assert report.parity_ok is True
    assert report.token_match_ok is True
    assert report.native_decoder_parity_ok is None
    assert report.native_decoder_generation_parity_ok is None
    assert report.paged_kv_bridged is True


def test_native_tier_head_layout_selects_largest_kernel_head_dim(serving):
    """S51: CPU attention accepts head_dim 32/64/128; pick the largest divisor."""
    from yirage.serving.vllm_e2e import native_tier_qwen2_head_layout

    cases = (
        (32, 1, 32),
        (64, 1, 64),
        (96, 3, 32),
        (128, 1, 128),
        (192, 3, 64),
        (256, 2, 128),
    )
    for hidden, heads, head_dim in cases:
        num_heads, num_kv_heads = native_tier_qwen2_head_layout(hidden)
        assert (num_heads, num_kv_heads) == (heads, heads)
        assert hidden // num_heads == head_dim


def test_native_tier_head_layout_rejects_kernel_unsafe_hidden_sizes(serving):
    """Sizes the pre-S51 quick default used (hidden 16) must degrade, not crash."""
    from yirage.serving.vllm_e2e import native_tier_qwen2_head_layout

    for hidden in (16, 24, 31, 48, 80):
        with pytest.raises(NotImplementedError, match="head_dim"):
            native_tier_qwen2_head_layout(hidden)


def test_vllm_native_mlp_forward_includes_residual(serving):
    """Reference is hidden + mlp(rmsnorm(hidden)), the block the RF capsule replaces."""
    serving.require_torch()
    import torch

    from yirage.serving.vllm_e2e import _vllm_native_mlp_forward

    class _Norm(torch.nn.Module):
        def forward(self, hidden):
            return hidden * 2

    class _Mlp(torch.nn.Module):
        def forward(self, hidden):
            return hidden + 1

    class _Layer:
        def __init__(self):
            self.post_attention_layernorm = _Norm()
            self.mlp = _Mlp()

    hidden = torch.tensor([[1.0, -2.0, 0.5]])
    layer = _Layer()
    got = _vllm_native_mlp_forward(layer, hidden)
    normed = layer.post_attention_layernorm(hidden)
    assert torch.equal(got, hidden + layer.mlp(normed))
    assert torch.equal(got, torch.tensor([[4.0, -5.0, 2.5]]))


def test_native_tier_or_none_degrades_wheel_errors_only(serving):
    """Wheel failures become None; programming errors still fail the bench."""
    from yirage.serving.vllm_paged_multistep_bench import _native_tier_or_none

    def _raise(exc_type):
        def fn(**kwargs):
            raise exc_type(kwargs.get("msg", "fail"))

        return fn

    assert _native_tier_or_none(lambda **kwargs: kwargs["n"] + 1, n=3) == 4
    for exc in (AttributeError, RuntimeError, NotImplementedError):
        assert _native_tier_or_none(_raise(exc), msg="wheel") is None
    with pytest.raises(ValueError):
        _native_tier_or_none(_raise(ValueError), msg="bug")
    with pytest.raises(TypeError):
        _native_tier_or_none(_raise(TypeError), msg="bug")


_NATIVE_TIER_RUNNERS = (
    "_run_vllm_native_paged_multistep",
    "_run_vllm_native_paged_multistep_full_layer",
    "_run_vllm_native_decoder_paged_multistep",
    "_run_vllm_native_decoder_generation",
)


def _patch_native_tier_runners(monkeypatch, fn):
    from yirage.serving import vllm_paged_multistep_bench as bench

    monkeypatch.setattr(bench, "is_vllm_available", lambda: True)
    for name in _NATIVE_TIER_RUNNERS:
        monkeypatch.setattr(bench, name, fn)
    return bench


def _run_tiny_paged_bench(bench, **overrides):
    kwargs = dict(
        quick=True,
        decode_steps=1,
        num_layers=1,
        batch=1,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=32,
        version="s45",
        try_native_full_layer=True,
        try_native_decoder=True,
    )
    kwargs.update(overrides)
    return bench.run_vllm_paged_multistep_bench(**kwargs)


def test_try_native_master_gate_skips_child_tiers(serving, monkeypatch):
    """Child flags stay on; try_native=False must still skip every native probe."""
    calls = []

    def _forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("native tier ran with try_native=False")

    bench = _patch_native_tier_runners(monkeypatch, _forbidden)
    report = _run_tiny_paged_bench(bench, try_native=False)
    assert calls == []
    assert report.parity_ok is True
    assert report.token_match_ok is True
    assert report.hidden_size == 32
    assert report.vllm_native_available is False
    assert report.native_parity_ok is None
    assert report.native_step_parity_ok == []
    assert report.native_full_layer_parity_ok is None
    assert report.native_full_layer_step_parity_ok == []
    assert report.native_decoder_parity_ok is None
    assert report.native_decoder_token_match_ok is None
    assert report.native_decoder_step_parity_ok == []
    assert report.native_decoder_ref_token_ids == []
    assert report.native_decoder_rf_token_ids == []
    assert report.native_decoder_generation_parity_ok is None
    assert report.native_decoder_generation_token_match_ok is None
    assert report.native_decoder_generation_step_parity_ok == []


def test_native_tier_wheel_failure_keeps_torch_gate(serving, monkeypatch):
    """NotImplementedError from a native probe degrades; the torch gate stays green."""
    calls = []

    def _unsupported(**kwargs):
        calls.append(1)
        raise NotImplementedError("unsupported CPU attention configuration")

    bench = _patch_native_tier_runners(monkeypatch, _unsupported)
    report = _run_tiny_paged_bench(bench, try_native=True)
    assert len(calls) == len(_NATIVE_TIER_RUNNERS)
    assert report.parity_ok is True
    assert report.token_match_ok is True
    assert report.paged_kv_bridged is True
    assert report.vllm_native_available is True
    assert report.native_parity_ok is None
    assert report.native_full_layer_parity_ok is None
    assert report.native_decoder_parity_ok is None
    assert report.native_decoder_generation_parity_ok is None
