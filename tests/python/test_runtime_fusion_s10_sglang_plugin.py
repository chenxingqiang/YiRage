# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S10: SGLang model-layer MLP RF hook + ForwardBatch meta (real torch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pytest

from serving_real_test_utils import serving, torch  # noqa: F401


@dataclass
class _ForwardBatchLike:
    extend_seq_lens: List[int]
    seq_lens: List[int]
    block_tables: Optional[List[List[int]]] = None
    page_size: int = 16


def test_is_sglang_available_bool(serving):
    assert isinstance(serving.is_sglang_available(), bool)


def test_rf_step_meta_from_forward_batch_like(serving):
    batch = _ForwardBatchLike(
        extend_seq_lens=[0, 2],
        seq_lens=[32, 18],
        block_tables=[[1, 2, -1], [3, 4, -1]],
    )
    meta = serving.rf_step_meta_from_forward_batch(
        batch,
        enabled={"mlp_layer_0"},
    )
    np.testing.assert_array_equal(meta["radix_hit_mask"], [True, False])
    assert meta["extras"]["sglang"]["extend_lens"] == [0, 2]
    assert "paged_kv" in meta["extras"]


def test_sglang_batch_torch_hook_partial_radix(serving, torch):
    layer = serving.TorchDecoderLayer(0, hidden_size=16, intermediate_size=32, seed=11)
    hook = serving.build_sglang_batch_torch_mlp_rf_hook(layer)
    batch = _ForwardBatchLike(extend_seq_lens=[0, 4, 0], seq_lens=[8, 12, 16])
    x = torch.randn(3, 16, device=layer.device)
    with torch.no_grad():
        h_attn = layer.attention_forward(x)
        expected = h_attn.clone()
        expected[1] = layer.mlp_forward(h_attn[1:2])[0]
        got = hook.forward_mlp(h_attn, forward_batch=batch)
    assert got.used_rf_mlp
    assert torch.allclose(got.hidden, expected, rtol=1e-5, atol=1e-5)


def test_sglang_batch_torch_hook_all_radix_skip(serving, torch):
    layer = serving.TorchDecoderLayer(1, hidden_size=8, intermediate_size=16, seed=12)
    hook = serving.build_sglang_batch_torch_mlp_rf_hook(layer)
    batch = _ForwardBatchLike(extend_seq_lens=[0, 0], seq_lens=[4, 4])
    x = torch.randn(2, 8, device=layer.device)
    with torch.no_grad():
        h_attn = layer.attention_forward(x)
        got = hook.forward_mlp(h_attn, forward_batch=batch)
    assert not got.used_rf_mlp
    assert torch.allclose(got.hidden, h_attn)


def test_sglang_qwen2_hook_raises_without_package(serving):
    if serving.is_sglang_available():
        pytest.skip("sglang installed; skip negative test")
    with pytest.raises(RuntimeError, match="sglang"):
        serving.build_sglang_qwen2_mlp_rf_hook(object())


@pytest.mark.skipif(
    not __import__("yirage.serving.sglang_plugin", fromlist=["is_sglang_available"]).is_sglang_available(),
    reason="requires installed sglang",
)
def test_sglang_plugin_requires_real_package(serving):
    serving.require_sglang()
    assert serving.is_sglang_available()


def test_rf_inspect_version_s10(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s17"
