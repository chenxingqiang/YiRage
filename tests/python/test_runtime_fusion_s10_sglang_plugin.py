# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S10: SGLang model-layer MLP RF hook + ForwardBatch meta (torch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pytest

from serving_test_utils import sglang_available, serving, torch  # noqa: F401


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


@pytest.mark.skipif(not sglang_available(), reason="requires installed sglang")
def test_sglang_plugin_requires_package(serving):
    serving.require_sglang()
    assert serving.is_sglang_available()


def test_rf_inspect_version_s10(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s50"


def test_split_gate_up_weight_keeps_gate_then_up(serving, torch):
    """Column-parallel (2I, H) splits rows; transposed (H, 2I) splits columns.

    Swapping the orientation, or splitting the short axis, exchanges gate and
    up. The RF capsule then computes silu(up)*gate and still has legal shapes.
    """
    split = serving.sglang_plugin._split_gate_up_weight
    column = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    gate, up = split(column)
    assert torch.equal(gate, column[:4])
    assert torch.equal(up, column[4:])
    gate_np, up_np = split(column.detach().numpy())
    np.testing.assert_array_equal(gate_np, column.numpy()[:4])
    np.testing.assert_array_equal(up_np, column.numpy()[4:])

    row = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    gate, up = split(row)
    assert torch.equal(gate, row[:, :4])
    assert torch.equal(up, row[:, 4:])
    gate_np, up_np = split(row.detach().numpy())
    np.testing.assert_array_equal(gate_np, row.numpy()[:, :4])
    np.testing.assert_array_equal(up_np, row.numpy()[:, 4:])

    # Odd hidden size is still a transposed fused proj when the last dim is
    # even and at least twice the hidden size.
    odd_hidden = np.arange(50, dtype=np.float32).reshape(5, 10)
    gate_np, up_np = split(odd_hidden)
    np.testing.assert_array_equal(gate_np, odd_hidden[:, :5])
    np.testing.assert_array_equal(up_np, odd_hidden[:, 5:])


def test_split_gate_up_weight_rejects_rank_and_ambiguous_shapes(serving):
    split = serving.sglang_plugin._split_gate_up_weight
    with pytest.raises(ValueError, match="rank-2"):
        split(np.arange(8))
    with pytest.raises(ValueError, match="rank-2"):
        split(np.arange(8).reshape(2, 2, 2))
    with pytest.raises(ValueError, match="cannot infer"):
        split(np.arange(16, dtype=np.float32).reshape(4, 4))
    with pytest.raises(ValueError, match="cannot infer"):
        split(np.arange(6, dtype=np.float32).reshape(2, 3))


def test_to_host_list_detaches_tensors_and_wraps_scalars(serving, torch):
    """Host conversion accepts engine tensors and a numpy scalar length.

    A tensor that requires grad must be detached before ``tolist``. A numpy
    scalar length is one request: extend 0 is a Radix hit, not a crash.
    """
    host = serving.sglang_plugin._to_host_list
    assert host(None) is None
    assert host(7) == [7]
    assert host(np.array([0, 4], dtype=np.int64)) == [0, 4]
    assert host(torch.tensor([0, 4], dtype=torch.int64)) == [0, 4]
    graded = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    assert host(graded) == [1.0, 2.0]
    assert graded.requires_grad

    @dataclass
    class _ScalarExtend:
        extend_lens: Any
        seq_lens: Any

    hit = serving.rf_step_meta_from_forward_batch(
        _ScalarExtend(extend_lens=np.int64(0), seq_lens=np.int64(8)),
        enabled={"mlp_layer_0"},
    )
    np.testing.assert_array_equal(hit["radix_hit_mask"], [True])
    assert hit["extras"]["sglang"]["extend_lens"] == [0]
    miss = serving.rf_step_meta_from_forward_batch(
        _ScalarExtend(extend_lens=np.int64(2), seq_lens=np.int64(8)),
        enabled={"mlp_layer_0"},
    )
    np.testing.assert_array_equal(miss["radix_hit_mask"], [False])
    assert miss["extras"]["sglang"]["extend_lens"] == [2]


def test_forward_batch_aliases_extend_lens_and_req_to_token(serving, torch):
    """SGLang names the same fields ``extend_lens`` and ``req_to_token``.

    ``extend_seq_lens`` wins when both are present. Dropping either alias
    drops Radix hits or the paged KV bridge while list-shaped tests stay green.
    """

    @dataclass
    class _AliasBatch:
        extend_lens: Any
        seq_lens: Any
        req_to_token: Any
        page_size: int = 16

    batch = _AliasBatch(
        extend_lens=torch.tensor([0, 2], dtype=torch.int64),
        seq_lens=torch.tensor([32, 18], dtype=torch.int64),
        req_to_token=torch.tensor([[1, 2, -1], [3, 4, -1]], dtype=torch.int64),
    )
    meta = serving.rf_step_meta_from_forward_batch(batch, enabled={"mlp_layer_0"})
    np.testing.assert_array_equal(meta["radix_hit_mask"], [True, False])
    assert meta["extras"]["sglang"]["extend_lens"] == [0, 2]
    assert meta["extras"]["paged_kv"]["block_tables"] == [[1, 2, -1], [3, 4, -1]]
    assert meta["extras"]["paged_kv"]["seq_lens"] == [32, 18]

    @dataclass
    class _BothNames:
        extend_seq_lens: List[int]
        extend_lens: List[int]
        seq_lens: List[int]

    preferred = serving.rf_step_meta_from_forward_batch(
        _BothNames(extend_seq_lens=[0, 1], extend_lens=[1, 0], seq_lens=[8, 8]),
        enabled={"mlp_layer_0"},
    )
    np.testing.assert_array_equal(preferred["radix_hit_mask"], [True, False])
    assert preferred["extras"]["sglang"]["extend_lens"] == [0, 1]


def test_forward_batch_all_miss_keeps_capsules_enabled(serving):
    """All-miss extend lengths leave capsules enabled (no force skip, no all-hit)."""
    batch = _ForwardBatchLike(extend_seq_lens=[1, 1], seq_lens=[8, 8])
    enabled = {"mlp_layer_0"}
    meta = serving.rf_step_meta_from_forward_batch(batch, enabled=enabled)
    assert not meta.get("force_skip_all", False)
    mask = np.asarray(meta["radix_hit_mask"], dtype=bool)
    assert mask.shape == (2,)
    assert not bool(np.all(mask))
    radix = serving.parse_radix_hit_mask(mask, batch_size=2)
    assert radix is not None and not radix.all_hit
    assert meta["enabled"] == enabled
    assert meta["extras"]["radix_hit"]["all_hit"] is False
    assert meta["extras"]["sglang"]["extend_lens"] == [1, 1]
