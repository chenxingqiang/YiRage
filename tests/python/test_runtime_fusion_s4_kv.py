# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S4 contracts: block_tables → paged_kv_* bridge + RF.step extras."""

from __future__ import annotations

import numpy as np
import pytest

from serving_test_utils import serving, torch  # noqa: F401


def test_block_tables_to_paged_kv_basic(serving):
    block_tables = np.array(
        [
            [10, 11, -1, -1],
            [7, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    seq_lens = np.array([20, 16], dtype=np.int32)
    paged = serving.block_tables_to_paged_kv(block_tables, seq_lens, page_size=16)
    np.testing.assert_array_equal(paged.paged_kv_indptr, [0, 2, 3])
    np.testing.assert_array_equal(paged.paged_kv_indices, [10, 11, 7])
    np.testing.assert_array_equal(paged.paged_kv_last_page_len, [4, 16])
    assert paged.batch_size == 2


def test_block_tables_rejects_insufficient_pages(serving):
    block_tables = np.array([[1, -1]], dtype=np.int32)
    seq_lens = np.array([33], dtype=np.int32)
    with pytest.raises(ValueError, match="needs >="):
        serving.block_tables_to_paged_kv(block_tables, seq_lens, page_size=16)


def test_attach_paged_kv_to_step_meta(serving):
    meta = serving.attach_paged_kv_to_step_meta(
        {"enabled": {"mlp_layer_0"}},
        block_tables=[[3, 4, -1]],
        seq_lens=[17],
        page_size=16,
    )
    assert "paged_kv" in meta["extras"]
    assert meta["extras"]["paged_kv"]["paged_kv_indptr"] == [0, 2]
    assert meta["extras"]["paged_kv"]["paged_kv_indices"] == [3, 4]
    assert meta["extras"]["paged_kv"]["paged_kv_last_page_len"] == [1]


def test_rf_step_auto_bridges_block_tables(serving, torch):
    seen = {}

    class _ProbeCapsule(serving.FusionCapsule):
        def execute(self, inputs, meta=None):
            seen["meta"] = dict(meta or {})
            return {"hidden": inputs["hidden"]}

    from yirage.serving.plan import FusionPlan

    plan = FusionPlan.mlp(name="probe", hidden_size=4, intermediate_size=8)
    cap = _ProbeCapsule(plan)
    rf = serving.RuntimeFusion([cap])
    x = torch.zeros(1, 4, dtype=torch.float32)
    result = rf.step(
        {"hidden": x},
        meta={
            "enabled": {"probe"},
            "block_tables": [[9, 8, -1]],
            "seq_lens": [20],
            "page_size": 16,
        },
    )
    assert result.ran == ["probe"]
    assert "paged_kv_indptr" in seen["meta"]
    np.testing.assert_array_equal(seen["meta"]["paged_kv_indptr"], [0, 2])
    np.testing.assert_array_equal(seen["meta"]["paged_kv_indices"], [9, 8])
    np.testing.assert_array_equal(seen["meta"]["paged_kv_last_page_len"], [4])
    assert result.meta is not None
    assert result.meta.extras["paged_kv"]["page_size"] == 16


def test_last_page_len_helper(serving):
    assert serving.last_page_len_from_seq(0, 16) == 0
    assert serving.last_page_len_from_seq(16, 16) == 16
    assert serving.last_page_len_from_seq(17, 16) == 1


def test_last_page_len_depends_on_page_size(serving):
    seq_len = 20
    len_16 = serving.last_page_len_from_seq(seq_len, 16)
    len_6 = serving.last_page_len_from_seq(seq_len, 6)
    assert len_16 == 4
    assert len_6 == 2
    assert len_16 != len_6
    for page_size, last_len in ((16, len_16), (6, len_6)):
        assert isinstance(last_len, int) and not isinstance(last_len, bool)
        assert 0 < last_len <= page_size


def test_block_tables_trim_to_seq_and_stop_at_pad(serving):
    """Indices follow seq_len pages and ignore ids after the first pad.

    The stored table keeps the raw row, including pad and anything after it.
    A regression that counts every non-negative id, or that emits every valid
    block, would address the wrong KV pages.
    """
    block_tables = np.array(
        [
            [10, 11, 12, -1, 99],
            [7, 8, 9, -1, -1],
        ],
        dtype=np.int32,
    )
    seq_lens = np.array([20, 0], dtype=np.int32)
    paged = serving.block_tables_to_paged_kv(block_tables, seq_lens, page_size=16)
    np.testing.assert_array_equal(paged.block_tables, block_tables)
    np.testing.assert_array_equal(paged.paged_kv_indptr, [0, 2, 5])
    np.testing.assert_array_equal(paged.paged_kv_indices, [10, 11, 7, 8, 9])
    np.testing.assert_array_equal(paged.paged_kv_last_page_len, [4, 0])

    # Ids after the pad must not satisfy a longer sequence.
    with pytest.raises(ValueError, match="needs >="):
        serving.block_tables_to_paged_kv(
            np.array([[1, -1, 2, 3]], dtype=np.int32),
            np.array([48], dtype=np.int32),
            page_size=16,
        )


def test_block_tables_rejects_malformed_kv_inputs(serving):
    tables = np.array([[10, 11, -1]], dtype=np.int32)
    with pytest.raises(ValueError, match="rank-2"):
        serving.block_tables_to_paged_kv(np.array([10, 11, -1]), [16], page_size=16)
    with pytest.raises(ValueError, match="seq_lens length"):
        serving.block_tables_to_paged_kv(tables, [16, 16], page_size=16)
    with pytest.raises(ValueError, match="non-negative"):
        serving.block_tables_to_paged_kv(tables, [-1], page_size=16)
    with pytest.raises(ValueError, match="page_size"):
        serving.last_page_len_from_seq(5, 0)
    with pytest.raises(ValueError, match="page_size"):
        serving.block_tables_to_paged_kv(tables, [20], page_size=-4)


def test_attach_paged_kv_keeps_unrelated_extras(serving):
    original = {"enabled": {"mlp_layer_0"}, "extras": {"note": "keep", "page_size": 1}}
    meta = serving.attach_paged_kv_to_step_meta(
        original,
        block_tables=[[3, 4, -1]],
        seq_lens=[16],
        page_size=16,
        slot_mapping=[100],
    )
    assert original["extras"] == {"note": "keep", "page_size": 1}
    assert meta["extras"]["note"] == "keep"
    assert meta["extras"]["page_size"] == 16
    assert meta["extras"]["paged_kv"]["paged_kv_indices"] == [3]
    np.testing.assert_array_equal(meta["extras"]["paged_kv_last_page_len"], [16])
    np.testing.assert_array_equal(meta["extras"]["paged_kv"]["slot_mapping"], [100])
