# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU same-backend: superoptimize correctness and execution on the local host."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.cpu_cert_utils import apply_plain_matmul_search_tractability

pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.fixture
def isolated_mugraph_home(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="yirage_cpu_value_") as tmp:
        monkeypatch.setenv("HOME", tmp)
        import yirage.storage.mugraph_store as ms

        ms._default_store = None
        yield


def _apply_plain_matmul_search_tractability() -> None:
    """Cap CPU search for cert superoptimize smoke (mirrors bench plain_matmul caps)."""
    apply_plain_matmul_search_tractability()


def test_cpu_superoptimize_correctness_and_same_backend(isolated_mugraph_home):
    import yirage as yr

    _apply_plain_matmul_search_tractability()
    g = yr.new_kernel_graph()
    a = g.new_input(dims=(8, 32), dtype=yr.float16)
    b = g.new_input(dims=(32, 64), dtype=yr.float16)
    g.mark_output(g.matmul(a, b))

    optimized = g.superoptimize(
        backend="cpu",
        griddims=[(1, 1, 1)],
        blockdims=[(32, 1, 1)],
        franges=[1],
        use_ray=False,
        use_graph_dataset=False,
        use_cached_graphs=False,
        use_persistent_cache=False,
        warmup_iters=1,
        profile_iters=10,
        verbose=False,
    )

    assert optimized is not None
    assert optimized.backend == "cpu"

    ref_a = torch.randn(8, 32, dtype=torch.float16)
    ref_b = torch.randn(32, 64, dtype=torch.float16)
    ref_out = torch.matmul(ref_a, ref_b)

    out = optimized(inputs=[ref_a, ref_b])
    assert len(out) == 1
    assert torch.allclose(out[0].float(), ref_out.float(), rtol=0.05, atol=0.05)


def test_cpu_runtime_config_matches_local_cores():
    from yirage.backends.cpu.config import get_cpu_runtime_config, get_cpu_search_config

    search = get_cpu_search_config()
    runtime = get_cpu_runtime_config()
    assert runtime["torch_num_threads"] >= 1
    assert runtime["torch_num_threads"] <= search["num_cores"]
    assert runtime["simd_type"] == search["simd_type"]
    assert "parallel_tb_grid" in runtime
