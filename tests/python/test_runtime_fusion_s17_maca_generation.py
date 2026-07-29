# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S17: yirage_maca generation latency archive + SGLang-metax fork e2e."""

from __future__ import annotations

import os

import pytest

from serving_test_utils import maca_integration_enabled, serving, torch  # noqa: F401


def test_resolve_yirage_maca_generation_backend(serving):
    be = serving.resolve_yirage_maca_generation_backend()
    assert be in (serving.BACKEND_TORCH, serving.BACKEND_YIRAGE_MACA)


def test_yirage_maca_generation_decode_loop(serving):
    report = serving.run_yirage_maca_generation_decode_loop(
        num_layers=2,
        hidden_size=16,
        intermediate_size=32,
        decode_steps=3,
        backend=serving.BACKEND_TORCH,
    )
    assert report.parity_ok
    assert report.decode_steps == 3
    assert report.backend_used == serving.BACKEND_TORCH
    assert report.maca_meta_bridged
    assert report.engine_per_step_ms > 0
    assert report.hybrid_per_step_ms > 0


def test_yirage_maca_generation_auto_entry(serving):
    report = serving.run_yirage_maca_generation_auto(
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        decode_steps=2,
    )
    assert report.parity_ok
    assert report.decode_steps == 2


def test_yirage_maca_generation_bench_archive(serving):
    archive = serving.run_yirage_maca_generation_bench_archive(
        num_layers=2,
        hidden_size=16,
        intermediate_size=32,
        decode_steps=3,
        warmup=1,
        iters=4,
        backend=serving.BACKEND_TORCH,
    )
    assert archive.version == "s19"
    hybrid = next(r for r in archive.rows if r.name == "hybrid_decode_step")
    assert hybrid.parity_ok
    assert hybrid.mean_ms > 0
    loop = next(r for r in archive.rows if r.name == "generation_loop_hybrid")
    assert loop.parity_ok


def test_sglang_metax_hybrid_full_e2e_auto(serving):
    report = serving.run_sglang_metax_hybrid_full_e2e_auto(
        num_layers=2,
        max_rf_mlp_layers=2,
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        bench=False,
    )
    assert report.parity_ok
    assert report.maca_meta_bridged


@pytest.mark.skipif(
    not maca_integration_enabled(),
    reason="requires sglang on MetaX host or YIRAGE_MACA_INTEGRATION=1",
)
def test_sglang_metax_fork_e2e(serving):
    fork = serving.run_sglang_metax_fork_e2e(
        hidden_size=16,
        intermediate_size=32,
        batch=2,
        num_layers=2,
        max_rf_mlp_layers=1,
        bench=False,
    )
    assert fork.fork
    assert fork.parity_ok
    assert fork.hook.plugin == "SglangMetaxQwen2MlpRfHook"


@pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)
def test_yirage_maca_generation_with_maca_backend(serving):
    if not serving.is_yirage_maca_available():
        pytest.skip("yirage_maca not available (requires YIRAGE_BACKEND=maca on MetaX VM)")
    report = serving.run_yirage_maca_generation_decode_loop(
        num_layers=2,
        hidden_size=16,
        intermediate_size=32,
        decode_steps=2,
        backend=serving.BACKEND_YIRAGE_MACA,
    )
    assert report.parity_ok
    assert report.yirage_maca_used


def test_rf_inspect_version_s17(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s19"
