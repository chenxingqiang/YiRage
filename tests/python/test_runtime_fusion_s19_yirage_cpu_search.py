# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S19: Qwen2-0.5B YiRage CPU superoptimize + DistributedSearchCoordinator tier."""

from __future__ import annotations

import os

import pytest

from serving_test_utils import import_serving, serving, torch  # noqa: F401


pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.fixture(scope="module")
def yirage_serving(serving):
    if not serving.is_yirage_core_available():
        pytest.skip("yirage.core not available")
    serving.require_yirage_core()
    return serving


def test_runtime_fusion_version_s19(yirage_serving):
    info = yirage_serving.RuntimeFusion([]).inspect()
    assert info["version"] == "s19"


def test_qwen05b_default_backend_yirage_cpu(yirage_serving):
    assert (
        yirage_serving.resolve_hf_qwen_mlp_backend(None)
        == yirage_serving.BACKEND_YIRAGE_CPU
    )


def test_superoptimize_down_never_fallback_tiny(yirage_serving, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    monkeypatch.delenv("YIRAGE_SERVING_USE_COORDINATOR", raising=False)
    yirage_serving.apply_serving_kn_down_matmul_tractability(use_ray=False)
    opt = yirage_serving.superoptimize_down_matmul_cpu(64, 128, quick=True)
    assert opt is not None
    assert opt.backend == "cpu"


def test_coordinator_local_smoke(yirage_serving, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_COORDINATOR", "1")
    monkeypatch.setenv("YIRAGE_SERVING_RAY_WORKERS", "2")
    import yirage as yr
    import yirage.serving.yirage_exec as yirage_exec

    graph = yr.new_kernel_graph()
    a = graph.new_input(dims=(1, 32), dtype=yr.float32)
    b = graph.new_input(dims=(32, 64), dtype=yr.float32)
    graph.mark_output(graph.matmul(a, b))

    # Local coordinator path (no Ray cluster) for cert smoke
    monkeypatch.setattr(yirage_exec, "resolve_serving_use_ray", lambda **_: False)
    opt = yirage_serving.superoptimize_down_matmul_via_coordinator(graph, quick=True)
    assert opt is not None
    assert opt.backend == "cpu"


def test_cpu_cert_manifest_includes_s19_and_ray_tier():
    import_serving()
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    base = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s19_contract" in base
    assert "qwen05b_contract" in base

    yc = [s.name for s in serving_cpu_cert_manifest(quick=True, yirage_core=True)]
    assert "serving_ray_contract" in yc
    assert "qwen05b_yirage_e2e" in yc
