# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Serving CPU superoptimize Ray opt-in contract tests."""

from __future__ import annotations

import pytest

from serving_test_utils import import_serving


@pytest.fixture(scope="module")
def yirage_exec():
    import_serving()
    from yirage.serving import yirage_exec as mod

    return mod


def test_resolve_serving_use_ray_default_off(yirage_exec, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    assert yirage_exec.resolve_serving_use_ray() is False
    assert yirage_exec.serving_superoptimize_ray_kwargs() == {"use_ray": False}


def test_resolve_serving_use_ray_env_on(yirage_exec, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    assert yirage_exec.resolve_serving_use_ray() is True
    assert yirage_exec.serving_superoptimize_ray_kwargs()["use_ray"] is True


def test_superoptimize_kwargs_ray_uses_auto_search_space(yirage_exec, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    kwargs = yirage_exec.superoptimize_kwargs(quick=True)
    assert kwargs["use_ray"] is True
    assert "griddims" not in kwargs
    assert "blockdims" not in kwargs


def test_apply_serving_tractability_ray_disables_seed_verify_env(yirage_exec, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    monkeypatch.delenv("YIRAGE_SERVING_KN_MATMUL_ONLY", raising=False)
    yirage_exec.apply_serving_kn_down_matmul_tractability(use_ray=True)
    import os

    assert os.environ.get("YIRAGE_SERVING_USE_RAY") == "1"
    assert "YIRAGE_SERVING_KN_MATMUL_ONLY" not in os.environ


def test_apply_serving_tractability_default_sets_seed_verify_env(yirage_exec, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    monkeypatch.delenv("YIRAGE_SERVING_KN_MATMUL_ONLY", raising=False)
    yirage_exec.apply_serving_kn_down_matmul_tractability(use_ray=False)
    import os

    assert os.environ.get("YIRAGE_SERVING_KN_MATMUL_ONLY") == "1"
    assert "YIRAGE_SERVING_USE_RAY" not in os.environ
