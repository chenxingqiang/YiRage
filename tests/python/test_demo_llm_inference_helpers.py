# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for demo/mps/llm_inference.py helpers (no MPS required)."""

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = PROJECT_ROOT / "demo" / "mps" / "llm_inference.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("llm_inference_demo", DEMO_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_device_cpu_when_mps_unavailable(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.torch.backends.mps,
        "is_available",
        lambda: False,
    )
    assert demo.resolve_device("auto") == "cpu"
    assert demo.resolve_device("cpu") == "cpu"


def test_resolve_device_rejects_mps_when_unavailable(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.torch.backends.mps,
        "is_available",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="MPS"):
        demo.resolve_device("mps")


def test_yirage_linear_adds_bias():
    demo = _load_demo_module()
    import torch

    graph = object()
    out = torch.ones(2, 3)
    bias = torch.tensor([1.0, 2.0, 3.0])
    result = demo.yirage_linear(graph, out, bias)
    assert torch.allclose(result, out + bias)
