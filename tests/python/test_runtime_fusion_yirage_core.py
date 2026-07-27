# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""yirage.core + CPU superoptimize integration for RuntimeFusion serving."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _import_serving():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules or not hasattr(sys.modules.get("yirage"), "core"):
        try:
            import yirage as yr  # noqa: F401
        except ImportError:
            stub = types.ModuleType("yirage")
            stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
            sys.modules["yirage"] = stub
    for key in list(sys.modules):
        if key == "yirage.serving" or key.startswith("yirage.serving."):
            del sys.modules[key]
    return importlib.import_module("yirage.serving")


pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.fixture(scope="module")
def serving():
    mod = _import_serving()
    if not mod.is_yirage_core_available():
        pytest.skip("yirage.core not available")
    mod.require_torch()
    return mod


def test_yirage_core_available(serving):
    assert serving.is_yirage_core_available()


def test_superoptimize_down_matmul_cpu(serving):
    opt = serving.superoptimize_down_matmul_cpu(32, 64, quick=True)
    assert opt is not None
    assert opt.backend == "cpu"


def test_yirage_mlp_capsule_parity(serving):
    import torch

    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=32,
        intermediate_size=64,
        seed=3,
        backend=serving.BACKEND_YIRAGE_CPU,
    )
    x = torch.randn(1, 32, dtype=torch.float32)
    y = cap.execute({"hidden": x})["hidden"]
    ref = cap._yirage_runner.forward_torch_reference(x)
    assert torch.allclose(y, ref, rtol=0.05, atol=0.05)


def test_gate_up_seed_execute(serving):
    import torch

    g = serving.build_gate_up_seed_graph(32, 64)
    x = torch.randn(1, 32)
    rw = torch.ones(1, 32)
    w = torch.randn(32, 128)
    out = g(inputs=[x, rw, w])[0]
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    h = x32 * rw.float() / (var + 1e-6).sqrt()
    ref = h @ w.float()
    assert torch.allclose(out.float(), ref, rtol=0.05, atol=0.05)
