#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S8 smoke: vLLM Qwen2 MLP RF plugin contract (duck mock when vllm absent).

Cloud-safe::

    PYTHONPATH=python python3 demo/serving/vllm_plugin_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path


def _bootstrap():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules or not hasattr(sys.modules["yirage"], "__path__"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    import yirage.serving as serving

    return serving


class _MockLinear:
    def __init__(self, weight):
        self.weight = weight


class _MockNorm:
    def __init__(self, weight):
        self.weight = weight


class _MockMlp:
    def __init__(self, gate, up, down):
        self.gate_proj = _MockLinear(gate)
        self.up_proj = _MockLinear(up)
        self.down_proj = _MockLinear(down)

    def __call__(self, hidden):
        import torch.nn.functional as F

        gate = hidden @ self.gate_proj.weight.t()
        up = hidden @ self.up_proj.weight.t()
        return F.silu(gate) * up @ self.down_proj.weight.t()


class _MockVllmQwen2Layer:
    def __init__(self, *, hidden=16, intermediate=32, seed=0, layer_id=0):
        import torch

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        scale = 0.02
        self.layer_id = layer_id
        self.post_attention_layernorm = _MockNorm(torch.ones(hidden))
        gate = torch.randn(intermediate, hidden, generator=gen) * scale
        up = torch.randn(intermediate, hidden, generator=gen) * scale
        down = torch.randn(hidden, intermediate, generator=gen) * scale
        self.mlp = _MockMlp(gate, up, down)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap()
    serving.require_torch()
    import torch

    mock = _MockVllmQwen2Layer(hidden=16, intermediate=32, seed=3, layer_id=2)
    hook = serving.build_vllm_qwen2_mlp_rf_hook(mock, layer_id=2)
    x = torch.randn(2, 16, dtype=torch.float32)
    h_attn = x + x @ (torch.randn(16, 16) * 0.01)

    ref = serving.mlp_torch(
        h_attn,
        rms_weight=mock.post_attention_layernorm.weight,
        w_gate=mock.mlp.gate_proj.weight.t(),
        w_up=mock.mlp.up_proj.weight.t(),
        w_down=mock.mlp.down_proj.weight.t(),
    )
    got = hook.forward_mlp(h_attn, rf_meta={"enabled": {hook.override.capsule_name}})

    view = serving.extract_qwen2_mlp_weights(mock, layer_id=2)
    rf_match = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-5))

    report = {
        "s8": True,
        "vllm_installed": serving.is_vllm_available(),
        "hook": hook.inspect(),
        "weight_view_hidden": view.hidden_size,
        "rf_mlp_match": rf_match,
        "used_rf_mlp": got.used_rf_mlp,
    }
    ok = rf_match and got.used_rf_mlp

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("S8 vLLM Qwen2 MLP RF plugin smoke")
        for k, v in report.items():
            if k != "hook":
                print(f"  {k}: {v}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
