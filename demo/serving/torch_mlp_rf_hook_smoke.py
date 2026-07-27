#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S8 smoke: real torch MLP RF hook (TorchDecoderLayer; no mock/stub).

Requires PyTorch::

    PYTHONPATH=python python3 demo/serving/torch_mlp_rf_hook_smoke.py
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

    serving.require_torch()
    return serving


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap()
    import torch

    layer = serving.TorchDecoderLayer(0, hidden_size=32, intermediate_size=64, seed=7)
    hook = serving.build_torch_mlp_rf_hook(layer)
    x = torch.randn(4, 32, dtype=torch.float32, device=layer.device)

    with torch.no_grad():
        h_attn = layer.attention_forward(x)
        ref = layer.mlp_forward(h_attn)
        got = hook.forward_mlp(
            h_attn,
            rf_meta={"enabled": {hook.override.capsule_name}},
        )
        skip = hook.forward_mlp(h_attn, rf_meta={"force_skip_all": True})

    rf_match = bool(torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-5))
    skip_match = bool(torch.allclose(skip.hidden, ref, rtol=1e-5, atol=1e-5))

    report = {
        "s8": True,
        "real_torch": True,
        "device": layer.device,
        "hook": hook.inspect(),
        "rf_mlp_match_engine_mlp": rf_match,
        "skip_fallback_match_engine_mlp": skip_match,
        "used_rf_mlp": got.used_rf_mlp,
    }
    ok = rf_match and skip_match and got.used_rf_mlp

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("S8 real torch MLP RF hook smoke")
        for k, v in report.items():
            if k != "hook":
                print(f"  {k}: {v}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
