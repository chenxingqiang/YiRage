#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S6 smoke: Radix hit meta skip/shrink on hybrid MLP path.

Cloud-safe::

    PYTHONPATH=python python3 demo/serving/radix_hit_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap()
    model = serving.EngineModelStub(2, hidden_size=16, intermediate_size=32, seed=8)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = np.random.default_rng(9).normal(size=(3, 16)).astype(np.float32)

    # No radix hits → same as engine full on RF layers.
    full = hybrid.forward(x)
    ref = model.forward_engine_full(x)
    no_hit_match = bool(np.allclose(full.hidden, ref, rtol=1e-5, atol=1e-6))

    # Partial hit on a single RF layer (batch shrink).
    model1 = serving.EngineModelStub(1, hidden_size=16, intermediate_size=32, seed=10)
    hybrid1 = serving.HybridModelOverride(model1, max_rf_mlp_layers=1)
    layer0 = model1.layers[0]
    h0 = layer0.attention_forward(x)
    expected_partial = h0.copy()
    expected_partial[1] = layer0.mlp_forward(h0[1:2])[0]
    partial = hybrid1.forward(x, rf_meta={"radix_hit_mask": [True, False, True]})
    partial_match = bool(np.allclose(partial.hidden, expected_partial, rtol=1e-5, atol=1e-6))

    # All-hit on both RF layers → MLP identity (post-attn only per layer).
    all_hit = hybrid.forward(x, rf_meta={"radix_hit_mask": [True, True, True]})
    h_after_l0 = model.layers[0].attention_forward(x)
    h_after_l1 = model.layers[1].attention_forward(h_after_l0)
    all_hit_match = bool(np.allclose(all_hit.hidden, h_after_l1, rtol=1e-5, atol=1e-6))
    all_hit_no_rf = all_hit.rf_layer_ids == []

    report = {
        "s6": True,
        "no_hit_matches_engine": no_hit_match,
        "partial_radix_match": partial_match,
        "all_hit_identity_match": all_hit_match,
        "all_hit_rf_layers": all_hit.rf_layer_ids,
        "all_hit_skipped_rf": all_hit_no_rf,
    }
    ok = no_hit_match and partial_match and all_hit_match and all_hit_no_rf

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("S6 Radix hit skip/shrink smoke")
        for k, v in report.items():
            print(f"  {k}: {v}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
