#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S2 smoke: one decoder layer — engine Attention + RF MLP Override.

Cloud-safe (numpy; no vLLM / yirage.core)::

    PYTHONPATH=python python3 demo/serving/vllm_mlp_override_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


def _bootstrap_serving():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules:
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    import yirage.serving as serving

    return serving


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--intermediate", type=int, default=128)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap_serving()
    layer = serving.EngineDecoderLayerStub(
        0, hidden_size=args.hidden, intermediate_size=args.intermediate, seed=0
    )
    cap = serving.build_layer_mlp_capsule(layer)
    rf = serving.RuntimeFusion([cap])
    ov = serving.RuntimeFusionMlpLayerOverride(layer, rf)

    x = np.random.default_rng(1).normal(0, 1, size=(2, args.hidden)).astype(np.float32)
    ref = layer.forward_engine_full(x)
    selected = ov.forward(x, rf_meta={"enabled": {cap.name}})
    skipped = ov.forward(x, rf_meta={"force_skip_all": True})

    report = {
        "s2": True,
        "override": ov.inspect(),
        "selected": selected.to_dict(),
        "skipped": skipped.to_dict(),
        "selected_matches_engine": bool(
            np.allclose(selected.hidden, ref, rtol=1e-5, atol=1e-6)
        ),
        "skip_matches_engine": bool(
            np.allclose(skipped.hidden, ref, rtol=1e-5, atol=1e-6)
        ),
    }
    ok = report["selected_matches_engine"] and report["skip_matches_engine"]
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("S2 RuntimeFusionMlpLayerOverride smoke")
        print(f"  capsule={cap.name} attention=engine mlp=RF|fallback")
        print(f"  selected_matches_engine={report['selected_matches_engine']}")
        print(f"  skip_matches_engine={report['skip_matches_engine']}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
