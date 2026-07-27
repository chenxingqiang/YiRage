#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S3 smoke: first-K decoder layers use RF MLP Capsules.

Cloud-safe::

    PYTHONPATH=python python3 demo/serving/hybrid_first_k_smoke.py --k 2
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
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--k", type=int, default=2, help="max_rf_mlp_layers")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--intermediate", type=int, default=64)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap_serving()
    model = serving.EngineModelStub(
        args.layers,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        seed=0,
    )
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=args.k)
    x = np.random.default_rng(2).normal(0, 1, size=(2, args.hidden)).astype(np.float32)
    result = hybrid.forward(x)
    ref = model.forward_engine_full(x)
    match = bool(np.allclose(result.hidden, ref, rtol=1e-5, atol=1e-6))
    report = {
        "s3": True,
        "inspect": hybrid.inspect(),
        "forward": result.to_dict(),
        "matches_engine_full": match,
        "expected_rf_layers": list(range(args.k)),
    }
    ok = match and result.rf_layer_ids == list(range(args.k))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("S3 HybridModelOverride first-K smoke")
        print(f"  layers={args.layers} k={args.k} rf_ids={result.rf_layer_ids}")
        print(f"  engine_mlp_ids={result.engine_mlp_layer_ids}")
        print(f"  matches_engine_full={match}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
