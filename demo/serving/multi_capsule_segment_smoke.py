#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S7 smoke: 2-Capsule MLP pipeline + decoder segment override.

Cloud-safe::

    PYTHONPATH=python python3 demo/serving/multi_capsule_segment_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    p.add_argument("--bench", action="store_true", help="optional latency micro-bench")
    args = p.parse_args()

    serving = _bootstrap()
    model = serving.EngineModelStub(4, hidden_size=32, intermediate_size=64, seed=12)
    x = np.random.default_rng(13).normal(size=(4, 32)).astype(np.float32)

    # Single layer split pipeline parity.
    layer0 = model.layers[0]
    rf = serving.build_split_mlp_runtime_fusion(layer0, backend=serving.BACKEND_NUMPY_REF)
    pipe = rf.step({"hidden": x}, meta=serving.pipeline_meta_for_layer(0))
    ref0 = serving.mlp_eager_numpy(
        x,
        rms_weight=layer0.rms_weight,
        w_gate=layer0.w_gate,
        w_up=layer0.w_up,
        w_down=layer0.w_down,
    )
    pipeline_parity = bool(np.allclose(pipe.outputs["hidden"], ref0, rtol=1e-5, atol=1e-6))
    pipeline_ran = pipe.ran == [
        serving.split_mlp_gate_up_name(0),
        serving.split_mlp_down_name(0),
    ]

    # Segment layers 1..3 vs engine full on those layers.
    segment = serving.DecoderSegmentOverride(model, layer_start=1, layer_end=4)
    seg_out = segment.forward_segment(x)
    ref_seg = x
    for lid in [1, 2, 3]:
        ref_seg = model.layers[lid].forward_engine_full(ref_seg)
    segment_parity = bool(np.allclose(seg_out.hidden, ref_seg, rtol=1e-5, atol=1e-6))

    # Hybrid: layer0 single-capsule RF + layers 1-2 split segment + layer3 engine.
    hybrid = serving.SegmentHybridModelOverride(
        model,
        segment_layer_ids=[1, 2],
        rf_mlp_layer_ids=[0],
    )
    hybrid_out = hybrid.forward(x)
    full_ref = model.forward_engine_full(x)
    hybrid_parity = bool(np.allclose(hybrid_out.hidden, full_ref, rtol=1e-5, atol=1e-6))

    bench_ms = None
    if args.bench:
        t0 = time.perf_counter()
        for _ in range(20):
            segment.forward_segment(x)
        bench_ms = (time.perf_counter() - t0) * 1000.0 / 20.0

    report = {
        "s7": True,
        "pipeline_parity": pipeline_parity,
        "pipeline_ran_both_capsules": pipeline_ran,
        "segment_parity": segment_parity,
        "hybrid_parity": hybrid_parity,
        "segment_layer_ids": segment.segment_layer_ids,
        "capsules_per_step": seg_out.capsules_per_step,
        "bench_ms_per_segment_forward": bench_ms,
    }
    ok = pipeline_parity and pipeline_ran and segment_parity and hybrid_parity

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("S7 multi-Capsule segment smoke")
        for k, v in report.items():
            print(f"  {k}: {v}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
