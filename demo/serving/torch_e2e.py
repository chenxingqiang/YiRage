#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Torch e2e: HybridModelOverride + RF + SM budget + latency.

Requires PyTorch (CPU or CUDA)::

    PYTHONPATH=python python3 demo/serving/torch_e2e.py
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
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--intermediate", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap()
    import torch

    model = serving.TorchEngineModel(
        args.layers,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        seed=0,
    )
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=args.k)
    x = torch.randn(args.batch, args.hidden, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        ref = model.forward_engine_full(x)
        full = hybrid.forward(
            x,
            rf_meta={"extras": {"total_sms": 108, "reserved_aux_sms": 8}},
        )
        tight = hybrid.forward(
            x,
            rf_meta={"sm_budget": 0, "extras": {"total_sms": 16, "reserved_aux_sms": 4}},
        )

        full_match = bool(torch.allclose(full.hidden, ref, rtol=1e-5, atol=1e-6))
        tight_match = bool(torch.allclose(tight.hidden, ref, rtol=1e-5, atol=1e-6))

        eng_bench = serving.bench_forward(
            lambda: model.forward_engine_full(x),
            name="engine_full",
            warmup=3,
            iters=15,
            device=model.device,
        )
        hyb_bench = serving.bench_forward(
            lambda: hybrid.forward(
                x,
                rf_meta={"extras": {"total_sms": 108, "reserved_aux_sms": 8}},
            ),
            name=f"hybrid_k{args.k}",
            warmup=3,
            iters=15,
            device=model.device,
        )

    report = {
        "torch_e2e": True,
        "device": model.device,
        "engine": model.inspect(),
        "hybrid": hybrid.inspect(),
        "full_match": full_match,
        "tight_sm_match": tight_match,
        "rf_layers_full": full.rf_layer_ids,
        "rf_layers_tight": tight.rf_layer_ids,
        "bench_ms": {
            eng_bench.name: eng_bench.mean_ms,
            hyb_bench.name: hyb_bench.mean_ms,
        },
    }
    ok = full_match and tight_match and full.rf_layer_ids == list(range(args.k))

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("Torch RuntimeFusion e2e")
        print(f"  device={model.device} layers={args.layers} k={args.k}")
        print(f"  full_match={full_match} rf={full.rf_layer_ids}")
        print(f"  tight_sm_match={tight_match} rf={tight.rf_layer_ids}")
        print(f"  bench engine={eng_bench.mean_ms:.3f}ms hybrid={hyb_bench.mean_ms:.3f}ms")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
