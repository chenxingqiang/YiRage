#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S30: MLP FusionCapsule micro-bench — parity + RF.step latency (G7 chain A).

Requires PyTorch only (no yirage.core)::

    export YIRAGE_BACKEND=cpu PYTHONPATH=python
    python3 benchmark/serving_mlp_capsule_bench.py --quick --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT / "python", _REPO_ROOT / "tests" / "python", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="", help="Write JSON report to path")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    from yirage.serving.mlp_capsule_bench import run_mlp_capsule_bench
    from yirage.serving.torch_exec import require_torch

    require_torch()
    report = run_mlp_capsule_bench(
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        batch=args.batch,
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
        quick=args.quick,
        version="s30",
    )
    payload = report.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ok = report.parity_ok and report.rows[0].mean_ms > 0 and report.rows[1].mean_ms > 0
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("S30 MLP capsule micro-bench (G7 chain A)")
        print(f"  device={report.device} parity_ok={report.parity_ok}")
        for row in report.rows:
            print(f"  {row.name}: {row.mean_ms:.3f}ms")
        print(f"  speedup_rf_vs_eager={report.speedup_rf_vs_eager:.4f}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
