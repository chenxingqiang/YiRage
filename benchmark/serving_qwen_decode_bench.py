#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S27: Qwen decode-step bench — native HF vs YiRage fused RF MLP on CPU.

Requires built ``yirage.core`` and ``transformers``::

    export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
    export YIRAGE_BACKEND=cpu PYTHONPATH=python
    python3 benchmark/serving_qwen_decode_bench.py --quick --json
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
    parser.add_argument("--model", default=None, help="HF model id (default Qwen/Qwen2-0.5B)")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-rf-mlp-layers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="", help="Write JSON report to path")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for sub in ("build/abstract_subexpr/release", "build/formal_verifier/release"):
        path = _REPO_ROOT / sub
        if path.exists() and str(path) not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{ld}"
            ld = os.environ["LD_LIBRARY_PATH"]

    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL
    from yirage.serving.qwen_decode_bench import run_qwen_decode_bench
    from yirage.serving.yirage_exec import require_yirage_core

    require_yirage_core()
    report = run_qwen_decode_bench(
        model_id=args.model or DEFAULT_QWEN05B_MODEL,
        prompt=args.prompt,
        max_rf_mlp_layers=args.max_rf_mlp_layers,
        warmup=args.warmup,
        iters=args.iters,
        quick=args.quick,
    )
    payload = report.to_dict()
    ok = bool(report.parity_ok and report.rows[0].mean_ms > 0)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Qwen decode-step bench (native HF vs YiRage RF yirage_cpu)")
        print(f"  model={report.model_id} rf_layers={report.max_rf_mlp_layers}")
        for row in report.rows:
            print(f"  {row.name}: {row.mean_ms:.4f} ms ({row.iters} iters)")
        print(f"  speedup_yirage_vs_native={report.speedup_yirage_vs_native:.3f}x")
        print(f"  superopt_s={report.superopt_elapsed_s_total:.4f}")
        print(f"  parity_ok={report.parity_ok}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
