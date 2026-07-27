#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S17: yirage_maca multi-step decode generation latency archive.

CPU CI uses torch hybrid + MACA serving meta. MetaX VM may set
``YIRAGE_BACKEND=maca`` for real ``yirage_maca`` capsules::

    PYTHONPATH=python python3 demo/serving/yirage_maca_generation_bench.py --quick --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path


def _bootstrap():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "yirage" not in sys.modules:
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    import yirage.serving as serving

    serving.require_torch()
    return serving, root


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--intermediate", type=int, default=128)
    p.add_argument("--decode-steps", type=int, default=4)
    p.add_argument("--quick", action="store_true", help="smaller shapes / fewer iters")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--json", action="store_true", help="print JSON archive to stdout")
    p.add_argument("--baseline-json", action="store_true", help="use mcPytorch baseline archive schema")
    args = p.parse_args()

    serving, root = _bootstrap()
    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for sub in ("build/abstract_subexpr/release", "build/formal_verifier/release"):
        path = root / sub
        if path.exists() and str(path) not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{ld}"
            ld = os.environ["LD_LIBRARY_PATH"]

    num_layers = 2 if args.quick else args.layers
    hidden_size = 16 if args.quick else args.hidden
    intermediate_size = 32 if args.quick else args.intermediate
    decode_steps = 2 if args.quick else args.decode_steps
    iters = 4 if args.quick else 8

    if args.baseline_json or args.json:
        baseline = serving.run_yirage_maca_generation_mcpytorch_baseline_archive(
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            decode_steps=decode_steps,
            warmup=1,
            iters=iters,
        )
        if args.output:
            baseline.write_json(Path(args.output))
        report = baseline.to_dict()
        hybrid_ms = baseline.summary.hybrid_decode_step_ms
        ok = baseline.summary.parity_ok and hybrid_ms > 0
    else:
        archive = serving.run_yirage_maca_generation_bench_archive(
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            decode_steps=decode_steps,
            warmup=1,
            iters=iters,
        )
        if args.output:
            archive.write_json(Path(args.output))
        report = archive.to_dict()
        hybrid = next(r for r in archive.rows if r.name == "hybrid_decode_step")
        ok = hybrid.parity_ok and hybrid.mean_ms > 0

    if args.baseline_json or args.json:
        print(json.dumps(report, indent=2))
    else:
        print("S17 yirage_maca generation bench archive")
        print(f"  device={report.get('device')} version={report.get('version')}")
        for row in report.get("rows", []):
            print(f"  {row['name']}: {row['mean_ms']:.3f}ms parity={row['parity_ok']}")
        print("PASS" if ok else "FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
