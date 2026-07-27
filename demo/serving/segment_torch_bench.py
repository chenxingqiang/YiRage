#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S8 smoke: torch segment hybrid bench archive (real measured ms).

Requires PyTorch::

    PYTHONPATH=python python3 demo/serving/segment_torch_bench.py --json
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
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--intermediate", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--output", type=str, default="")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap()
    archive = serving.run_segment_torch_bench_archive(
        num_layers=args.layers,
        segment_layer_ids=(1, 2),
        rf_mlp_layer_ids=(0,),
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        batch=args.batch,
        iters=args.iters,
    )
    if args.output:
        archive.write_json(Path(args.output))

    report = archive.to_dict()
    hybrid = next(r for r in archive.rows if r.name == "segment_hybrid_torch")
    ok = hybrid.parity_ok and hybrid.mean_ms > 0

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("S8 segment torch bench archive")
        print(f"  device={archive.device} version={archive.version}")
        for row in archive.rows:
            print(f"  {row.name}: {row.mean_ms:.3f}ms parity={row.parity_ok}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
