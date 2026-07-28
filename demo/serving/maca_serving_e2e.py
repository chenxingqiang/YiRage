#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""MACA serving meta + full-layer MLP RF e2e (torch).

::

    PYTHONPATH=python python3 demo/serving/maca_serving_e2e.py
    PYTHONPATH=python python3 demo/serving/maca_serving_e2e.py --json
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
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap()
    report = serving.run_maca_serving_full_layer_e2e_auto(
        num_layers=args.layers,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        batch=args.batch,
        bench=True,
    )
    payload = report.to_dict()
    ok = bool(report.parity_ok and report.all_layers_rf and report.maca_meta_bridged)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("MACA serving full-layer MLP RF e2e")
        print(f"  plugin={payload['plugin']}")
        print(f"  parity_ok={report.parity_ok} all_layers_rf={report.all_layers_rf}")
        print(f"  maca_meta_bridged={report.maca_meta_bridged} warp_size={payload['warp_size']}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
