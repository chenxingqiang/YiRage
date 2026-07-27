#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""SGLang-metax MLP RF e2e with MACA serving meta (real torch measured path).

Torch surrogate runs on CPU CI; real SGLang-metax tier when ``sglang`` + MetaX host::

    PYTHONPATH=python python3 demo/serving/sglang_metax_e2e.py --quick
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
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "yirage" not in sys.modules:
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    import yirage.serving as serving

    serving.require_torch()
    return serving


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="smaller shapes / skip bench")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap()
    hidden_size = 16 if args.quick else 64
    intermediate_size = 32 if args.quick else 128
    batch = 2 if args.quick else 4

    hook_report = serving.run_sglang_metax_mlp_rf_e2e_auto(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=not args.quick,
    )
    hybrid_report = serving.run_torch_sglang_metax_hybrid_full_e2e(
        num_layers=2 if args.quick else 4,
        max_rf_mlp_layers=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=batch,
        bench=not args.quick,
    )
    payload = {
        "hook": hook_report.to_dict(),
        "hybrid": hybrid_report.to_dict(),
        "sglang_metax_tier": serving.is_sglang_metax_available(),
    }

    ok = hook_report.parity_ok and hybrid_report.parity_ok
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("SGLang-metax MLP RF e2e")
        for section, report in (("hook", hook_report), ("hybrid", hybrid_report)):
            print(f"  [{section}]")
            for k, v in report.to_dict().items():
                print(f"    {k}: {v}")
        print(f"  sglang_metax_tier: {payload['sglang_metax_tier']}")
        print("PASS" if ok else "FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
