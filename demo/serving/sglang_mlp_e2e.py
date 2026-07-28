#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""SGLang ForwardBatch MLP RF full-path e2e (torch; sglang when installed).

::

    PYTHONPATH=python python3 demo/serving/sglang_mlp_e2e.py
    PYTHONPATH=python python3 demo/serving/sglang_mlp_e2e.py --json
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
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--intermediate", type=int, default=128)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap()
    report = serving.run_sglang_mlp_rf_e2e_auto(
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        batch=args.batch,
        bench=True,
    )
    payload = report.to_dict()
    ok = bool(report.parity_ok)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("SGLang ForwardBatch MLP RF full-path e2e")
        print(f"  plugin={payload['plugin']}")
        print(f"  parity_ok={report.parity_ok}")
        if "extend_seq_lens" in payload:
            print(f"  extend_seq_lens={payload['extend_seq_lens']}")
        if "rf_layer_ids" in payload:
            print(f"  rf_layers={payload['rf_layer_ids']}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
