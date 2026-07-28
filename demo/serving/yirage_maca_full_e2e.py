#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""YiRage MACA full-layer hybrid MLP e2e (MetaX VM).

Every decoder layer routes MLP through RuntimeFusion ``yirage_maca`` capsules.
Requires ``YIRAGE_BACKEND=maca`` and built ``yirage.core`` on MetaX GPU VM::

    export MACA_PATH=/opt/maca
    export LD_LIBRARY_PATH=...
    export YIRAGE_BACKEND=maca PYTHONPATH=python
    python3 demo/serving/yirage_maca_full_e2e.py --quick
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
    if "yirage" not in sys.modules or not hasattr(sys.modules.get("yirage"), "core"):
        try:
            import yirage as yr  # noqa: F401
        except ImportError:
            stub = types.ModuleType("yirage")
            stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
            sys.modules["yirage"] = stub
    import yirage.serving as serving

    serving.require_yirage_maca()
    serving.require_torch()
    return serving, root


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--intermediate", type=int, default=128)
    p.add_argument("--quick", action="store_true", help="smaller shapes / skip bench")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving, root = _bootstrap()
    os.environ.setdefault("YIRAGE_BACKEND", "maca")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for sub in ("build/abstract_subexpr/release", "build/formal_verifier/release"):
        path = root / sub
        if path.exists() and str(path) not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{ld}"
            ld = os.environ["LD_LIBRARY_PATH"]

    num_layers = 2 if args.quick else args.layers
    hidden_size = 16 if args.quick else args.hidden
    intermediate_size = 32 if args.quick else args.intermediate

    report = serving.run_yirage_maca_full_layer_e2e(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        batch=1,
        bench=not args.quick,
    )
    payload = report.to_dict()
    payload["yirage_maca"] = True

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("YiRage MACA full-layer hybrid MLP e2e")
        for k, v in payload.items():
            print(f"  {k}: {v}")
        print("PASS" if report.parity_ok else "FAIL")

    return 0 if report.parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
