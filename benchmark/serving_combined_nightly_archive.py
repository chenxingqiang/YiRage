#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S34: Combined Serving nightly archive CLI (decode + G1 + multistep).

Torch-only quick smoke (G1 subsection only)::

    export YIRAGE_BACKEND=cpu PYTHONPATH=python
    python3 benchmark/serving_combined_nightly_archive.py --quick --g1-only --json

Full archive requires ``yirage.core`` + transformers::

    export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
    export YIRAGE_BACKEND=cpu PYTHONPATH=python:tests/python
    python3 benchmark/serving_combined_nightly_archive.py --quick --json
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
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--g1-only", action="store_true")
    parser.add_argument("--output", default="", help="Write JSON archive to path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    if args.g1_only:
        from yirage.serving.engine_g1_regression import run_engine_g1_regression
        from yirage.serving.torch_exec import require_torch

        require_torch()
        g1 = run_engine_g1_regression(quick=args.quick, version="s34").to_dict()
        payload = {
            "serving_combined_nightly_archive": True,
            "version": "s34",
            "parity_ok": g1.get("parity_ok"),
            "quick": args.quick,
            "functional_chains": ["chain_c_vllm_torch", "chain_d_sglang_torch"],
            "decode": None,
            "engine_g1": g1,
            "multistep": None,
        }
    else:
        from yirage.serving.combined_nightly_archive import run_serving_combined_nightly_archive
        from yirage.serving.yirage_exec import require_yirage_core

        require_yirage_core()
        payload = run_serving_combined_nightly_archive(quick=args.quick, version="s34")

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
