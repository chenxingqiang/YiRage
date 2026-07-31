#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S32: vLLM + SGLang G1 engine-cooperative regression (G7 chains C/D).

Requires PyTorch only for cert gate::

    export YIRAGE_BACKEND=cpu PYTHONPATH=python
    python3 benchmark/serving_engine_g1_regression.py --quick --json
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
    parser.add_argument("--output", default="", help="Write JSON report to path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    from yirage.serving.engine_g1_regression import run_engine_g1_regression
    from yirage.serving.torch_exec import require_torch

    require_torch()
    report = run_engine_g1_regression(quick=args.quick, version="s32")
    payload = report.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    ok = report.parity_ok
    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    else:
        print("S32 engine G1 regression")
        for chain in report.chains:
            print(f"  {chain.chain_id}: parity={chain.parity_ok} plugin={chain.plugin}")
        print(f"  vllm_native={report.vllm_native_available} sglang_native={report.sglang_native_available}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
