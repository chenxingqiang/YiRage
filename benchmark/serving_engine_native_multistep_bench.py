#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S35: Engine-native multistep MLP bench CLI (G7 chains C/D multistep).

Torch-only cert gate::

    export YIRAGE_BACKEND=cpu PYTHONPATH=python
    python3 benchmark/serving_engine_native_multistep_bench.py --quick --json
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
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-native", action="store_true", help="Skip native vllm/sglang tiers")
    parser.add_argument("--output", default="", help="Write JSON report to path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")

    from yirage.serving.engine_native_multistep_bench import run_engine_native_multistep_bench
    from yirage.serving.torch_exec import require_torch

    require_torch()
    report = run_engine_native_multistep_bench(
        decode_steps=args.decode_steps,
        quick=args.quick,
        try_native=not args.no_native,
        version="s35",
    )
    payload = report.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
