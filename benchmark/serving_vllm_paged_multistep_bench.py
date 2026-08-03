#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S37: vLLM PagedAttention multistep decode bench CLI (G7 chain C paged).

Requires PyTorch only (no yirage.core)::

    export YIRAGE_BACKEND=cpu PYTHONPATH=python
    python3 benchmark/serving_vllm_paged_multistep_bench.py --quick --json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT / "python", _REPO_ROOT / "tests" / "python", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _install_yirage_stub() -> None:
    yirage_dir = _REPO_ROOT / "python" / "yirage"
    if "yirage" not in sys.modules or not hasattr(sys.modules.get("yirage"), "core"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="", help="Write JSON report to path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    _install_yirage_stub()

    bench = importlib.import_module("yirage.serving.vllm_paged_multistep_bench")
    torch_exec = importlib.import_module("yirage.serving.torch_exec")

    torch_exec.require_torch()
    report = bench.run_vllm_paged_multistep_bench(
        decode_steps=args.decode_steps,
        quick=args.quick,
        version="s37",
    )
    payload = report.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
