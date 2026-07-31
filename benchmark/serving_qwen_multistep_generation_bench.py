#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S33: HF multi-step greedy generation bench — native vs RF MLP (G7 chain B).

Requires ``transformers``; ``yirage.core`` when ``--mlp-backend yirage_cpu``::

    export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
    export YIRAGE_BACKEND=cpu PYTHONPATH=python:tests/python
    python3 benchmark/serving_qwen_multistep_generation_bench.py --quick --json
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
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-rf-mlp-layers", type=int, default=1)
    parser.add_argument("--all-rf-layers", action="store_true")
    parser.add_argument(
        "--mlp-backend",
        default=None,
        choices=["torch", "yirage_cpu"],
        help="RF MLP backend (default: yirage_cpu if built else torch)",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for sub in ("build/abstract_subexpr/release", "build/formal_verifier/release"):
        path = _REPO_ROOT / sub
        if path.exists() and str(path) not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{ld}"
            ld = os.environ["LD_LIBRARY_PATH"]

    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, require_transformers
    from yirage.serving.qwen_multistep_generation_bench import run_qwen_multistep_generation_bench

    require_transformers()
    report = run_qwen_multistep_generation_bench(
        model_id=args.model or DEFAULT_QWEN05B_MODEL,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_rf_mlp_layers=args.max_rf_mlp_layers,
        all_rf_layers=args.all_rf_layers,
        mlp_backend=args.mlp_backend,
        quick=args.quick,
        version="s33",
    )
    payload = report.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    ok = report.parity_ok and report.token_match_ok
    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    else:
        print("S33 Qwen multistep generation bench (G7 chain B)")
        print(f"  token_match_ok={report.token_match_ok} backend={report.mlp_backend}")
        print(f"  max_new_tokens={report.max_new_tokens} native_ms={report.native_generate_ms:.2f}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
