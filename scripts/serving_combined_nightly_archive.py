#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Generate combined Serving nightly archive JSON (S34: decode + G1 + multistep).

Usage::

    export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
    export YIRAGE_BACKEND=cpu PYTHONPATH=python:tests/python
    python3 scripts/serving_combined_nightly_archive.py --json --quick --output artifacts/serving-combined-nightly.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root / "python", root / "tests" / "python", root):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for sub in ("build/abstract_subexpr/release", "build/formal_verifier/release"):
        p = root / sub
        if p.exists() and str(p) not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{p}:{ld}"
            ld = os.environ["LD_LIBRARY_PATH"]


def main() -> int:
    _bootstrap()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-rf-mlp-layers", type=int, default=1)
    parser.add_argument("--all-rf-layers", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--quick", action="store_true", default=True)
    parser.add_argument("--no-quick", action="store_false", dest="quick")
    parser.add_argument("--output", default="", help="Write JSON archive to path")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    parser.add_argument(
        "--g1-only",
        action="store_true",
        help="Skip decode/multistep (torch-only G1 subsection; contract smoke)",
    )
    args = parser.parse_args()

    from yirage.serving.combined_nightly_archive import run_serving_combined_nightly_archive
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL

    if args.g1_only:
        from yirage.serving.engine_g1_regression import run_engine_g1_regression
        from yirage.serving.torch_exec import require_torch

        require_torch()
        g1 = run_engine_g1_regression(quick=args.quick, version="s35").to_dict()
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
        from yirage.serving.yirage_exec import require_yirage_core

        require_yirage_core()
        payload = run_serving_combined_nightly_archive(
            model_id=args.model or DEFAULT_QWEN05B_MODEL,
            prompt=args.prompt,
            max_rf_mlp_layers=args.max_rf_mlp_layers,
            all_rf_layers=args.all_rf_layers,
            max_new_tokens=args.max_new_tokens,
            quick=args.quick,
            version="s35",
        )

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
