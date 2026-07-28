#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Qwen2-0.5B CPU full-model e2e: HF generate + RuntimeFusion MLP parity.

Requires ``transformers`` and PyTorch on CPU::

    PYTHONPATH=python python3 demo/serving/qwen05b_cpu_e2e.py --quick
    PYTHONPATH=python python3 demo/serving/qwen05b_cpu_e2e.py --json
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
    if str(root / "tests" / "python") not in sys.path:
        sys.path.insert(0, str(root / "tests" / "python"))
    if "yirage" not in sys.modules or not hasattr(sys.modules["yirage"], "__path__"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        require_transformers,
        run_hf_qwen05b_cpu_e2e,
    )

    require_transformers()
    return run_hf_qwen05b_cpu_e2e, DEFAULT_QWEN05B_MODEL


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="HF model id (default Qwen/Qwen2-0.5B)")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-rf-mlp-layers", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    run_e2e, default_model = _bootstrap()
    report = run_e2e(
        model_id=args.model or default_model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_rf_mlp_layers=args.max_rf_mlp_layers,
        quick=args.quick,
    )
    payload = report.to_dict()
    ok = bool(report.parity_ok and report.num_layers >= 1)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("Qwen2-0.5B CPU full-model e2e")
        print(f"  model={payload['model_id']} layers={payload['num_layers']}")
        print(f"  prefill_parity_ok={report.prefill_parity_ok} decode_parity_ok={report.decode_parity_ok}")
        print(f"  generated={payload['generated_text'][:120]!r}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
