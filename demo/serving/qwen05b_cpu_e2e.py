#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Qwen2-0.5B CPU full-model e2e: HF generate + YiRage superoptimize decode.

Default path uses ``yirage_cpu`` (gate_up via ``yirage.core`` + down ``superoptimize``).
Requires built ``yirage.core`` — see ``scripts/setup_serving_yirage_core.sh``::

    export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
    export YIRAGE_BACKEND=cpu PYTHONPATH=python
    python3 demo/serving/qwen05b_cpu_e2e.py --quick
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
    if str(root / "tests" / "python") not in sys.path:
        sys.path.insert(0, str(root / "tests" / "python"))
    if "yirage" not in sys.modules or not hasattr(sys.modules.get("yirage"), "core"):
        try:
            import yirage as yr  # noqa: F401
        except ImportError:
            stub = types.ModuleType("yirage")
            stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
            sys.modules["yirage"] = stub
    from yirage.serving.exec_backend import BACKEND_TORCH, BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        require_transformers,
        resolve_hf_qwen_mlp_backend,
        run_hf_qwen05b_cpu_e2e,
    )
    from yirage.serving.yirage_exec import require_yirage_core

    require_transformers()
    return (
        run_hf_qwen05b_cpu_e2e,
        DEFAULT_QWEN05B_MODEL,
        BACKEND_TORCH,
        BACKEND_YIRAGE_CPU,
        require_yirage_core,
        resolve_hf_qwen_mlp_backend,
        root,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="HF model id (default Qwen/Qwen2-0.5B)")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-rf-mlp-layers", type=int, default=2)
    p.add_argument(
        "--mlp-backend",
        choices=("yirage_cpu", "torch"),
        default=None,
        help="decode MLP backend (default: yirage_cpu when yirage.core built)",
    )
    p.add_argument(
        "--use-ray",
        action="store_true",
        help="Use DistributedSearchCoordinator for down matmul superoptimize",
    )
    p.add_argument(
        "--ray-workers",
        type=int,
        default=None,
        help="Ray/coordinator worker count (YIRAGE_SERVING_RAY_WORKERS)",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    (
        run_e2e,
        default_model,
        BACKEND_TORCH,
        BACKEND_YIRAGE_CPU,
        require_yirage_core,
        resolve_backend,
        root,
    ) = _bootstrap()

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for sub in ("build/abstract_subexpr/release", "build/formal_verifier/release"):
        path = root / sub
        if path.exists() and str(path) not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{ld}"
            ld = os.environ["LD_LIBRARY_PATH"]

    mlp_backend = args.mlp_backend
    if args.use_ray:
        os.environ["YIRAGE_SERVING_USE_RAY"] = "1"
        os.environ.setdefault("YIRAGE_SERVING_USE_COORDINATOR", "1")
    if args.ray_workers is not None:
        os.environ["YIRAGE_SERVING_RAY_WORKERS"] = str(max(1, args.ray_workers))

    if mlp_backend == "yirage_cpu":
        mlp_backend = BACKEND_YIRAGE_CPU
        require_yirage_core()
    elif mlp_backend == "torch":
        mlp_backend = BACKEND_TORCH
    else:
        mlp_backend = resolve_backend(None)
        if mlp_backend == BACKEND_YIRAGE_CPU:
            require_yirage_core()

    report = run_e2e(
        model_id=args.model or default_model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_rf_mlp_layers=args.max_rf_mlp_layers,
        mlp_backend=mlp_backend,
        quick=args.quick,
    )
    payload = report.to_dict()
    ok = bool(report.parity_ok and report.num_layers >= 1)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("Qwen2-0.5B CPU full-model e2e (YiRage optimize path)")
        print(f"  model={payload['model_id']} layers={payload['num_layers']}")
        print(f"  mlp_backend={payload['mlp_backend']} yirage_core={payload['yirage_core_used']}")
        print(
            f"  prefill_parity={report.prefill_parity_ok} "
            f"yirage_decode_parity={report.yirage_decode_parity_ok} "
            f"token_match={report.generate_token_match_ok}"
        )
        print(f"  superopt_s={payload['superopt_elapsed_s_total']}")
        print(f"  generated={payload['generated_text'][:120]!r}")
        if report.yirage_core_used:
            print(f"  yirage_generated={payload['yirage_generated_text'][:120]!r}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
