#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""YiRage core + CPU superoptimize e2e for RuntimeFusion MLP (real execution).

Requires built ``yirage.core`` (see ``scripts/setup_serving_yirage_core.sh``)::

    export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
    export YIRAGE_BACKEND=cpu PYTHONPATH=python
    python3 demo/serving/yirage_superopt_e2e.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

    serving.require_yirage_core()
    serving.require_torch()
    return serving, root


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--intermediate", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="smaller shapes / fewer bench iters")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving, root = _bootstrap()
    import torch

    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for sub in ("build/abstract_subexpr/release", "build/formal_verifier/release"):
        path = root / sub
        if path.exists() and str(path) not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{ld}"
            ld = os.environ["LD_LIBRARY_PATH"]

    hidden_size = 32 if args.quick else args.hidden
    intermediate_size = 64 if args.quick else args.intermediate

    t_compile = time.perf_counter()
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=args.seed,
        backend=serving.BACKEND_YIRAGE_CPU,
    )
    compile_s = time.perf_counter() - t_compile
    superopt_s = cap._yirage_runner.superopt_elapsed_s

    x = torch.randn(1, hidden_size, dtype=torch.float32)
    with torch.no_grad():
        y_ref = cap._yirage_runner.forward_torch_reference(x)
        t0 = time.perf_counter()
        y = cap.execute({"hidden": x})["hidden"]
        forward_s = time.perf_counter() - t0
        parity = bool(torch.allclose(y, y_ref, rtol=0.05, atol=0.05))

        bench = serving.bench_forward(
            lambda: cap.execute({"hidden": x})["hidden"],
            name="yirage_cpu_mlp",
            warmup=2 if args.quick else 5,
            iters=10 if args.quick else 50,
            device=cap._device,
        )

    report = {
        "backend": serving.BACKEND_YIRAGE_CPU,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "superopt_elapsed_s": round(superopt_s, 4),
        "capsule_init_s": round(compile_s, 4),
        "forward_once_s": round(forward_s, 6),
        "parity_vs_torch": parity,
        "bench_ms": round(bench.mean_ms, 4),
        "yirage_core": True,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("YiRage core MLP superoptimize e2e")
        for k, v in report.items():
            print(f"  {k}: {v}")
        print("PASS" if parity else "FAIL")

    return 0 if parity else 1


if __name__ == "__main__":
    raise SystemExit(main())
