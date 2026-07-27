#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Standalone S1 smoke: MLP FusionCapsule + RuntimeFusion.step select/skip.

Cloud-safe (numpy only; no yirage.core)::

    PYTHONPATH=python python3 demo/serving/mlp_capsule_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


def _bootstrap_serving():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules:
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    import yirage.serving as serving

    return serving


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    serving = _bootstrap_serving()
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        seed=args.seed,
    )
    rf = serving.RuntimeFusion([cap])
    x = np.random.default_rng(args.seed + 1).normal(
        0, 1, size=(args.batch, args.hidden)
    ).astype(np.float32)

    selected = rf.step({"hidden": x}, meta={"enabled": {cap.name}})
    skipped = rf.step({"hidden": x}, meta={"force_skip_all": True})

    report = {
        "runtime": "RuntimeFusion",
        "s1": True,
        "capsule": cap.inspect(),
        "selected": selected.to_dict(),
        "skipped": skipped.to_dict(),
        "selected_changed_hidden": bool(
            not np.allclose(selected.outputs["hidden"], x)
        ),
        "skipped_identity": bool(np.array_equal(skipped.outputs["hidden"], x)),
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("RuntimeFusion S1 MLP FusionCapsule smoke")
        print(f"  capsule={cap.name} kind={cap.kind}")
        print(f"  plan={cap.plan.name} backend={cap.plan.backend}")
        print(f"  selected ran={selected.ran} skipped={selected.skipped}")
        print(f"  force_skip ran={skipped.ran} skipped={skipped.skipped}")
        print(f"  selected_changed_hidden={report['selected_changed_hidden']}")
        print(f"  skipped_identity={report['skipped_identity']}")
        ok = report["selected_changed_hidden"] and report["skipped_identity"]
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
