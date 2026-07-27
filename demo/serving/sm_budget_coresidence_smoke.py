#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S5 smoke: SM budget co-residence — RF skips over-budget capsules; engine fallback.

Cloud-safe::

    PYTHONPATH=python python3 demo/serving/sm_budget_coresidence_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


def _bootstrap():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules or not hasattr(sys.modules["yirage"], "__path__"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    import yirage.serving as serving

    return serving


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap()
    model = serving.EngineModelStub(2, hidden_size=16, intermediate_size=32, seed=3)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = np.random.default_rng(4).normal(0, 1, size=(2, 16)).astype(np.float32)

    # Normal budget: RF runs both MLP layers.
    ok_full = hybrid.forward(
        x,
        rf_meta={"extras": {"total_sms": 32, "reserved_aux_sms": 4}},
    )
    ref = model.forward_engine_full(x)
    full_match = bool(np.allclose(ok_full.hidden, ref, rtol=1e-5, atol=1e-6))
    full_rf = ok_full.rf_layer_ids == [0, 1]

    # Zero capsule budget: RF skips → engine MLP fallback, still numerically correct.
    tight = hybrid.forward(
        x,
        rf_meta={"sm_budget": 0, "extras": {"total_sms": 16, "reserved_aux_sms": 4}},
    )
    tight_match = bool(np.allclose(tight.hidden, ref, rtol=1e-5, atol=1e-6))
    tight_engine = tight.rf_layer_ids == [] and tight.engine_mlp_layer_ids == [0, 1]

    # Partial budget: first capsule sm_cost=1 runs; second layer also cost=1 with budget=1
    # → layer0 RF, layer1 engine fallback (still matches full engine path).
    partial = hybrid.forward(
        x,
        rf_meta={"sm_budget": 1, "extras": {"total_sms": 16, "reserved_aux_sms": 4}},
    )
    partial_match = bool(np.allclose(partial.hidden, ref, rtol=1e-5, atol=1e-6))

    report = {
        "s5": True,
        "full_rf_layers": ok_full.rf_layer_ids,
        "full_matches_engine": full_match,
        "tight_engine_only": tight_engine,
        "tight_matches_engine": tight_match,
        "partial_rf_layers": partial.rf_layer_ids,
        "partial_matches_engine": partial_match,
        "quota_default": serving.resolve_sm_worker_quota().to_dict(),
    }
    ok = full_match and full_rf and tight_match and tight_engine and partial_match

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("S5 SM budget co-residence smoke")
        print(f"  full_rf={ok_full.rf_layer_ids} match={full_match}")
        print(f"  tight_engine={tight.engine_mlp_layer_ids} match={tight_match}")
        print(f"  partial_rf={partial.rf_layer_ids} match={partial_match}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
