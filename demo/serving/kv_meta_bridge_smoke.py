#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S4 smoke: vLLM block_tables → paged_kv_* → RF.step extras.

    PYTHONPATH=python python3 demo/serving/kv_meta_bridge_smoke.py
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    serving = _bootstrap_serving()
    block_tables = np.array([[1, 2, 3, -1], [4, -1, -1, -1]], dtype=np.int32)
    seq_lens = np.array([40, 8], dtype=np.int32)  # page=16 → last=[8,8]
    paged = serving.block_tables_to_paged_kv(block_tables, seq_lens, page_size=16)

    cap = serving.MlpFusionCapsule.from_random(hidden_size=8, intermediate_size=16, seed=0)
    rf = serving.RuntimeFusion([cap])
    x = np.zeros((2, 8), dtype=np.float32)
    result = rf.step(
        {"hidden": x},
        meta={
            "enabled": {cap.name},
            "block_tables": block_tables,
            "seq_lens": seq_lens,
            "page_size": 16,
        },
    )
    extras = result.meta.extras if result.meta else {}
    ok = (
        list(paged.paged_kv_indptr) == [0, 3, 4]
        and list(paged.paged_kv_indices) == [1, 2, 3, 4]
        and list(paged.paged_kv_last_page_len) == [8, 8]
        and "paged_kv" in extras
        and result.ran == [cap.name]
    )
    report = {
        "s4": True,
        "paged_kv": paged.to_dict(),
        "rf_ran": result.ran,
        "extras_has_paged_kv": "paged_kv" in extras,
        "ok": ok,
    }
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("S4 KV meta bridge smoke")
        print(f"  indptr={paged.paged_kv_indptr.tolist()}")
        print(f"  indices={paged.paged_kv_indices.tolist()}")
        print(f"  last_page_len={paged.paged_kv_last_page_len.tolist()}")
        print(f"  rf_extras_paged_kv={report['extras_has_paged_kv']}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
