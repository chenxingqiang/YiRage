#!/usr/bin/env python3
"""
MACA attention smoke — aligned to ``benchmark/end-to-end/maca/chameleon_maca.py``.

  - ``--inspect-only``: Cloud/CI scaffold report (no GPU, no yirage.core).
  - Default / ``--quick``: MetaX mcPytorch + ``superoptimize(backend=maca, config=attention)``.

MetaX VM:
  export MACA_PATH=/opt/maca YIRAGE_BACKEND=maca PYTHONPATH=.
  python3 demo/maca/attention_smoke.py --inspect-only
  python3 demo/maca/attention_smoke.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo.maca.attention_utils import (  # noqa: E402
    AttentionScaffold,
    inspect_maca_attention_scaffold,
    maca_attention_superoptimize_smoke,
)


def _apply_maca_env() -> None:
    os.environ.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    os.environ.setdefault("MACA_PATH", "/opt/maca")
    os.environ.setdefault("YIRAGE_BACKEND", "maca")


def _is_maca_device() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0)
    if "MetaX" in name:
        return True
    return os.environ.get("YIRAGE_MACA_ALLOW_NON_METAX", "") == "1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MACA attention smoke (chameleon_maca get_chameleon_attention aligned)"
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print scaffold JSON and exit (no GPU required)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Tractable superoptimize smoke (default on)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report",
    )
    args = parser.parse_args()

    _apply_maca_env()

    scaffold = AttentionScaffold()
    report: dict = {"scaffold": inspect_maca_attention_scaffold(scaffold)}

    if args.inspect_only:
        report["status"] = "inspect_only"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA attention scaffold (inspect-only)")
            print("=" * 70)
            for key, val in report["scaffold"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — inspect-only (no MetaX GPU required)")
        return 0

    if not _is_maca_device():
        print("✗ MetaX MACA GPU not detected; use --inspect-only on Cloud VM", file=sys.stderr)
        return 1

    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    report["gpu"] = {
        "device_name": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
    }
    report["superoptimize"] = maca_attention_superoptimize_smoke(scaffold, quick=args.quick)
    report["status"] = "pass"

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 70)
        print("MACA attention superoptimize smoke")
        print("=" * 70)
        print(f"  device: {report['gpu']['device_name']}")
        print(f"  superoptimize: {report['superoptimize']}")
        print()
        print("PASS — MACA attention superoptimize smoke")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
