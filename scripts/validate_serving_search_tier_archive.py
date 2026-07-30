#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Validate a Serving search-tier bench archive JSON (S26).

Usage:
  PYTHONPATH=python python3 scripts/validate_serving_search_tier_archive.py artifacts/serving-tier.json
  PYTHONPATH=python python3 scripts/validate_serving_search_tier_archive.py archive.json \
    --metadata-output artifacts/serving-tier.meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to serving bench archive JSON")
    parser.add_argument(
        "--metadata-output",
        metavar="PATH",
        help="Write artifact sidecar metadata JSON after successful validation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Record quick=true in metadata sidecar",
    )
    args = parser.parse_args()

    from yirage.serving.search_tier_archive import (
        load_serving_bench_archive,
        serving_bench_archive_metadata,
        serving_multi_tier_bench_archive_metadata,
        validate_serving_search_tier_archive,
        is_serving_multi_tier_bench_archive,
    )

    payload = load_serving_bench_archive(args.path)
    errors = validate_serving_search_tier_archive(payload)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    summary = {"ok": True}
    if is_serving_multi_tier_bench_archive(payload):
        compare = payload.get("compare") if isinstance(payload.get("compare"), dict) else {}
        summary["multi_tier"] = True
        summary["compare_ok"] = bool(compare.get("ok"))
        summary["superopt_slowdown_vs_baseline"] = compare.get("superopt_slowdown_vs_baseline")
    else:
        summary["tier"] = (payload.get("search_tier") or {}).get("tier")
    if args.metadata_output:
        if is_serving_multi_tier_bench_archive(payload):
            metadata = serving_multi_tier_bench_archive_metadata(
                payload,
                archive_path=args.path,
                validation_ok=True,
                quick=args.quick,
            )
        else:
            metadata = serving_bench_archive_metadata(
                payload,
                archive_path=args.path,
                validation_ok=True,
                quick=args.quick,
            )
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summary["metadata_output"] = args.metadata_output
        summary["quick"] = metadata.get("quick")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
