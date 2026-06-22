#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Validate a loop-close JSON archive file (Loop R71/R74).

Usage:
  PYTHONPATH=. python3 scripts/validate_loop_close_archive.py artifacts/cpu-loop-close.json
  PYTHONPATH=. python3 scripts/validate_loop_close_archive.py archive.json \
    --metadata-output artifacts/cpu-loop-close.meta.json
"""

from __future__ import annotations

import argparse
import json
import sys

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to loop-close JSON (--output file)")
    parser.add_argument(
        "--metadata-output",
        metavar="PATH",
        help="Write artifact sidecar metadata JSON after successful validation",
    )
    args = parser.parse_args()

    from scripts.cpu_cert_utils import (
        load_loop_close_archive,
        loop_close_archive_metadata,
        validate_loop_close_archive,
    )

    report = load_loop_close_archive(args.path)
    errors = validate_loop_close_archive(report)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    summary = {"ok": True, "mode": report.get("mode")}
    if args.metadata_output:
        metadata = loop_close_archive_metadata(
            report,
            archive_path=args.path,
            validation_ok=True,
        )
        with open(args.metadata_output, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
            fh.write("\n")
        summary["metadata_output"] = args.metadata_output
        summary["bench_quick"] = metadata.get("bench_quick")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
