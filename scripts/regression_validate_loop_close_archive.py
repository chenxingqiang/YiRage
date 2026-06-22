#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Simulate downloaded loop-close archive regression validate (Loop R83).

Usage:
  PYTHONPATH=. python3 scripts/regression_validate_loop_close_archive.py \
    --source-archive artifacts/cpu-loop-close.json \
    --source-meta artifacts/cpu-loop-close.meta.json \
    --dest-dir artifacts/downloaded-regression
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, help="Built loop-close archive JSON")
    parser.add_argument("--source-meta", required=True, help="Metadata sidecar JSON")
    parser.add_argument(
        "--dest-dir",
        required=True,
        help="Simulated download destination directory",
    )
    parser.add_argument(
        "--check-stage-timeouts",
        action="store_true",
        help="Fail full archives when stage elapsed exceeds CI ceiling",
    )
    parser.add_argument(
        "--require-alert-annotation",
        action="store_true",
        help="Require timeout_alert_emitted when stage_timeout_warning_count >= 1",
    )
    args = parser.parse_args()

    from scripts.cpu_cert_utils import simulate_downloaded_loop_close_regression_validate

    errors = simulate_downloaded_loop_close_regression_validate(
        args.source_archive,
        args.source_meta,
        args.dest_dir,
        check_stage_timeouts=args.check_stage_timeouts,
        require_alert_annotation=args.require_alert_annotation,
    )
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"ok": True, "dest_dir": args.dest_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
