#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Validate a loop-close artifact metadata sidecar (Loop R75).

Usage:
  PYTHONPATH=. python3 scripts/validate_loop_close_archive_metadata.py meta.json
  PYTHONPATH=. python3 scripts/validate_loop_close_archive_metadata.py meta.json \
    --archive archive.json
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
    parser.add_argument("metadata_path", help="Path to .meta.json sidecar")
    parser.add_argument(
        "--archive",
        metavar="PATH",
        help="Cross-check metadata against loop-close archive JSON",
    )
    parser.add_argument(
        "--check-stage-timeouts",
        action="store_true",
        help="When archive mode is full, fail on stage elapsed ceilings",
    )
    parser.add_argument(
        "--require-alert-annotation",
        action="store_true",
        help="Require timeout_alert_emitted when stage_timeout_warning_count >= 1 (R80 post-alert)",
    )
    args = parser.parse_args()

    from scripts.cpu_cert_utils import (
        load_loop_close_archive,
        load_loop_close_archive_metadata,
        validate_loop_close_archive_hash,
        validate_loop_close_archive_metadata,
        validate_loop_close_archive_stage_timeouts,
    )

    metadata = load_loop_close_archive_metadata(args.metadata_path)
    archive = load_loop_close_archive(args.archive) if args.archive else None
    errors = validate_loop_close_archive_metadata(
        metadata,
        archive=archive,
        require_alert_annotation=args.require_alert_annotation,
    )
    if args.archive:
        errors.extend(validate_loop_close_archive_hash(metadata, archive_path=args.archive))
    if args.check_stage_timeouts and archive is not None:
        errors.extend(validate_loop_close_archive_stage_timeouts(archive))

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "schema": metadata.get("schema"),
                "bench_quick": metadata.get("bench_quick"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
