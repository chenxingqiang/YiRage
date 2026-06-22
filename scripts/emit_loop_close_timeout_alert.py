#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Emit loop-close soft timeout alert placeholder from metadata sidecar (Loop R77).

Usage:
  PYTHONPATH=. python3 scripts/emit_loop_close_timeout_alert.py meta.json
  PYTHONPATH=. python3 scripts/emit_loop_close_timeout_alert.py meta.json \
    --output artifacts/loop-close-timeout-alert.json
  PYTHONPATH=. python3 scripts/emit_loop_close_timeout_alert.py meta.json --github-summary
  PYTHONPATH=. python3 scripts/emit_loop_close_timeout_alert.py meta.json \
    --annotate-metadata
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
    parser.add_argument("metadata_path", help="Path to .meta.json sidecar")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write alert JSON when warnings present (no-op when empty)",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Append alert summary to GITHUB_STEP_SUMMARY when warnings present",
    )
    parser.add_argument(
        "--annotate-metadata",
        action="store_true",
        help="Set timeout_alert_emitted on metadata sidecar when alert is emitted",
    )
    args = parser.parse_args()

    from scripts.cpu_cert_utils import (
        load_loop_close_archive_metadata,
        loop_close_timeout_alert_payload,
    )

    metadata = load_loop_close_archive_metadata(args.metadata_path)
    alert = loop_close_timeout_alert_payload(metadata)
    if alert is None:
        print(json.dumps({"ok": True, "alert": None}, indent=2))
        return 0

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(alert, fh, indent=2)
            fh.write("\n")

    if args.annotate_metadata:
        metadata["timeout_alert_emitted"] = True
        metadata["timeout_alert_pending"] = False
        with open(args.metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
            fh.write("\n")

    if args.github_summary:
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write("\n### Loop-close soft timeout alert (placeholder)\n\n")
                fh.write(f"{alert['summary']}\n\n")
                fh.write("```json\n")
                fh.write(json.dumps(alert["warnings"], indent=2))
                fh.write("\n```\n")

    print(json.dumps({"ok": True, "alert": alert}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
