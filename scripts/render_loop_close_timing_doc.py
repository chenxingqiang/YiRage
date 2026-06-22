#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Render loop-close timing markdown table from single source (Loop R79/R80/R96).

Usage:
  PYTHONPATH=. python3 scripts/render_loop_close_timing_doc.py
  PYTHONPATH=. python3 scripts/render_loop_close_timing_doc.py --check
  PYTHONPATH=. python3 scripts/render_loop_close_timing_doc.py --write
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DOC_PATH = os.path.join(_REPO, "docs", "HARDWARE_OPTIMIZATION.md")


def replace_timing_table_markers(text: str, table: str) -> str:
    """Replace content between marker comments with generated table (R106 paired lookup)."""
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_TIMING_TABLE_BEGIN,
        LOOP_CLOSE_TIMING_TABLE_END,
        replace_loop_close_doc_marker_block,
    )

    if LOOP_CLOSE_TIMING_TABLE_BEGIN not in text or LOOP_CLOSE_TIMING_TABLE_END not in text:
        raise ValueError(
            "timing table markers missing in doc "
            f"(need {LOOP_CLOSE_TIMING_TABLE_BEGIN} / {LOOP_CLOSE_TIMING_TABLE_END})"
        )
    return replace_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_TIMING_TABLE_BEGIN,
        LOOP_CLOSE_TIMING_TABLE_END,
        table,
    )


def check_timing_intro_line(text: str) -> int:
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_TIMING_TABLE_BEGIN,
        line_before_marker,
        loop_close_timing_table_doc_intro_line,
    )

    got = line_before_marker(text, LOOP_CLOSE_TIMING_TABLE_BEGIN)
    expected = loop_close_timing_table_doc_intro_line()
    if got != expected:
        print(f"timing doc intro drift: {got!r} != {expected!r}", file=sys.stderr)
        return 1
    return 0


def write_timing_table_to_doc(path: str = DOC_PATH) -> str:
    """Write generated table into doc between markers; returns updated doc text."""
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_TIMING_TABLE_BEGIN,
        loop_close_timing_markdown_table,
        loop_close_timing_table_doc_intro_line,
        replace_line_before_marker,
    )

    table = loop_close_timing_markdown_table()
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_TIMING_TABLE_BEGIN,
        loop_close_timing_table_doc_intro_line(),
    )
    updated = replace_timing_table_markers(text, table)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(updated)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify HARDWARE_OPTIMIZATION.md table matches generated output",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace timing table block in HARDWARE_OPTIMIZATION.md from single source",
    )
    parser.add_argument(
        "--doc-path",
        default=DOC_PATH,
        help="HARDWARE_OPTIMIZATION.md path for --check/--write (default: repo doc)",
    )
    args = parser.parse_args()

    from scripts.cpu_cert_utils import (
        loop_close_timing_doc_rows,
        loop_close_timing_markdown_table,
        parse_hardware_optimization_timing_table,
    )

    generated = loop_close_timing_markdown_table()

    if args.write:
        write_timing_table_to_doc(args.doc_path)
        print(json.dumps({"ok": True, "write": args.doc_path}))
        return 0

    if args.check:
        with open(args.doc_path, encoding="utf-8") as fh:
            text = fh.read()
        intro_err = check_timing_intro_line(text)
        if intro_err:
            return intro_err
        parsed = parse_hardware_optimization_timing_table(text)
        expected_rows = loop_close_timing_doc_rows()
        if len(parsed) != len(expected_rows):
            print(
                f"timing table row count mismatch: doc {len(parsed)} != expected {len(expected_rows)}",
                file=sys.stderr,
            )
            return 1
        for got, want in zip(parsed, expected_rows):
            if got != want:
                print(f"timing table drift: {got} != {want}", file=sys.stderr)
                return 1
        print(json.dumps({"ok": True, "check": "timing_table_sync"}))
        return 0

    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
