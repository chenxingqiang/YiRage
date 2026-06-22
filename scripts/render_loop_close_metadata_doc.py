#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Render loop-close metadata sidecar field table (Loop R83/R96).

Usage:
  PYTHONPATH=. python3 scripts/render_loop_close_metadata_doc.py --check
  PYTHONPATH=. python3 scripts/render_loop_close_metadata_doc.py --write
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


def replace_metadata_table_markers(text: str, table: str) -> str:
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_METADATA_FIELDS_BEGIN,
        LOOP_CLOSE_METADATA_FIELDS_END,
        replace_loop_close_doc_marker_block,
    )

    if LOOP_CLOSE_METADATA_FIELDS_BEGIN not in text or LOOP_CLOSE_METADATA_FIELDS_END not in text:
        raise ValueError(
            "metadata field markers missing in doc "
            f"(need {LOOP_CLOSE_METADATA_FIELDS_BEGIN} / {LOOP_CLOSE_METADATA_FIELDS_END})"
        )
    return replace_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_METADATA_FIELDS_BEGIN,
        LOOP_CLOSE_METADATA_FIELDS_END,
        table,
    )


def check_metadata_intro_line(text: str) -> int:
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_METADATA_FIELDS_BEGIN,
        line_before_marker,
        loop_close_metadata_table_doc_intro_line,
    )

    got = line_before_marker(text, LOOP_CLOSE_METADATA_FIELDS_BEGIN)
    expected = loop_close_metadata_table_doc_intro_line()
    if got != expected:
        print(f"metadata doc intro drift: {got!r} != {expected!r}", file=sys.stderr)
        return 1
    return 0


def write_metadata_table_to_doc(path: str = DOC_PATH) -> str:
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_METADATA_FIELDS_BEGIN,
        loop_close_metadata_doc_markdown_table,
        loop_close_metadata_table_doc_intro_line,
        replace_line_before_marker,
    )

    table = loop_close_metadata_doc_markdown_table()
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_METADATA_FIELDS_BEGIN,
        loop_close_metadata_table_doc_intro_line(),
    )
    updated = replace_metadata_table_markers(text, table)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(updated)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify HARDWARE_OPTIMIZATION.md metadata table matches single source",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace metadata field table block in HARDWARE_OPTIMIZATION.md",
    )
    parser.add_argument(
        "--doc-path",
        default=DOC_PATH,
        help="HARDWARE_OPTIMIZATION.md path for --check/--write (default: repo doc)",
    )
    args = parser.parse_args()

    from scripts.cpu_cert_utils import (
        loop_close_metadata_doc_markdown_table,
        loop_close_metadata_doc_rows,
        parse_hardware_optimization_metadata_doc_table,
    )

    generated = loop_close_metadata_doc_markdown_table()

    if args.write:
        write_metadata_table_to_doc(args.doc_path)
        print(json.dumps({"ok": True, "write": args.doc_path}))
        return 0

    if args.check:
        with open(args.doc_path, encoding="utf-8") as fh:
            text = fh.read()
        intro_err = check_metadata_intro_line(text)
        if intro_err:
            return intro_err
        parsed = parse_hardware_optimization_metadata_doc_table(text)
        expected = loop_close_metadata_doc_rows()
        if len(parsed) != len(expected):
            print(
                f"metadata table row count mismatch: doc {len(parsed)} != expected {len(expected)}",
                file=sys.stderr,
            )
            return 1
        for got, want in zip(parsed, expected):
            if got != want:
                print(f"metadata table drift: {got} != {want}", file=sys.stderr)
                return 1
        print(json.dumps({"ok": True, "check": "metadata_table_sync"}))
        return 0

    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
