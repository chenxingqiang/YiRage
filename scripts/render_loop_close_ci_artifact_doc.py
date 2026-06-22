#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Render loop-close CI artifact + workflow/make mapping tables (Loop R94).

Usage:
  PYTHONPATH=. python3 scripts/render_loop_close_ci_artifact_doc.py --check
  PYTHONPATH=. python3 scripts/render_loop_close_ci_artifact_doc.py --write
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


def replace_marker_block(text: str, begin: str, end: str, table: str) -> str:
    """Replace content between paired marker comments (R112 spec dispatch)."""
    from scripts.cpu_cert_utils import (
        apply_loop_close_doc_render_write_block_replace,
        loop_close_doc_render_write_block_specs,
        replace_loop_close_doc_marker_block,
    )

    if begin not in text or end not in text:
        raise ValueError(f"doc markers missing (need {begin} / {end})")
    for spec in loop_close_doc_render_write_block_specs():
        if spec["marker_begin"] == begin and spec["marker_end"] == end:
            return apply_loop_close_doc_render_write_block_replace(spec, text, table)
    return replace_loop_close_doc_marker_block(text, begin, end, table)


def write_ci_artifact_tables_to_doc(path: str = DOC_PATH) -> str:
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_END,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END,
        cpu_ci_artifact_doc_intro_line,
        cpu_ci_artifact_doc_markdown_table,
        cpu_ci_path_symmetry_doc_intro_line,
        cpu_ci_path_symmetry_doc_markdown_table,
        cpu_ci_workflow_make_doc_intro_line,
        cpu_ci_workflow_make_step_doc_markdown_table,
        cpu_mlir_ci_bundle_contract_doc_intro_line,
        cpu_mlir_ci_bundle_contract_doc_markdown_table,
        loop_close_doc_intro_line_doc_intro_line,
        loop_close_doc_intro_line_doc_markdown_table,
        loop_close_doc_makefile_helpers_doc_intro_line,
        loop_close_doc_makefile_helpers_doc_markdown_table,
        loop_close_doc_render_check_write_crossref_doc_intro_line,
        loop_close_doc_render_check_write_crossref_doc_markdown_table,
        loop_close_doc_render_write_block_doc_intro_line,
        loop_close_doc_render_write_block_doc_markdown_table,
        replace_line_before_marker,
    )

    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN,
        loop_close_doc_makefile_helpers_doc_intro_line(),
    )
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN,
        cpu_mlir_ci_bundle_contract_doc_intro_line(),
    )
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN,
        loop_close_doc_render_write_block_doc_intro_line(),
    )
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
        loop_close_doc_render_check_write_crossref_doc_intro_line(),
    )
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
        loop_close_doc_intro_line_doc_intro_line(),
    )
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
        cpu_ci_path_symmetry_doc_intro_line(),
    )
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN,
        cpu_ci_artifact_doc_intro_line(),
    )
    text = replace_line_before_marker(
        text,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
        cpu_ci_workflow_make_doc_intro_line(),
    )
    text = replace_marker_block(
        text,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END,
        loop_close_doc_makefile_helpers_doc_markdown_table(),
    )
    text = replace_marker_block(
        text,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END,
        cpu_mlir_ci_bundle_contract_doc_markdown_table(),
    )
    text = replace_marker_block(
        text,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END,
        loop_close_doc_render_write_block_doc_markdown_table(),
    )
    text = replace_marker_block(
        text,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END,
        loop_close_doc_render_check_write_crossref_doc_markdown_table(),
    )
    text = replace_marker_block(
        text,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
        loop_close_doc_intro_line_doc_markdown_table(),
    )
    text = replace_marker_block(
        text,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END,
        cpu_ci_path_symmetry_doc_markdown_table(),
    )
    text = replace_marker_block(
        text,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_END,
        cpu_ci_artifact_doc_markdown_table(),
    )
    text = replace_marker_block(
        text,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END,
        cpu_ci_workflow_make_step_doc_markdown_table(),
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def check_ci_artifact_intro_lines(text: str) -> int:
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN,
        cpu_ci_artifact_doc_intro_line,
        cpu_ci_path_symmetry_doc_intro_line,
        cpu_ci_workflow_make_doc_intro_line,
        cpu_mlir_ci_bundle_contract_doc_intro_line,
        loop_close_doc_intro_line_doc_intro_line,
        loop_close_doc_makefile_helpers_doc_intro_line,
        loop_close_doc_render_check_write_crossref_doc_intro_line,
        loop_close_doc_render_write_block_doc_intro_line,
        line_before_marker,
    )

    for marker, expected in (
        (LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN, loop_close_doc_makefile_helpers_doc_intro_line()),
        (LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN, cpu_mlir_ci_bundle_contract_doc_intro_line()),
        (LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN, loop_close_doc_render_write_block_doc_intro_line()),
        (
            LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
            loop_close_doc_render_check_write_crossref_doc_intro_line(),
        ),
        (LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN, loop_close_doc_intro_line_doc_intro_line()),
        (LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN, cpu_ci_path_symmetry_doc_intro_line()),
        (LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN, cpu_ci_artifact_doc_intro_line()),
        (LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN, cpu_ci_workflow_make_doc_intro_line()),
    ):
        got = line_before_marker(text, marker)
        if got != expected:
            print(f"CI doc intro drift before {marker}: {got!r} != {expected!r}", file=sys.stderr)
            return 1
    return 0


def check_ci_artifact_tables(path: str = DOC_PATH) -> int:
    from scripts.cpu_cert_utils import (
        cpu_ci_artifact_doc_rows,
        cpu_ci_workflow_make_step_doc_rows,
        cpu_ci_workflow_path_symmetry_doc_rows,
        cpu_mlir_ci_bundle_contract_doc_rows,
        loop_close_doc_intro_line_doc_rows,
        loop_close_doc_makefile_helpers_doc_rows,
        loop_close_doc_render_check_write_crossref_rows,
        loop_close_doc_render_write_block_doc_rows,
        parse_hardware_optimization_ci_artifact_table,
        parse_hardware_optimization_ci_path_symmetry_table,
        parse_hardware_optimization_ci_workflow_make_table,
        parse_hardware_optimization_doc_intro_line_table,
        parse_hardware_optimization_doc_render_check_write_crossref_table,
        parse_hardware_optimization_doc_render_write_block_table,
        parse_hardware_optimization_makefile_helpers_table,
        parse_hardware_optimization_mlir_ci_bundle_contract_table,
    )

    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    intro_err = check_ci_artifact_intro_lines(text)
    if intro_err:
        return intro_err

    parsed_helpers = parse_hardware_optimization_makefile_helpers_table(text)
    expected_helpers = loop_close_doc_makefile_helpers_doc_rows()
    if len(parsed_helpers) != len(expected_helpers):
        print(
            f"makefile helpers table row count mismatch: doc {len(parsed_helpers)} "
            f"!= expected {len(expected_helpers)}",
            file=sys.stderr,
        )
        return 1
    for got, want in zip(parsed_helpers, expected_helpers):
        if got != want:
            print(f"makefile helpers table drift: {got} != {want}", file=sys.stderr)
            return 1

    parsed_bundle = parse_hardware_optimization_mlir_ci_bundle_contract_table(text)
    expected_bundle = cpu_mlir_ci_bundle_contract_doc_rows()
    if len(parsed_bundle) != len(expected_bundle):
        print(
            f"bundle contract table row count mismatch: doc {len(parsed_bundle)} "
            f"!= expected {len(expected_bundle)}",
            file=sys.stderr,
        )
        return 1
    for got, want in zip(parsed_bundle, expected_bundle):
        if got != want:
            print(f"bundle contract table drift: {got} != {want}", file=sys.stderr)
            return 1

    parsed_render_blocks = parse_hardware_optimization_doc_render_write_block_table(text)
    expected_render_blocks = loop_close_doc_render_write_block_doc_rows()
    if len(parsed_render_blocks) != len(expected_render_blocks):
        print(
            f"render write block table row count mismatch: doc {len(parsed_render_blocks)} "
            f"!= expected {len(expected_render_blocks)}",
            file=sys.stderr,
        )
        return 1
    for got, want in zip(parsed_render_blocks, expected_render_blocks):
        if got != want:
            print(f"render write block table drift: {got} != {want}", file=sys.stderr)
            return 1

    parsed_crossref = parse_hardware_optimization_doc_render_check_write_crossref_table(text)
    expected_crossref = loop_close_doc_render_check_write_crossref_rows()
    if len(parsed_crossref) != len(expected_crossref):
        print(
            f"check/write cross-ref table row count mismatch: doc {len(parsed_crossref)} "
            f"!= expected {len(expected_crossref)}",
            file=sys.stderr,
        )
        return 1
    for got, want in zip(parsed_crossref, expected_crossref):
        if got != want:
            print(f"check/write cross-ref table drift: {got} != {want}", file=sys.stderr)
            return 1

    parsed_intro_lines = parse_hardware_optimization_doc_intro_line_table(text)
    expected_intro_lines = loop_close_doc_intro_line_doc_rows()
    if len(parsed_intro_lines) != len(expected_intro_lines):
        print(
            f"doc intro line table row count mismatch: doc {len(parsed_intro_lines)} "
            f"!= expected {len(expected_intro_lines)}",
            file=sys.stderr,
        )
        return 1
    for got, want in zip(parsed_intro_lines, expected_intro_lines):
        if got != want:
            print(f"doc intro line table drift: {got} != {want}", file=sys.stderr)
            return 1

    parsed_symmetry = parse_hardware_optimization_ci_path_symmetry_table(text)
    expected_symmetry = cpu_ci_workflow_path_symmetry_doc_rows()
    if len(parsed_symmetry) != len(expected_symmetry):
        print(
            f"path symmetry table row count mismatch: doc {len(parsed_symmetry)} "
            f"!= expected {len(expected_symmetry)}",
            file=sys.stderr,
        )
        return 1
    for got, want in zip(parsed_symmetry, expected_symmetry):
        if got != want:
            print(f"path symmetry table drift: {got} != {want}", file=sys.stderr)
            return 1

    parsed_artifacts = parse_hardware_optimization_ci_artifact_table(text)
    expected_artifacts = cpu_ci_artifact_doc_rows()
    if len(parsed_artifacts) != len(expected_artifacts):
        print(
            f"CI artifact table row count mismatch: doc {len(parsed_artifacts)} "
            f"!= expected {len(expected_artifacts)}",
            file=sys.stderr,
        )
        return 1
    for got, want in zip(parsed_artifacts, expected_artifacts):
        if got != want:
            print(f"CI artifact table drift: {got} != {want}", file=sys.stderr)
            return 1

    parsed_steps = parse_hardware_optimization_ci_workflow_make_table(text)
    expected_steps = cpu_ci_workflow_make_step_doc_rows()
    if len(parsed_steps) != len(expected_steps):
        print(
            f"workflow/make table row count mismatch: doc {len(parsed_steps)} "
            f"!= expected {len(expected_steps)}",
            file=sys.stderr,
        )
        return 1
    for got, want in zip(parsed_steps, expected_steps):
        if got != want:
            print(f"workflow/make table drift: {got} != {want}", file=sys.stderr)
            return 1

    from scripts.cpu_cert_utils import (
        loop_close_ci_artifact_doc_bundle_sync_gate_check,
        loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet,
        loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled,
    )

    if loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled():
        print(
            loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()
            + " (manifest row count + blocks summary + combined sync gates)",
            file=sys.stderr,
        )
        return 1

    if not loop_close_ci_artifact_doc_bundle_sync_gate_check(text):
        print(
            loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()
            + " (manifest row count + blocks summary + combined sync gates)",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "check": (
                    "mlir_ci_bundle_contract_render_write_blocks_"
                    "path_symmetry_artifact_and_workflow_make_table_sync"
                ),
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify HARDWARE_OPTIMIZATION.md bundle contract + render write blocks + "
            "path symmetry + CI artifact + workflow/make tables"
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Replace bundle contract + render write blocks + path symmetry + CI artifact + "
            "workflow/make table blocks in HARDWARE_OPTIMIZATION.md"
        ),
    )
    parser.add_argument(
        "--doc-path",
        default=DOC_PATH,
        help="HARDWARE_OPTIMIZATION.md path for --check/--write (default: repo doc)",
    )
    args = parser.parse_args()

    if args.write:
        write_ci_artifact_tables_to_doc(args.doc_path)
        print(json.dumps({"ok": True, "write": args.doc_path}))
        return 0

    if args.check:
        return check_ci_artifact_tables(args.doc_path)

    from scripts.cpu_cert_utils import (
        cpu_ci_artifact_doc_markdown_table,
        cpu_ci_path_symmetry_doc_markdown_table,
        cpu_ci_workflow_make_step_doc_markdown_table,
        cpu_mlir_ci_bundle_contract_doc_markdown_table,
        loop_close_doc_render_write_block_doc_markdown_table,
    )

    print(cpu_mlir_ci_bundle_contract_doc_markdown_table())
    print()
    print(loop_close_doc_render_write_block_doc_markdown_table())
    print()
    print(cpu_ci_path_symmetry_doc_markdown_table())
    print()
    print(cpu_ci_artifact_doc_markdown_table())
    print()
    print(cpu_ci_workflow_make_step_doc_markdown_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
