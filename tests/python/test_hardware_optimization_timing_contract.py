# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Docs drift guard for loop-close timing table (Loop R78)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.cpu_cert_utils import (  # noqa: E402
    LOOP_CLOSE_TIMING_TABLE_BEGIN,
    LOOP_CLOSE_TIMING_TABLE_END,
    loop_close_timing_doc_rows,
    loop_close_timing_markdown_table,
    parse_hardware_optimization_timing_table,
)


def test_hardware_optimization_timing_contract_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    parsed = parse_hardware_optimization_timing_table(text)
    expected = loop_close_timing_doc_rows()
    assert parsed, "timing table missing from HARDWARE_OPTIMIZATION.md"
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got["mode"] == want["mode"]
        assert got["stage"] == want["stage"]
        assert got["soft_limit_s"] == want["soft_limit_s"]
        assert got["hard_ceiling_s"] == want["hard_ceiling_s"]


def test_loop_close_timing_markdown_table_matches_doc_rows():
    parsed = parse_hardware_optimization_timing_table(loop_close_timing_markdown_table())
    expected = loop_close_timing_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want


def test_hardware_optimization_timing_table_parse_uses_marker_block_only():
    decoy = (
        "| Mode | Stage | Soft limit (s) | Hard ceiling (s) |\n"
        "|------|-------|----------------|------------------|\n"
        "| quick | demos | 999 | 999 |\n"
    )
    real_table = loop_close_timing_markdown_table()
    sample = (
        f"decoy\n{decoy}\n"
        f"{LOOP_CLOSE_TIMING_TABLE_BEGIN}\n{real_table}\n{LOOP_CLOSE_TIMING_TABLE_END}\n"
    )
    parsed = parse_hardware_optimization_timing_table(sample)
    expected = loop_close_timing_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want
    assert all(row["soft_limit_s"] != 999.0 for row in parsed)


def test_makefile_check_loop_close_timing_doc():
    import subprocess

    proc = subprocess.run(
        ["make", "check-loop-close-timing-doc"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_replace_timing_table_markers_roundtrip():
    from scripts.render_loop_close_timing_doc import replace_timing_table_markers

    table = loop_close_timing_markdown_table()
    sample = (
        f"before\n{LOOP_CLOSE_TIMING_TABLE_BEGIN}\nold\n{LOOP_CLOSE_TIMING_TABLE_END}\nafter"
    )
    updated = replace_timing_table_markers(sample, table)
    assert "old" not in updated
    assert table in updated
    assert updated.startswith("before\n")
    assert updated.endswith("after")


def test_replace_timing_table_markers_paired_lookup_with_label_decoy():
    from scripts.cpu_cert_utils import _loop_close_marker_doc_label
    from scripts.render_loop_close_timing_doc import replace_timing_table_markers

    label = _loop_close_marker_doc_label(LOOP_CLOSE_TIMING_TABLE_BEGIN)
    table = loop_close_timing_markdown_table()
    sample = (
        f"| timing | `{label}` | fn |\n"
        f"before\n{LOOP_CLOSE_TIMING_TABLE_BEGIN}\nold\n{LOOP_CLOSE_TIMING_TABLE_END}\nafter"
    )
    updated = replace_timing_table_markers(sample, table)
    assert "old" not in updated
    assert table in updated
    assert updated.endswith("after")
