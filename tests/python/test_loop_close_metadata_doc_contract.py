# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Docs drift guard for loop-close metadata sidecar fields (Loop R82)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.cpu_cert_utils import (  # noqa: E402
    LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN,
    LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
    LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
    LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
    LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
    LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
    LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END,
    LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN,
    LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END,
    LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN,
    LOOP_CLOSE_METADATA_DOC_SCHEMA,
    LOOP_CLOSE_METADATA_FIELDS_BEGIN,
    LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN,
    LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END,
    line_before_marker,
    cpu_ci_artifact_doc_markdown_table,
    cpu_ci_artifact_doc_rows,
    cpu_ci_artifact_manifest,
    cpu_ci_path_symmetry_doc_markdown_table,
    cpu_ci_workflow_make_step_doc_markdown_table,
    cpu_ci_workflow_make_step_doc_rows,
    cpu_ci_workflow_make_target_manifest,
    cpu_ci_workflow_path_symmetry_doc_rows,
    cpu_mlir_ci_bundle_contract_doc_markdown_table,
    cpu_mlir_ci_bundle_contract_doc_rows,
    cpu_mlir_ci_bundle_test_contract_manifest,
    cpu_mlir_ci_bundle_test_contract_manifest_row_count,
    cpu_mlir_ci_bundle_contract_doc_sync_gate_ok,
    cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok,
    cpu_mlir_ci_workflow_path_symmetry_doc_rows,
    extract_loop_close_doc_marker_block,
    replace_loop_close_doc_marker_block,
    apply_loop_close_doc_render_write_block_replace,
    loop_close_archive_metadata,
    loop_close_ci_doc_render_path_triggers,
    loop_close_ci_doc_render_path_triggers_crossref_scripts,
    loop_close_ci_artifact_doc_bundle_sync_gate_check,
    loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet,
    loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env,
    loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled,
    loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref,
    loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref_assign_fragment,
    loop_close_doc_force_fail_crossref_and_check_row_parity_ok,
    loop_close_doc_force_fail_three_way_intro_parity_ok,
    loop_close_doc_force_fail_env_stripped_subprocess_env,
    loop_close_doc_makefile_helpers_check_row_force_fail_purpose_fragment,
    loop_close_doc_makefile_helpers_manifest_new_helpers_crossref,
    loop_close_doc_makefile_helpers_manifest_new_helpers_crossref_ok,
    loop_close_doc_makefile_helpers_manifest_helpers_crossref,
    loop_close_doc_makefile_helpers_manifest_helpers_parity_ok,
    loop_close_doc_makefile_helpers_manifest_helpers_parity_purpose_fragment,
    loop_close_doc_bundle_intro_manifest_helpers_parity_fragment,
    loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok,
    loop_close_doc_intro_line_three_way_parity_ok,
    loop_close_doc_intro_line_bundle_manifest_parity_ok,
    loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok,
    loop_close_doc_manifest_parity_three_way_ok,
    loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan,
    loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_python_snippet,
    loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv,
    loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches,
    loop_close_doc_check_loop_close_docs_make_subprocess_argv,
    loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok,
    loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text,
    loop_close_doc_render_check_subprocess_argv_chain,
    loop_close_docs_smoke_check_make_subprocess_argv,
    loop_close_doc_render_check_write_crossref_force_fail_intro_fragment,
    loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table,
    loop_close_ci_docs_gate_step_names,
    loop_close_ci_docs_gate_workflows,
    loop_close_doc_bundle_loop_revision,
    loop_close_doc_intro_line_specs,
    loop_close_doc_intro_line_doc_row_count,
    loop_close_doc_intro_line_doc_rows,
    loop_close_doc_makefile_helpers_doc_intro_line,
    loop_close_doc_makefile_helpers_doc_rows,
    loop_close_doc_render_check_specs,
    loop_close_doc_render_check_script_doc_crossref,
    normalize_loop_close_doc_render_check_script_doc_label,
    loop_close_doc_render_check_write_crossref_block_count_parity,
    loop_close_doc_render_check_write_crossref_blocks_summary_parity,
    loop_close_doc_render_check_write_crossref_doc_markdown_table,
    loop_close_doc_render_check_write_crossref_doc_intro_line,
    loop_close_doc_render_check_write_crossref_rows,
    loop_close_doc_render_write_block_counts_by_write_spec,
    loop_close_doc_render_write_block_counts_summary,
    loop_close_doc_render_write_block_doc_markdown_table,
    loop_close_doc_render_write_block_doc_rows,
    loop_close_doc_render_write_block_specs,
    loop_close_doc_render_write_specs,
    loop_close_docs_smoke_make_target,
    loop_close_metadata_doc_field_names,
    loop_close_metadata_doc_markdown_table,
    loop_close_metadata_doc_rows,
    loop_close_timing_markdown_table,
    parse_ci_artifact_markdown_table,
    parse_ci_path_symmetry_markdown_table,
    parse_ci_workflow_make_step_markdown_table,
    parse_hardware_optimization_ci_artifact_table,
    parse_hardware_optimization_ci_path_symmetry_table,
    parse_hardware_optimization_ci_workflow_make_table,
    parse_hardware_optimization_doc_render_write_block_table,
    parse_hardware_optimization_doc_intro_line_table,
    parse_hardware_optimization_makefile_helpers_table,
    parse_hardware_optimization_doc_render_check_write_crossref_table,
    parse_hardware_optimization_metadata_doc_fields,
    parse_hardware_optimization_metadata_doc_table,
    parse_doc_render_write_block_markdown_table,
    parse_hardware_optimization_mlir_ci_bundle_contract_table,
    resolve_loop_close_doc_render_block_table,
    resolve_loop_close_doc_render_write_block_replace_fn,
    resolve_loop_close_doc_render_write_fn,
    simulate_downloaded_loop_close_regression_validate,
)


def _sample_report(*, slow_demos: bool = False):
    from tests.integration.test_cpu_loop_close import _sample_mlir_bench_profile_stage

    elapsed = 55.0 if slow_demos else 30.0
    return {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {"mlir_bench_profile": _sample_mlir_bench_profile_stage()},
        "profile": {
            "stage_elapsed_s": {"demos": elapsed, "mlir_bench_profile": 2.0},
            "total_elapsed_s": elapsed + 2.0,
            "stages_ok": 3,
            "demos_passed": 29,
            "mlir_bench_profile_ok": True,
        },
    }


def test_hardware_optimization_metadata_doc_fields_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    documented = parse_hardware_optimization_metadata_doc_fields(text)
    expected = set(loop_close_metadata_doc_field_names()) | {LOOP_CLOSE_METADATA_DOC_SCHEMA}
    missing = expected - documented
    assert not missing, f"HARDWARE_OPTIMIZATION.md missing documented fields: {sorted(missing)}"


def test_hardware_optimization_metadata_table_parse_uses_marker_block_only():
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_METADATA_FIELDS_BEGIN,
        LOOP_CLOSE_METADATA_FIELDS_END,
        loop_close_metadata_doc_markdown_table,
    )

    decoy = (
        "| Field | Description |\n"
        "|-------|-------------|\n"
        "| decoy_field | should not parse |\n"
    )
    real_table = loop_close_metadata_doc_markdown_table()
    sample = (
        f"decoy\n{decoy}\n"
        f"{LOOP_CLOSE_METADATA_FIELDS_BEGIN}\n{real_table}\n{LOOP_CLOSE_METADATA_FIELDS_END}\n"
    )
    parsed = parse_hardware_optimization_metadata_doc_table(sample)
    expected = loop_close_metadata_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want
    assert all(row["field"] != "decoy_field" for row in parsed)


def test_hardware_optimization_metadata_doc_table_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    parsed = parse_hardware_optimization_metadata_doc_table(text)
    expected = loop_close_metadata_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want


def test_loop_close_metadata_doc_markdown_table_matches_rows():
    parsed = parse_hardware_optimization_metadata_doc_table(
        loop_close_metadata_doc_markdown_table()
    )
    assert parsed == loop_close_metadata_doc_rows()


def test_makefile_smoke_check_loop_close_docs():
    import subprocess

    proc = subprocess.run(
        ["make", "smoke-check-loop-close-docs"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    for spec in loop_close_doc_render_check_specs():
        assert spec["script"] in proc.stdout or "check-loop-close" in proc.stdout
    assert '"check": "timing_table_sync"' in proc.stdout
    assert '"check": "metadata_table_sync"' in proc.stdout
    assert "render_loop_close_ci_artifact_doc.py" in proc.stdout
    assert "mlir_ci_bundle_contract" in proc.stdout
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet() not in proc.stderr
    assert not loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled()
    assert (
        os.environ.get(loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env())
        != "1"
    )


def test_hardware_optimization_mlir_ci_bundle_contract_helpers_documented():
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert "cpu_mlir_ci_bundle_test_contract_manifest()" in text
    for row in cpu_mlir_ci_bundle_test_contract_manifest():
        assert row["helper"] in text, f"missing documented helper {row['helper']}"
        assert row["test_module"] in text, f"missing documented module {row['test_module']}"
    for rel in loop_close_ci_docs_gate_workflows():
        assert rel.replace(".github/workflows/", "") in text, f"missing docs gate workflow {rel}"


def test_hardware_optimization_mlir_ci_bundle_contract_table_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_mlir_ci_bundle_contract_table(
        doc_path.read_text(encoding="utf-8")
    )
    expected = cpu_mlir_ci_bundle_contract_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want


def test_hardware_optimization_mlir_ci_bundle_contract_manifest_row_count_sync():
    manifest_count = cpu_mlir_ci_bundle_test_contract_manifest_row_count()
    manifest = cpu_mlir_ci_bundle_test_contract_manifest()
    assert manifest_count == len(manifest)
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_mlir_ci_bundle_contract_table(
        doc_path.read_text(encoding="utf-8")
    )
    assert len(parsed) == manifest_count
    assert manifest_count == len(cpu_mlir_ci_bundle_contract_doc_rows())


def test_cpu_mlir_ci_bundle_contract_doc_sync_gate_ok():
    assert cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()
    assert loop_close_doc_render_check_write_crossref_block_count_parity()
    manifest_count = cpu_mlir_ci_bundle_test_contract_manifest_row_count()
    assert manifest_count == len(cpu_mlir_ci_bundle_test_contract_manifest())
    intro_count = loop_close_doc_intro_line_doc_row_count()
    assert intro_count == len(loop_close_doc_intro_line_doc_rows())
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_mlir_ci_bundle_contract_table(
        doc_path.read_text(encoding="utf-8")
    )
    assert len(parsed) == manifest_count


def test_loop_close_doc_intro_line_doc_row_count_sync():
    intro_count = loop_close_doc_intro_line_doc_row_count()
    expected_rows = loop_close_doc_intro_line_doc_rows()
    assert intro_count == len(expected_rows)
    assert intro_count == len(loop_close_doc_intro_line_specs()) - 1
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_doc_intro_line_table(
        doc_path.read_text(encoding="utf-8")
    )
    assert len(parsed) == intro_count


def test_loop_close_doc_intro_line_and_bundle_contract_row_count_parity():
    intro_count = loop_close_doc_intro_line_doc_row_count()
    bundle_count = cpu_mlir_ci_bundle_test_contract_manifest_row_count()
    assert intro_count != bundle_count
    assert intro_count == len(loop_close_doc_intro_line_specs()) - 1
    assert bundle_count >= intro_count


def test_loop_close_doc_render_check_write_crossref_aligns_with_write_and_blocks():
    crossref = loop_close_doc_render_check_write_crossref_rows()
    check_specs = loop_close_doc_render_check_specs()
    write_specs = loop_close_doc_render_write_specs()
    assert len(crossref) == len(check_specs) == len(write_specs)
    write_by_name = {spec["name"]: spec for spec in write_specs}
    for row, check in zip(crossref, check_specs):
        assert row["check_name"] == check["name"]
        assert row["check_script"] == loop_close_doc_render_check_script_doc_crossref(check)
        assert "--doc-path" in row["check_script"]
        write = write_by_name[check["name"]]
        assert row["write_module"] == write["module"]
        assert row["write_fn"] == write["write_fn"]
        blocks = [
            block
            for block in loop_close_doc_render_write_block_specs()
            if block["write_spec"] == check["name"]
        ]
        assert row["block_count"] == str(len(blocks))
        assert set(row["replace_fns"].split(", ")) == {block["replace_fn"] for block in blocks}


def test_hardware_optimization_makefile_helpers_includes_manifest_row_count():
    rows = loop_close_doc_makefile_helpers_doc_rows()
    test_row = next(row for row in rows if row["target"] == "make test-cpu-mlir-ci-bundle")
    assert "cpu_mlir_ci_bundle_test_contract_manifest_row_count()" in test_row["purpose"]
    manifest_count = cpu_mlir_ci_bundle_test_contract_manifest_row_count()
    assert f"{manifest_count} rows" in test_row["purpose"]
    assert manifest_count == len(cpu_mlir_ci_bundle_test_contract_manifest())
    assert loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()
    assert loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()
    assert loop_close_doc_intro_line_bundle_manifest_parity_ok()
    assert loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()
    assert loop_close_doc_manifest_parity_three_way_ok()
    assert loop_close_doc_makefile_helpers_manifest_new_helpers_crossref_ok()
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    assert loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok(
        doc_path.read_text(encoding="utf-8")
    )
    helpers_intro = loop_close_doc_makefile_helpers_doc_intro_line()
    assert "--doc-path" in helpers_intro
    assert "must be unset on smoke" in helpers_intro
    assert (
        loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()
        in helpers_intro
    )
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert helpers_intro == line_before_marker(text, LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN)
    assert "cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()" in test_row["purpose"]
    assert "loop_close_doc_intro_line_doc_row_count()" in test_row["purpose"]
    assert "cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()" in test_row[
        "purpose"
    ]
    check_row = next(row for row in rows if row["target"] == "make check-loop-close-docs")
    assert "loop_close_doc_intro_line_doc_row_count()" in check_row["purpose"]
    assert "cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()" in check_row["purpose"]
    assert "cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()" in check_row[
        "purpose"
    ]
    smoke = loop_close_docs_smoke_make_target()
    smoke_row = next(row for row in rows if row["target"] == f"make {smoke}")
    assert "loop_close_doc_render_check_write_crossref_rows()" in smoke_row["purpose"]
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check()" in smoke_row["purpose"]
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()" in smoke_row[
        "purpose"
    ]
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env() in smoke_row[
        "purpose"
    ]
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env() in check_row[
        "purpose"
    ]
    assert "must be unset on smoke" in check_row["purpose"]
    assert loop_close_doc_force_fail_crossref_and_check_row_parity_ok()
    timing_row = next(
        row for row in rows if row["target"] == "make render-loop-close-timing-doc"
    )
    metadata_row = next(
        row for row in rows if row["target"] == "make render-loop-close-metadata-doc"
    )
    ci_artifact_row = next(
        row for row in rows if row["target"] == "make render-loop-close-ci-artifact-doc"
    )
    assert "--doc-path" in timing_row["purpose"]
    assert "--doc-path" in metadata_row["purpose"]
    assert "--doc-path" in ci_artifact_row["purpose"]


def test_loop_close_doc_render_check_hook_three_way_intro_parity():
    from scripts.cpu_cert_utils import (
        cpu_mlir_ci_bundle_contract_doc_intro_line,
        loop_close_doc_makefile_helpers_doc_intro_line,
        loop_close_doc_render_check_write_crossref_doc_intro_line,
    )

    hook = "loop_close_ci_artifact_doc_bundle_sync_gate_check()"
    snippet = "loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()"
    force_env = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()
    helpers_intro = loop_close_doc_makefile_helpers_doc_intro_line()
    bundle_intro = cpu_mlir_ci_bundle_contract_doc_intro_line()
    crossref_intro = loop_close_doc_render_check_write_crossref_doc_intro_line()
    for label, intro in (
        ("helpers", helpers_intro),
        ("bundle", bundle_intro),
        ("crossref", crossref_intro),
    ):
        assert hook in intro, f"{label} intro missing render check hook"
    assert snippet in crossref_intro
    assert force_env in crossref_intro
    smoke = loop_close_docs_smoke_make_target()
    smoke_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == f"make {smoke}"
    )
    assert force_env in smoke_row["purpose"]
    assert "must be unset" in smoke_row["purpose"]
    assert (
        loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()
        in smoke_row["purpose"]
    )
    assert "must be unset on smoke" in helpers_intro
    assert "--doc-path" in helpers_intro
    assert "--doc-path" in crossref_intro
    check_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make check-loop-close-docs"
    )
    assert "must be unset on smoke" in check_row["purpose"]
    assert loop_close_doc_force_fail_crossref_and_check_row_parity_ok()
    assert loop_close_doc_force_fail_three_way_intro_parity_ok()
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert helpers_intro == line_before_marker(text, LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN)
    assert bundle_intro == line_before_marker(text, LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN)
    assert crossref_intro == line_before_marker(
        text, LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN
    )


def test_loop_close_doc_smoke_helpers_and_crossref_intro_render_check_hook_parity():
    from scripts.cpu_cert_utils import loop_close_doc_render_check_write_crossref_doc_intro_line

    hook = "loop_close_ci_artifact_doc_bundle_sync_gate_check()"
    smoke = loop_close_docs_smoke_make_target()
    smoke_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == f"make {smoke}"
    )
    crossref_intro = loop_close_doc_render_check_write_crossref_doc_intro_line()
    assert hook in smoke_row["purpose"]
    assert hook in crossref_intro
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert crossref_intro == line_before_marker(
        text, LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN
    )


def test_loop_close_doc_makefile_helpers_and_bundle_intro_render_check_hook_parity():
    from scripts.cpu_cert_utils import (
        cpu_mlir_ci_bundle_contract_doc_intro_line,
        loop_close_doc_makefile_helpers_doc_intro_line,
    )

    hook = "loop_close_ci_artifact_doc_bundle_sync_gate_check()"
    helpers_intro = loop_close_doc_makefile_helpers_doc_intro_line()
    bundle_intro = cpu_mlir_ci_bundle_contract_doc_intro_line()
    assert hook in helpers_intro
    assert hook in bundle_intro
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert helpers_intro == line_before_marker(text, LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN)
    assert bundle_intro == line_before_marker(text, LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN)


def test_hardware_optimization_mlir_ci_bundle_contract_table_marker_block_only():
    decoy = (
        "| Test module | Single-source helper | Contract |\n"
        "|-------------|----------------------|----------|\n"
        "| `tests/decoy.py` | `decoy_helper()` | decoy contract |\n"
    )
    real_table = cpu_mlir_ci_bundle_contract_doc_markdown_table()
    sample = (
        f"decoy\n{decoy}\n"
        f"{LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN}\n{real_table}\n"
        f"{LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END}\n"
    )
    parsed = parse_hardware_optimization_mlir_ci_bundle_contract_table(sample)
    assert parsed == cpu_mlir_ci_bundle_contract_doc_rows()
    assert len(parsed) == cpu_mlir_ci_bundle_test_contract_manifest_row_count()


def test_hardware_optimization_mlir_ci_bundle_contract_decoy_and_sync_gate_on_full_doc():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    decoy = (
        "| Test module | Single-source helper | Contract |\n"
        "|-------------|----------------------|----------|\n"
        "| `tests/decoy.py` | `decoy_helper()` | decoy contract |\n"
    )
    begin = text.index(LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN)
    sample = text[:begin] + f"decoy\n{decoy}\n" + text[begin:]
    parsed = parse_hardware_optimization_mlir_ci_bundle_contract_table(sample)
    assert parsed == cpu_mlir_ci_bundle_contract_doc_rows()
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check(sample)
    assert cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()


def test_render_loop_close_timing_doc_check_subprocess_with_doc_path(tmp_path: Path):
    import subprocess

    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    copy_path = tmp_path / "HARDWARE_OPTIMIZATION.md"
    copy_path.write_text(doc_path.read_text(encoding="utf-8"), encoding="utf-8")
    proc = subprocess.run(
        [
            "python3",
            "scripts/render_loop_close_timing_doc.py",
            "--check",
            "--doc-path",
            str(copy_path),
        ],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert '"check": "timing_table_sync"' in proc.stdout


def test_render_loop_close_metadata_doc_check_subprocess_with_doc_path(tmp_path: Path):
    import subprocess

    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    copy_path = tmp_path / "HARDWARE_OPTIMIZATION.md"
    copy_path.write_text(doc_path.read_text(encoding="utf-8"), encoding="utf-8")
    proc = subprocess.run(
        [
            "python3",
            "scripts/render_loop_close_metadata_doc.py",
            "--check",
            "--doc-path",
            str(copy_path),
        ],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert '"check": "metadata_table_sync"' in proc.stdout


def test_loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet():
    assert (
        loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()
        == "bundle contract doc sync gate check failed"
    )


def _bad_mlir_ci_bundle_doc_missing_last_row(tmp_path: Path) -> Path:
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    bad_doc = tmp_path / "bad.md"
    text = doc_path.read_text(encoding="utf-8")
    begin = text.index(LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN)
    end = text.index(LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END)
    block = text[begin:end]
    lines = block.splitlines()
    manifest_rows = len(cpu_mlir_ci_bundle_test_contract_manifest())
    kept: list[str] = []
    data_rows = 0
    for line in lines:
        if line.startswith("| `tests/"):
            data_rows += 1
            if data_rows == manifest_rows:
                continue
        kept.append(line)
    bad_text = text[:begin] + "\n".join(kept) + text[end:]
    bad_doc.write_text(bad_text, encoding="utf-8")
    return bad_doc


def test_render_ci_artifact_check_sync_gate_failure_snippet_stderr(capsys, tmp_path: Path):
    from unittest.mock import patch

    from scripts.render_loop_close_ci_artifact_doc import check_ci_artifact_tables

    doc_path = tmp_path / "HARDWARE_OPTIMIZATION.md"
    doc_path.write_text(
        (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with patch(
        "scripts.cpu_cert_utils.loop_close_ci_artifact_doc_bundle_sync_gate_check",
        return_value=False,
    ):
        rc = check_ci_artifact_tables(str(doc_path))
    assert rc != 0
    captured = capsys.readouterr()
    assert (
        loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet() in captured.err
    )


def test_smoke_render_checks_sequence_ci_artifact_failure_stderr_subprocess(tmp_path: Path):
    import subprocess

    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    good_doc = tmp_path / "good.md"
    good_doc.write_text(doc_path.read_text(encoding="utf-8"), encoding="utf-8")
    bad_doc = _bad_mlir_ci_bundle_doc_missing_last_row(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(_REPO)}
    snippet = loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()

    for script, expected in (
        ("scripts/render_loop_close_timing_doc.py", "timing_table_sync"),
        ("scripts/render_loop_close_metadata_doc.py", "metadata_table_sync"),
    ):
        proc = subprocess.run(
            ["python3", script, "--check", "--doc-path", str(good_doc)],
            cwd=_REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        assert expected in proc.stdout

    proc = subprocess.run(
        [
            "python3",
            "scripts/render_loop_close_ci_artifact_doc.py",
            "--check",
            "--doc-path",
            str(bad_doc),
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert snippet in combined or "bundle contract table row count mismatch" in combined


def test_make_check_loop_close_docs_ci_artifact_failure_stderr_subprocess():
    import subprocess

    snippet = loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()
    force_env = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()
    proc = subprocess.run(
        ["make", "check-loop-close-docs"],
        cwd=_REPO,
        env={**os.environ, force_env: "1", "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert snippet in proc.stderr
    assert "render_loop_close_ci_artifact_doc.py" in proc.stderr + proc.stdout
    assert "bundle contract table row count mismatch" not in proc.stderr


def test_make_check_loop_close_docs_success_force_fail_env_not_inherited_subprocess():
    import subprocess

    force_env = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()
    env = loop_close_doc_force_fail_env_stripped_subprocess_env()
    proc = subprocess.run(
        ["make", "check-loop-close-docs"],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet() not in proc.stderr
    assert '"check": "timing_table_sync"' in proc.stdout
    assert '"check": "metadata_table_sync"' in proc.stdout
    assert "mlir_ci_bundle_contract" in proc.stdout
    assert env.get(force_env) != "1"


def test_smoke_and_check_loop_close_docs_success_dual_force_fail_env_subprocess():
    import subprocess

    force_env = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()
    snippet = loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()
    env = loop_close_doc_force_fail_env_stripped_subprocess_env()
    for target in ("smoke-check-loop-close-docs", "check-loop-close-docs"):
        proc = subprocess.run(
            ["make", target],
            cwd=_REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{target} failed: {proc.stderr or proc.stdout}"
        assert snippet not in proc.stderr
        assert not loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled()
        assert env.get(force_env) != "1"


def test_loop_close_doc_force_fail_env_stripped_subprocess_env():
    force_env = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()
    env = loop_close_doc_force_fail_env_stripped_subprocess_env(
        {force_env: "1", "PYTHONPATH": "/tmp", "HOME": "/home"}
    )
    assert force_env not in env
    assert env["PYTHONPATH"] == str(_REPO)
    assert env["HOME"] == "/home"


def test_loop_close_doc_force_fail_crossref_and_check_row_parity_ok():
    assert loop_close_doc_force_fail_crossref_and_check_row_parity_ok()
    check_fragment = loop_close_doc_makefile_helpers_check_row_force_fail_purpose_fragment()
    crossref_fragment = loop_close_doc_render_check_write_crossref_force_fail_intro_fragment()
    assign_fragment = (
        loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref_assign_fragment()
    )
    doc_crossref = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()
    assert assign_fragment == f"= {doc_crossref}"
    assert doc_crossref in check_fragment
    assert assign_fragment in crossref_fragment
    check_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make check-loop-close-docs"
    )
    crossref_intro = loop_close_doc_render_check_write_crossref_doc_intro_line()
    assert check_fragment in check_row["purpose"]
    assert crossref_fragment in crossref_intro


def test_loop_close_doc_force_fail_three_way_intro_parity_ok():
    assert loop_close_doc_force_fail_three_way_intro_parity_ok()
    from scripts.cpu_cert_utils import cpu_mlir_ci_bundle_contract_doc_intro_line

    bundle_intro = cpu_mlir_ci_bundle_contract_doc_intro_line()
    crossref_fragment = loop_close_doc_render_check_write_crossref_force_fail_intro_fragment()
    assert crossref_fragment in loop_close_doc_render_check_write_crossref_doc_intro_line()
    assert "loop_close_doc_force_fail_crossref_and_check_row_parity_ok()" in bundle_intro
    assert "loop_close_doc_force_fail_three_way_intro_parity_ok()" in bundle_intro


def test_loop_close_doc_makefile_helpers_manifest_helpers_parity_ok():
    assert loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()
    assert loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()
    assert loop_close_doc_makefile_helpers_manifest_new_helpers_crossref_ok()
    test_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make test-cpu-mlir-ci-bundle"
    )
    for helper in loop_close_doc_makefile_helpers_manifest_helpers_crossref():
        assert helper in test_row["purpose"]


def test_loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok():
    from scripts.cpu_cert_utils import cpu_mlir_ci_bundle_contract_doc_intro_line

    assert loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()
    bundle_intro = cpu_mlir_ci_bundle_contract_doc_intro_line()
    assert "loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()" in bundle_intro
    assert "loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()" in bundle_intro


def test_loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok():
    from scripts.cpu_cert_utils import cpu_mlir_ci_bundle_contract_doc_intro_line

    assert loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()
    assert loop_close_doc_intro_line_bundle_manifest_parity_ok()
    assert loop_close_doc_manifest_parity_three_way_ok()
    bundle_rows = [
        row
        for row in loop_close_doc_intro_line_doc_rows()
        if row["intro_fn"] == "cpu_mlir_ci_bundle_contract_doc_intro_line()"
    ]
    assert len(bundle_rows) == 1
    assert (
        bundle_rows[0]["manifest_parity_gate"]
        == "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()"
    )
    fragment = loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()
    test_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make test-cpu-mlir-ci-bundle"
    )
    assert fragment in test_row["purpose"]
    assert fragment in cpu_mlir_ci_bundle_contract_doc_intro_line()


def test_loop_close_doc_manifest_parity_three_way_ok():
    from scripts.cpu_cert_utils import cpu_mlir_ci_bundle_contract_doc_intro_line

    assert loop_close_doc_manifest_parity_three_way_ok()
    assert loop_close_doc_intro_line_bundle_manifest_parity_ok()
    assert loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()
    fragment = loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()
    test_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make test-cpu-mlir-ci-bundle"
    )
    assert fragment in test_row["purpose"]
    assert fragment in cpu_mlir_ci_bundle_contract_doc_intro_line()


def test_loop_close_doc_intro_line_bundle_manifest_parity_ok():
    from scripts.cpu_cert_utils import cpu_mlir_ci_bundle_contract_doc_intro_line

    assert loop_close_doc_intro_line_bundle_manifest_parity_ok()
    bundle_rows = [
        row
        for row in loop_close_doc_intro_line_doc_rows()
        if row["intro_fn"] == "cpu_mlir_ci_bundle_contract_doc_intro_line()"
    ]
    assert len(bundle_rows) == 1
    assert (
        bundle_rows[0]["manifest_parity_gate"]
        == "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()"
    )
    assert (
        loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()
        in cpu_mlir_ci_bundle_contract_doc_intro_line()
    )


def test_loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    assert loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok(text)
    fragment = loop_close_doc_makefile_helpers_manifest_helpers_parity_purpose_fragment()
    parsed = parse_hardware_optimization_makefile_helpers_table(text)
    test_row = next(
        row for row in parsed if row["target"] == "make test-cpu-mlir-ci-bundle"
    )
    assert fragment in test_row["purpose"]


def test_makefile_helpers_test_row_manifest_parity_purpose_fragment_subprocess():
    import subprocess

    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    env = loop_close_doc_force_fail_env_stripped_subprocess_env()
    proc = subprocess.run(
        [
            "python3",
            "-c",
            loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_python_snippet(
                str(doc_path)
            ),
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert loop_close_doc_manifest_parity_three_way_ok()


def test_loop_close_doc_intro_line_three_way_parity_ok():
    assert loop_close_doc_intro_line_three_way_parity_ok()
    assert loop_close_doc_force_fail_three_way_intro_parity_ok()
    parity_intro_fns = {
        "loop_close_doc_makefile_helpers_doc_intro_line()",
        "cpu_mlir_ci_bundle_contract_doc_intro_line()",
        "loop_close_doc_render_check_write_crossref_doc_intro_line()",
    }
    parity_rows = [
        row
        for row in loop_close_doc_intro_line_doc_rows()
        if row["intro_fn"] in parity_intro_fns
    ]
    assert len(parity_rows) == 3
    for row in parity_rows:
        assert row["intro_parity_gate"] == "loop_close_doc_force_fail_three_way_intro_parity_ok()"


def test_mixed_parse_check_loop_close_docs_full_chain_subprocess(tmp_path: Path):
    import subprocess

    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    patched = loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text(text)
    copy_path = tmp_path / "HARDWARE_OPTIMIZATION.md"
    copy_path.write_text(patched, encoding="utf-8")
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check(patched)
    env = loop_close_doc_force_fail_env_stripped_subprocess_env()
    snippet = loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()
    plan = loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan(
        str(copy_path),
        str(doc_path),
    )
    assert plan["canonical_doc_path"] == str(doc_path)
    assert plan["canonical_check_argv"] == loop_close_doc_check_loop_close_docs_make_subprocess_argv()
    assert plan["full_argv_batches"] == (
        loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches(str(copy_path))
    )
    argv_batches = plan["argv_batches"]
    assert len(argv_batches) == len(loop_close_doc_render_check_specs()) + 1
    full_argv_batches = plan["full_argv_batches"]
    assert len(full_argv_batches) == len(argv_batches) + 1
    for argv in full_argv_batches:
        proc = subprocess.run(
            argv,
            cwd=_REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        assert snippet not in proc.stderr
        if argv[0] == "python3":
            assert '"ok": true' in proc.stdout
        elif argv[-1] == "check-loop-close-docs":
            assert '"check": "timing_table_sync"' in proc.stdout
        else:
            assert '"check": "timing_table_sync"' in proc.stdout
    parity_proc = subprocess.run(
        [
            "python3",
            "-c",
            loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_python_snippet(
                str(doc_path)
            ),
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert parity_proc.returncode == 0, parity_proc.stderr or parity_proc.stdout


def test_loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches():
    patched = "/tmp/patched/HARDWARE_OPTIMIZATION.md"
    batches = loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches(patched)
    assert batches[:-1] == loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv(patched)
    assert batches[-1] == loop_close_doc_check_loop_close_docs_make_subprocess_argv()


def test_loop_close_doc_check_loop_close_docs_make_subprocess_argv():
    assert loop_close_doc_check_loop_close_docs_make_subprocess_argv() == [
        "make",
        "check-loop-close-docs",
    ]


def test_loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan():
    patched = "/tmp/patched/HARDWARE_OPTIMIZATION.md"
    canonical = "/tmp/canonical/HARDWARE_OPTIMIZATION.md"
    plan = loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan(
        patched,
        canonical,
    )
    assert plan["argv_batches"] == loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv(
        patched
    )
    assert plan["full_argv_batches"] == (
        loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches(patched)
    )
    assert plan["canonical_doc_path"] == canonical
    assert plan["canonical_check_argv"] == loop_close_doc_check_loop_close_docs_make_subprocess_argv()
    snippet = loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_python_snippet(
        canonical
    )
    assert "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()" in snippet
    assert canonical in snippet


def test_loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    patched = loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text(text)
    parsed = parse_hardware_optimization_doc_render_check_write_crossref_table(patched)
    assert parsed == loop_close_doc_render_check_write_crossref_rows()
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check(patched)


def test_loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv():
    doc_path = "/tmp/HARDWARE_OPTIMIZATION.md"
    batches = loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv(doc_path)
    assert batches[:-1] == loop_close_doc_render_check_subprocess_argv_chain(doc_path)
    assert batches[-1] == ["make", loop_close_docs_smoke_make_target()]


def test_mixed_parse_full_doc_sync_gate_subprocess(tmp_path: Path):
    import subprocess

    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    patched = loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text(text)
    copy_path = tmp_path / "HARDWARE_OPTIMIZATION.md"
    copy_path.write_text(patched, encoding="utf-8")
    parsed_mixed = parse_hardware_optimization_doc_render_check_write_crossref_table(
        patched
    )
    assert parsed_mixed == loop_close_doc_render_check_write_crossref_rows()
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check(patched)
    env = loop_close_doc_force_fail_env_stripped_subprocess_env()
    proc = subprocess.run(
        [
            "python3",
            "scripts/render_loop_close_ci_artifact_doc.py",
            "--check",
            "--doc-path",
            str(copy_path),
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert '"ok": true' in proc.stdout
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet() not in proc.stderr


def test_render_loop_close_ci_artifact_doc_check_subprocess_with_doc_path(tmp_path: Path):
    import subprocess

    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    copy_path = tmp_path / "HARDWARE_OPTIMIZATION.md"
    copy_path.write_text(doc_path.read_text(encoding="utf-8"), encoding="utf-8")
    proc = subprocess.run(
        [
            "python3",
            "scripts/render_loop_close_ci_artifact_doc.py",
            "--check",
            "--doc-path",
            str(copy_path),
        ],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert '"ok": true' in proc.stdout


def test_render_loop_close_ci_artifact_doc_check_sync_gate_failure_subprocess(tmp_path: Path):
    import subprocess

    bad_doc = _bad_mlir_ci_bundle_doc_missing_last_row(tmp_path)

    proc = subprocess.run(
        [
            "python3",
            "scripts/render_loop_close_ci_artifact_doc.py",
            "--check",
            "--doc-path",
            str(bad_doc),
        ],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    snippet = loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()
    assert (
        snippet in combined
        or "bundle contract table row count mismatch" in combined
    )


def test_hardware_optimization_doc_render_write_block_table_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_doc_render_write_block_table(
        doc_path.read_text(encoding="utf-8")
    )
    expected = loop_close_doc_render_write_block_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want


def test_loop_close_doc_render_write_block_doc_markdown_table_matches_rows():
    parsed = parse_hardware_optimization_doc_render_write_block_table(
        loop_close_doc_render_write_block_doc_markdown_table()
    )
    assert parsed == loop_close_doc_render_write_block_doc_rows()


def test_loop_close_mlir_ci_bundle_contract_doc_markdown_table_matches_rows():
    parsed = parse_hardware_optimization_mlir_ci_bundle_contract_table(
        cpu_mlir_ci_bundle_contract_doc_markdown_table()
    )
    assert parsed == cpu_mlir_ci_bundle_contract_doc_rows()


def test_loop_close_path_symmetry_cross_pairs_documented_with_pytest_names():
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    for row in cpu_ci_workflow_path_symmetry_doc_rows():
        if "test_loop_close_" not in row["contract"]:
            continue
        assert row["left"] in text
        assert row["right"] in text
        assert row["contract"] in text, f"missing cross symmetry contract {row['contract']}"


def test_loop_close_doc_bundle_loop_revision_in_intro_lines():
    rev = loop_close_doc_bundle_loop_revision()
    assert rev == "R136"
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    for marker_begin, intro_fn in loop_close_doc_intro_line_specs():
        intro = intro_fn()
        if intro_fn.__name__ == "loop_close_doc_makefile_helpers_doc_intro_line":
            assert f"R84–{rev}" in intro
        else:
            assert f"Loop {rev}" in intro, f"{intro_fn.__name__} missing Loop {rev}"
        assert intro == line_before_marker(text, marker_begin)


def test_loop_close_doc_marker_paired_lookup_skips_label_before_real_marker():
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_TIMING_TABLE_BEGIN,
        LOOP_CLOSE_TIMING_TABLE_END,
        _loop_close_marker_doc_label,
        replace_loop_close_doc_marker_block,
    )

    label = _loop_close_marker_doc_label(LOOP_CLOSE_TIMING_TABLE_BEGIN)
    table = loop_close_timing_markdown_table()
    sample = (
        f"| timing | `{label}` | fn |\n"
        f"intro\n{LOOP_CLOSE_TIMING_TABLE_BEGIN}\nold\n{LOOP_CLOSE_TIMING_TABLE_END}\n"
    )
    updated = replace_loop_close_doc_marker_block(
        sample,
        LOOP_CLOSE_TIMING_TABLE_BEGIN,
        LOOP_CLOSE_TIMING_TABLE_END,
        table,
    )
    assert "old" not in updated
    assert table in updated
    assert updated.index(label) < updated.index(LOOP_CLOSE_TIMING_TABLE_BEGIN)


def test_loop_close_timing_metadata_intro_lines_use_bundle_revision():
    rev = loop_close_doc_bundle_loop_revision()
    from scripts.cpu_cert_utils import (
        loop_close_metadata_table_doc_intro_line,
        loop_close_timing_table_doc_intro_line,
    )

    assert f"Loop {rev}" in loop_close_timing_table_doc_intro_line()
    assert f"Loop {rev}" in loop_close_metadata_table_doc_intro_line()


def test_hardware_optimization_makefile_helpers_loop_range_sync():
    from scripts.cpu_cert_utils import loop_close_doc_makefile_helpers_loop_range

    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    expected = loop_close_doc_makefile_helpers_loop_range()
    assert f"Loop {expected}" in text


def test_loop_close_doc_render_write_block_specs_includes_makefile_helpers_and_intro_line():
    markers = {spec["marker_begin"] for spec in loop_close_doc_render_write_block_specs()}
    assert LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN in markers
    assert LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN in markers
    assert LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN in markers
    assert LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN in markers
    assert LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN in markers


def test_hardware_optimization_metadata_doc_fields_marker_section_only():
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_METADATA_FIELDS_BEGIN,
        LOOP_CLOSE_METADATA_FIELDS_END,
        loop_close_metadata_doc_markdown_table,
        loop_close_metadata_table_doc_intro_line,
    )

    fake = "fake_outside_field"
    real_table = loop_close_metadata_doc_markdown_table()
    intro = loop_close_metadata_table_doc_intro_line()
    sample = (
        f"Prose ``{fake}`` must not count.\n\n"
        f"{intro}\n\n"
        f"{LOOP_CLOSE_METADATA_FIELDS_BEGIN}\n{real_table}\n{LOOP_CLOSE_METADATA_FIELDS_END}\n"
    )
    documented = parse_hardware_optimization_metadata_doc_fields(sample)
    assert fake not in documented
    assert LOOP_CLOSE_METADATA_DOC_SCHEMA in documented
    assert set(loop_close_metadata_doc_field_names()).issubset(documented)

    prose_only = f"``bench_quick`` and ``{LOOP_CLOSE_METADATA_DOC_SCHEMA}`` in prose only.\n"
    assert parse_hardware_optimization_metadata_doc_fields(prose_only) == set()


def test_hardware_optimization_mlir_ci_bundle_contract_includes_helpers_parse_and_metadata_fields():
    helpers = {row["helper"] for row in cpu_mlir_ci_bundle_test_contract_manifest()}
    assert "parse_hardware_optimization_makefile_helpers_table()" in helpers
    assert "parse_hardware_optimization_metadata_doc_fields()" in helpers


def test_loop_close_doc_intro_line_table_idempotent_in_render_write_blocks(tmp_path: Path):
    import shutil

    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
        loop_close_doc_intro_line_doc_markdown_table,
    )

    doc_copy = tmp_path / "HARDWARE_OPTIMIZATION.md"
    shutil.copy(_REPO / "docs" / "HARDWARE_OPTIMIZATION.md", doc_copy)
    write_fn = resolve_loop_close_doc_render_write_fn(
        next(spec for spec in loop_close_doc_render_write_specs() if spec["name"] == "ci_artifact")
    )
    write_fn(str(doc_copy))
    text = doc_copy.read_text(encoding="utf-8")
    _assert_doc_marker_block_table(
        text,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
        loop_close_doc_intro_line_doc_markdown_table(),
    )


def test_hardware_optimization_mlir_ci_bundle_contract_includes_makefile_helpers_range():
    helpers = {
        row["helper"] for row in cpu_mlir_ci_bundle_test_contract_manifest()
    }
    assert "loop_close_doc_makefile_helpers_loop_range()" in helpers
    assert "loop_close_doc_intro_line_doc_rows()" in helpers


def test_hardware_optimization_makefile_helpers_table_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    from scripts.cpu_cert_utils import (
        loop_close_doc_makefile_helpers_doc_rows,
        parse_hardware_optimization_makefile_helpers_table,
    )

    parsed = parse_hardware_optimization_makefile_helpers_table(doc_path.read_text(encoding="utf-8"))
    expected = loop_close_doc_makefile_helpers_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want


def test_hardware_optimization_doc_intro_line_table_sync():
    from scripts.cpu_cert_utils import (
        loop_close_doc_intro_line_doc_rows,
        loop_close_doc_makefile_helpers_loop_range,
        parse_hardware_optimization_doc_intro_line_table,
    )

    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    parsed = parse_hardware_optimization_doc_intro_line_table(text)
    expected = loop_close_doc_intro_line_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want
    makefile_rows = [row for row in parsed if row["intro_fn"] == "loop_close_doc_makefile_helpers_doc_intro_line()"]
    assert len(makefile_rows) == 1
    assert makefile_rows[0]["loop_label"] == loop_close_doc_makefile_helpers_loop_range()
    metadata_rows = [
        row for row in parsed if row["intro_fn"] == "loop_close_metadata_table_doc_intro_line()"
    ]
    assert len(metadata_rows) == 1
    assert metadata_rows[0]["schema"] == LOOP_CLOSE_METADATA_DOC_SCHEMA
    assert metadata_rows[0]["marker_section_fn"] == "loop_close_metadata_doc_marker_section()"
    assert all(row.get("schema", "-") == "-" for row in parsed if row not in metadata_rows)
    assert all(
        row.get("marker_section_fn", "-") == "-"
        for row in parsed
        if row not in metadata_rows
    )
    parity_rows = [
        row
        for row in parsed
        if row["intro_fn"]
        in {
            "loop_close_doc_makefile_helpers_doc_intro_line()",
            "cpu_mlir_ci_bundle_contract_doc_intro_line()",
            "loop_close_doc_render_check_write_crossref_doc_intro_line()",
        }
    ]
    assert len(parity_rows) == 3
    assert all(
        row.get("intro_parity_gate")
        == "loop_close_doc_force_fail_three_way_intro_parity_ok()"
        for row in parity_rows
    )
    assert all(
        row.get("intro_parity_gate", "-") == "-"
        for row in parsed
        if row not in parity_rows
    )
    assert loop_close_doc_intro_line_three_way_parity_ok()
    bundle_rows = [
        row
        for row in parsed
        if row["intro_fn"] == "cpu_mlir_ci_bundle_contract_doc_intro_line()"
    ]
    assert len(bundle_rows) == 1
    assert (
        bundle_rows[0].get("manifest_parity_gate")
        == "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()"
    )
    assert all(
        row.get("manifest_parity_gate", "-") == "-"
        for row in parsed
        if row not in bundle_rows
    )
    assert loop_close_doc_intro_line_bundle_manifest_parity_ok()
    assert loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()
    assert loop_close_doc_manifest_parity_three_way_ok()


def test_hardware_optimization_doc_intro_line_table_triple_decoy_parse():
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
        loop_close_doc_intro_line_doc_markdown_table,
        loop_close_doc_intro_line_doc_rows,
        parse_hardware_optimization_doc_intro_line_table,
    )

    parity_gates_decoy = (
        "| Marker begin | Intro function | Loop label | Schema | Marker section | "
        "Intro parity gate | Manifest parity gate |\n"
        "|--------------|----------------|------------|--------|----------------|"
        "-------------------|----------------------|\n"
        "| `DECOY_MARKER` | `decoy_intro()` | `R0` | `-` | `-` | `decoy_intro_parity()` | "
        "`decoy_manifest_parity()` |\n"
    )
    marker_section_decoy = (
        "| Marker begin | Intro function | Loop label | Schema | Marker section |\n"
        "|--------------|----------------|------------|--------|----------------|\n"
        "| `DECOY_MARKER` | `decoy_intro()` | `R0` | `-` | `decoy_marker_section()` |\n"
    )
    schema_decoy = (
        "| Marker begin | Intro function | Loop label | Schema |\n"
        "|--------------|----------------|------------|--------|\n"
        "| `DECOY_MARKER` | `decoy_intro()` | `R0` | `decoy_schema` |\n"
    )
    real_table = loop_close_doc_intro_line_doc_markdown_table()
    sample = (
        f"decoy\n{parity_gates_decoy}\n{marker_section_decoy}\n{schema_decoy}\n"
        f"{LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN}\n{real_table}\n"
        f"{LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END}\n"
    )
    parsed = parse_hardware_optimization_doc_intro_line_table(sample)
    expected = loop_close_doc_intro_line_doc_rows()
    assert parsed == expected
    intro_parity_rows = [
        row
        for row in parsed
        if row["intro_parity_gate"] == "loop_close_doc_force_fail_three_way_intro_parity_ok()"
    ]
    assert len(intro_parity_rows) == 3
    bundle_rows = [
        row
        for row in parsed
        if row["intro_fn"] == "cpu_mlir_ci_bundle_contract_doc_intro_line()"
    ]
    assert len(bundle_rows) == 1
    assert (
        bundle_rows[0]["manifest_parity_gate"]
        == "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()"
    )
    assert all(
        row.get("manifest_parity_gate", "-") == "-"
        for row in parsed
        if row not in bundle_rows
    )
    metadata_rows = [
        row for row in parsed if row["intro_fn"] == "loop_close_metadata_table_doc_intro_line()"
    ]
    assert len(metadata_rows) == 1
    assert metadata_rows[0]["schema"] == LOOP_CLOSE_METADATA_DOC_SCHEMA
    assert metadata_rows[0]["marker_section_fn"] == "loop_close_metadata_doc_marker_section()"


def test_hardware_optimization_doc_intro_line_table_six_column_backward_compat_parse():
    from scripts.cpu_cert_utils import _parse_doc_intro_line_markdown_table

    legacy = (
        "| Marker begin | Intro function | Loop label | Schema | Marker section | Intro parity gate |\n"
        "|--------------|----------------|------------|--------|----------------|-------------------|\n"
        "| `LEGACY_MARKER` | `legacy_intro()` | `R0` | `-` | `-` | "
        "`loop_close_doc_force_fail_three_way_intro_parity_ok()` |\n"
    )
    parsed = _parse_doc_intro_line_markdown_table(legacy)
    assert len(parsed) == 1
    assert parsed[0]["manifest_parity_gate"] == "-"
    assert parsed[0]["intro_parity_gate"] == "loop_close_doc_force_fail_three_way_intro_parity_ok()"


def test_hardware_optimization_doc_render_check_write_crossref_table_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_doc_render_check_write_crossref_table(
        doc_path.read_text(encoding="utf-8")
    )
    expected = loop_close_doc_render_check_write_crossref_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want


def test_loop_close_doc_render_check_write_crossref_doc_markdown_table_matches_rows():
    parsed = parse_hardware_optimization_doc_render_check_write_crossref_table(
        loop_close_doc_render_check_write_crossref_doc_markdown_table()
    )
    assert parsed == loop_close_doc_render_check_write_crossref_rows()


def test_parse_doc_render_check_write_crossref_markdown_table_check_script_suffix_backward_compat():
    legacy = (
        "| Check name | Check script | Write module | Write fn | Replace fns | Blocks |\n"
        "|------------|--------------|--------------|----------|-------------|--------|\n"
        "| `timing` | `scripts/render_loop_close_timing_doc.py` | "
        "`scripts.render_loop_close_timing_doc` | `write_timing_table_to_doc()` | "
        "`replace_timing_table_markers` | `1` |\n"
    )
    parsed = parse_hardware_optimization_doc_render_check_write_crossref_table(legacy)
    expected = loop_close_doc_render_check_write_crossref_rows()
    assert parsed[0]["check_script"] == expected[0]["check_script"]
    assert parsed[0]["check_script"] == loop_close_doc_render_check_script_doc_crossref(
        loop_close_doc_render_check_specs()[0]
    )
    assert normalize_loop_close_doc_render_check_script_doc_label(
        "scripts/render_loop_close_timing_doc.py"
    ) == expected[0]["check_script"]


def test_parse_doc_render_check_write_crossref_markdown_table_five_column_backward_compat():
    from scripts.cpu_cert_utils import _parse_doc_render_check_write_crossref_markdown_table

    legacy = (
        "| Check name | Check script | Write module | Write fn | Replace fns |\n"
        "|------------|--------------|--------------|----------|-------------|\n"
        "| `timing` | `scripts/render_loop_close_timing_doc.py` | "
        "`scripts.render_loop_close_timing_doc` | `write_timing_table_to_doc()` | "
        "`replace_timing_table_markers` |\n"
    )
    parsed = _parse_doc_render_check_write_crossref_markdown_table(legacy)
    assert len(parsed) == 1
    assert parsed[0]["block_count"] == "-"
    assert parsed[0]["check_name"] == "timing"


def test_loop_close_doc_render_check_write_crossref_intro_includes_block_counts_summary():
    from scripts.cpu_cert_utils import loop_close_doc_render_check_write_crossref_doc_intro_line

    intro = loop_close_doc_render_check_write_crossref_doc_intro_line()
    summary = loop_close_doc_render_write_block_counts_summary()
    assert "loop_close_doc_render_write_block_counts_by_write_spec()" in intro
    assert summary in intro
    assert "loop_close_doc_render_check_write_crossref_blocks_summary_parity()" in intro
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()" in intro
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env() in intro
    assert loop_close_doc_render_check_write_crossref_force_fail_intro_fragment() in intro
    assert (
        loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref_assign_fragment()
        in intro
    )
    assert "--doc-path" in intro
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert intro == line_before_marker(text, LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN)


def test_loop_close_doc_render_check_write_crossref_blocks_summary_parity_with_doc_table():
    assert loop_close_doc_render_check_write_crossref_blocks_summary_parity()
    counts = loop_close_doc_render_write_block_counts_by_write_spec()
    summary = loop_close_doc_render_write_block_counts_summary()
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_doc_render_check_write_crossref_table(
        doc_path.read_text(encoding="utf-8")
    )
    for row in parsed:
        assert row["block_count"] == str(counts[row["check_name"]])
        assert f"{row['check_name']}={row['block_count']}" in summary


def test_cpu_mlir_ci_bundle_manifest_row_count_and_blocks_summary_parity():
    assert cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()
    assert loop_close_doc_render_check_write_crossref_blocks_summary_parity()
    manifest_count = cpu_mlir_ci_bundle_test_contract_manifest_row_count()
    assert manifest_count == len(cpu_mlir_ci_bundle_test_contract_manifest())
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_mlir_ci_bundle_contract_table(
        doc_path.read_text(encoding="utf-8")
    )
    assert len(parsed) == manifest_count
    assert cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()


def test_hardware_optimization_doc_render_check_write_crossref_table_marker_block_only():
    decoy = (
        "| Check name | Check script | Write module | Write fn | Replace fns | Blocks |\n"
        "|------------|--------------|--------------|----------|-------------|--------|\n"
        "| `decoy` | `scripts/decoy.py` | `scripts.decoy` | `decoy_write()` | `decoy_replace` | `99` |\n"
    )
    real_table = loop_close_doc_render_check_write_crossref_doc_markdown_table()
    sample = (
        f"decoy\n{decoy}\n"
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN}\n{real_table}\n"
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END}\n"
    )
    parsed = parse_hardware_optimization_doc_render_check_write_crossref_table(sample)
    assert parsed == loop_close_doc_render_check_write_crossref_rows()
    assert loop_close_doc_render_check_write_crossref_blocks_summary_parity()
    from scripts.cpu_cert_utils import loop_close_doc_render_check_write_crossref_doc_intro_line

    intro = loop_close_doc_render_check_write_crossref_doc_intro_line()
    assert "loop_close_doc_render_check_write_crossref_blocks_summary_parity()" in intro


def test_hardware_optimization_doc_render_check_write_crossref_decoy_and_normalize_check_script():
    decoy = (
        "| Check name | Check script | Write module | Write fn | Replace fns | Blocks |\n"
        "|------------|--------------|--------------|----------|-------------|--------|\n"
        "| `decoy` | `scripts/decoy.py --check --doc-path` | `scripts.decoy` | "
        "`decoy_write()` | `decoy_replace` | `99` |\n"
    )
    legacy_rows = []
    for check in loop_close_doc_render_check_specs():
        write = next(
            spec for spec in loop_close_doc_render_write_specs() if spec["name"] == check["name"]
        )
        blocks = [
            block
            for block in loop_close_doc_render_write_block_specs()
            if block["write_spec"] == check["name"]
        ]
        replace_fns = ", ".join(sorted({block["replace_fn"] for block in blocks}))
        legacy_rows.append(
            f"| `{check['name']}` | `{check['script']}` | `{write['module']}` | "
            f"`{write['write_fn']}()` | `{replace_fns}` | `{len(blocks)}` |"
        )
    legacy_table = (
        "| Check name | Check script | Write module | Write fn | Replace fns | Blocks |\n"
        "|------------|--------------|--------------|----------|-------------|--------|\n"
        + "\n".join(legacy_rows)
    )
    real_table = loop_close_doc_render_check_write_crossref_doc_markdown_table()
    decoy_sample = (
        f"decoy\n{decoy}\n"
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN}\n{real_table}\n"
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END}\n"
    )
    parsed_decoy = parse_hardware_optimization_doc_render_check_write_crossref_table(
        decoy_sample
    )
    assert parsed_decoy == loop_close_doc_render_check_write_crossref_rows()
    legacy_sample = (
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN}\n{legacy_table}\n"
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END}\n"
    )
    parsed_legacy = parse_hardware_optimization_doc_render_check_write_crossref_table(
        legacy_sample
    )
    assert parsed_legacy == loop_close_doc_render_check_write_crossref_rows()
    for row in parsed_legacy:
        assert row["check_script"].endswith(" --check --doc-path")


def test_hardware_optimization_doc_render_check_write_crossref_legacy_and_suffix_decoy_dual_path():
    legacy_decoy = (
        "| Check name | Check script | Write module | Write fn | Replace fns | Blocks |\n"
        "|------------|--------------|--------------|----------|-------------|--------|\n"
        "| `decoy_legacy` | `scripts/decoy_legacy.py` | `scripts.decoy_legacy` | "
        "`decoy_legacy_write()` | `decoy_legacy_replace` | `88` |\n"
    )
    suffix_decoy = (
        "| Check name | Check script | Write module | Write fn | Replace fns | Blocks |\n"
        "|------------|--------------|--------------|----------|-------------|--------|\n"
        "| `decoy_suffix` | `scripts/decoy_suffix.py --check --doc-path` | "
        "`scripts.decoy_suffix` | `decoy_suffix_write()` | `decoy_suffix_replace` | `77` |\n"
    )
    mixed_rows = []
    for index, check in enumerate(loop_close_doc_render_check_specs()):
        write = next(
            spec for spec in loop_close_doc_render_write_specs() if spec["name"] == check["name"]
        )
        blocks = [
            block
            for block in loop_close_doc_render_write_block_specs()
            if block["write_spec"] == check["name"]
        ]
        replace_fns = ", ".join(sorted({block["replace_fn"] for block in blocks}))
        if index % 2 == 0:
            check_script = check["script"]
        else:
            check_script = loop_close_doc_render_check_script_doc_crossref(check)
        mixed_rows.append(
            f"| `{check['name']}` | `{check_script}` | `{write['module']}` | "
            f"`{write['write_fn']}()` | `{replace_fns}` | `{len(blocks)}` |"
        )
    mixed_table = (
        "| Check name | Check script | Write module | Write fn | Replace fns | Blocks |\n"
        "|------------|--------------|--------------|----------|-------------|--------|\n"
        + "\n".join(mixed_rows)
    )
    real_table = loop_close_doc_render_check_write_crossref_doc_markdown_table()
    decoy_sample = (
        f"{legacy_decoy}\n{suffix_decoy}\n"
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN}\n{real_table}\n"
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END}\n"
    )
    parsed_decoy = parse_hardware_optimization_doc_render_check_write_crossref_table(
        decoy_sample
    )
    assert parsed_decoy == loop_close_doc_render_check_write_crossref_rows()
    mixed_sample = (
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN}\n{mixed_table}\n"
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END}\n"
    )
    parsed_mixed = parse_hardware_optimization_doc_render_check_write_crossref_table(
        mixed_sample
    )
    assert parsed_mixed == loop_close_doc_render_check_write_crossref_rows()


def test_hardware_optimization_doc_render_check_write_crossref_mixed_parse_and_sync_gate():
    mixed_table = loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table()
    mixed_sample = (
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN}\n{mixed_table}\n"
        f"{LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END}\n"
    )
    parsed_mixed = parse_hardware_optimization_doc_render_check_write_crossref_table(
        mixed_sample
    )
    assert parsed_mixed == loop_close_doc_render_check_write_crossref_rows()
    for row in parsed_mixed:
        assert row["check_script"].endswith(" --check --doc-path")
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check(text)
    assert cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()
    parsed_doc = parse_hardware_optimization_doc_render_check_write_crossref_table(text)
    assert parsed_doc == loop_close_doc_render_check_write_crossref_rows()
    assert loop_close_doc_force_fail_crossref_and_check_row_parity_ok()


def test_loop_close_ci_artifact_doc_bundle_sync_gate_check_on_hardware_optimization_doc():
    from scripts.cpu_cert_utils import cpu_mlir_ci_bundle_contract_doc_intro_line

    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    text = doc_path.read_text(encoding="utf-8")
    assert loop_close_ci_artifact_doc_bundle_sync_gate_check(text)
    assert cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()
    intro = cpu_mlir_ci_bundle_contract_doc_intro_line()
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check()" in intro


def test_loop_close_doc_render_check_write_crossref_block_count_parity_with_render_write_blocks():
    assert loop_close_doc_render_check_write_crossref_block_count_parity()
    counts = loop_close_doc_render_write_block_counts_by_write_spec()
    for row in loop_close_doc_render_check_write_crossref_rows():
        assert int(row["block_count"]) == counts[row["check_name"]]
    render_rows = loop_close_doc_render_write_block_doc_rows()
    for write_spec, expected in counts.items():
        actual = sum(1 for row in render_rows if row["write_spec"] == write_spec)
        assert actual == expected
        crossref_row = next(
            row for row in loop_close_doc_render_check_write_crossref_rows()
            if row["check_name"] == write_spec
        )
        assert crossref_row["block_count"] == str(expected)


def test_hardware_optimization_mlir_ci_bundle_contract_intro_includes_intro_line_row_count():
    from scripts.cpu_cert_utils import cpu_mlir_ci_bundle_contract_doc_intro_line

    intro = cpu_mlir_ci_bundle_contract_doc_intro_line()
    assert "loop_close_doc_intro_line_doc_row_count()" in intro
    assert str(loop_close_doc_intro_line_doc_row_count()) in intro
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert intro == line_before_marker(text, LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN)


def test_loop_close_ci_doc_render_path_triggers_cover_crossref_scripts():
    triggers = set(loop_close_ci_doc_render_path_triggers())
    crossref_scripts = loop_close_ci_doc_render_path_triggers_crossref_scripts()
    assert crossref_scripts
    assert set(crossref_scripts) <= triggers
    check_scripts = {spec["script"] for spec in loop_close_doc_render_check_specs()}
    write_scripts = {
        f"{spec['module'].replace('.', '/')}.py" for spec in loop_close_doc_render_write_specs()
    }
    assert set(crossref_scripts) == check_scripts | write_scripts


def test_loop_close_doc_render_write_block_ci_artifact_replace_fn():
    rows = loop_close_doc_render_write_block_doc_rows()
    ci_rows = [row for row in rows if row["write_spec"] == "ci_artifact"]
    assert ci_rows, "expected ci_artifact render write block rows"
    assert all(
        row["replace_fn"] == "replace_loop_close_doc_marker_block()"
        for row in ci_rows
    )
    timing_rows = [row for row in rows if row["write_spec"] == "timing"]
    assert timing_rows[0]["replace_fn"] == "replace_timing_table_markers()"
    metadata_rows = [row for row in rows if row["write_spec"] == "metadata"]
    assert metadata_rows[0]["replace_fn"] == "replace_metadata_table_markers()"


def test_resolve_loop_close_doc_render_write_block_replace_fn_matches_spec():
    for spec in loop_close_doc_render_write_block_specs():
        replace_fn = resolve_loop_close_doc_render_write_block_replace_fn(spec)
        assert replace_fn.__name__ == spec["replace_fn"]


def test_apply_loop_close_doc_render_write_block_replace_per_block():
    for spec in loop_close_doc_render_write_block_specs():
        table = resolve_loop_close_doc_render_block_table(spec)
        sample = (
            f"intro\n{spec['marker_begin']}\nold content\n{spec['marker_end']}\n"
        )
        updated = apply_loop_close_doc_render_write_block_replace(spec, sample, table)
        assert "old content" not in updated
        assert table.strip() in updated


def test_hardware_optimization_doc_render_write_block_table_marker_block_only():
    decoy = (
        "| Write spec | Marker begin | Table function |\n"
        "|------------|--------------|----------------|\n"
        "| `decoy` | `DECOY_MARKER` | `decoy_table()` |\n"
    )
    real_table = loop_close_doc_render_write_block_doc_markdown_table()
    sample = (
        f"decoy\n{decoy}\n"
        f"{LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN}\n{real_table}\n"
        f"{LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END}\n"
    )
    parsed = parse_hardware_optimization_doc_render_write_block_table(sample)
    assert parsed == loop_close_doc_render_write_block_doc_rows()


def test_parse_doc_render_write_block_markdown_table_three_column_backward_compat():
    legacy = (
        "| Write spec | Marker begin | Table function |\n"
        "|------------|--------------|----------------|\n"
        "| `ci_artifact` | `LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN` | `cpu_ci_artifact_doc_markdown_table()` |\n"
    )
    parsed = parse_doc_render_write_block_markdown_table(legacy)
    assert len(parsed) == 1
    assert parsed[0]["replace_fn"] == "replace_loop_close_doc_marker_block()"


def test_hardware_optimization_mlir_ci_bundle_contract_includes_metadata_marker_section():
    helpers = {row["helper"] for row in cpu_mlir_ci_bundle_test_contract_manifest()}
    assert "loop_close_metadata_doc_marker_section()" in helpers
    assert "parse_hardware_optimization_doc_intro_line_table()" in helpers
    assert "apply_loop_close_doc_render_write_block_replace()" in helpers
    assert "parse_hardware_optimization_doc_render_write_block_table()" in helpers
    assert "cpu_mlir_ci_bundle_test_contract_manifest_row_count()" in helpers
    assert "loop_close_doc_render_check_write_crossref_rows()" in helpers
    assert "loop_close_doc_intro_line_doc_row_count()" in helpers
    assert "parse_hardware_optimization_doc_render_check_write_crossref_table()" in helpers
    assert "loop_close_ci_doc_render_path_triggers_crossref_scripts()" in helpers
    assert "loop_close_doc_render_check_write_crossref_block_count_parity()" in helpers
    assert "cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()" in helpers
    assert "loop_close_doc_render_write_block_counts_by_write_spec()" in helpers
    assert "loop_close_doc_render_check_write_crossref_blocks_summary_parity()" in helpers
    assert "cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()" in helpers
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check()" in helpers
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()" in helpers
    assert "loop_close_doc_render_check_script_doc_crossref()" in helpers
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()" in helpers
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled()" in helpers
    assert (
        "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()"
        in helpers
    )
    assert "loop_close_doc_force_fail_crossref_and_check_row_parity_ok()" in helpers
    assert "loop_close_doc_force_fail_env_stripped_subprocess_env()" in helpers
    assert "loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table()" in helpers
    assert "loop_close_doc_makefile_helpers_manifest_new_helpers_crossref_ok()" in helpers
    assert "loop_close_doc_force_fail_three_way_intro_parity_ok()" in helpers
    assert "loop_close_doc_render_check_subprocess_argv_chain()" in helpers
    assert "loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()" in helpers
    assert "loop_close_doc_intro_line_three_way_parity_ok()" in helpers
    assert "loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()" in helpers
    assert "loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv()" in helpers
    assert "loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text()" in helpers
    assert "loop_close_doc_intro_line_bundle_manifest_parity_ok()" in helpers
    assert "loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok()" in helpers
    assert "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()" in helpers
    assert "loop_close_doc_manifest_parity_three_way_ok()" in helpers
    assert "loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()" in helpers
    assert "loop_close_doc_check_loop_close_docs_make_subprocess_argv()" in helpers
    assert "loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches()" in helpers
    manifest_row = next(
        row for row in cpu_mlir_ci_bundle_test_contract_manifest()
        if row["helper"] == "cpu_mlir_ci_bundle_test_contract_manifest_row_count()"
    )
    assert "cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()" in manifest_row[
        "contract"
    ]
    intro_row = next(
        row for row in cpu_mlir_ci_bundle_test_contract_manifest()
        if row["helper"] == "loop_close_doc_intro_line_doc_row_count()"
    )
    assert "cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()" in intro_row["contract"]
    sync_row = next(
        row for row in cpu_mlir_ci_bundle_test_contract_manifest()
        if row["helper"] == "cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()"
    )
    assert "loop_close_doc_intro_line_doc_row_count()" in sync_row["contract"]
    sync_gate_row = next(
        row for row in cpu_mlir_ci_bundle_test_contract_manifest()
        if row["helper"] == "loop_close_ci_artifact_doc_bundle_sync_gate_check()"
    )
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()" in sync_gate_row[
        "contract"
    ]
    snippet_row = next(
        row for row in cpu_mlir_ci_bundle_test_contract_manifest()
        if row["helper"] == "loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()"
    )
    assert "loop_close_ci_artifact_doc_bundle_sync_gate_check()" in snippet_row["contract"]
    crossref_script_row = next(
        row for row in cpu_mlir_ci_bundle_test_contract_manifest()
        if row["helper"] == "loop_close_doc_render_check_script_doc_crossref()"
    )
    assert "normalize_loop_close_doc_render_check_script_doc_label()" in crossref_script_row[
        "contract"
    ]
    force_env_row = next(
        row for row in cpu_mlir_ci_bundle_test_contract_manifest()
        if row["helper"] == "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()"
    )
    assert (
        "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()"
        in force_env_row["contract"]
    )
    force_enabled_row = next(
        row for row in cpu_mlir_ci_bundle_test_contract_manifest()
        if row["helper"] == "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled()"
    )
    assert "smoke-check-loop-close-docs" in force_enabled_row["contract"]
    doc_crossref_row = next(
        row for row in cpu_mlir_ci_bundle_test_contract_manifest()
        if row["helper"]
        == "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()"
    )
    assert "must be unset" in doc_crossref_row["contract"]


def test_ci_artifact_replace_marker_block_paired_lookup_with_label_decoy():
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END,
        _loop_close_marker_doc_label,
        cpu_ci_path_symmetry_doc_markdown_table,
    )
    from scripts.render_loop_close_ci_artifact_doc import replace_marker_block

    label = _loop_close_marker_doc_label(LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN)
    table = cpu_ci_path_symmetry_doc_markdown_table()
    sample = (
        f"| x | `{label}` | y |\n"
        f"intro\n{LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN}\nold\n"
        f"{LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END}\n"
    )
    updated = replace_marker_block(
        sample,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END,
        table,
    )
    assert "old" not in updated
    assert table in updated


def test_loop_close_doc_intro_lines_sync():
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    for marker_begin, intro_fn in loop_close_doc_intro_line_specs():
        assert line_before_marker(text, marker_begin) == intro_fn()


def _assert_doc_marker_block_table(
    doc_text: str, marker_begin: str, marker_end: str, expected_table: str
) -> None:
    block = extract_loop_close_doc_marker_block(doc_text, marker_begin, marker_end)
    assert block is not None, f"missing marker block {marker_begin}"
    assert block.strip() == expected_table.strip()


def test_loop_close_doc_render_write_specs_blocks_match_apply_replace_dispatch(tmp_path: Path):
    """Each --write spec refreshes blocks via the same replace_fn as apply_* dispatch (R113)."""
    import shutil

    base_doc = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    for write_spec in loop_close_doc_render_write_specs():
        doc_copy = tmp_path / f"HARDWARE_OPTIMIZATION_{write_spec['name']}.md"
        shutil.copy(base_doc, doc_copy)
        write_fn = resolve_loop_close_doc_render_write_fn(write_spec)
        write_fn(str(doc_copy))
        text_after_write = doc_copy.read_text(encoding="utf-8")

        for block_spec in loop_close_doc_render_write_block_specs():
            if block_spec["write_spec"] != write_spec["name"]:
                continue
            expected_table = resolve_loop_close_doc_render_block_table(block_spec)
            _assert_doc_marker_block_table(
                text_after_write,
                block_spec["marker_begin"],
                block_spec["marker_end"],
                expected_table,
            )
            stale = (
                f"intro\n{block_spec['marker_begin']}\nSTALE\n{block_spec['marker_end']}\n"
            )
            refreshed = apply_loop_close_doc_render_write_block_replace(
                block_spec, stale, expected_table
            )
            apply_block = extract_loop_close_doc_marker_block(
                refreshed,
                block_spec["marker_begin"],
                block_spec["marker_end"],
            )
            assert apply_block is not None
            assert apply_block.strip() == expected_table.strip()


def test_loop_close_doc_render_writes_are_idempotent(tmp_path: Path):
    import shutil

    doc_copy = tmp_path / "HARDWARE_OPTIMIZATION.md"
    shutil.copy(_REPO / "docs" / "HARDWARE_OPTIMIZATION.md", doc_copy)
    for spec in loop_close_doc_render_write_specs():
        write_fn = resolve_loop_close_doc_render_write_fn(spec)
        write_fn(str(doc_copy))
        text = doc_copy.read_text(encoding="utf-8")
        for block_spec in loop_close_doc_render_write_block_specs():
            if block_spec["write_spec"] != spec["name"]:
                continue
            expected_table = resolve_loop_close_doc_render_block_table(block_spec)
            _assert_doc_marker_block_table(
                text,
                block_spec["marker_begin"],
                block_spec["marker_end"],
                expected_table,
            )
        write_fn(str(doc_copy))
        assert text == doc_copy.read_text(encoding="utf-8")


def test_simulate_downloaded_loop_close_regression_validate(tmp_path: Path):
    import json

    from tests.integration.test_cpu_loop_close import _sample_mlir_bench_profile_stage

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": _sample_mlir_bench_profile_stage(),
            "mlir_bench_contract": {"ok": True},
        },
        "profile": {
            "stage_elapsed_s": {"demos": 30.0, "mlir_bench_profile": 2.0},
            "total_elapsed_s": 32.0,
        },
    }
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "archive.json"
    archive.write_text(json.dumps(report) + "\n", encoding="utf-8")
    meta = loop_close_archive_metadata(report, archive_path=str(archive), validation_ok=True)
    meta_path = source / "archive.meta.json"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    dest = tmp_path / "downloaded-regression"
    assert (
        simulate_downloaded_loop_close_regression_validate(
            str(archive), str(meta_path), str(dest)
        )
        == []
    )
    assert (dest / "cpu-loop-close.json").is_file()
    assert (dest / "cpu-loop-close.meta.json").is_file()


def test_loop_close_archive_metadata_matches_doc_fields_at_creation():
    meta = loop_close_archive_metadata(_sample_report(), validation_ok=True)
    for field in loop_close_metadata_doc_field_names():
        if field in ("timeout_alert_emitted", "archive_sha256"):
            continue
        assert field in meta, f"metadata missing documented field {field}"
    assert meta["schema"] == LOOP_CLOSE_METADATA_DOC_SCHEMA


def test_simulate_downloaded_loop_close_regression_post_alert(tmp_path: Path):
    import json
    import subprocess

    from tests.integration.test_cpu_loop_close import _sample_mlir_bench_profile_stage

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": _sample_mlir_bench_profile_stage(),
            "mlir_bench_contract": {"ok": True},
        },
        "profile": {
            "stage_elapsed_s": {"demos": 55.0, "mlir_bench_profile": 2.0},
            "total_elapsed_s": 57.0,
        },
    }
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "archive.json"
    archive.write_text(json.dumps(report) + "\n", encoding="utf-8")
    meta = loop_close_archive_metadata(report, archive_path=str(archive), validation_ok=True)
    meta["timeout_alert_emitted"] = True
    meta["timeout_alert_pending"] = False
    meta_path = source / "archive.meta.json"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    dest = tmp_path / "downloaded-regression-post-alert"
    proc = subprocess.run(
        [
            "make",
            "validate-loop-close-metadata-post-alert",
            f"ARCHIVE={archive}",
            f"META={meta_path}",
            f"DEST={dest}",
        ],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_simulate_downloaded_loop_close_regression_pre_alert(tmp_path: Path):
    import json
    import subprocess

    from tests.integration.test_cpu_loop_close import _sample_mlir_bench_profile_stage

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": _sample_mlir_bench_profile_stage(),
            "mlir_bench_contract": {"ok": True},
        },
        "profile": {
            "stage_elapsed_s": {"demos": 30.0, "mlir_bench_profile": 2.0},
            "total_elapsed_s": 32.0,
        },
    }
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "archive.json"
    archive.write_text(json.dumps(report) + "\n", encoding="utf-8")
    meta = loop_close_archive_metadata(report, archive_path=str(archive), validation_ok=True)
    meta_path = source / "archive.meta.json"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    dest = tmp_path / "downloaded-regression-pre-alert"
    proc = subprocess.run(
        [
            "make",
            "validate-loop-close-metadata-pre-alert",
            f"ARCHIVE={archive}",
            f"META={meta_path}",
            f"DEST={dest}",
        ],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_makefile_validate_loop_close_metadata_post_alert_alias(tmp_path: Path):
    import json
    import subprocess

    from tests.integration.test_cpu_loop_close import _sample_mlir_bench_profile_stage

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": _sample_mlir_bench_profile_stage(),
            "mlir_bench_contract": {"ok": True},
        },
        "profile": {
            "stage_elapsed_s": {"demos": 55.0, "mlir_bench_profile": 2.0},
            "total_elapsed_s": 57.0,
        },
    }
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "archive.json"
    archive.write_text(json.dumps(report) + "\n", encoding="utf-8")
    meta = loop_close_archive_metadata(report, archive_path=str(archive), validation_ok=True)
    meta["timeout_alert_emitted"] = True
    meta["timeout_alert_pending"] = False
    meta_path = source / "archive.meta.json"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    dest = tmp_path / "post-alert-alias"
    proc = subprocess.run(
        [
            "make",
            "-n",
            "validate-loop-close-metadata-post-alert",
            f"ARCHIVE={archive}",
            f"META={meta_path}",
            f"DEST={dest}",
            "CHECK_STAGE_TIMEOUTS=1",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "REQUIRE_ALERT_ANNOTATION=1" in proc.stdout
    assert "CHECK_STAGE_TIMEOUTS=1" in proc.stdout


def test_makefile_validate_loop_close_metadata_pre_alert_alias(tmp_path: Path):
    import json
    import subprocess

    report = _sample_report(slow_demos=False)
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "archive.json"
    archive.write_text(json.dumps(report) + "\n", encoding="utf-8")
    meta = loop_close_archive_metadata(report, archive_path=str(archive), validation_ok=True)
    meta_path = source / "archive.meta.json"
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    dest = tmp_path / "pre-alert-alias"
    proc = subprocess.run(
        [
            "make",
            "-n",
            "validate-loop-close-metadata-pre-alert",
            f"ARCHIVE={archive}",
            f"META={meta_path}",
            f"DEST={dest}",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "CHECK_STAGE_TIMEOUTS=1" in proc.stdout
    recursive = [
        line
        for line in proc.stdout.splitlines()
        if line.strip().startswith("make regression-validate-loop-close-archive")
    ]
    assert recursive
    assert "REQUIRE_ALERT_ANNOTATION=1" not in recursive[0]


def test_timeout_alert_pending_doc_semantics_no_warnings():
    meta = loop_close_archive_metadata(_sample_report(slow_demos=False), validation_ok=True)
    assert meta["stage_timeout_warning_count"] == 0
    assert meta["timeout_alert_pending"] is False


def test_timeout_alert_pending_doc_semantics_with_warnings():
    meta = loop_close_archive_metadata(_sample_report(slow_demos=True), validation_ok=True)
    assert meta["stage_timeout_warning_count"] >= 1
    assert meta["timeout_alert_pending"] is True
    assert "timeout_alert_emitted" not in meta


def test_loop_close_workflows_upload_post_alert_regression_artifact():
    post_alert_upload = "artifacts/downloaded-regression-post-alert/"
    for rel in (
        ".github/workflows/cpu-loop-close-nightly.yml",
        ".github/workflows/cpu-loop-close-pr.yml",
    ):
        text = (_REPO / rel).read_text(encoding="utf-8")
        assert post_alert_upload in text, f"{rel} missing post-alert artifact upload path"
        assert "validate-loop-close-metadata-post-alert" in text


def test_loop_close_workflows_upload_pre_alert_regression_artifact():
    pre_alert_upload = "artifacts/downloaded-regression/"
    for rel in (
        ".github/workflows/cpu-loop-close-nightly.yml",
        ".github/workflows/cpu-loop-close-pr.yml",
    ):
        text = (_REPO / rel).read_text(encoding="utf-8")
        assert pre_alert_upload in text, f"{rel} missing pre-alert artifact upload path"
        assert "validate-loop-close-metadata-pre-alert" in text


def test_mlir_ci_workflows_include_manifest_only_smoke():
    smoke = "smoke-build-mlir-ci-bundle-manifest"
    for rel, workflow in (
        (".github/workflows/cpu-mlir-jit-contract.yml", "cpu-mlir-jit-contract"),
        (".github/workflows/cpu-mlir-ci-nightly.yml", "cpu-mlir-ci-nightly"),
    ):
        text = (_REPO / rel).read_text(encoding="utf-8")
        assert smoke in text, f"{rel} missing manifest-only smoke step"
        assert f"WORKFLOW={workflow}" in text


def test_cpu_loop_close_pr_paths_cover_post_alert_contract():
    text = (_REPO / ".github/workflows/cpu-loop-close-pr.yml").read_text(encoding="utf-8")
    for needle in (
        "tests/python/test_loop_close_metadata_doc_contract.py",
        "scripts/regression_validate_loop_close_archive.py",
        ".github/workflows/cpu-loop-close-nightly.yml",
    ):
        assert needle in text


def test_mlir_ci_nightly_paths_cover_manifest_smoke_contract():
    text = (_REPO / ".github/workflows/cpu-mlir-ci-nightly.yml").read_text(encoding="utf-8")
    for needle in (
        "smoke-build-mlir-ci-bundle-manifest",
        "scripts/build_mlir_ci_bundle.py",
        "tests/python/test_loop_close_metadata_doc_contract.py",
        "Makefile",
    ):
        assert needle in text


def test_mlir_jit_contract_paths_cover_nightly_manifest_smoke():
    text = (_REPO / ".github/workflows/cpu-mlir-jit-contract.yml").read_text(encoding="utf-8")
    for needle in (
        ".github/workflows/cpu-mlir-ci-nightly.yml",
        "tests/python/test_loop_close_metadata_doc_contract.py",
        "Makefile",
    ):
        assert needle in text
    for spec in loop_close_doc_render_check_specs():
        assert spec["script"] in text, f"jit contract missing path trigger {spec['script']}"


def test_loop_close_ci_workflows_docs_gate_uses_smoke_make_target():
    smoke = loop_close_docs_smoke_make_target()
    for rel in loop_close_ci_docs_gate_workflows():
        text = (_REPO / rel).read_text(encoding="utf-8")
        assert f"make {smoke}" in text, f"{rel} missing docs smoke make target"
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("run:") and "make" in stripped:
                assert "make check-loop-close-docs" not in stripped, (
                    f"{rel} must not invoke check-loop-close-docs in CI steps: {stripped}"
                )


def test_mlir_jit_contract_docs_gate_smoke_make_target_sync():
    rel = ".github/workflows/cpu-mlir-jit-contract.yml"
    smoke = loop_close_docs_smoke_make_target()
    text = (_REPO / rel).read_text(encoding="utf-8")
    assert "name: Loop-close docs sync gate" in text
    assert f"run: make {smoke}" in text
    jit_docs = [
        entry
        for entry in cpu_ci_workflow_make_target_manifest()
        if entry["workflow"] == rel and "docs sync gate" in entry["step_name"]
    ]
    assert len(jit_docs) == 1
    assert jit_docs[0]["make"] == smoke


def test_loop_close_ci_workflows_docs_gate_step_names_sync():
    for rel, step_name in loop_close_ci_docs_gate_step_names().items():
        text = (_REPO / rel).read_text(encoding="utf-8")
        assert f"name: {step_name}" in text, f"{rel} missing docs gate step name {step_name!r}"


def test_loop_close_ci_workflow_paths_include_doc_render_triggers():
    triggers = set(loop_close_ci_doc_render_path_triggers())
    check_scripts = {spec["script"] for spec in loop_close_doc_render_check_specs()}
    write_scripts = {
        f"{spec['module'].replace('.', '/')}.py" for spec in loop_close_doc_render_write_specs()
    }
    assert check_scripts <= triggers
    assert write_scripts <= triggers
    makefile_text = (_REPO / "Makefile").read_text(encoding="utf-8")
    smoke = loop_close_docs_smoke_make_target()
    assert f"{smoke}:" in makefile_text, f"Makefile missing {smoke} target"
    assert "Makefile" in triggers
    for rel in loop_close_ci_docs_gate_workflows():
        for paths in (
            _workflow_pull_request_paths(rel),
            _workflow_push_paths(rel),
        ):
            missing = triggers - paths
            assert not missing, f"{rel} paths missing doc render triggers {missing}"


def test_cpu_ci_path_symmetry_doc_markdown_table_matches_rows():
    parsed = parse_ci_path_symmetry_markdown_table(cpu_ci_path_symmetry_doc_markdown_table())
    assert parsed == cpu_ci_workflow_path_symmetry_doc_rows()


def test_hardware_optimization_ci_path_symmetry_table_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_ci_path_symmetry_table(
        doc_path.read_text(encoding="utf-8")
    )
    expected = cpu_ci_workflow_path_symmetry_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want


def test_hardware_optimization_workflow_make_intro_mentions_docs_gate_step_names():
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
        cpu_ci_workflow_make_doc_intro_line,
    )

    intro = cpu_ci_workflow_make_doc_intro_line()
    assert "loop_close_ci_docs_gate_step_names()" in intro
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert (
        line_before_marker(text, LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN) == intro
    )


def test_hardware_optimization_mlir_ci_path_symmetry_documented():
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    assert "cpu_ci_workflow_path_symmetry_doc_rows()" in text
    for row in cpu_ci_workflow_path_symmetry_doc_rows():
        assert row["left"] in text, f"missing symmetry left {row['left']}"
        assert row["right"] in text, f"missing symmetry right {row['right']}"
    mlir_rows = cpu_mlir_ci_workflow_path_symmetry_doc_rows()
    assert len(mlir_rows) == 3


def test_mlir_ci_workflows_upload_downloaded_regression_artifact():
    mlir_regression_upload = "artifacts/downloaded-regression-mlir/"
    expected_names = {
        ".github/workflows/cpu-mlir-jit-contract.yml": "mlir-downloaded-regression-${{ github.run_id }}",
        ".github/workflows/cpu-mlir-ci-nightly.yml": "mlir-downloaded-regression-nightly-${{ github.run_id }}",
    }
    for rel, artifact_name in expected_names.items():
        text = (_REPO / rel).read_text(encoding="utf-8")
        assert mlir_regression_upload in text, f"{rel} missing MLIR regression artifact upload"
        assert artifact_name in text, f"{rel} missing independent artifact name {artifact_name}"
        assert "validate-mlir-ci-metadata-download" in text


def _workflow_paths(rel: str, trigger: str) -> set[str]:
    import re

    text = (_REPO / rel).read_text(encoding="utf-8")
    match = re.search(
        rf"{trigger}:\s*\n(?:\s*[^\n]+\n)*?\s*paths:\s*\n((?:\s+- \"[^\"]+\"\n)+)",
        text,
    )
    assert match, f"{rel} missing {trigger} paths block"
    return set(re.findall(r"- \"([^\"]+)\"", match.group(1)))


def _workflow_pull_request_paths(rel: str) -> set[str]:
    return _workflow_paths(rel, "pull_request")


def _workflow_push_paths(rel: str) -> set[str]:
    return _workflow_paths(rel, "push")


def test_loop_close_nightly_push_paths_symmetric_with_pr_pull_request():
    nightly_push = _workflow_push_paths(".github/workflows/cpu-loop-close-nightly.yml")
    pr_pr = _workflow_pull_request_paths(".github/workflows/cpu-loop-close-pr.yml")
    assert nightly_push == pr_pr


def test_loop_close_pr_push_paths_symmetric_with_nightly_pull_request():
    pr_push = _workflow_push_paths(".github/workflows/cpu-loop-close-pr.yml")
    nightly_pr = _workflow_pull_request_paths(".github/workflows/cpu-loop-close-nightly.yml")
    assert pr_push == nightly_pr


def test_loop_close_nightly_paths_symmetric_with_pr():
    nightly_paths = _workflow_pull_request_paths(".github/workflows/cpu-loop-close-nightly.yml")
    pr_paths = _workflow_pull_request_paths(".github/workflows/cpu-loop-close-pr.yml")
    assert nightly_paths == pr_paths


def test_loop_close_nightly_push_paths_symmetric_with_pull_request():
    nightly_push = _workflow_push_paths(".github/workflows/cpu-loop-close-nightly.yml")
    nightly_pr = _workflow_pull_request_paths(".github/workflows/cpu-loop-close-nightly.yml")
    assert nightly_push == nightly_pr


def test_loop_close_pr_and_nightly_push_paths_match():
    nightly_push = _workflow_push_paths(".github/workflows/cpu-loop-close-nightly.yml")
    pr_push = _workflow_push_paths(".github/workflows/cpu-loop-close-pr.yml")
    assert nightly_push == pr_push


def test_makefile_validate_mlir_ci_metadata_download_alias(tmp_path: Path):
    import subprocess

    src = tmp_path / "mlir-ci"
    src.mkdir()
    dest = tmp_path / "downloaded-regression-mlir"
    proc = subprocess.run(
        [
            "make",
            "-n",
            "validate-mlir-ci-metadata-download",
            f"SRC={src}",
            f"DEST={dest}",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    recursive = [
        line
        for line in proc.stdout.splitlines()
        if line.strip().startswith("make regression-validate-mlir-ci-bundle")
    ]
    assert recursive
    assert f'SRC="{src}"' in recursive[0] or f"SRC={src}" in proc.stdout
    assert f'DEST="{dest}"' in recursive[0] or f"DEST={dest}" in proc.stdout


def test_mlir_ci_nightly_push_paths_symmetric_with_pull_request():
    nightly_push = _workflow_push_paths(".github/workflows/cpu-mlir-ci-nightly.yml")
    nightly_pr = _workflow_pull_request_paths(".github/workflows/cpu-mlir-ci-nightly.yml")
    assert nightly_push == nightly_pr


def test_mlir_ci_nightly_push_paths_symmetric_with_jit_contract_pr():
    nightly_push = _workflow_push_paths(".github/workflows/cpu-mlir-ci-nightly.yml")
    jit_pr = _workflow_pull_request_paths(".github/workflows/cpu-mlir-jit-contract.yml")
    assert nightly_push == jit_pr


def test_mlir_ci_nightly_pull_request_paths_symmetric_with_jit_contract_pr():
    nightly_pr = _workflow_pull_request_paths(".github/workflows/cpu-mlir-ci-nightly.yml")
    jit_pr = _workflow_pull_request_paths(".github/workflows/cpu-mlir-jit-contract.yml")
    assert nightly_pr == jit_pr


def test_mlir_jit_contract_push_paths_symmetric_with_mlir_ci_nightly_pull_request():
    jit_push = _workflow_push_paths(".github/workflows/cpu-mlir-jit-contract.yml")
    nightly_pr = _workflow_pull_request_paths(".github/workflows/cpu-mlir-ci-nightly.yml")
    assert jit_push == nightly_pr


def test_mlir_jit_contract_push_paths_symmetric_with_pull_request():
    jit_push = _workflow_push_paths(".github/workflows/cpu-mlir-jit-contract.yml")
    jit_pr = _workflow_pull_request_paths(".github/workflows/cpu-mlir-jit-contract.yml")
    assert jit_push == jit_pr


def test_cpu_ci_artifact_manifest_loop_close_names():
    loop_close = [
        row
        for row in cpu_ci_artifact_manifest()
        if row["workflow"].startswith("cpu-loop-close")
    ]
    assert len(loop_close) == 2
    names = {row["artifact_name"] for row in loop_close}
    assert "cpu-loop-close-quick-pr-${{ github.run_id }}" in names
    assert "cpu-loop-close-json-${{ github.run_id }}" in names


def test_cpu_ci_artifact_doc_table_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_ci_artifact_table(doc_path.read_text(encoding="utf-8"))
    expected = cpu_ci_artifact_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want


def test_cpu_ci_artifact_doc_markdown_table_matches_rows():
    parsed = parse_ci_artifact_markdown_table(cpu_ci_artifact_doc_markdown_table())
    assert parsed == cpu_ci_artifact_doc_rows()


def test_cpu_ci_workflows_artifact_names_match_manifest():
    expected_by_workflow: dict[str, set[str]] = {}
    for row in cpu_ci_artifact_manifest():
        expected_by_workflow.setdefault(row["workflow"], set()).add(row["artifact_name"])
    for workflow, names in expected_by_workflow.items():
        text = (_REPO / ".github/workflows" / workflow).read_text(encoding="utf-8")
        for name in names:
            assert name in text, f"{workflow} missing artifact name {name}"


def test_cpu_ci_workflow_make_targets_present():
    for entry in cpu_ci_workflow_make_target_manifest():
        text = (_REPO / entry["workflow"]).read_text(encoding="utf-8")
        needle = f"make {entry['make']}"
        assert needle in text, f"{entry['workflow']} missing {needle}"
        assert entry["step_name"] in text, f"{entry['workflow']} missing step {entry['step_name']}"


def test_cpu_ci_workflow_make_step_doc_table_sync():
    doc_path = _REPO / "docs" / "HARDWARE_OPTIMIZATION.md"
    parsed = parse_hardware_optimization_ci_workflow_make_table(
        doc_path.read_text(encoding="utf-8")
    )
    expected = cpu_ci_workflow_make_step_doc_rows()
    assert len(parsed) == len(expected)
    for got, want in zip(parsed, expected):
        assert got == want


def test_cpu_ci_workflow_make_step_doc_markdown_table_matches_rows():
    parsed = parse_ci_workflow_make_step_markdown_table(
        cpu_ci_workflow_make_step_doc_markdown_table()
    )
    assert parsed == cpu_ci_workflow_make_step_doc_rows()


def test_hardware_optimization_ci_artifact_markers_present():
    from scripts.cpu_cert_utils import (
        LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_END,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END,
    )

    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    for marker in (
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_END,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END,
    ):
        assert marker in text, f"missing doc marker {marker}"


def test_cpu_loop_close_pr_push_paths_symmetric_with_pull_request():
    pr_paths = _workflow_pull_request_paths(".github/workflows/cpu-loop-close-pr.yml")
    push_paths = _workflow_push_paths(".github/workflows/cpu-loop-close-pr.yml")
    assert push_paths == pr_paths


def test_hardware_optimization_ci_artifact_names_documented():
    text = (_REPO / "docs" / "HARDWARE_OPTIMIZATION.md").read_text(encoding="utf-8")
    for row in cpu_ci_artifact_manifest():
        assert row["artifact_name"] in text
    for upload_path in (
        "downloaded-regression/",
        "downloaded-regression-post-alert/",
        "downloaded-regression-mlir/",
    ):
        assert upload_path in text, f"missing documented upload path {upload_path}"
