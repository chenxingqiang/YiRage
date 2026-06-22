# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for CPU certification and value-verify reporting."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, TypedDict


def apply_plain_matmul_search_tractability() -> None:
    """Cap CPU search for plain matmul cert/bench smoke (mirrors bench plain_matmul)."""
    os.environ["YIRAGE_CPU_MAX_KN_GRAPH_OP"] = "4"
    os.environ["YIRAGE_CPU_MAX_TB_GRAPH_OP"] = "5"
    os.environ["YIRAGE_CPU_BENCH_MINIMAL_EXPLORE"] = "1"


def apply_rms_matmul_search_tractability() -> None:
    """Cap CPU search for fused rms_norm+matmul (mirrors bench ``rms_norm_matmul`` / MLIR e2e)."""
    os.environ["YIRAGE_CPU_MAX_KN_GRAPH_OP"] = "4"
    os.environ["YIRAGE_CPU_MAX_TB_GRAPH_OP"] = "6"
    os.environ["YIRAGE_CPU_BENCH_MINIMAL_EXPLORE"] = "1"


def parse_pytest_summary(stdout: str) -> Dict[str, Optional[int]]:
    """Parse trailing ``N passed[, M skipped]`` line from pytest output."""
    text = stdout or ""
    match = re.search(
        r"(\d+)\s+passed(?:,\s+(\d+)\s+skipped)?(?:,\s+(\d+)\s+failed)?",
        text,
    )
    if not match:
        return {"passed": None, "skipped": None, "failed": None}
    failed = int(match.group(3)) if match.group(3) else 0
    return {
        "passed": int(match.group(1)),
        "skipped": int(match.group(2)) if match.group(2) else 0,
        "failed": failed,
    }


def parse_json_marker(stdout: str, begin: str, end: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object between ``BEGIN`` / ``END`` marker lines."""
    import json

    text = stdout or ""
    start = text.find(begin)
    end_idx = text.find(end)
    if start == -1 or end_idx == -1 or end_idx <= start:
        return None
    payload = text[start + len(begin) : end_idx].strip()
    if not payload:
        return None
    return json.loads(payload)


def parse_loop_close_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse ``YIRAGE_CPU_LOOP_CLOSE_JSON_*`` archive from stdout."""
    return parse_json_marker(
        stdout,
        "YIRAGE_CPU_LOOP_CLOSE_JSON_BEGIN",
        "YIRAGE_CPU_LOOP_CLOSE_JSON_END",
    )


def parse_mlir_bench_profile_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse ``YIRAGE_MLIR_BENCH_PROFILE_JSON_*`` archive from stdout."""
    return parse_json_marker(
        stdout,
        "YIRAGE_MLIR_BENCH_PROFILE_JSON_BEGIN",
        "YIRAGE_MLIR_BENCH_PROFILE_JSON_END",
    )


def validate_loop_close_archive(report: Dict[str, Any]) -> List[str]:
    """Validate loop-close JSON archive structure (Loop R70/R72/R73 regression)."""
    errors: List[str] = []
    if report.get("backend") != "cpu":
        errors.append("backend must be cpu")
    mode = report.get("mode")
    if mode not in ("quick", "full"):
        errors.append("mode must be quick or full")
    stages = report.get("stages")
    if not isinstance(stages, dict):
        errors.append("stages must be a dict")
        return errors
    for name in ("demos", "mlir_bench_profile", "mlir_bench_contract"):
        if name not in stages:
            errors.append(f"missing stage {name}")
    profile = report.get("profile")
    if not isinstance(profile, dict):
        errors.append("profile must be a dict")
    elif "stage_elapsed_s" not in profile:
        errors.append("profile.stage_elapsed_s required")
    if mode == "full" and "cert_e2e" not in stages:
        errors.append("full mode requires cert_e2e stage")
    if report.get("ok") is not True:
        errors.append("report.ok must be true")

    mlir_stage = stages.get("mlir_bench_profile")
    if isinstance(mlir_stage, dict):
        bench_quick = mode == "quick"
        errors.extend(_validate_loop_close_mlir_bench_stage(mlir_stage, bench_quick=bench_quick))

    return errors


def _validate_loop_close_mlir_bench_stage(
    stage: Dict[str, Any],
    *,
    bench_quick: bool,
) -> List[str]:
    """Bench JSON shape gate on archived mlir_bench_profile rows (Loop R72/R73)."""
    from scripts.cpu_bench_shapes import validate_bench_json_row_shapes

    errors: List[str] = []
    rows = stage.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("mlir_bench_profile.rows must be a non-empty list")
        return errors

    shape_errors = validate_bench_json_row_shapes(rows, quick=bench_quick)
    errors.extend(shape_errors)

    mlir_profile = stage.get("profile") or {}
    errors.extend(_validate_shape_validation_errors_field(mlir_profile, prefix="mlir_bench_profile.profile"))

    if mlir_profile.get("shape_contract_ok") is not True:
        errors.append("mlir_bench_profile.profile.shape_contract_ok must be true")
    expected_bench_quick = mlir_profile.get("bench_quick")
    if expected_bench_quick is not None and expected_bench_quick is not bench_quick:
        errors.append(
            f"mlir_bench_profile.profile.bench_quick {expected_bench_quick!r} "
            f"!= archive mode bench_quick {bench_quick!r}"
        )
    if stage.get("ok") is not True:
        errors.append("mlir_bench_profile.ok must be true")
    return errors


def _validate_shape_validation_errors_field(
    profile: Dict[str, Any],
    *,
    prefix: str,
) -> List[str]:
    """Require ``shape_validation_errors`` list consistent with ``shape_contract_ok`` (R73)."""
    errors: List[str] = []
    field_errors = profile.get("shape_validation_errors")
    if not isinstance(field_errors, list):
        errors.append(f"{prefix}.shape_validation_errors must be a list")
        return errors
    if field_errors:
        errors.extend(field_errors)
    contract_ok = profile.get("shape_contract_ok")
    if field_errors and contract_ok is True:
        errors.append(
            f"{prefix}.shape_contract_ok must not be true when shape_validation_errors is non-empty"
        )
    if not field_errors and contract_ok is False:
        errors.append(
            f"{prefix}.shape_contract_ok must be true when shape_validation_errors is empty"
        )
    return errors


def load_loop_close_archive(path: str) -> Dict[str, Any]:
    """Load loop-close JSON from ``--output`` file."""
    import json

    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cert_inventory_summary() -> Dict[str, Any]:
    from tests.integration.cpu_inventory import planned_value_verify_count, registry_sizes

    planned = planned_value_verify_count()
    return {
        "planned_value_verify_count": planned,
        "registry_sizes": registry_sizes(),
    }


def cert_profile_from_stages(
    stages: Dict[str, Any],
    *,
    planned_value_verify: int,
) -> Dict[str, Any]:
    """Build certification profile JSON from per-stage runner results."""
    vv = stages.get("value_verify_all") or {}
    passed = (vv.get("pytest") or {}).get("passed")
    stage_elapsed = {
        name: round(float(stage.get("elapsed_s", 0)), 2)
        for name, stage in stages.items()
    }
    stages_ok = sum(1 for stage in stages.values() if stage.get("ok"))
    profile: Dict[str, Any] = {
        "total_elapsed_s": round(sum(stage_elapsed.values()), 2),
        "stage_elapsed_s": stage_elapsed,
        "stages_run": len(stages),
        "stages_ok": stages_ok,
        "value_verify_passed": passed,
        "value_verify_planned": planned_value_verify,
        "value_verify_aligned": passed == planned_value_verify if passed is not None else None,
    }
    wt = stages.get("walkthrough") or {}
    substages = wt.get("walkthrough_substage_elapsed_s")
    if substages:
        profile["walkthrough_substage_elapsed_s"] = substages
        profile["walkthrough_elapsed_s"] = stage_elapsed.get("walkthrough")
    mlir = stages.get("mlir_bench_profile") or {}
    mlir_profile = mlir.get("profile")
    if mlir_profile:
        profile["mlir_bench_profile"] = mlir_profile
        profile["mlir_bench_profile_ok"] = mlir.get("ok")
        profile["mlir_bench_elapsed_s"] = stage_elapsed.get("mlir_bench_profile")
    return profile


def cpu_demo_loop_manifest() -> list[Dict[str, Any]]:
    """CPU demos exercised in the infinite optimization loop (see AGENTS.md)."""
    return [
        {
            "id": "backend_selection",
            "script": "demo/backend_selection_demo.py",
            "layer": "perceive",
            "framework_tier": "same-backend",
            "pytest": "test_backend_selection_demo_runs_on_cpu",
            "notes": "Confirms auto backend is cpu on Linux VM",
        },
        {
            "id": "demo_jit",
            "script": "demo/demo_jit.py",
            "layer": "verify",
            "framework_tier": "P1",
            "pytest": "test_demo_jit_runs_on_cpu",
            "notes": "CPU --device cpu correctness smoke",
        },
        {
            "id": "demo_rms_norm",
            "script": "demo/demo_rms_norm.py",
            "layer": "verify",
            "framework_tier": "P0/P1",
            "pytest": "test_demo_rms_norm_smoke_on_cpu",
            "notes": "RMSNorm+matmul execute on CPU (no superoptimize on CPU path)",
        },
        {
            "id": "demo_lora",
            "script": "demo/demo_lora.py",
            "layer": "verify",
            "framework_tier": "P1",
            "pytest": "test_demo_lora_smoke_on_cpu",
            "notes": "LoRA blocked GEMM on CPU (no superoptimize; concat_matmul path)",
        },
        {
            "id": "reference_mugraph_rms_norm",
            "script": "demo/reference_mugraphs/rms_norm.py",
            "layer": "verify",
            "framework_tier": "P1",
            "pytest": "test_reference_mugraph_rms_norm_smoke_on_cpu",
            "notes": "Fused customized RMS+matmul reference graph; --quick on CPU",
        },
        {
            "id": "reference_mugraph_lora",
            "script": "demo/reference_mugraphs/lora.py",
            "layer": "verify",
            "framework_tier": "P1",
            "pytest": "test_reference_mugraph_lora_smoke_on_cpu",
            "notes": "LoRA blocked GEMM reference graph; --quick on CPU",
        },
        {
            "id": "reference_mugraph_gated_mlp",
            "script": "demo/reference_mugraphs/gated_mlp.py",
            "layer": "verify",
            "framework_tier": "P1",
            "pytest": "test_reference_mugraph_gated_mlp_smoke_on_cpu",
            "notes": "Gated MLP (SiLU gate) reference graph; --quick on CPU",
        },
        {
            "id": "reference_mugraph_plain_matmul",
            "script": "demo/reference_mugraphs/plain_matmul.py",
            "layer": "verify",
            "framework_tier": "P0",
            "pytest": "test_reference_mugraph_plain_matmul_smoke_on_cpu",
            "notes": "KN plain matmul reference graph; --quick on CPU (cpu_matmul fast path)",
        },
        {
            "id": "reference_mugraph_matmul_chain",
            "script": "demo/reference_mugraphs/matmul_chain.py",
            "layer": "verify",
            "framework_tier": "P0",
            "pytest": "test_reference_mugraph_matmul_chain_smoke_on_cpu",
            "notes": "Two-matmul chain reference graph; --quick on CPU (cpu_matmul_chain fast path)",
        },
        {
            "id": "reference_mugraph_concat_matmul",
            "script": "demo/reference_mugraphs/concat_matmul.py",
            "layer": "verify",
            "framework_tier": "P0",
            "pytest": "test_reference_mugraph_concat_matmul_smoke_on_cpu",
            "notes": "Dual-concat matmul reference (bench concat_matmul); --quick on CPU",
        },
        {
            "id": "submission_validate",
            "script": "examples/submission.py",
            "layer": "verify",
            "framework_tier": "contract",
            "pytest": "test_submission_validate_runs_on_cpu",
            "notes": "Examples submission --validate",
        },
        {
            "id": "llama3b_moe_pytorch",
            "script": "demo/llama3b_moe/demo.py",
            "layer": "evolve",
            "framework_tier": "P1",
            "pytest": "test_llama3b_moe_demo_runs_on_cpu",
            "notes": "--pytorch-only small shapes for CI tractability",
        },
        {
            "id": "llama3b_moe_benchmark",
            "script": "benchmark/end-to-end/llama3b_moe_cpu.py",
            "layer": "evolve",
            "framework_tier": "P1",
            "pytest": "test_llama3b_moe_benchmark_runs_on_cpu",
            "notes": "--skip-search e2e benchmark smoke",
        },
    ]


def cpu_bench_workload_reference_map() -> Dict[str, str]:
    """Map bench_fused_vs_mkl_baseline workloads to reference_mugraphs manifest ids."""
    return {
        "plain_matmul": "reference_mugraph_plain_matmul",
        "rms_norm_matmul": "reference_mugraph_rms_norm",
        "matmul_chain": "reference_mugraph_matmul_chain",
        "concat_matmul": "reference_mugraph_concat_matmul",
    }


def cpu_bench_reference_shape_contract() -> Dict[str, Dict[str, Any]]:
    """Bench ``--quick`` vs reference ``--quick`` dimension fields (Loop R67/R68)."""
    from scripts.cpu_bench_shapes import shape_contract

    return shape_contract()


def cpu_loop_close_manifest() -> list[Dict[str, Any]]:
    """Stages aggregated by ``make test-cpu-loop-close`` (Loop R66)."""
    return [
        {
            "id": "demos",
            "make": "test-cpu-demos",
            "layer": "verify",
            "notes": "All cpu_demo_loop_manifest demos + manifest contract",
        },
        {
            "id": "mlir_bench_contract",
            "make": "test-cpu-mlir-bench-contract",
            "layer": "verify",
            "notes": "MLIR JIT + concat deferred JSON unit tests + run_mlir_bench_profile smoke",
        },
        {
            "id": "cert_e2e_profile",
            "make": "test-cpu-cert-e2e-profile",
            "layer": "evolve",
            "notes": "Full cert JSON: value verify + walkthrough quick + mlir_bench_profile",
        },
        {
            "id": "loop_close_archive",
            "make": "test-cpu-loop-close-archive",
            "layer": "evolve",
            "notes": "Full loop-close JSON archive (cpu_loop_close.py --json)",
        },
    ]


def cpu_ci_artifact_manifest() -> List[Dict[str, Any]]:
    """CI workflow artifact names and upload path summaries (Loop R93 single source)."""
    return [
        {
            "workflow": "cpu-loop-close-pr.yml",
            "artifact_name": "cpu-loop-close-quick-pr-${{ github.run_id }}",
            "upload_paths_doc": (
                "quick JSON/meta, timeout alert, "
                "`downloaded-regression/`, `downloaded-regression-post-alert/`"
            ),
        },
        {
            "workflow": "cpu-loop-close-nightly.yml",
            "artifact_name": "cpu-loop-close-json-${{ github.run_id }}",
            "upload_paths_doc": (
                "full JSON/meta, timeout alert, "
                "`downloaded-regression/`, `downloaded-regression-post-alert/`"
            ),
        },
        {
            "workflow": "cpu-mlir-jit-contract.yml",
            "artifact_name": "mlir-ci-bundle-${{ github.run_id }}",
            "upload_paths_doc": "full MLIR CI bundle directory",
        },
        {
            "workflow": "cpu-mlir-jit-contract.yml",
            "artifact_name": "mlir-downloaded-regression-${{ github.run_id }}",
            "upload_paths_doc": "`downloaded-regression-mlir/`",
        },
        {
            "workflow": "cpu-mlir-ci-nightly.yml",
            "artifact_name": "mlir-ci-nightly-${{ github.run_id }}",
            "upload_paths_doc": "full MLIR CI bundle directory",
        },
        {
            "workflow": "cpu-mlir-ci-nightly.yml",
            "artifact_name": "mlir-downloaded-regression-nightly-${{ github.run_id }}",
            "upload_paths_doc": "`downloaded-regression-mlir/`",
        },
    ]


def cpu_ci_artifact_doc_rows() -> List[Dict[str, str]]:
    return [
        {
            "workflow": row["workflow"],
            "artifact_name": row["artifact_name"],
            "upload_paths": row["upload_paths_doc"],
        }
        for row in cpu_ci_artifact_manifest()
    ]


def cpu_ci_artifact_doc_markdown_table() -> str:
    """Markdown table for CI workflow artifacts (R93 single source)."""
    header = "| Workflow | Artifact name | Upload paths (under `artifacts/`) |"
    sep = "|----------|---------------|-----------------------------------|"
    body = [
        f"| `{row['workflow']}` | `{row['artifact_name']}` | {row['upload_paths']} |"
        for row in cpu_ci_artifact_doc_rows()
    ]
    return "\n".join([header, sep, *body])


def parse_ci_artifact_markdown_table(text: str) -> List[Dict[str, str]]:
    """Parse CI artifact markdown table rows."""
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Workflow | Artifact name" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 3:
            continue
        workflow, artifact_name, upload_paths = parts
        rows.append(
            {
                "workflow": workflow.strip("`"),
                "artifact_name": artifact_name.strip("`"),
                "upload_paths": upload_paths,
            }
        )
    return rows


def parse_hardware_optimization_ci_artifact_table(text: str) -> List[Dict[str, str]]:
    """Parse CI artifact markdown table from ``HARDWARE_OPTIMIZATION.md``."""
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_END,
    )
    if block is not None:
        return parse_ci_artifact_markdown_table(block)
    marker = "CI workflow artifact names"
    start = text.find(marker)
    if start == -1:
        return parse_ci_artifact_markdown_table(text)
    end = text.find("**Loop-close archive", start)
    section = text[start:end] if end != -1 else text[start:]
    return parse_ci_artifact_markdown_table(section)


LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN = "<!-- LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN -->"
LOOP_CLOSE_CI_ARTIFACT_TABLE_END = "<!-- LOOP_CLOSE_CI_ARTIFACT_TABLE_END -->"
LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN = "<!-- LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN -->"
LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END = "<!-- LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END -->"
LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN = "<!-- LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN -->"
LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END = "<!-- LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END -->"
LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN = (
    "<!-- LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN -->"
)
LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END = "<!-- LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END -->"
LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN = (
    "<!-- LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN -->"
)
LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END = "<!-- LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END -->"
LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN = "<!-- LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN -->"
LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END = "<!-- LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END -->"
LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN = "<!-- LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN -->"
LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END = "<!-- LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END -->"
LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN = (
    "<!-- LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN -->"
)
LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END = (
    "<!-- LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END -->"
)


def cpu_ci_artifact_doc_intro_line() -> str:
    """Intro line immediately above CI artifact marker block (R105 single source)."""
    rev = loop_close_doc_bundle_loop_revision()
    return (
        f"CI workflow artifact names (Loop {rev}; single source: "
        "``cpu_ci_artifact_manifest()``; regenerate with "
        "``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):"
    )


def cpu_ci_workflow_make_doc_intro_line() -> str:
    """Intro line immediately above workflow/make marker block (R105 single source)."""
    rev = loop_close_doc_bundle_loop_revision()
    return (
        f"CI workflow step ↔ Makefile target mapping (Loop {rev}; single source: "
        "``cpu_ci_workflow_make_target_manifest()``; docs sync gate step names from "
        "``loop_close_ci_docs_gate_step_names()``; regenerate with "
        "``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):"
    )


def cpu_ci_path_symmetry_doc_intro_line() -> str:
    """Intro line immediately above path symmetry marker block (R104 single source)."""
    rev = loop_close_doc_bundle_loop_revision()
    return (
        f"CI workflow path-filter symmetry (Loop {rev}; single source: "
        "``cpu_ci_workflow_path_symmetry_doc_rows()``; regenerate with "
        "``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):"
    )


class LoopCloseDocRenderWriteSpec(TypedDict):
    """Contract spec for loop-close doc render ``--write`` idempotence tests (R97)."""

    name: str
    module: str
    write_fn: str


def loop_close_doc_render_write_specs() -> List[LoopCloseDocRenderWriteSpec]:
    """Return render script write specs for intro+table idempotent contract tests."""
    return [
        {
            "name": "timing",
            "module": "scripts.render_loop_close_timing_doc",
            "write_fn": "write_timing_table_to_doc",
        },
        {
            "name": "metadata",
            "module": "scripts.render_loop_close_metadata_doc",
            "write_fn": "write_metadata_table_to_doc",
        },
        {
            "name": "ci_artifact",
            "module": "scripts.render_loop_close_ci_artifact_doc",
            "write_fn": "write_ci_artifact_tables_to_doc",
        },
    ]


class LoopCloseDocRenderWriteBlockSpec(TypedDict):
    """Marker block refreshed by a render write spec (Loop R104 sub-block contract)."""

    write_spec: str
    marker_begin: str
    marker_end: str
    table_fn: str
    replace_fn: str


def loop_close_doc_render_write_block_specs() -> List[LoopCloseDocRenderWriteBlockSpec]:
    """Return marker blocks asserted after each ``loop_close_doc_render_write_specs`` write."""
    return [
        {
            "write_spec": "timing",
            "marker_begin": LOOP_CLOSE_TIMING_TABLE_BEGIN,
            "marker_end": LOOP_CLOSE_TIMING_TABLE_END,
            "table_fn": "loop_close_timing_markdown_table",
            "replace_fn": "replace_timing_table_markers",
        },
        {
            "write_spec": "metadata",
            "marker_begin": LOOP_CLOSE_METADATA_FIELDS_BEGIN,
            "marker_end": LOOP_CLOSE_METADATA_FIELDS_END,
            "table_fn": "loop_close_metadata_doc_markdown_table",
            "replace_fn": "replace_metadata_table_markers",
        },
        {
            "write_spec": "ci_artifact",
            "marker_begin": LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN,
            "marker_end": LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END,
            "table_fn": "loop_close_doc_makefile_helpers_doc_markdown_table",
            "replace_fn": "replace_loop_close_doc_marker_block",
        },
        {
            "write_spec": "ci_artifact",
            "marker_begin": LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN,
            "marker_end": LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END,
            "table_fn": "cpu_mlir_ci_bundle_contract_doc_markdown_table",
            "replace_fn": "replace_loop_close_doc_marker_block",
        },
        {
            "write_spec": "ci_artifact",
            "marker_begin": LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN,
            "marker_end": LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END,
            "table_fn": "loop_close_doc_render_write_block_doc_markdown_table",
            "replace_fn": "replace_loop_close_doc_marker_block",
        },
        {
            "write_spec": "ci_artifact",
            "marker_begin": LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
            "marker_end": LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END,
            "table_fn": "loop_close_doc_render_check_write_crossref_doc_markdown_table",
            "replace_fn": "replace_loop_close_doc_marker_block",
        },
        {
            "write_spec": "ci_artifact",
            "marker_begin": LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
            "marker_end": LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
            "table_fn": "loop_close_doc_intro_line_doc_markdown_table",
            "replace_fn": "replace_loop_close_doc_marker_block",
        },
        {
            "write_spec": "ci_artifact",
            "marker_begin": LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
            "marker_end": LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END,
            "table_fn": "cpu_ci_path_symmetry_doc_markdown_table",
            "replace_fn": "replace_loop_close_doc_marker_block",
        },
        {
            "write_spec": "ci_artifact",
            "marker_begin": LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN,
            "marker_end": LOOP_CLOSE_CI_ARTIFACT_TABLE_END,
            "table_fn": "cpu_ci_artifact_doc_markdown_table",
            "replace_fn": "replace_loop_close_doc_marker_block",
        },
        {
            "write_spec": "ci_artifact",
            "marker_begin": LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
            "marker_end": LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END,
            "table_fn": "cpu_ci_workflow_make_step_doc_markdown_table",
            "replace_fn": "replace_loop_close_doc_marker_block",
        },
    ]


def resolve_loop_close_doc_render_block_table(spec: LoopCloseDocRenderWriteBlockSpec) -> str:
    """Return generated markdown table for a render write block spec."""
    import scripts.cpu_cert_utils as cert_utils

    return getattr(cert_utils, spec["table_fn"])()


def resolve_loop_close_doc_render_write_block_replace_fn(
    spec: LoopCloseDocRenderWriteBlockSpec,
) -> Callable[..., str]:
    """Import and return the ``replace_fn`` callable declared on ``spec`` (R112 dispatch)."""
    import importlib

    replace_fn_name = spec["replace_fn"]
    if replace_fn_name == "replace_loop_close_doc_marker_block":
        return replace_loop_close_doc_marker_block
    write_spec = spec["write_spec"]
    render_spec = next(
        entry for entry in loop_close_doc_render_write_specs() if entry["name"] == write_spec
    )
    mod = importlib.import_module(render_spec["module"])
    return getattr(mod, replace_fn_name)


def apply_loop_close_doc_render_write_block_replace(
    spec: LoopCloseDocRenderWriteBlockSpec, text: str, table: str
) -> str:
    """Apply ``spec``'s ``replace_fn`` to refresh one marker block (R112 single dispatch path)."""
    replace_fn = resolve_loop_close_doc_render_write_block_replace_fn(spec)
    if spec["replace_fn"] == "replace_loop_close_doc_marker_block":
        return replace_fn(text, spec["marker_begin"], spec["marker_end"], table)
    return replace_fn(text, table)


def loop_close_doc_render_write_block_counts_by_write_spec() -> Dict[str, int]:
    """Block counts per write spec from ``loop_close_doc_render_write_block_specs()`` (R116)."""
    counts: Dict[str, int] = {}
    for spec in loop_close_doc_render_write_block_specs():
        counts[spec["write_spec"]] = counts.get(spec["write_spec"], 0) + 1
    return counts


def loop_close_doc_render_check_write_crossref_block_count_parity() -> bool:
    """True when cross-ref ``block_count`` matches render write block spec counts (R116)."""
    counts = loop_close_doc_render_write_block_counts_by_write_spec()
    for row in loop_close_doc_render_check_write_crossref_rows():
        if int(row["block_count"]) != counts[row["check_name"]]:
            return False
    return True


def loop_close_doc_render_write_block_counts_summary() -> str:
    """Human-readable block counts per write spec for cross-ref intro (R117)."""
    counts = loop_close_doc_render_write_block_counts_by_write_spec()
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def loop_close_doc_render_check_write_crossref_blocks_summary_parity() -> bool:
    """True when cross-ref ``Blocks`` column matches ``block_counts_summary`` tokens (R118)."""
    counts = loop_close_doc_render_write_block_counts_by_write_spec()
    summary = loop_close_doc_render_write_block_counts_summary()
    expected_summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    if summary != expected_summary:
        return False
    for row in loop_close_doc_render_check_write_crossref_rows():
        token = f"{row['check_name']}={row['block_count']}"
        if token not in summary:
            return False
        if row["block_count"] != str(counts[row["check_name"]]):
            return False
    return True


def loop_close_doc_bundle_loop_revision() -> str:
    """Loop revision string for loop-close doc intro lines and bundle tables (R136)."""
    return "R136"


def loop_close_doc_makefile_helpers_loop_range() -> str:
    """Loop range label for Makefile helpers section in HARDWARE_OPTIMIZATION.md."""
    return f"R84–{loop_close_doc_bundle_loop_revision()}"


def loop_close_doc_makefile_helpers_doc_intro_line() -> str:
    """Intro line immediately above Makefile helpers marker block (R107 single source)."""
    loop_range = loop_close_doc_makefile_helpers_loop_range()
    smoke = loop_close_docs_smoke_make_target()
    return (
        f"Loop-close and MLIR CI Makefile helpers (Loop {loop_range}; "
        f"``make {smoke}`` render check "
        "``loop_close_ci_artifact_doc_bundle_sync_gate_check()``; "
        "render ``--check`` scripts support ``--doc-path``; "
        f"test hook {loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()} "
        "must be unset on smoke):"
    )


def loop_close_doc_makefile_helpers_doc_rows() -> List[Dict[str, str]]:
    """Makefile helper targets documented under ``LOOP_CLOSE_MAKEFILE_HELPERS_TABLE`` (R108)."""
    smoke = loop_close_docs_smoke_make_target()
    return [
        {
            "target": "make check-loop-close-docs",
            "purpose": (
                "Verify timing + metadata + CI artifact/workflow-make tables and intro lines "
                f"(``loop_close_doc_intro_line_doc_row_count()`` = "
                f"{loop_close_doc_intro_line_doc_row_count()} rows; sync gate "
                "``cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()``; manifest/blocks "
                "``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()``; "
                f"{loop_close_doc_makefile_helpers_check_row_force_fail_purpose_fragment()})"
            ),
        },
        {
            "target": f"make {smoke}",
            "purpose": (
                "Single smoke entry for all three render ``--check`` scripts "
                "(single source: ``loop_close_docs_smoke_make_target()``; bundle sync gate "
                "``loop_close_ci_artifact_doc_bundle_sync_gate_check()`` via ci_artifact "
                "render ``--check`` (stderr "
                "``loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()``; "
                f"test hook {loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()} "
                "must be unset); cross-ref "
                "``loop_close_doc_render_check_write_crossref_rows()`` → "
                "``loop_close_doc_render_write_specs()`` + "
                "``apply_loop_close_doc_render_write_block_replace``)"
            ),
        },
        {
            "target": "make render-loop-close-docs",
            "purpose": "Regenerate all loop-close doc tables + intro lines from single source",
        },
        {
            "target": "make render-loop-close-timing-doc",
            "purpose": (
                "Regenerate timing table + ``loop_close_timing_table_doc_intro_line()`` "
                "(``render_loop_close_timing_doc.py`` ``--check``/``--write`` + ``--doc-path``)"
            ),
        },
        {
            "target": "make render-loop-close-metadata-doc",
            "purpose": (
                "Regenerate metadata field table + ``loop_close_metadata_table_doc_intro_line()`` "
                "(``render_loop_close_metadata_doc.py`` ``--check``/``--write`` + ``--doc-path``)"
            ),
        },
        {
            "target": "make render-loop-close-ci-artifact-doc",
            "purpose": (
                "Regenerate CI artifact/workflow-make tables + ``cpu_ci_*_doc_intro_line()`` "
                "(``render_loop_close_ci_artifact_doc.py`` ``--check``/``--write`` + ``--doc-path``)"
            ),
        },
        {
            "target": "make build-mlir-ci-bundle BUNDLE=... WORKFLOW=...",
            "purpose": "Build MLIR CI bundle (`MANIFEST_ONLY=1` writes manifest only)",
        },
        {
            "target": "make smoke-build-mlir-ci-bundle-manifest ...",
            "purpose": "Alias for `build-mlir-ci-bundle ... MANIFEST_ONLY=1`",
        },
        {
            "target": "make regression-validate-loop-close-archive ARCHIVE=... META=... DEST=...",
            "purpose": (
                "Simulate download validate (`CHECK_STAGE_TIMEOUTS=1`, "
                "optional `REQUIRE_ALERT_ANNOTATION=1`)"
            ),
        },
        {
            "target": "make validate-loop-close-metadata-pre-alert ARCHIVE=... META=... DEST=...",
            "purpose": "Pre-alert download validate (`CHECK_STAGE_TIMEOUTS=1`)",
        },
        {
            "target": "make validate-loop-close-metadata-post-alert ARCHIVE=... META=... DEST=...",
            "purpose": (
                "Post-alert download validate (`REQUIRE_ALERT_ANNOTATION=1`; "
                "optional `CHECK_STAGE_TIMEOUTS=1`)"
            ),
        },
        {
            "target": "make regression-validate-mlir-ci-bundle SRC=... DEST=...",
            "purpose": "Simulate MLIR bundle download validate",
        },
        {
            "target": "make validate-mlir-ci-metadata-download SRC=... DEST=...",
            "purpose": "Alias for MLIR bundle download validate",
        },
        {
            "target": "make test-cpu-mlir-ci-bundle",
            "purpose": (
                "MLIR bundle + timing + metadata docs + **workflow artifact/smoke contract** tests "
                f"(``cpu_mlir_ci_bundle_test_contract_manifest_row_count()`` = "
                f"{cpu_mlir_ci_bundle_test_contract_manifest_row_count()} rows; "
                "manifest/blocks "
                f"``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()``; "
                f"sync gate ``cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()`` incl. intro "
                f"``loop_close_doc_intro_line_doc_row_count()`` = "
                f"{loop_close_doc_intro_line_doc_row_count()} rows; "
                f"{loop_close_doc_makefile_helpers_manifest_helpers_parity_purpose_fragment()}; "
                "force-fail parity "
                "``loop_close_doc_force_fail_crossref_and_check_row_parity_ok()``; subprocess "
                "``loop_close_doc_force_fail_env_stripped_subprocess_env()``; mixed parse "
                "``loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table()``; "
                "three-way intro ``loop_close_doc_intro_line_three_way_parity_ok()``; check/smoke "
                "``loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()`` / "
                "``loop_close_doc_check_loop_close_docs_make_subprocess_argv()`` / "
                "``loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches()``; "
                "manifest parity three-way "
                "``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()`` / "
                "``loop_close_doc_manifest_parity_three_way_ok()``)"
            ),
        },
    ]


def loop_close_doc_makefile_helpers_doc_markdown_table() -> str:
    header = "| Target | Purpose |"
    sep = "|--------|---------|"
    body = [
        f"| `{row['target']}` | {row['purpose']} |"
        for row in loop_close_doc_makefile_helpers_doc_rows()
    ]
    return "\n".join([header, sep, *body])


class LoopCloseDocRenderCheckSpec(TypedDict):
    """Contract spec for loop-close doc render ``--check`` smoke (R98)."""

    name: str
    script: str


def loop_close_doc_render_check_specs() -> List[LoopCloseDocRenderCheckSpec]:
    """Return render script ``--check`` specs merged by ``make smoke-check-loop-close-docs``."""
    return [
        {"name": "timing", "script": "scripts/render_loop_close_timing_doc.py"},
        {"name": "metadata", "script": "scripts/render_loop_close_metadata_doc.py"},
        {"name": "ci_artifact", "script": "scripts/render_loop_close_ci_artifact_doc.py"},
    ]


def loop_close_doc_render_check_script_doc_crossref(
    check: LoopCloseDocRenderCheckSpec,
) -> str:
    """Doc table Check script column label including ``--check`` + ``--doc-path`` (R124)."""
    return f"{check['script']} --check --doc-path"


def normalize_loop_close_doc_render_check_script_doc_label(label: str) -> str:
    """Normalize Check script column (legacy path-only or ``--check --doc-path`` suffix) (R125)."""
    stripped = label.strip("`")
    base = stripped
    suffix = " --check --doc-path"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    for check in loop_close_doc_render_check_specs():
        if check["script"] == base:
            return loop_close_doc_render_check_script_doc_crossref(check)
    return stripped


def loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env() -> str:
    """Env var name to force sync gate failure stderr in render ``--check`` (R125 test hook)."""
    return "YIRAGE_LOOP_CLOSE_DOC_FORCE_SYNC_GATE_FAIL"


def loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref() -> str:
    """Doc cross-ref for force-fail env contract tests (R126)."""
    return f"``{loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()}``=1"


def loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled() -> bool:
    """Return True when render ``--check`` should emit sync gate failure stderr (R125)."""
    import os

    return os.environ.get(loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()) == "1"


def loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref_assign_fragment() -> str:
    """Assign fragment for force-fail doc cross-ref in crossref intro (R129)."""
    return f"= {loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()}"


def loop_close_doc_makefile_helpers_check_row_force_fail_purpose_fragment() -> str:
    """Force-fail purpose fragment for ``make check-loop-close-docs`` helpers row (R129)."""
    doc_crossref = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()
    return (
        f"ci_artifact test hook {doc_crossref} must be unset on smoke; "
        "``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()`` "
        "for sync gate stderr contract"
    )


def loop_close_doc_render_check_write_crossref_force_fail_intro_fragment() -> str:
    """Force-fail intro fragment for check/write cross-ref doc intro line (R129)."""
    return (
        "doc cross-ref "
        "``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()`` "
        f"{loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref_assign_fragment()}"
    )


def loop_close_doc_force_fail_crossref_and_check_row_parity_ok() -> bool:
    """True when crossref intro and check helpers row share force-fail doc cross-ref (R129)."""
    doc_crossref = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()
    check_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make check-loop-close-docs"
    )
    crossref_intro = loop_close_doc_render_check_write_crossref_doc_intro_line()
    check_fragment = loop_close_doc_makefile_helpers_check_row_force_fail_purpose_fragment()
    crossref_fragment = loop_close_doc_render_check_write_crossref_force_fail_intro_fragment()
    if check_fragment not in check_row["purpose"]:
        return False
    if crossref_fragment not in crossref_intro:
        return False
    return doc_crossref in check_row["purpose"] and doc_crossref in crossref_intro


def loop_close_doc_makefile_helpers_manifest_new_helpers_crossref() -> List[str]:
    """R129 manifest helpers cross-ref'd from ``make test-cpu-mlir-ci-bundle`` helpers row (R130)."""
    return [
        "loop_close_doc_force_fail_crossref_and_check_row_parity_ok()",
        "loop_close_doc_force_fail_env_stripped_subprocess_env()",
        "loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table()",
    ]


def loop_close_doc_makefile_helpers_manifest_helpers_crossref() -> List[str]:
    """Manifest helpers cross-ref'd from Makefile helpers test row (R129–R132)."""
    return [
        *loop_close_doc_makefile_helpers_manifest_new_helpers_crossref(),
        "loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()",
        "loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()",
        "loop_close_doc_intro_line_three_way_parity_ok()",
        "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()",
        "loop_close_doc_manifest_parity_three_way_ok()",
        "loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()",
        "loop_close_doc_check_loop_close_docs_make_subprocess_argv()",
        "loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches()",
    ]


def loop_close_doc_makefile_helpers_manifest_new_helpers_crossref_ok() -> bool:
    """True when Makefile helpers test row cross-ref's R129 manifest helpers (R130)."""
    test_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make test-cpu-mlir-ci-bundle"
    )
    return all(helper in test_row["purpose"] for helper in loop_close_doc_makefile_helpers_manifest_new_helpers_crossref())


def loop_close_doc_makefile_helpers_manifest_helpers_parity_ok() -> bool:
    """True when Makefile helpers test row cross-ref's all manifest doc-contract helpers (R131)."""
    test_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make test-cpu-mlir-ci-bundle"
    )
    return all(
        helper in test_row["purpose"]
        for helper in loop_close_doc_makefile_helpers_manifest_helpers_crossref()
    )


def loop_close_doc_makefile_helpers_manifest_helpers_parity_purpose_fragment() -> str:
    """Purpose fragment for manifest/helpers parity on Makefile helpers test row (R132)."""
    return (
        "manifest/helpers parity "
        "``loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()`` / "
        "``loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()``; bundle intro cross-ref "
        f"{loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()}"
    )


def loop_close_doc_bundle_intro_manifest_helpers_parity_fragment() -> str:
    """Bundle intro fragment cross-ref'ing manifest/helpers parity gate (R132)."""
    return (
        "manifest/helpers parity "
        "``loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()`` / "
        "``loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()``"
    )


def loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok() -> bool:
    """True when helpers test row and bundle intro share manifest/helpers parity cross-ref (R132)."""
    if not loop_close_doc_makefile_helpers_manifest_helpers_parity_ok():
        return False
    bundle_intro = cpu_mlir_ci_bundle_contract_doc_intro_line()
    fragment = loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()
    if fragment not in bundle_intro:
        return False
    return "loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()" in bundle_intro


def loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok(text: str) -> bool:
    """True when doc Makefile helpers test row contains manifest parity purpose fragment (R133)."""
    parsed = parse_hardware_optimization_makefile_helpers_table(text)
    test_row = next(
        row for row in parsed if row["target"] == "make test-cpu-mlir-ci-bundle"
    )
    expected = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make test-cpu-mlir-ci-bundle"
    )
    fragment = loop_close_doc_makefile_helpers_manifest_helpers_parity_purpose_fragment()
    if test_row["purpose"] != expected["purpose"]:
        return False
    return fragment in test_row["purpose"]


def loop_close_doc_force_fail_three_way_intro_parity_ok() -> bool:
    """True when helpers/bundle/crossref intros share render check + force-fail parity (R130)."""
    hook = "loop_close_ci_artifact_doc_bundle_sync_gate_check()"
    helpers_intro = loop_close_doc_makefile_helpers_doc_intro_line()
    bundle_intro = cpu_mlir_ci_bundle_contract_doc_intro_line()
    crossref_intro = loop_close_doc_render_check_write_crossref_doc_intro_line()
    if hook not in helpers_intro or hook not in bundle_intro or hook not in crossref_intro:
        return False
    crossref_fragment = loop_close_doc_render_check_write_crossref_force_fail_intro_fragment()
    if crossref_fragment not in crossref_intro:
        return False
    doc_crossref = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()
    if doc_crossref not in helpers_intro:
        return False
    if "must be unset on smoke" not in helpers_intro:
        return False
    if "loop_close_doc_force_fail_crossref_and_check_row_parity_ok()" not in bundle_intro:
        return False
    if "loop_close_doc_force_fail_three_way_intro_parity_ok()" not in bundle_intro:
        return False
    return loop_close_doc_force_fail_crossref_and_check_row_parity_ok()


def loop_close_doc_render_check_subprocess_argv(doc_path: str, check_name: str) -> List[str]:
    """Argv for one render ``--check`` script with ``--doc-path`` (R130 ``make check-loop-close-docs`` chain)."""
    spec = next(item for item in loop_close_doc_render_check_specs() if item["name"] == check_name)
    return ["python3", spec["script"], "--check", "--doc-path", doc_path]


def loop_close_doc_render_check_subprocess_argv_chain(doc_path: str) -> List[List[str]]:
    """Argv list mirroring ``make check-loop-close-docs`` render ``--check`` chain (R130)."""
    return [
        loop_close_doc_render_check_subprocess_argv(doc_path, spec["name"])
        for spec in loop_close_doc_render_check_specs()
    ]


def loop_close_docs_smoke_check_make_subprocess_argv() -> List[str]:
    """Argv for ``make smoke-check-loop-close-docs`` subprocess contract tests (R131)."""
    return ["make", loop_close_docs_smoke_make_target()]


def loop_close_doc_check_loop_close_docs_make_subprocess_argv() -> List[str]:
    """Argv for ``make check-loop-close-docs`` canonical doc subprocess tests (R135)."""
    return ["make", "check-loop-close-docs"]


def loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv(doc_path: str) -> List[List[str]]:
    """Argv batches: render ``--check`` chain + smoke make for mixed-parse subprocess tests (R132)."""
    return [
        *loop_close_doc_render_check_subprocess_argv_chain(doc_path),
        loop_close_docs_smoke_check_make_subprocess_argv(),
    ]


def loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches(
    patched_doc_path: str,
) -> List[List[str]]:
    """Argv batches: patched check+smoke chain + canonical ``make check-loop-close-docs`` (R136)."""
    return [
        *loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv(patched_doc_path),
        loop_close_doc_check_loop_close_docs_make_subprocess_argv(),
    ]


def loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text(
    text: str,
) -> str:
    """Replace cross-ref marker block with mixed legacy/suffix table in doc text (R133)."""
    mixed_table = loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table()
    return replace_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END,
        mixed_table,
    )


def loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan(
    patched_doc_path: str,
    canonical_doc_path: str,
) -> Dict[str, object]:
    """Subprocess plan: mixed-parse patched + full smoke/check argv + manifest parity (R136)."""
    return {
        "argv_batches": loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv(
            patched_doc_path
        ),
        "full_argv_batches": loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches(
            patched_doc_path
        ),
        "canonical_doc_path": canonical_doc_path,
        "canonical_check_argv": loop_close_doc_check_loop_close_docs_make_subprocess_argv(),
    }


def loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_python_snippet(
    canonical_doc_path: str,
) -> str:
    """Python ``-c`` snippet asserting manifest parity on canonical doc (R134)."""
    repo = str(Path(__file__).resolve().parents[1])
    return (
        f"import sys; sys.path.insert(0, {repo!r}); "
        "from pathlib import Path; "
        "from scripts.cpu_cert_utils import ("
        "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok, "
        "loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok"
        "); "
        f"text = Path({canonical_doc_path!r}).read_text(encoding='utf-8'); "
        "assert loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok(); "
        "assert loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok(text)"
    )


def loop_close_doc_force_fail_env_stripped_subprocess_env(
    base: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Subprocess env with force-fail hook unset and ``PYTHONPATH`` set (R129)."""
    import os

    force_env = loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()
    source = dict(base if base is not None else os.environ)
    env = {key: value for key, value in source.items() if key != force_env}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return env


def loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table() -> str:
    """Cross-ref table with alternating legacy/suffix Check script labels (R129)."""
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
    header = (
        "| Check name | Check script | Write module | Write fn | Replace fns | Blocks |"
    )
    sep = "|------------|--------------|--------------|----------|-------------|--------|"
    return "\n".join([header, sep, *mixed_rows])


def loop_close_doc_render_check_write_crossref_rows() -> List[Dict[str, str]]:
    """Cross-reference render ``--check`` specs with ``--write`` and replace_fn dispatch (R114)."""
    write_by_name = {spec["name"]: spec for spec in loop_close_doc_render_write_specs()}
    rows: List[Dict[str, str]] = []
    for check in loop_close_doc_render_check_specs():
        write = write_by_name[check["name"]]
        blocks = [
            block
            for block in loop_close_doc_render_write_block_specs()
            if block["write_spec"] == check["name"]
        ]
        replace_fns = ", ".join(sorted({block["replace_fn"] for block in blocks}))
        rows.append(
            {
                "check_name": check["name"],
                "check_script": loop_close_doc_render_check_script_doc_crossref(check),
                "write_module": write["module"],
                "write_fn": write["write_fn"],
                "replace_fns": replace_fns,
                "block_count": str(len(blocks)),
            }
        )
    return rows


def loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet() -> str:
    """Stderr fragment when ``loop_close_ci_artifact_doc_bundle_sync_gate_check`` fails (R122)."""
    return "bundle contract doc sync gate check failed"


def loop_close_doc_render_check_write_crossref_doc_intro_line() -> str:
    """Intro line above check/write cross-ref marker block (R115 single source)."""
    rev = loop_close_doc_bundle_loop_revision()
    row_count = len(loop_close_doc_render_check_write_crossref_rows())
    block_summary = loop_close_doc_render_write_block_counts_summary()
    return (
        f"Loop-close render check/write cross-reference (Loop {rev}; single source: "
        "``loop_close_doc_render_check_write_crossref_rows()``; "
        f"{row_count} rows; block counts via "
        f"``loop_close_doc_render_write_block_counts_by_write_spec()`` ({block_summary}); "
        "ci_artifact bundle sync gate "
        "``loop_close_ci_artifact_doc_bundle_sync_gate_check()`` "
        "(stderr ``loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()``; "
        f"{loop_close_doc_render_check_write_crossref_force_fail_intro_fragment()}); "
        "Check script column includes ``--check`` + ``--doc-path``; "
        "``Blocks`` parity via "
        "``loop_close_doc_render_check_write_crossref_blocks_summary_parity()``; "
        "regenerate with ``render_loop_close_ci_artifact_doc.py`` or "
        "``make check-loop-close-docs``):"
    )


def loop_close_doc_render_check_write_crossref_doc_markdown_table() -> str:
    header = (
        "| Check name | Check script | Write module | Write fn | Replace fns | Blocks |"
    )
    sep = "|------------|--------------|--------------|----------|-------------|--------|"
    body = [
        (
            f"| `{row['check_name']}` | `{row['check_script']}` | `{row['write_module']}` | "
            f"`{row['write_fn']}()` | `{row['replace_fns']}` | `{row['block_count']}` |"
        )
        for row in loop_close_doc_render_check_write_crossref_rows()
    ]
    return "\n".join([header, sep, *body])


def loop_close_ci_doc_render_path_triggers_crossref_scripts() -> List[str]:
    """Render scripts referenced by ``loop_close_doc_render_check_write_crossref_rows()`` (R115)."""
    scripts: set[str] = {check["script"] for check in loop_close_doc_render_check_specs()}
    for row in loop_close_doc_render_check_write_crossref_rows():
        scripts.add(f"{row['write_module'].replace('.', '/')}.py")
    return sorted(scripts)


def loop_close_docs_smoke_make_target() -> str:
    """Makefile target for merged render ``--check`` smoke (R99 single source)."""
    return "smoke-check-loop-close-docs"


def loop_close_ci_docs_gate_workflows() -> List[str]:
    """Workflows whose docs sync gate must invoke ``loop_close_docs_smoke_make_target()``."""
    return [
        ".github/workflows/cpu-loop-close-pr.yml",
        ".github/workflows/cpu-loop-close-nightly.yml",
        ".github/workflows/cpu-mlir-jit-contract.yml",
        ".github/workflows/cpu-mlir-ci-nightly.yml",
    ]


def loop_close_ci_render_check_path_trigger_workflows() -> List[str]:
    """Workflows whose pull_request/push paths must list render check spec scripts."""
    return loop_close_ci_docs_gate_workflows()


def loop_close_ci_doc_render_path_triggers() -> List[str]:
    """Unified CI path triggers for render check/write scripts and smoke Makefile (R101)."""
    triggers = {spec["script"] for spec in loop_close_doc_render_check_specs()}
    triggers.add("Makefile")
    for spec in loop_close_doc_render_write_specs():
        triggers.add(f"{spec['module'].replace('.', '/')}.py")
    return sorted(triggers)


def loop_close_docs_smoke_path_triggers() -> List[str]:
    """Path triggers when smoke target or render scripts change (delegates to R101 unified list)."""
    return loop_close_ci_doc_render_path_triggers()


def loop_close_ci_docs_gate_step_names() -> Dict[str, str]:
    """Human step names for docs sync gates keyed by workflow rel path (R101 single source)."""
    return {
        ".github/workflows/cpu-loop-close-pr.yml": "Loop-close docs sync gate (PR)",
        ".github/workflows/cpu-loop-close-nightly.yml": "Loop-close docs sync gate (nightly)",
        ".github/workflows/cpu-mlir-jit-contract.yml": "Loop-close docs sync gate",
        ".github/workflows/cpu-mlir-ci-nightly.yml": "Loop-close docs sync gate",
    }


def _cpu_ci_path_symmetry_contract(left: str, right: str) -> str:
    """Contract column for path symmetry rows; cross pairs cite pytest names (R105)."""
    cross_tests = {
        (
            "cpu-loop-close-pr.yml push",
            "cpu-loop-close-nightly.yml pull_request",
        ): "test_loop_close_pr_push_paths_symmetric_with_nightly_pull_request",
        (
            "cpu-loop-close-nightly.yml push",
            "cpu-loop-close-pr.yml pull_request",
        ): "test_loop_close_nightly_push_paths_symmetric_with_pr_pull_request",
    }
    base = "Identical path filters"
    test = cross_tests.get((left, right))
    if test:
        return f"{base}; asserted by ``{test}``"
    return base


def cpu_ci_workflow_path_symmetry_doc_rows() -> List[Dict[str, str]]:
    """Documented CI workflow path-filter symmetry pairs (Loop R105 single source)."""
    pairs = [
        ("cpu-loop-close-nightly.yml pull_request", "cpu-loop-close-pr.yml pull_request"),
        ("cpu-loop-close-pr.yml push", "cpu-loop-close-nightly.yml pull_request"),
        ("cpu-loop-close-nightly.yml push", "cpu-loop-close-pr.yml pull_request"),
        ("cpu-loop-close-nightly.yml push", "cpu-loop-close-nightly.yml pull_request"),
        ("cpu-loop-close-pr.yml push", "cpu-loop-close-pr.yml pull_request"),
        ("cpu-mlir-jit-contract.yml push", "cpu-mlir-ci-nightly.yml pull_request"),
        ("cpu-mlir-jit-contract.yml pull_request", "cpu-mlir-ci-nightly.yml pull_request"),
        ("cpu-mlir-ci-nightly.yml push", "cpu-mlir-jit-contract.yml pull_request"),
    ]
    return [
        {"left": left, "right": right, "contract": _cpu_ci_path_symmetry_contract(left, right)}
        for left, right in pairs
    ]


def cpu_mlir_ci_workflow_path_symmetry_doc_rows() -> List[Dict[str, str]]:
    """MLIR subset of ``cpu_ci_workflow_path_symmetry_doc_rows()`` (Loop R101 compat)."""
    return [
        row
        for row in cpu_ci_workflow_path_symmetry_doc_rows()
        if "mlir" in row["left"] or "mlir" in row["right"]
    ]


def cpu_mlir_ci_bundle_test_contract_manifest() -> List[Dict[str, str]]:
    """Helpers exercised by ``make test-cpu-mlir-ci-bundle`` (Loop R105 doc single source)."""
    return [
        {
            "test_module": "tests/python/test_cpu_mlir_ci_bundle.py",
            "helper": "build_mlir_ci_bundle_manifest()",
            "contract": "MLIR CI bundle schema, manifest-only smoke, download regression",
        },
        {
            "test_module": "tests/python/test_hardware_optimization_timing_contract.py",
            "helper": "loop_close_timing_contract()",
            "contract": "Timing threshold table doc sync",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_write_specs()",
            "contract": (
                "Render intro+table idempotence + ``apply_loop_close_doc_render_write_block_replace`` "
                "dispatch per write spec (timing/metadata/ci_artifact)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_check_write_crossref_rows()",
            "contract": (
                "Render ``--check`` specs cross-ref ``--write`` specs and ``replace_fn`` dispatch "
                "(``loop_close_doc_render_check_specs`` ↔ ``loop_close_doc_render_write_block_specs``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_check_script_doc_crossref()",
            "contract": (
                "Check/write cross-ref Check script column (``--check`` + ``--doc-path``; "
                "``normalize_loop_close_doc_render_check_script_doc_label()`` backward compat)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "parse_hardware_optimization_doc_render_check_write_crossref_table()",
            "contract": (
                "Check/write cross-ref table parse sync "
                "(``LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE``; 5/6-column + Check script "
                "suffix backward compat via ``normalize_loop_close_doc_render_check_script_doc_label()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_ci_doc_render_path_triggers_crossref_scripts()",
            "contract": (
                "CI path triggers cover all scripts in "
                "``loop_close_doc_render_check_write_crossref_rows()``"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_write_block_specs()",
            "contract": (
                "Render write marker sub-blocks incl. makefile helpers, intro line registry, "
                "and paired ``replace_fn`` column (ci_artifact → ``replace_loop_close_doc_marker_block``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_intro_line_specs()",
            "contract": "Doc intro lines sync with marker blocks (incl. ``loop_close_doc_makefile_helpers_loop_range()``)",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_intro_line_doc_rows()",
            "contract": (
                "Doc intro marker ↔ intro_fn ↔ loop label ↔ schema ↔ marker_section cross-ref "
                "(``LOOP_CLOSE_DOC_INTRO_LINE_TABLE``; metadata → ``loop_close_metadata_doc_marker_section()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_intro_line_doc_row_count()",
            "contract": (
                "Intro line registry row count + ``cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()`` "
                "(``LOOP_CLOSE_DOC_INTRO_LINE_TABLE``; combined with manifest/crossref parity)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "parse_hardware_optimization_doc_intro_line_table()",
            "contract": (
                "Intro line table parse incl. Schema + Marker section columns "
                "(metadata row ``marker_section_fn`` → ``loop_close_metadata_doc_marker_section()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "apply_loop_close_doc_render_write_block_replace()",
            "contract": (
                "Render write block ``replace_fn`` dispatch per "
                "``loop_close_doc_render_write_block_specs()`` "
                "(timing/metadata render modules; ci_artifact → ``replace_loop_close_doc_marker_block``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "parse_hardware_optimization_doc_render_write_block_table()",
            "contract": (
                "Render write sub-block table parse incl. Replace function column "
                "(``LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE``; 3/4-column backward compat)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_check_write_crossref_block_count_parity()",
            "contract": (
                "Cross-ref ``block_count`` column matches render write sub-block counts per "
                "``loop_close_doc_render_write_block_specs()`` write spec"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()",
            "contract": (
                "Bundle manifest row count + crossref ``block_count`` parity + "
                "``loop_close_doc_intro_line_doc_row_count()`` combined sync gate"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_check_write_crossref_blocks_summary_parity()",
            "contract": (
                "Cross-ref intro ``block_counts_summary`` matches doc ``Blocks`` column per row; "
                "combined with manifest row count via "
                "``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()``"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()",
            "contract": (
                "Bundle manifest row count + crossref blocks summary parity combined sync gate "
                "(Makefile helpers cross-ref on ``check-loop-close-docs`` / ``test-cpu-mlir-ci-bundle``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_ci_artifact_doc_bundle_sync_gate_check()",
            "contract": (
                "Render ``--check`` hook for bundle manifest doc row count + combined sync gates "
                "(stderr ``loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()",
            "contract": (
                "Render ``--check`` stderr fragment when bundle sync gate fails "
                "(``render_loop_close_ci_artifact_doc.py``; cross-ref "
                "``loop_close_ci_artifact_doc_bundle_sync_gate_check()``; test hook "
                "``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()",
            "contract": (
                "Env var to force sync gate failure stderr in render ``--check`` "
                "(``make check-loop-close-docs`` contract tests; doc cross-ref "
                "``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()",
            "contract": (
                "Doc cross-ref label for force-fail env contract tests "
                "(Makefile helpers smoke row ``must be unset`` parity)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled()",
            "contract": (
                "Returns True when ``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()`` "
                "is ``1`` (must be unset during normal ``make smoke-check-loop-close-docs``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_force_fail_crossref_and_check_row_parity_ok()",
            "contract": (
                "Crossref intro and ``make check-loop-close-docs`` helpers row share force-fail "
                "doc cross-ref fragments (``loop_close_doc_makefile_helpers_check_row_force_fail_purpose_fragment()`` "
                "+ ``loop_close_doc_render_check_write_crossref_force_fail_intro_fragment()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_force_fail_env_stripped_subprocess_env()",
            "contract": (
                "Subprocess env with force-fail hook unset + ``PYTHONPATH`` "
                "(``make smoke-check-loop-close-docs`` / ``make check-loop-close-docs`` success tests)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table()",
            "contract": (
                "Mixed legacy/suffix Check script labels for crossref parse backward-compat "
                "(``parse_hardware_optimization_doc_render_check_write_crossref_table()`` + sync gate)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_makefile_helpers_manifest_new_helpers_crossref_ok()",
            "contract": (
                "Makefile helpers ``make test-cpu-mlir-ci-bundle`` row cross-ref's R129 manifest helpers "
                "(``loop_close_doc_makefile_helpers_manifest_new_helpers_crossref()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_force_fail_three_way_intro_parity_ok()",
            "contract": (
                "Helpers/bundle/crossref intro lines share render check hook + force-fail parity "
                "(``loop_close_doc_render_check_write_crossref_force_fail_intro_fragment()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_check_subprocess_argv_chain()",
            "contract": (
                "Render ``--check`` argv chain mirroring ``make check-loop-close-docs`` "
                "(``loop_close_doc_render_check_subprocess_argv()`` + ``--doc-path``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()",
            "contract": (
                "Makefile helpers ``make test-cpu-mlir-ci-bundle`` row cross-ref's all manifest "
                "doc-contract helpers (``loop_close_doc_makefile_helpers_manifest_helpers_crossref()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_intro_line_three_way_parity_ok()",
            "contract": (
                "Intro line registry rows mark three-way force-fail parity gate "
                "(``loop_close_doc_force_fail_three_way_intro_parity_ok()`` on helpers/bundle/crossref)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_docs_smoke_check_make_subprocess_argv()",
            "contract": (
                "Argv for ``make smoke-check-loop-close-docs`` subprocess mixed-parse chain tests "
                "(``loop_close_docs_smoke_make_target()``; see "
                "``loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()",
            "contract": (
                "Makefile helpers test row + bundle intro share manifest/helpers parity cross-ref "
                "(``loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv()",
            "contract": (
                "Mixed-parse subprocess argv: ``loop_close_doc_render_check_subprocess_argv_chain()`` "
                "+ ``loop_close_docs_smoke_check_make_subprocess_argv()``; doc patch via "
                "``loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text()``"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text()",
            "contract": (
                "Mixed legacy/suffix cross-ref table patched into doc text for subprocess tests "
                "(``replace_loop_close_doc_marker_block`` + mixed table helper)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_intro_line_bundle_manifest_parity_ok()",
            "contract": (
                "Bundle intro registry row marks merged manifest parity three-way gate "
                "(``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()``; "
                "``loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok()",
            "contract": (
                "Doc Makefile helpers ``make test-cpu-mlir-ci-bundle`` row matches single source "
                "purpose fragment (``loop_close_doc_makefile_helpers_manifest_helpers_parity_purpose_fragment()``; "
                "see ``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()",
            "contract": (
                "Intro registry manifest gate + bundle intro + helpers test row three-way "
                "(``loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()``; "
                "see ``loop_close_doc_manifest_parity_three_way_ok()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_manifest_parity_three_way_ok()",
            "contract": (
                "Alias for merged intro registry + bundle + helpers manifest parity gate "
                "(``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_check_loop_close_docs_make_subprocess_argv()",
            "contract": (
                "Argv for ``make check-loop-close-docs`` canonical doc subprocess mixed-parse plan "
                "(``loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches()",
            "contract": (
                "Full patched smoke+check argv batches ending with canonical "
                "``make check-loop-close-docs`` "
                "(``loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()",
            "contract": (
                "Mixed-parse patched doc subprocess plan: patched + full smoke/check argv batches "
                "+ manifest parity "
                "(``loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_python_snippet()``; "
                "``loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_render_write_block_counts_by_write_spec()",
            "contract": (
                "Render write sub-block counts per write spec (cross-ref ``Blocks`` column source)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "cpu_mlir_ci_bundle_test_contract_manifest_row_count()",
            "contract": (
                "Bundle contract doc table row count matches manifest "
                "(``LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE``; "
                "``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()``)"
            ),
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_bundle_loop_revision()",
            "contract": "Loop revision for all doc intro lines + bundle/render marker tables",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "parse_hardware_optimization_makefile_helpers_table()",
            "contract": "Makefile helpers marker table parse sync (``LOOP_CLOSE_MAKEFILE_HELPERS_TABLE``)",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "parse_hardware_optimization_metadata_doc_fields()",
            "contract": "Metadata field names from marker block section only (no full-doc backtick scan)",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_metadata_doc_marker_section()",
            "contract": "Bounded metadata intro + marker block for field/schema parse (``_metadata_doc_marker_section``)",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_doc_makefile_helpers_loop_range()",
            "contract": "Makefile helpers section Loop range label (``loop_close_doc_makefile_helpers_doc_intro_line()``)",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_ci_docs_gate_workflows()",
            "contract": "Workflows whose docs sync gate invokes ``loop_close_docs_smoke_make_target()``",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_docs_smoke_make_target()",
            "contract": "CI docs sync gate Makefile target (``make smoke-check-loop-close-docs``)",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_ci_doc_render_path_triggers()",
            "contract": "Unified Makefile + render check/write script CI path triggers",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "loop_close_ci_docs_gate_step_names()",
            "contract": "Docs sync gate human step names in workflow YAML",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "cpu_ci_workflow_path_symmetry_doc_rows()",
            "contract": "Loop-close + MLIR workflow path-filter symmetry (render marker block)",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "cpu_ci_artifact_manifest()",
            "contract": "CI artifact names documented and workflow-aligned",
        },
        {
            "test_module": "tests/python/test_loop_close_metadata_doc_contract.py",
            "helper": "cpu_ci_workflow_make_target_manifest()",
            "contract": "Workflow step ↔ Makefile target mapping",
        },
    ]


def cpu_mlir_ci_bundle_test_contract_manifest_row_count() -> int:
    """Row count gate for ``LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE`` doc sync (R113)."""
    return len(cpu_mlir_ci_bundle_test_contract_manifest())


def cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok() -> bool:
    """Manifest row count + blocks summary parity combined gate (R119)."""
    if not loop_close_doc_render_check_write_crossref_blocks_summary_parity():
        return False
    manifest_count = cpu_mlir_ci_bundle_test_contract_manifest_row_count()
    return manifest_count == len(cpu_mlir_ci_bundle_test_contract_manifest())


def loop_close_ci_artifact_doc_bundle_sync_gate_check(text: str) -> bool:
    """Render ``--check`` hook: manifest doc row count + bundle sync gates (R120)."""
    parsed = parse_hardware_optimization_mlir_ci_bundle_contract_table(text)
    manifest_count = cpu_mlir_ci_bundle_test_contract_manifest_row_count()
    if len(parsed) != manifest_count:
        return False
    if not cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok():
        return False
    return cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()


def cpu_mlir_ci_bundle_contract_doc_sync_gate_ok() -> bool:
    """Combined manifest, crossref block_count, blocks summary, intro row count gate (R119)."""
    if not loop_close_doc_render_check_write_crossref_block_count_parity():
        return False
    if not loop_close_doc_render_check_write_crossref_blocks_summary_parity():
        return False
    if not cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok():
        return False
    manifest_count = cpu_mlir_ci_bundle_test_contract_manifest_row_count()
    if manifest_count != len(cpu_mlir_ci_bundle_test_contract_manifest()):
        return False
    intro_count = loop_close_doc_intro_line_doc_row_count()
    return intro_count == len(loop_close_doc_intro_line_doc_rows())


def cpu_mlir_ci_bundle_contract_doc_intro_line() -> str:
    """Intro line above bundle contract marker block (R105 single source)."""
    rev = loop_close_doc_bundle_loop_revision()
    return (
        f"``make test-cpu-mlir-ci-bundle`` contract helpers (Loop {rev}; single source: "
        "``cpu_mlir_ci_bundle_test_contract_manifest()``; "
        f"{cpu_mlir_ci_bundle_test_contract_manifest_row_count()} rows; intro registry "
        f"``loop_close_doc_intro_line_doc_row_count()`` = "
        f"{loop_close_doc_intro_line_doc_row_count()} rows; sync gate "
        "``cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()``; render check "
        "``loop_close_ci_artifact_doc_bundle_sync_gate_check()``; force-fail parity "
        "``loop_close_doc_force_fail_crossref_and_check_row_parity_ok()`` / "
        "``loop_close_doc_force_fail_three_way_intro_parity_ok()``; "
        f"{loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()}; manifest parity "
        "three-way ``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()`` / "
        "``loop_close_doc_manifest_parity_three_way_ok()``; render write blocks from "
        "``loop_close_doc_render_write_block_specs()``; regenerate with "
        "``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):"
    )


def cpu_mlir_ci_bundle_contract_doc_rows() -> List[Dict[str, str]]:
    return list(cpu_mlir_ci_bundle_test_contract_manifest())


def cpu_mlir_ci_bundle_contract_doc_markdown_table() -> str:
    header = "| Test module | Single-source helper | Contract |"
    sep = "|-------------|----------------------|----------|"
    body = [
        f"| `{row['test_module']}` | `{row['helper']}` | {row['contract']} |"
        for row in cpu_mlir_ci_bundle_contract_doc_rows()
    ]
    return "\n".join([header, sep, *body])


def loop_close_doc_render_write_block_doc_intro_line() -> str:
    """Intro line above render write block marker table (R111 single source)."""
    rev = loop_close_doc_bundle_loop_revision()
    return (
        f"Loop-close doc render write marker sub-blocks (Loop {rev}; single source: "
        "``loop_close_doc_render_write_block_specs()``; check/write cross-ref table at "
        "``LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE``; ci_artifact blocks use paired "
        "``replace_loop_close_doc_marker_block`` via ``render_loop_close_ci_artifact_doc``; "
        "regenerate with ``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):"
    )


def loop_close_doc_render_write_block_doc_rows() -> List[Dict[str, str]]:
    return [
        {
            "write_spec": spec["write_spec"],
            "marker": _loop_close_marker_doc_label(spec["marker_begin"]),
            "table_fn": f"{spec['table_fn']}()",
            "replace_fn": f"{spec['replace_fn']}()",
        }
        for spec in loop_close_doc_render_write_block_specs()
    ]


def _loop_close_marker_doc_label(marker_begin: str) -> str:
    """Doc-safe marker label without HTML comments (avoids nested marker false matches)."""
    return marker_begin.removeprefix("<!-- ").removesuffix(" -->")


def loop_close_doc_render_write_block_doc_markdown_table() -> str:
    header = "| Write spec | Marker begin | Table function | Replace function |"
    sep = "|------------|--------------|----------------|------------------|"
    body = [
        (
            f"| `{row['write_spec']}` | `{row['marker']}` | `{row['table_fn']}` | "
            f"`{row['replace_fn']}` |"
        )
        for row in loop_close_doc_render_write_block_doc_rows()
    ]
    return "\n".join([header, sep, *body])


def parse_mlir_ci_bundle_contract_markdown_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Test module | Single-source" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 3:
            continue
        test_module, helper, contract = parts
        rows.append(
            {
                "test_module": test_module.strip("`"),
                "helper": helper.strip("`"),
                "contract": contract,
            }
        )
    return rows


def parse_doc_render_write_block_markdown_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Write spec | Marker" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) not in (3, 4):
            continue
        write_spec, marker, table_fn = parts[:3]
        replace_fn = (
            parts[3].strip("`")
            if len(parts) == 4
            else "replace_loop_close_doc_marker_block()"
        )
        rows.append(
            {
                "write_spec": write_spec.strip("`"),
                "marker": marker.strip("`"),
                "table_fn": table_fn.strip("`"),
                "replace_fn": replace_fn,
            }
        )
    return rows


def parse_hardware_optimization_mlir_ci_bundle_contract_table(
    text: str,
) -> List[Dict[str, str]]:
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN,
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END,
    )
    if block is not None:
        return parse_mlir_ci_bundle_contract_markdown_table(block)
    marker = "make test-cpu-mlir-ci-bundle`` contract helpers"
    start = text.find(marker)
    if start == -1:
        return parse_mlir_ci_bundle_contract_markdown_table(text)
    end = text.find("CI workflow path-filter symmetry", start)
    section = text[start:end] if end != -1 else text[start:]
    return parse_mlir_ci_bundle_contract_markdown_table(section)


def parse_hardware_optimization_doc_render_write_block_table(
    text: str,
) -> List[Dict[str, str]]:
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END,
    )
    if block is not None:
        return parse_doc_render_write_block_markdown_table(block)
    marker = "Loop-close doc render write marker sub-blocks"
    start = text.find(marker)
    if start == -1:
        return parse_doc_render_write_block_markdown_table(text)
    end = text.find("Loop-close render check/write cross-reference", start)
    if end == -1:
        end = text.find("Loop-close doc intro line registry", start)
    section = text[start:end] if end != -1 else text[start:]
    return parse_doc_render_write_block_markdown_table(section)


def _parse_doc_render_check_write_crossref_markdown_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Check name | Check script" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) not in (5, 6):
            continue
        check_name, check_script, write_module, write_fn, replace_fns = parts[:5]
        block_count = parts[5].strip("`") if len(parts) == 6 else "-"
        rows.append(
            {
                "check_name": check_name.strip("`"),
                "check_script": normalize_loop_close_doc_render_check_script_doc_label(
                    check_script
                ),
                "write_module": write_module.strip("`"),
                "write_fn": write_fn.strip("`").removesuffix("()"),
                "replace_fns": replace_fns.strip("`"),
                "block_count": block_count.strip("`"),
            }
        )
    return rows


def parse_hardware_optimization_doc_render_check_write_crossref_table(
    text: str,
) -> List[Dict[str, str]]:
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END,
    )
    if block is not None:
        return _parse_doc_render_check_write_crossref_markdown_table(block)
    marker = "Loop-close render check/write cross-reference"
    start = text.find(marker)
    if start == -1:
        return _parse_doc_render_check_write_crossref_markdown_table(text)
    end = text.find("Loop-close doc intro line registry", start)
    section = text[start:end] if end != -1 else text[start:]
    return _parse_doc_render_check_write_crossref_markdown_table(section)


def resolve_loop_close_doc_render_write_fn(
    spec: LoopCloseDocRenderWriteSpec,
) -> Callable[[str], str]:
    """Import and return a render script ``--write`` callable from ``spec``."""
    import importlib

    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["write_fn"])


def loop_close_doc_intro_line_specs() -> List[tuple[str, Callable[[], str]]]:
    """Return (marker_begin, intro_line_fn) pairs for doc intro sync tests (R97/R108)."""
    return [
        (LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN, loop_close_doc_makefile_helpers_doc_intro_line),
        (LOOP_CLOSE_TIMING_TABLE_BEGIN, loop_close_timing_table_doc_intro_line),
        (LOOP_CLOSE_METADATA_FIELDS_BEGIN, loop_close_metadata_table_doc_intro_line),
        (LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN, cpu_mlir_ci_bundle_contract_doc_intro_line),
        (LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN, loop_close_doc_render_write_block_doc_intro_line),
        (
            LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN,
            loop_close_doc_render_check_write_crossref_doc_intro_line,
        ),
        (LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN, loop_close_doc_intro_line_doc_intro_line),
        (LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN, cpu_ci_path_symmetry_doc_intro_line),
        (LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN, cpu_ci_artifact_doc_intro_line),
        (LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN, cpu_ci_workflow_make_doc_intro_line),
    ]


def loop_close_doc_intro_line_doc_intro_line() -> str:
    """Intro line above doc intro line marker table (R111 single source)."""
    rev = loop_close_doc_bundle_loop_revision()
    return (
        f"Loop-close doc intro line registry (Loop {rev}; single source: "
        "``loop_close_doc_intro_line_doc_rows()``; makefile helpers range via "
        "``loop_close_doc_makefile_helpers_loop_range()``; metadata schema + "
        "``loop_close_metadata_doc_marker_section()`` cross-ref on metadata row; "
        "three-way intro parity via "
        "``loop_close_doc_intro_line_three_way_parity_ok()``; bundle manifest parity gate via "
        "``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()`` / "
        "``loop_close_doc_intro_line_bundle_manifest_parity_ok()`` / "
        "``loop_close_doc_manifest_parity_three_way_ok()``; "
        "regenerate with ``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):"
    )


def _loop_close_doc_intro_line_schema_label(intro_fn_name: str) -> str:
    if intro_fn_name == "loop_close_metadata_table_doc_intro_line":
        return LOOP_CLOSE_METADATA_DOC_SCHEMA
    return "-"


def _loop_close_doc_intro_line_marker_section_fn(intro_fn_name: str) -> str:
    if intro_fn_name == "loop_close_metadata_table_doc_intro_line":
        return "loop_close_metadata_doc_marker_section()"
    return "-"


def _loop_close_doc_intro_line_three_way_parity_intro_fns() -> List[str]:
    """Intro functions participating in force-fail three-way parity gate (R131)."""
    return [
        "loop_close_doc_makefile_helpers_doc_intro_line",
        "cpu_mlir_ci_bundle_contract_doc_intro_line",
        "loop_close_doc_render_check_write_crossref_doc_intro_line",
    ]


def _loop_close_doc_intro_line_parity_gate_label(intro_fn_name: str) -> str:
    if intro_fn_name in _loop_close_doc_intro_line_three_way_parity_intro_fns():
        return "loop_close_doc_force_fail_three_way_intro_parity_ok()"
    return "-"


def _loop_close_doc_intro_line_manifest_parity_gate_label(intro_fn_name: str) -> str:
    """Manifest parity gate label for bundle intro registry row (R136 merged three-way)."""
    if intro_fn_name == "cpu_mlir_ci_bundle_contract_doc_intro_line":
        return "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()"
    return "-"


def loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok() -> bool:
    """True when intro registry, bundle intro, and helpers test row share manifest parity (R136)."""
    if not loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok():
        return False
    bundle_rows = [
        row
        for row in loop_close_doc_intro_line_doc_rows()
        if row["intro_fn"] == "cpu_mlir_ci_bundle_contract_doc_intro_line()"
    ]
    if len(bundle_rows) != 1:
        return False
    row = bundle_rows[0]
    if (
        row["manifest_parity_gate"]
        != "loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()"
    ):
        return False
    fragment = loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()
    if fragment not in cpu_mlir_ci_bundle_contract_doc_intro_line():
        return False
    test_row = next(
        row
        for row in loop_close_doc_makefile_helpers_doc_rows()
        if row["target"] == "make test-cpu-mlir-ci-bundle"
    )
    return fragment in test_row["purpose"]


def loop_close_doc_intro_line_bundle_manifest_parity_ok() -> bool:
    """True when bundle intro registry row marks merged manifest parity three-way gate (R136)."""
    return loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()


def loop_close_doc_manifest_parity_three_way_ok() -> bool:
    """Alias for merged intro registry + bundle + helpers manifest parity gate (R136)."""
    return loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()


def loop_close_doc_intro_line_three_way_parity_ok() -> bool:
    """True when intro line registry rows mark three-way parity gate (R131)."""
    if not loop_close_doc_force_fail_three_way_intro_parity_ok():
        return False
    parity_rows = [
        row
        for row in loop_close_doc_intro_line_doc_rows()
        if row["intro_fn"].strip("`")
        in {f"{name}()" for name in _loop_close_doc_intro_line_three_way_parity_intro_fns()}
    ]
    if len(parity_rows) != len(_loop_close_doc_intro_line_three_way_parity_intro_fns()):
        return False
    return all(
        row["intro_parity_gate"] == "loop_close_doc_force_fail_three_way_intro_parity_ok()"
        for row in parity_rows
    )


def loop_close_doc_intro_line_doc_rows() -> List[Dict[str, str]]:
    """Rows for ``loop_close_doc_intro_line_specs()`` documentation table (R111)."""
    rows: List[Dict[str, str]] = []
    for marker_begin, intro_fn in loop_close_doc_intro_line_specs():
        if marker_begin == LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN:
            continue
        intro_fn_name = intro_fn.__name__
        if intro_fn_name == "loop_close_doc_makefile_helpers_doc_intro_line":
            loop_label = loop_close_doc_makefile_helpers_loop_range()
        else:
            loop_label = loop_close_doc_bundle_loop_revision()
        rows.append(
            {
                "marker": _loop_close_marker_doc_label(marker_begin),
                "intro_fn": f"{intro_fn_name}()",
                "loop_label": loop_label,
                "schema": _loop_close_doc_intro_line_schema_label(intro_fn_name),
                "marker_section_fn": _loop_close_doc_intro_line_marker_section_fn(intro_fn_name),
                "intro_parity_gate": _loop_close_doc_intro_line_parity_gate_label(intro_fn_name),
                "manifest_parity_gate": _loop_close_doc_intro_line_manifest_parity_gate_label(
                    intro_fn_name
                ),
            }
        )
    return rows


def loop_close_doc_intro_line_doc_row_count() -> int:
    """Row count gate for ``LOOP_CLOSE_DOC_INTRO_LINE_TABLE`` doc sync (R114)."""
    return len(loop_close_doc_intro_line_doc_rows())


def loop_close_doc_intro_line_doc_markdown_table() -> str:
    header = (
        "| Marker begin | Intro function | Loop label | Schema | Marker section | "
        "Intro parity gate | Manifest parity gate |"
    )
    sep = (
        "|--------------|----------------|------------|--------|----------------|"
        "-------------------|----------------------|"
    )
    body = [
        (
            f"| `{row['marker']}` | `{row['intro_fn']}` | `{row['loop_label']}` | "
            f"`{row['schema']}` | `{row['marker_section_fn']}` | `{row['intro_parity_gate']}` | "
            f"`{row['manifest_parity_gate']}` |"
        )
        for row in loop_close_doc_intro_line_doc_rows()
    ]
    return "\n".join([header, sep, *body])


def _parse_makefile_helpers_markdown_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Target | Purpose" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 2:
            continue
        target, purpose = parts
        rows.append({"target": target.strip("`"), "purpose": purpose})
    return rows


def _parse_doc_intro_line_markdown_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Marker begin | Intro" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) not in (3, 4, 5, 6, 7):
            continue
        marker, intro_fn, loop_label = parts[:3]
        schema = parts[3].strip("`") if len(parts) >= 4 else "-"
        marker_section_fn = parts[4].strip("`") if len(parts) >= 5 else "-"
        intro_parity_gate = parts[5].strip("`") if len(parts) >= 6 else "-"
        manifest_parity_gate = parts[6].strip("`") if len(parts) >= 7 else "-"
        rows.append(
            {
                "marker": marker.strip("`"),
                "intro_fn": intro_fn.strip("`"),
                "loop_label": loop_label.strip("`"),
                "schema": schema,
                "marker_section_fn": marker_section_fn,
                "intro_parity_gate": intro_parity_gate,
                "manifest_parity_gate": manifest_parity_gate,
            }
        )
    return rows


def parse_hardware_optimization_makefile_helpers_table(text: str) -> List[Dict[str, str]]:
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN,
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END,
    )
    if block is not None:
        return _parse_makefile_helpers_markdown_table(block)
    marker = "Loop-close and MLIR CI Makefile helpers"
    start = text.find(marker)
    if start == -1:
        return _parse_makefile_helpers_markdown_table(text)
    end = text.find("make test-cpu-mlir-ci-bundle`` contract helpers", start)
    section = text[start:end] if end != -1 else text[start:]
    return _parse_makefile_helpers_markdown_table(section)


def parse_hardware_optimization_doc_intro_line_table(text: str) -> List[Dict[str, str]]:
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
    )
    if block is not None:
        return _parse_doc_intro_line_markdown_table(block)
    marker = "Loop-close doc intro line registry"
    start = text.find(marker)
    if start == -1:
        return _parse_doc_intro_line_markdown_table(text)
    end = text.find("CI workflow path-filter symmetry", start)
    section = text[start:end] if end != -1 else text[start:]
    return _parse_doc_intro_line_markdown_table(section)


def line_before_marker(text: str, marker_begin: str) -> str:
    begin_idx = _loop_close_doc_marker_begin_index(text, marker_begin)
    pos = begin_idx
    while pos > 0 and text[pos - 1] in "\r\n":
        pos -= 1
    line_end = pos
    line_start = text.rfind("\n", 0, line_end) + 1
    return text[line_start:line_end]


def replace_line_before_marker(text: str, marker_begin: str, new_line: str) -> str:
    idx = _loop_close_doc_marker_begin_index(text, marker_begin)
    pos = idx
    while pos > 0 and text[pos - 1] in "\r\n":
        pos -= 1
    line_start = text.rfind("\n", 0, pos) + 1
    return text[:line_start] + new_line + "\n\n" + text[idx:]


def extract_loop_close_doc_marker_block(text: str, begin: str, end: str) -> str | None:
    if begin not in text or end not in text:
        return None
    begin_idx = _loop_close_doc_marker_begin_index(text, begin, end)
    end_idx = text.find(end, begin_idx + len(begin))
    if end_idx == -1:
        return None
    return text[begin_idx + len(begin) : end_idx]


def cpu_ci_workflow_make_target_manifest() -> List[Dict[str, str]]:
    """Makefile targets each CI workflow must invoke (Loop R93/R94 step alignment)."""
    docs_smoke = loop_close_docs_smoke_make_target()
    docs_gate_steps = loop_close_ci_docs_gate_step_names()
    return [
        {
            "workflow": ".github/workflows/cpu-loop-close-pr.yml",
            "step_name": docs_gate_steps[".github/workflows/cpu-loop-close-pr.yml"],
            "make": docs_smoke,
        },
        {
            "workflow": ".github/workflows/cpu-loop-close-pr.yml",
            "step_name": "Pre-alert metadata validate (PR)",
            "make": "validate-loop-close-metadata-pre-alert",
        },
        {
            "workflow": ".github/workflows/cpu-loop-close-pr.yml",
            "step_name": "Post-alert metadata validate (PR)",
            "make": "validate-loop-close-metadata-post-alert",
        },
        {
            "workflow": ".github/workflows/cpu-loop-close-pr.yml",
            "step_name": "MLIR CI bundle regression tests",
            "make": "test-cpu-mlir-ci-bundle",
        },
        {
            "workflow": ".github/workflows/cpu-loop-close-nightly.yml",
            "step_name": docs_gate_steps[".github/workflows/cpu-loop-close-nightly.yml"],
            "make": docs_smoke,
        },
        {
            "workflow": ".github/workflows/cpu-loop-close-nightly.yml",
            "step_name": "Pre-alert metadata validate (nightly)",
            "make": "validate-loop-close-metadata-pre-alert",
        },
        {
            "workflow": ".github/workflows/cpu-loop-close-nightly.yml",
            "step_name": "Post-alert metadata validate (nightly)",
            "make": "validate-loop-close-metadata-post-alert",
        },
        {
            "workflow": ".github/workflows/cpu-mlir-jit-contract.yml",
            "step_name": docs_gate_steps[".github/workflows/cpu-mlir-jit-contract.yml"],
            "make": docs_smoke,
        },
        {
            "workflow": ".github/workflows/cpu-mlir-jit-contract.yml",
            "step_name": "MLIR CI artifact bundle (dialect smoke log + bench profile JSON)",
            "make": "build-mlir-ci-bundle",
        },
        {
            "workflow": ".github/workflows/cpu-mlir-jit-contract.yml",
            "step_name": "MLIR CI bundle manifest-only smoke (same bundle dir)",
            "make": "smoke-build-mlir-ci-bundle-manifest",
        },
        {
            "workflow": ".github/workflows/cpu-mlir-jit-contract.yml",
            "step_name": "Simulate downloaded MLIR bundle regression validate",
            "make": "validate-mlir-ci-metadata-download",
        },
        {
            "workflow": ".github/workflows/cpu-mlir-ci-nightly.yml",
            "step_name": docs_gate_steps[".github/workflows/cpu-mlir-ci-nightly.yml"],
            "make": docs_smoke,
        },
        {
            "workflow": ".github/workflows/cpu-mlir-ci-nightly.yml",
            "step_name": "MLIR CI artifact bundle (nightly)",
            "make": "build-mlir-ci-bundle",
        },
        {
            "workflow": ".github/workflows/cpu-mlir-ci-nightly.yml",
            "step_name": "MLIR CI bundle manifest-only smoke (same bundle dir)",
            "make": "smoke-build-mlir-ci-bundle-manifest",
        },
        {
            "workflow": ".github/workflows/cpu-mlir-ci-nightly.yml",
            "step_name": "Simulate downloaded MLIR bundle regression validate (nightly)",
            "make": "validate-mlir-ci-metadata-download",
        },
    ]


def cpu_ci_workflow_make_step_doc_rows() -> List[Dict[str, str]]:
    import os

    return [
        {
            "workflow": os.path.basename(entry["workflow"]),
            "step_name": entry["step_name"],
            "make": entry["make"],
        }
        for entry in cpu_ci_workflow_make_target_manifest()
    ]


def cpu_ci_workflow_make_step_doc_markdown_table() -> str:
    """Markdown table mapping CI workflow steps to Makefile targets (R94)."""
    header = "| Workflow | Step (human name) | Makefile target |"
    sep = "|----------|-------------------|-----------------|"
    body = [
        f"| `{row['workflow']}` | {row['step_name']} | `make {row['make']}` |"
        for row in cpu_ci_workflow_make_step_doc_rows()
    ]
    return "\n".join([header, sep, *body])


def cpu_ci_path_symmetry_doc_markdown_table() -> str:
    """Markdown table for CI workflow path-filter symmetry pairs (R102)."""
    header = "| Left | Right | Contract |"
    sep = "|------|-------|----------|"
    body = [
        f"| `{row['left']}` | `{row['right']}` | {row['contract']} |"
        for row in cpu_ci_workflow_path_symmetry_doc_rows()
    ]
    return "\n".join([header, sep, *body])


def parse_ci_path_symmetry_markdown_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Left | Right" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 3:
            continue
        left, right, contract = parts
        rows.append(
            {
                "left": left.strip("`"),
                "right": right.strip("`"),
                "contract": contract,
            }
        )
    return rows


def parse_hardware_optimization_ci_path_symmetry_table(text: str) -> List[Dict[str, str]]:
    """Parse path symmetry markdown table from ``HARDWARE_OPTIMIZATION.md``."""
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END,
    )
    if block is not None:
        return parse_ci_path_symmetry_markdown_table(block)
    marker = "CI workflow path-filter symmetry"
    start = text.find(marker)
    if start == -1:
        return parse_ci_path_symmetry_markdown_table(text)
    end = text.find("CI workflow artifact names", start)
    section = text[start:end] if end != -1 else text[start:]
    return parse_ci_path_symmetry_markdown_table(section)


def parse_ci_workflow_make_step_markdown_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Workflow | Step" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 3:
            continue
        workflow, step_name, make_target = parts
        make_target = make_target.strip("`")
        if make_target.startswith("make "):
            make_target = make_target[5:]
        rows.append(
            {
                "workflow": workflow.strip("`"),
                "step_name": step_name,
                "make": make_target,
            }
        )
    return rows


def parse_hardware_optimization_ci_workflow_make_table(text: str) -> List[Dict[str, str]]:
    """Parse workflow step ↔ make target table from ``HARDWARE_OPTIMIZATION.md``."""
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END,
    )
    if block is not None:
        return parse_ci_workflow_make_step_markdown_table(block)
    marker = "CI workflow step"
    start = text.find(marker)
    if start == -1:
        return parse_ci_workflow_make_step_markdown_table(text)
    end = text.find("**Loop-close archive", start)
    section = text[start:end] if end != -1 else text[start:]
    return parse_ci_workflow_make_step_markdown_table(section)


def loop_profile_from_stages(stages: Dict[str, Any]) -> Dict[str, Any]:
    """Build loop-close archive profile from per-stage runner results."""
    stage_elapsed = {
        name: round(float(stage.get("elapsed_s", 0)), 2)
        for name, stage in stages.items()
    }
    stages_ok = sum(1 for stage in stages.values() if stage.get("ok"))
    profile: Dict[str, Any] = {
        "total_elapsed_s": round(sum(stage_elapsed.values()), 2),
        "stage_elapsed_s": stage_elapsed,
        "stages_run": len(stages),
        "stages_ok": stages_ok,
    }
    demos = stages.get("demos") or {}
    if demos.get("pytest"):
        profile["demos_passed"] = demos["pytest"].get("passed")
    mlir = stages.get("mlir_bench_profile") or {}
    mlir_profile = mlir.get("profile") or {}
    if mlir_profile:
        profile["mlir_bench_profile"] = mlir_profile
        profile["mlir_bench_profile_ok"] = mlir.get("ok")
    cert = stages.get("cert_e2e") or {}
    cert_profile = cert.get("profile")
    if cert_profile:
        profile["cert_e2e_profile"] = cert_profile
        profile["cert_e2e_ok"] = cert.get("ok")
        profile["value_verify_aligned"] = cert_profile.get("value_verify_aligned")
    return profile


def loop_close_archive_metadata(
    report: Dict[str, Any],
    *,
    archive_path: str | None = None,
    validation_ok: bool | None = None,
) -> Dict[str, Any]:
    """Build artifact sidecar metadata for loop-close archives (Loop R74)."""
    profile = report.get("profile") or {}
    mlir_stage = (report.get("stages") or {}).get("mlir_bench_profile") or {}
    mlir_profile = mlir_stage.get("profile") or profile.get("mlir_bench_profile") or {}
    mode = report.get("mode")
    bench_quick = mlir_profile.get("bench_quick")
    if bench_quick is None and mode in ("quick", "full"):
        bench_quick = mode == "quick"

    meta: Dict[str, Any] = {
        "schema": "loop_close_artifact_metadata_v2",
        "backend": report.get("backend"),
        "mode": mode,
        "ok": report.get("ok"),
        "bench_quick": bench_quick,
        "stage_elapsed_s": profile.get("stage_elapsed_s") or {},
        "total_elapsed_s": profile.get("total_elapsed_s"),
        "stages_ok": profile.get("stages_ok"),
        "demos_passed": profile.get("demos_passed"),
        "mlir_bench_profile_ok": profile.get("mlir_bench_profile_ok"),
        "shape_contract_ok": mlir_profile.get("shape_contract_ok"),
        "shape_validation_errors": mlir_profile.get("shape_validation_errors"),
        "cert_e2e_ok": profile.get("cert_e2e_ok"),
        "value_verify_aligned": profile.get("value_verify_aligned"),
        "validation_ok": validation_ok,
        "stage_timeout_warnings": (warnings := loop_close_stage_timeout_warnings(report)),
        "stage_timeout_warning_count": (warning_count := len(warnings)),
        "timeout_alert_pending": warning_count >= 1,
    }
    if archive_path:
        meta["archive_path"] = archive_path
        import os

        if os.path.isfile(archive_path):
            meta["archive_sha256"] = loop_close_archive_file_sha256(archive_path)
    return meta


LOOP_CLOSE_METADATA_SCHEMA = "loop_close_artifact_metadata_v2"
LOOP_CLOSE_METADATA_REQUIRED_KEYS = (
    "schema",
    "backend",
    "mode",
    "ok",
    "bench_quick",
    "stage_elapsed_s",
    "validation_ok",
    "stage_timeout_warning_count",
    "timeout_alert_pending",
)

LOOP_CLOSE_METADATA_DOC_SCHEMA = LOOP_CLOSE_METADATA_SCHEMA
LOOP_CLOSE_METADATA_DOC_FIELD_NAMES = (
    "bench_quick",
    "stage_elapsed_s",
    "validation_ok",
    "stage_timeout_warning_count",
    "timeout_alert_pending",
    "timeout_alert_emitted",
    "stage_timeout_warnings",
    "archive_sha256",
)

LOOP_CLOSE_METADATA_DOC_FIELD_DESCRIPTIONS: Dict[str, str] = {
    "bench_quick": "Quick vs full archive mode for embedded MLIR bench profile",
    "stage_elapsed_s": "Per-stage elapsed seconds copied from archive profile",
    "validation_ok": "True when archive JSON passed validate_loop_close_archive",
    "stage_timeout_warning_count": "Count of soft stage timeout warnings",
    "timeout_alert_pending": "True when warnings exist and alert not yet emitted",
    "timeout_alert_emitted": "Set by emit_loop_close_timeout_alert --annotate-metadata",
    "stage_timeout_warnings": "Soft warnings when elapsed exceeds doc baseline below hard ceiling",
    "archive_sha256": "SHA-256 digest of archive file when archive_path exists on disk",
}


def loop_close_metadata_doc_field_names() -> tuple[str, ...]:
    """Fields documented in ``HARDWARE_OPTIMIZATION.md`` metadata sidecar section (R82)."""
    return LOOP_CLOSE_METADATA_DOC_FIELD_NAMES


def loop_close_metadata_doc_rows() -> List[Dict[str, str]]:
    """Rows for metadata sidecar field table in ``HARDWARE_OPTIMIZATION.md`` (R83)."""
    return [
        {"field": field, "description": LOOP_CLOSE_METADATA_DOC_FIELD_DESCRIPTIONS[field]}
        for field in LOOP_CLOSE_METADATA_DOC_FIELD_NAMES
    ]


def loop_close_metadata_doc_markdown_table() -> str:
    """Markdown table fragment for metadata sidecar fields (R83 single source)."""
    header = "| Field | Description |"
    sep = "|-------|-------------|"
    body = [
        f"| {row['field']} | {row['description']} |"
        for row in loop_close_metadata_doc_rows()
    ]
    return "\n".join([header, sep, *body])


LOOP_CLOSE_METADATA_FIELDS_BEGIN = "<!-- LOOP_CLOSE_METADATA_FIELDS_BEGIN -->"
LOOP_CLOSE_METADATA_FIELDS_END = "<!-- LOOP_CLOSE_METADATA_FIELDS_END -->"


def loop_close_metadata_table_doc_intro_line() -> str:
    """Intro line immediately above metadata field marker block (R109 single source)."""
    rev = loop_close_doc_bundle_loop_revision()
    return (
        f"Loop-close archive metadata sidecar fields (Loop {rev}; schema "
        f"``{LOOP_CLOSE_METADATA_DOC_SCHEMA}``; single source: "
        "``loop_close_metadata_doc_rows()``; regenerate with "
        "``render_loop_close_metadata_doc.py`` or ``make check-loop-close-docs``):"
    )


def _metadata_doc_marker_section(text: str) -> str:
    """Return intro line + marker block body for metadata fields (R109 bounded parse)."""
    if LOOP_CLOSE_METADATA_FIELDS_BEGIN not in text:
        return ""
    intro = line_before_marker(text, LOOP_CLOSE_METADATA_FIELDS_BEGIN)
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_METADATA_FIELDS_BEGIN,
        LOOP_CLOSE_METADATA_FIELDS_END,
    )
    if block is None:
        return intro
    return f"{intro}\n{block}"


def loop_close_metadata_doc_marker_section(text: str) -> str:
    """Public wrapper for metadata intro + marker block section (R110 single source)."""
    return _metadata_doc_marker_section(text)


def _parse_metadata_markdown_table(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Field | Description" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 2:
            continue
        field, description = parts
        if field in LOOP_CLOSE_METADATA_DOC_FIELD_NAMES:
            rows.append({"field": field, "description": description})
    return rows


def parse_hardware_optimization_metadata_doc_table(text: str) -> List[Dict[str, str]]:
    """Parse metadata field markdown table from ``HARDWARE_OPTIMIZATION.md`` marker block."""
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_METADATA_FIELDS_BEGIN,
        LOOP_CLOSE_METADATA_FIELDS_END,
    )
    if block is not None:
        return _parse_metadata_markdown_table(block)
    marker = "Loop-close archive metadata sidecar fields"
    start = text.find(marker)
    if start == -1:
        return _parse_metadata_markdown_table(text)
    end = text.find("| `test-cpu-mlir-dialect-smoke`", start)
    section = text[start:end] if end != -1 else text[start:]
    return _parse_metadata_markdown_table(section)


def parse_hardware_optimization_metadata_doc_fields(text: str) -> set[str]:
    """Return metadata field names documented in metadata marker section only (R109)."""
    documented = {row["field"] for row in parse_hardware_optimization_metadata_doc_table(text)}
    section = _metadata_doc_marker_section(text)
    if section and f"``{LOOP_CLOSE_METADATA_DOC_SCHEMA}``" in section:
        documented.add(LOOP_CLOSE_METADATA_DOC_SCHEMA)
    return documented


def simulate_downloaded_loop_close_regression_validate(
    source_archive: str,
    source_meta: str,
    dest_dir: str,
    *,
    check_stage_timeouts: bool = False,
    require_alert_annotation: bool = False,
) -> List[str]:
    """Copy archive+meta to download path and validate (R83 nightly/PR regression)."""
    import os
    import shutil

    os.makedirs(dest_dir, exist_ok=True)
    dest_archive = os.path.join(dest_dir, "cpu-loop-close.json")
    dest_meta = os.path.join(dest_dir, "cpu-loop-close.meta.json")
    shutil.copy2(source_archive, dest_archive)
    shutil.copy2(source_meta, dest_meta)

    archive = load_loop_close_archive(dest_archive)
    metadata = load_loop_close_archive_metadata(dest_meta)
    errors = validate_loop_close_archive(archive)
    errors.extend(
        validate_loop_close_archive_metadata(
            metadata,
            archive=archive,
            require_alert_annotation=require_alert_annotation,
        )
    )
    errors.extend(validate_loop_close_archive_hash(metadata, archive_path=dest_archive))
    if check_stage_timeouts:
        errors.extend(validate_loop_close_archive_stage_timeouts(archive))
    return errors


LOOP_CLOSE_STAGE_CEILING_S: Dict[str, float] = {
    "demos": 90.0,
    "mlir_bench_profile": 60.0,
    "mlir_bench_contract": 20.0,
    "cert_e2e": 150.0,
}
LOOP_CLOSE_TOTAL_CEILING_S = 220.0


def validate_loop_close_archive_metadata(
    metadata: Dict[str, Any],
    *,
    archive: Dict[str, Any] | None = None,
    require_alert_annotation: bool = False,
) -> List[str]:
    """Validate loop-close artifact sidecar metadata (Loop R75/R80)."""
    errors: List[str] = []
    if metadata.get("schema") != LOOP_CLOSE_METADATA_SCHEMA:
        errors.append(f"schema must be {LOOP_CLOSE_METADATA_SCHEMA!r}")
    for key in LOOP_CLOSE_METADATA_REQUIRED_KEYS:
        if key not in metadata:
            errors.append(f"missing metadata field {key}")
    if metadata.get("backend") != "cpu":
        errors.append("metadata.backend must be cpu")
    if metadata.get("mode") not in ("quick", "full"):
        errors.append("metadata.mode must be quick or full")
    if metadata.get("validation_ok") is not True:
        errors.append("metadata.validation_ok must be true")
    stage_elapsed = metadata.get("stage_elapsed_s")
    if not isinstance(stage_elapsed, dict):
        errors.append("metadata.stage_elapsed_s must be a dict")
    warnings = metadata.get("stage_timeout_warnings")
    if warnings is not None and not isinstance(warnings, list):
        errors.append("metadata.stage_timeout_warnings must be a list")
    warning_count = metadata.get("stage_timeout_warning_count")
    if warning_count is not None:
        if not isinstance(warning_count, int) or warning_count < 0:
            errors.append("metadata.stage_timeout_warning_count must be a non-negative int")
        elif warnings is not None and warning_count != len(warnings):
            errors.append(
                "metadata.stage_timeout_warning_count must match len(stage_timeout_warnings)"
            )
    alert_emitted = metadata.get("timeout_alert_emitted")
    if alert_emitted is not None and alert_emitted is not True:
        errors.append("metadata.timeout_alert_emitted must be true when set")
    if alert_emitted is True:
        if warning_count is None or warning_count < 1:
            errors.append(
                "metadata.timeout_alert_emitted requires stage_timeout_warning_count >= 1"
            )
    if require_alert_annotation and warning_count is not None and warning_count >= 1:
        if alert_emitted is not True:
            errors.append(
                "metadata.timeout_alert_emitted required when stage_timeout_warning_count >= 1"
            )
    alert_pending = metadata.get("timeout_alert_pending")
    if alert_pending is not None and not isinstance(alert_pending, bool):
        errors.append("metadata.timeout_alert_pending must be a bool")
    if warning_count is not None and alert_pending is not None:
        if warning_count == 0 and alert_pending is not False:
            errors.append(
                "metadata.timeout_alert_pending must be false when stage_timeout_warning_count is 0"
            )
        if alert_emitted is True and alert_pending is not False:
            errors.append(
                "metadata.timeout_alert_pending must be false when timeout_alert_emitted is true"
            )
        if (
            warning_count >= 1
            and alert_emitted is not True
            and not require_alert_annotation
            and alert_pending is not True
        ):
            errors.append(
                "metadata.timeout_alert_pending must be true when warnings exist and alert not emitted"
            )
        if require_alert_annotation and warning_count >= 1 and alert_pending is not False:
            errors.append(
                "metadata.timeout_alert_pending must be false after alert annotation"
            )

    if archive is not None:
        if metadata.get("mode") != archive.get("mode"):
            errors.append("metadata.mode must match archive mode")
        if metadata.get("ok") != archive.get("ok"):
            errors.append("metadata.ok must match archive ok")
        expected_bench_quick = archive.get("mode") == "quick"
        if metadata.get("bench_quick") is not expected_bench_quick:
            errors.append(
                f"metadata.bench_quick {metadata.get('bench_quick')!r} "
                f"!= expected {expected_bench_quick!r} for mode {archive.get('mode')!r}"
            )
        archive_profile = archive.get("profile") or {}
        if stage_elapsed and archive_profile.get("stage_elapsed_s"):
            for name, elapsed in archive_profile["stage_elapsed_s"].items():
                meta_elapsed = stage_elapsed.get(name)
                if meta_elapsed is not None and float(meta_elapsed) != float(elapsed):
                    errors.append(
                        f"metadata.stage_elapsed_s[{name}] {meta_elapsed} "
                        f"!= archive {elapsed}"
                    )
    return errors


def validate_loop_close_archive_hash(
    metadata: Dict[str, Any],
    *,
    archive_path: str | None = None,
) -> List[str]:
    """Verify metadata ``archive_sha256`` matches archive file (Loop R76)."""
    import os

    errors: List[str] = []
    expected = metadata.get("archive_sha256")
    path = archive_path or metadata.get("archive_path")
    if not expected:
        if path:
            errors.append("metadata.archive_sha256 required when archive_path is set")
        return errors
    if not path or not os.path.isfile(path):
        return errors
    actual = loop_close_archive_file_sha256(path)
    if actual != expected:
        errors.append(
            f"archive_sha256 mismatch: metadata {expected[:12]}... != file {actual[:12]}..."
        )
    return errors


def validate_loop_close_archive_stage_timeouts(report: Dict[str, Any]) -> List[str]:
    """Fail full archives when any stage exceeds CI ceiling (Loop R75)."""
    if report.get("mode") != "full":
        return []
    profile = report.get("profile") or {}
    stage_elapsed = profile.get("stage_elapsed_s") or {}
    errors: List[str] = []
    for stage, ceiling in LOOP_CLOSE_STAGE_CEILING_S.items():
        elapsed = stage_elapsed.get(stage)
        if elapsed is not None and float(elapsed) > ceiling:
            errors.append(
                f"stage {stage} elapsed {elapsed}s exceeds ceiling {ceiling}s"
            )
    total = profile.get("total_elapsed_s")
    if total is not None and float(total) > LOOP_CLOSE_TOTAL_CEILING_S:
        errors.append(
            f"total_elapsed_s {total}s exceeds ceiling {LOOP_CLOSE_TOTAL_CEILING_S}s"
        )
    return errors


def load_loop_close_archive_metadata(path: str) -> Dict[str, Any]:
    """Load loop-close metadata sidecar JSON."""
    import json

    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# VM reference baselines (Loop R59/R74); used for docs/tests — not enforced at runtime.
CERT_TIMING_BASELINE_QUICK_S = 35.0
CERT_TIMING_BASELINE_FULL_NO_WALKTHROUGH_S = 38.0
CERT_TIMING_BASELINE_WALKTHROUGH_QUICK_S = 30.0
CERT_TIMING_BASELINE_E2E_FULL_S = 70.0
CERT_TIMING_BASELINE_LOOP_CLOSE_S = 100.0
CERT_TIMING_BASELINE_LOOP_CLOSE_PROFILE_QUICK_S = 45.0
CERT_TIMING_BASELINE_LOOP_CLOSE_PROFILE_VALIDATE_S = 50.0
CERT_TIMING_BASELINE_LOOP_CLOSE_ARCHIVE_S = 110.0
CERT_TIMING_BASELINE_LOOP_CLOSE_ARCHIVE_FULL_BENCH_S = 130.0

LOOP_CLOSE_STAGE_SOFT_LIMIT_S: Dict[str, Dict[str, float]] = {}
LOOP_CLOSE_TOTAL_SOFT_LIMIT_S: Dict[str, float] = {}


def loop_close_stage_soft_limit_table() -> Dict[str, Dict[str, float]]:
    """Single source for metadata soft warnings and HARDWARE_OPTIMIZATION (R77)."""
    return {
        "quick": {
            "demos": CERT_TIMING_BASELINE_LOOP_CLOSE_PROFILE_QUICK_S,
            "mlir_bench_profile": 8.0,
            "mlir_bench_contract": 12.0,
        },
        "full": {
            "demos": 50.0,
            "mlir_bench_profile": 35.0,
            "mlir_bench_contract": 12.0,
            "cert_e2e": CERT_TIMING_BASELINE_E2E_FULL_S,
        },
    }


def loop_close_total_soft_limit_table() -> Dict[str, float]:
    return {
        "quick": CERT_TIMING_BASELINE_LOOP_CLOSE_PROFILE_VALIDATE_S,
        "full": CERT_TIMING_BASELINE_LOOP_CLOSE_ARCHIVE_FULL_BENCH_S,
    }


def loop_close_timing_contract() -> Dict[str, Any]:
    """Export soft/hard stage limits for docs and regression tests (R77)."""
    return {
        "stage_soft_limit_s": loop_close_stage_soft_limit_table(),
        "total_soft_limit_s": loop_close_total_soft_limit_table(),
        "stage_ceiling_s": dict(LOOP_CLOSE_STAGE_CEILING_S),
        "total_ceiling_s": LOOP_CLOSE_TOTAL_CEILING_S,
    }


def loop_close_timing_doc_rows() -> List[Dict[str, Any]]:
    """Rows for ``HARDWARE_OPTIMIZATION.md`` soft/hard timing table (R78 single source)."""
    contract = loop_close_timing_contract()
    rows: List[Dict[str, Any]] = []
    for mode in ("quick", "full"):
        for stage, soft in contract["stage_soft_limit_s"][mode].items():
            rows.append(
                {
                    "mode": mode,
                    "stage": stage,
                    "soft_limit_s": soft,
                    "hard_ceiling_s": contract["stage_ceiling_s"][stage],
                }
            )
        rows.append(
            {
                "mode": mode,
                "stage": "total",
                "soft_limit_s": contract["total_soft_limit_s"][mode],
                "hard_ceiling_s": contract["total_ceiling_s"],
            }
        )
    return rows


def loop_close_timing_markdown_table() -> str:
    """Markdown table fragment for ``HARDWARE_OPTIMIZATION.md`` (R79 single source)."""
    header = "| Mode | Stage | Soft limit (s) | Hard ceiling (s) |"
    sep = "|------|-------|----------------|------------------|"
    body = [
        f"| {row['mode']} | {row['stage']} | {row['soft_limit_s']:.0f} | {row['hard_ceiling_s']:.0f} |"
        for row in loop_close_timing_doc_rows()
    ]
    return "\n".join([header, sep, *body])


LOOP_CLOSE_TIMING_TABLE_BEGIN = "<!-- LOOP_CLOSE_TIMING_TABLE_BEGIN -->"
LOOP_CLOSE_TIMING_TABLE_END = "<!-- LOOP_CLOSE_TIMING_TABLE_END -->"


def loop_close_timing_table_doc_intro_line() -> str:
    """Intro line immediately above timing marker block (R106 single source)."""
    rev = loop_close_doc_bundle_loop_revision()
    return (
        f"Loop-close archive timing thresholds (Loop {rev}; single source: "
        "``loop_close_timing_contract()``; regenerate with "
        "``render_loop_close_timing_doc.py`` or ``make check-loop-close-docs``):"
    )


def _loop_close_doc_marker_end_by_begin() -> Dict[str, str]:
    """Map marker begin comments to paired end markers for doc render lookup."""
    return {
        LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN: LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END,
        LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN: LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END,
        LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN: (
            LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END
        ),
        LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN: LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END,
        LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN: LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END,
        LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN: LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END,
        LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN: LOOP_CLOSE_CI_ARTIFACT_TABLE_END,
        LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN: LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END,
        LOOP_CLOSE_METADATA_FIELDS_BEGIN: LOOP_CLOSE_METADATA_FIELDS_END,
        LOOP_CLOSE_TIMING_TABLE_BEGIN: LOOP_CLOSE_TIMING_TABLE_END,
    }


def _loop_close_doc_marker_begin_index(text: str, begin: str, end: str | None = None) -> int:
    """Return begin index for the first valid begin/end pair (paired lookup)."""
    marker_end = end or _loop_close_doc_marker_end_by_begin().get(begin)
    if marker_end is None:
        if begin not in text:
            raise ValueError(f"marker missing: {begin}")
        return text.index(begin)
    search = 0
    while True:
        begin_idx = text.find(begin, search)
        if begin_idx == -1:
            raise ValueError(f"marker missing: {begin}")
        end_idx = text.find(marker_end, begin_idx + len(begin))
        if end_idx != -1:
            return begin_idx
        search = begin_idx + len(begin)


def find_loop_close_doc_marker_span(text: str, begin: str, end: str) -> tuple[int, int]:
    """Return ``(begin_idx, end_idx_after_end)`` for the first valid marker pair."""
    begin_idx = _loop_close_doc_marker_begin_index(text, begin, end)
    end_idx = text.find(end, begin_idx + len(begin))
    if end_idx == -1:
        raise ValueError(f"marker end missing after {begin}: {end}")
    return begin_idx, end_idx + len(end)


def replace_loop_close_doc_marker_block(text: str, begin: str, end: str, table: str) -> str:
    """Replace content between paired marker comments with generated table."""
    begin_idx, end_idx = find_loop_close_doc_marker_span(text, begin, end)
    replacement = f"{begin}\n{table}\n{end}"
    return text[:begin_idx] + replacement + text[end_idx:]


def _parse_timing_markdown_table(text: str) -> List[Dict[str, Any]]:
    """Parse timing threshold markdown table body."""
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "Mode | Stage" in line or "|---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 4:
            continue
        mode, stage, soft, hard = parts
        if mode not in ("quick", "full"):
            continue
        rows.append(
            {
                "mode": mode,
                "stage": stage,
                "soft_limit_s": float(soft),
                "hard_ceiling_s": float(hard),
            }
        )
    return rows


def parse_hardware_optimization_timing_table(text: str) -> List[Dict[str, Any]]:
    """Parse soft/hard limit markdown table from ``HARDWARE_OPTIMIZATION.md`` marker block."""
    block = extract_loop_close_doc_marker_block(
        text,
        LOOP_CLOSE_TIMING_TABLE_BEGIN,
        LOOP_CLOSE_TIMING_TABLE_END,
    )
    if block is not None:
        return _parse_timing_markdown_table(block)
    marker = "Loop-close archive timing thresholds"
    start = text.find(marker)
    if start == -1:
        return _parse_timing_markdown_table(text)
    end = text.find("Nightly runs", start)
    section = text[start:end] if end != -1 else text[start:]
    return _parse_timing_markdown_table(section)


def _init_loop_close_soft_limits() -> None:
    global LOOP_CLOSE_STAGE_SOFT_LIMIT_S, LOOP_CLOSE_TOTAL_SOFT_LIMIT_S
    LOOP_CLOSE_STAGE_SOFT_LIMIT_S = loop_close_stage_soft_limit_table()
    LOOP_CLOSE_TOTAL_SOFT_LIMIT_S = loop_close_total_soft_limit_table()


_init_loop_close_soft_limits()


def loop_close_timeout_alert_payload(metadata: Dict[str, Any]) -> Dict[str, Any] | None:
    """Build notify placeholder when metadata has soft stage warnings (R77)."""
    warnings = metadata.get("stage_timeout_warnings") or []
    if not warnings:
        return None
    stages = ", ".join(w["stage"] for w in warnings)
    return {
        "schema": "loop_close_timeout_alert_v1",
        "action": "notify_placeholder",
        "channels": {
            "slack": "disabled_set_LOOP_CLOSE_SLACK_WEBHOOK",
            "github_issue": "disabled_set_LOOP_CLOSE_GITHUB_ISSUE",
        },
        "mode": metadata.get("mode"),
        "bench_quick": metadata.get("bench_quick"),
        "warnings": warnings,
        "summary": f"loop-close {metadata.get('mode')} archive exceeded soft limits: {stages}",
    }


def loop_close_archive_file_sha256(path: str) -> str:
    """Return SHA-256 hex digest of a loop-close archive file (Loop R76)."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loop_close_stage_timeout_warnings(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Soft warnings when stage elapsed exceeds doc baseline but below hard ceiling (R76)."""
    mode = report.get("mode")
    if mode not in LOOP_CLOSE_STAGE_SOFT_LIMIT_S:
        return []
    profile = report.get("profile") or {}
    stage_elapsed = profile.get("stage_elapsed_s") or {}
    limits = LOOP_CLOSE_STAGE_SOFT_LIMIT_S[mode]
    warnings: List[Dict[str, Any]] = []
    for stage, soft_limit in limits.items():
        elapsed = stage_elapsed.get(stage)
        if elapsed is None:
            continue
        elapsed_f = float(elapsed)
        if elapsed_f > soft_limit:
            hard = LOOP_CLOSE_STAGE_CEILING_S.get(stage)
            if hard is not None and elapsed_f > hard:
                continue
            warnings.append(
                {
                    "stage": stage,
                    "elapsed_s": round(elapsed_f, 2),
                    "soft_limit_s": soft_limit,
                    "level": "soft",
                }
            )
    total = profile.get("total_elapsed_s")
    total_soft = LOOP_CLOSE_TOTAL_SOFT_LIMIT_S.get(mode)
    if total is not None and total_soft is not None:
        total_f = float(total)
        if total_f > total_soft and total_f <= LOOP_CLOSE_TOTAL_CEILING_S:
            warnings.append(
                {
                    "stage": "total",
                    "elapsed_s": round(total_f, 2),
                    "soft_limit_s": total_soft,
                    "level": "soft",
                }
            )
    return warnings
