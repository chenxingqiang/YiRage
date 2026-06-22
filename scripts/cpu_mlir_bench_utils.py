# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Parse fused-vs-MKL bench JSON and build MLIR archive profiles (Loop R64)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def parse_bench_json(stdout: str) -> List[Dict[str, Any]]:
    """Parse ``YIRAGE_BENCH_JSON_BEGIN`` … ``END`` payload from bench stdout."""
    text = stdout or ""
    match = re.search(
        r"YIRAGE_BENCH_JSON_BEGIN\s*(\[.*?\])\s*YIRAGE_BENCH_JSON_END",
        text,
        re.DOTALL,
    )
    if not match:
        return []
    import json

    return json.loads(match.group(1))


def mlir_bench_profile_from_rows(
    rows: List[Dict[str, Any]],
    *,
    mlir_jit_requested: bool,
    elapsed_s: float,
    quick: bool = True,
) -> Dict[str, Any]:
    """Build archive profile from bench JSON rows."""
    from scripts.bench_fused_vs_mkl_baseline import (
        MLIR_JIT_WORKLOADS,
        validate_concat_matmul_bench_row,
        validate_mlir_jit_bench_row,
    )

    by_workload = {r.get("workload"): r for r in rows if r.get("workload")}
    rms = by_workload.get("rms_norm_matmul")
    concat = by_workload.get("concat_matmul")

    rms_errors: List[str] = []
    concat_errors: List[str] = []
    if rms and mlir_jit_requested:
        rms_errors = validate_mlir_jit_bench_row(rms)
    if concat and mlir_jit_requested:
        concat_errors = validate_concat_matmul_bench_row(concat)

    shape_errors = []
    if rows:
        from scripts.cpu_bench_shapes import validate_bench_json_row_shapes

        shape_errors = validate_bench_json_row_shapes(rows, quick=quick)

    return {
        "elapsed_s": round(elapsed_s, 2),
        "bench_quick": quick,
        "mlir_jit_requested": mlir_jit_requested,
        "workloads_run": sorted(by_workload.keys()),
        "mlir_jit_workloads": sorted(MLIR_JIT_WORKLOADS),
        "rms_norm_matmul": {
            "present": rms is not None,
            "ok": (rms or {}).get("ok"),
            "mlir_jit": (rms or {}).get("mlir_jit"),
            "mlir_jit_emit_path": (rms or {}).get("mlir_jit_emit_path"),
            "validation_errors": rms_errors,
            "contract_ok": len(rms_errors) == 0,
        },
        "concat_matmul": {
            "present": concat is not None,
            "ok": (concat or {}).get("ok"),
            "mlir_jit_applicable": (concat or {}).get("mlir_jit_applicable"),
            "concat_matmul_fast_path": (concat or {}).get("concat_matmul_fast_path"),
            "validation_errors": concat_errors,
            "contract_ok": len(concat_errors) == 0,
        },
        "shape_validation_errors": shape_errors,
        "shape_contract_ok": len(shape_errors) == 0,
        "profile_ok": (
            len(rms_errors) == 0
            and len(concat_errors) == 0
            and len(shape_errors) == 0
        ),
    }


def run_mlir_bench_profile(*, skip_rms_mlir: bool = False, quick: bool = True) -> Dict[str, Any]:
    """Run quick bench with MLIR contract validation; returns cert-stage payload."""
    import os
    import subprocess
    import sys
    import time

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    use_mlir = not skip_rms_mlir
    workloads = ["concat_matmul", "rms_norm_matmul"] if use_mlir else ["concat_matmul"]
    cmd = [
        sys.executable,
        "scripts/bench_fused_vs_mkl_baseline.py",
        "--json",
        "--workloads",
        *workloads,
    ]
    if quick:
        cmd.append("--quick")
    else:
        cmd.append("--full")
    if use_mlir:
        cmd.append("--mlir-jit")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    rows = parse_bench_json(proc.stdout + proc.stderr)
    elapsed_s = round(time.perf_counter() - t0, 2)
    profile = mlir_bench_profile_from_rows(
        rows,
        mlir_jit_requested=use_mlir,
        elapsed_s=elapsed_s,
        quick=quick,
    )
    try:
        from yirage.kernel.cpu_mlir_jit import is_mlir_jit_available
    except Exception:
        mlir_available = False
    else:
        mlir_available = is_mlir_jit_available()
    profile["mlir_jit_available"] = mlir_available
    if use_mlir and not mlir_available:
        profile["rms_norm_matmul"]["contract_skipped"] = True
        profile["rms_norm_matmul"]["contract_ok"] = None
        profile["profile_ok"] = (
            profile["concat_matmul"]["contract_ok"]
            and profile.get("shape_contract_ok", True)
        )
    elif use_mlir and mlir_available:
        profile["rms_norm_matmul"]["contract_skipped"] = False
        profile["profile_ok"] = (
            profile["concat_matmul"]["contract_ok"]
            and profile["rms_norm_matmul"]["contract_ok"]
            and profile.get("shape_contract_ok", True)
        )
    stage_ok = proc.returncode == 0 and bool(rows)
    profile_ok = profile.get("profile_ok") is not False
    return {
        "ok": stage_ok and profile_ok,
        "elapsed_s": elapsed_s,
        "returncode": proc.returncode,
        "profile": profile,
        "rows": rows,
    }


def validate_mlir_bench_profile_archive(
    report: Dict[str, Any],
    *,
    bench_quick: bool = True,
) -> List[str]:
    """Validate archived MLIR bench profile JSON (Loop R72/R73)."""
    from scripts.cpu_bench_shapes import validate_bench_json_row_shapes
    from scripts.cpu_cert_utils import _validate_shape_validation_errors_field

    errors: List[str] = []
    if report.get("backend") != "cpu":
        errors.append("backend must be cpu")
    if report.get("mode") != "mlir_bench_profile":
        errors.append("mode must be mlir_bench_profile")
    if report.get("ok") is not True:
        errors.append("report.ok must be true")

    stage = report.get("stage")
    if not isinstance(stage, dict):
        errors.append("stage must be a dict")
        return errors
    rows = stage.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("stage.rows must be a non-empty list")
        return errors

    errors.extend(validate_bench_json_row_shapes(rows, quick=bench_quick))

    profile = report.get("profile") or stage.get("profile") or {}
    errors.extend(_validate_shape_validation_errors_field(profile, prefix="profile"))
    if profile.get("shape_contract_ok") is not True:
        errors.append("profile.shape_contract_ok must be true")
    if profile.get("profile_ok") is False:
        errors.append("profile.profile_ok must not be false")
    profile_bench_quick = profile.get("bench_quick")
    if profile_bench_quick is not None and profile_bench_quick is not bench_quick:
        errors.append(
            f"profile.bench_quick {profile_bench_quick!r} != expected {bench_quick!r}"
        )
    return errors


def load_mlir_bench_profile_archive(path: str) -> Dict[str, Any]:
    """Load MLIR bench profile JSON from ``--output`` file."""
    import json

    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mlir_profile_file_sha256(path: str) -> str:
    """SHA-256 digest of MLIR bench profile JSON (R77, same as loop-close archive hash)."""
    from scripts.cpu_cert_utils import loop_close_archive_file_sha256

    return loop_close_archive_file_sha256(path)


MLIR_CI_BUNDLE_MANIFEST_SCHEMA = "mlir_ci_bundle_manifest_v3"
MLIR_CI_BUNDLE_WORKFLOWS = ("cpu-mlir-jit-contract", "cpu-mlir-ci-nightly")
MLIR_CI_BUNDLE_MANIFEST_REQUIRED_KEYS = (
    "schema",
    "workflow",
    "bench_quick",
    "profile_sha256",
    "shape_contract_ok",
    "shape_validation_errors_count",
    "shape_validation_errors_summary",
    "files",
)


def mlir_ci_bundle_shape_summary(profile_report: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize profile shape_validation_errors for bundle manifest (R79)."""
    profile = profile_report.get("profile") or {}
    errors = profile.get("shape_validation_errors")
    if not isinstance(errors, list):
        errors = []
    return {
        "shape_contract_ok": profile.get("shape_contract_ok"),
        "shape_validation_errors_count": len(errors),
        "shape_validation_errors_summary": errors[:5],
    }


def build_mlir_ci_bundle_manifest(
    bundle_dir: str,
    *,
    workflow: str,
    run_id: str | None = None,
    sha: str | None = None,
) -> Dict[str, Any]:
    """Build MLIR CI bundle manifest from on-disk profile JSON (R82 shared builder)."""
    import json
    import os

    if workflow not in MLIR_CI_BUNDLE_WORKFLOWS:
        raise ValueError(f"workflow must be one of {MLIR_CI_BUNDLE_WORKFLOWS!r}")
    profile_path = os.path.join(bundle_dir, "mlir-bench-profile.json")
    with open(profile_path, encoding="utf-8") as fh:
        profile_report = json.load(fh)
    bench_quick = (profile_report.get("profile") or {}).get("bench_quick", True)
    manifest: Dict[str, Any] = {
        "schema": MLIR_CI_BUNDLE_MANIFEST_SCHEMA,
        "workflow": workflow,
        "bench_quick": bench_quick,
        "profile_sha256": mlir_profile_file_sha256(profile_path),
        **mlir_ci_bundle_shape_summary(profile_report),
        "files": ["mlir-dialect-smoke.log", "mlir-bench-profile.json"],
    }
    if run_id is not None:
        manifest["run_id"] = run_id
    if sha is not None:
        manifest["sha"] = sha
    return manifest


def write_mlir_ci_bundle_manifest(
    bundle_dir: str,
    *,
    workflow: str,
    run_id: str | None = None,
    sha: str | None = None,
) -> str:
    """Write ``bundle-manifest.json`` under ``bundle_dir``; returns manifest path."""
    import json
    import os

    manifest = build_mlir_ci_bundle_manifest(
        bundle_dir, workflow=workflow, run_id=run_id, sha=sha
    )
    path = os.path.join(bundle_dir, "bundle-manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return path


def simulate_downloaded_mlir_ci_bundle_validate(
    source_dir: str,
    dest_dir: str,
) -> List[str]:
    """Copy bundle to a download path and validate (R82 nightly/PR regression)."""
    import os
    import shutil

    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    return validate_mlir_ci_bundle(dest_dir)


def validate_mlir_ci_bundle_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Validate MLIR CI bundle manifest schema (Loop R78)."""
    errors: List[str] = []
    if manifest.get("schema") != MLIR_CI_BUNDLE_MANIFEST_SCHEMA:
        errors.append(f"schema must be {MLIR_CI_BUNDLE_MANIFEST_SCHEMA!r}")
    for key in MLIR_CI_BUNDLE_MANIFEST_REQUIRED_KEYS:
        if key not in manifest:
            errors.append(f"missing manifest field {key}")
    if manifest.get("workflow") not in MLIR_CI_BUNDLE_WORKFLOWS:
        errors.append(
            f"bundle-manifest.workflow must be one of {MLIR_CI_BUNDLE_WORKFLOWS!r}"
        )
    if manifest.get("bench_quick") is not None and not isinstance(
        manifest.get("bench_quick"), bool
    ):
        errors.append("bundle-manifest.bench_quick must be a bool")
    shape_ok = manifest.get("shape_contract_ok")
    if shape_ok is not None and not isinstance(shape_ok, bool):
        errors.append("bundle-manifest.shape_contract_ok must be a bool")
    err_count = manifest.get("shape_validation_errors_count")
    if err_count is not None:
        if not isinstance(err_count, int) or err_count < 0:
            errors.append(
                "bundle-manifest.shape_validation_errors_count must be a non-negative int"
            )
    err_summary = manifest.get("shape_validation_errors_summary")
    if err_summary is not None and not isinstance(err_summary, list):
        errors.append("bundle-manifest.shape_validation_errors_summary must be a list")
    if err_count is not None and err_summary is not None:
        if err_count == 0 and err_summary:
            errors.append(
                "bundle-manifest.shape_validation_errors_summary must be empty when count is 0"
            )
        if err_count > 0 and not err_summary:
            errors.append(
                "bundle-manifest.shape_validation_errors_summary required when count > 0"
            )
        if err_summary and len(err_summary) > min(err_count, 5):
            errors.append(
                "bundle-manifest.shape_validation_errors_summary exceeds 5 entries"
            )
    profile_sha = manifest.get("profile_sha256")
    if profile_sha is not None and not isinstance(profile_sha, str):
        errors.append("bundle-manifest.profile_sha256 must be a string")
    files = manifest.get("files")
    if files is not None and not isinstance(files, list):
        errors.append("bundle-manifest.files must be a list")
    return errors


def validate_mlir_ci_bundle(bundle_dir: str) -> List[str]:
    """Validate MLIR CI artifact bundle (manifest + profile + dialect log, R72)."""
    import json
    import os

    errors: List[str] = []
    manifest_path = os.path.join(bundle_dir, "bundle-manifest.json")
    if not os.path.isfile(manifest_path):
        errors.append("bundle-manifest.json missing")
        return errors

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    errors.extend(validate_mlir_ci_bundle_manifest(manifest))

    manifest_bench_quick = manifest.get("bench_quick")
    manifest_profile_sha = manifest.get("profile_sha256")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append("bundle-manifest.files must be a non-empty list")
        return errors

    for name in files:
        path = os.path.join(bundle_dir, name)
        if not os.path.isfile(path):
            errors.append(f"missing bundle file {name}")

    profile_path = os.path.join(bundle_dir, "mlir-bench-profile.json")
    profile_bench_quick = None
    if os.path.isfile(profile_path):
        report = load_mlir_bench_profile_archive(profile_path)
        profile_bench_quick = (report.get("profile") or {}).get("bench_quick")
        if profile_bench_quick is None:
            profile_bench_quick = (report.get("stage") or {}).get("profile", {}).get(
                "bench_quick"
            )
        errors.extend(validate_mlir_bench_profile_archive(report))
        if manifest_bench_quick is not None and profile_bench_quick is not None:
            if manifest_bench_quick is not profile_bench_quick:
                errors.append(
                    f"bundle-manifest.bench_quick {manifest_bench_quick!r} "
                    f"!= profile.bench_quick {profile_bench_quick!r}"
                )
        elif manifest_bench_quick is None:
            errors.append("bundle-manifest.bench_quick required (R75)")
        if manifest_profile_sha is not None:
            actual = mlir_profile_file_sha256(profile_path)
            if actual != manifest_profile_sha:
                errors.append(
                    f"bundle-manifest.profile_sha256 {manifest_profile_sha[:12]}... "
                    f"!= file {actual[:12]}..."
                )
        elif os.path.isfile(profile_path):
            errors.append("bundle-manifest.profile_sha256 required (R77)")
        profile = report.get("profile") or {}
        shape_summary = mlir_ci_bundle_shape_summary(report)
        if manifest.get("shape_contract_ok") is not None:
            if manifest.get("shape_contract_ok") is not profile.get("shape_contract_ok"):
                errors.append(
                    "bundle-manifest.shape_contract_ok must match profile.shape_contract_ok"
                )
        profile_errors = profile.get("shape_validation_errors") or []
        if not isinstance(profile_errors, list):
            profile_errors = []
        expected_count = len(profile_errors)
        manifest_count = manifest.get("shape_validation_errors_count")
        if manifest_count is not None and manifest_count != expected_count:
            errors.append(
                f"bundle-manifest.shape_validation_errors_count {manifest_count} "
                f"!= profile count {expected_count}"
            )
        manifest_summary = manifest.get("shape_validation_errors_summary")
        if manifest_summary is not None:
            expected_summary = shape_summary["shape_validation_errors_summary"]
            if manifest_summary != expected_summary:
                errors.append(
                    "bundle-manifest.shape_validation_errors_summary must match "
                    "profile shape_validation_errors (truncated to 5)"
                )

    dialect_log = os.path.join(bundle_dir, "mlir-dialect-smoke.log")
    if os.path.isfile(dialect_log):
        with open(dialect_log, encoding="utf-8") as fh:
            log_text = fh.read()
        if "passed" not in log_text.lower():
            errors.append("mlir-dialect-smoke.log must contain pytest passed summary")
        if " failed" in log_text.lower() or "FAILED" in log_text:
            errors.append("mlir-dialect-smoke.log must not contain failed tests")

    return errors
