# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""MLIR CI bundle and profile archive validation (Loop R72)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.bench_fused_vs_mkl_baseline import (  # noqa: E402
    CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON,
)
from scripts.cpu_bench_shapes import bench_shape_label  # noqa: E402
from scripts.cpu_mlir_bench_utils import (  # noqa: E402
    MLIR_CI_BUNDLE_MANIFEST_SCHEMA,
    build_mlir_ci_bundle_manifest,
    mlir_ci_bundle_shape_summary,
    mlir_profile_file_sha256,
    simulate_downloaded_mlir_ci_bundle_validate,
    validate_mlir_bench_profile_archive,
    validate_mlir_ci_bundle,
    validate_mlir_ci_bundle_manifest,
    write_mlir_ci_bundle_manifest,
)


def _sample_bench_rows():
    return [
        {
            "workload": "concat_matmul",
            "ok": True,
            "shapes": bench_shape_label("concat_matmul", quick=True),
            "mlir_jit_applicable": False,
            "mlir_jit": False,
            "mlir_jit_deferred_reason": CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON,
            "concat_matmul_fast_path": True,
        },
        {
            "workload": "rms_norm_matmul",
            "ok": True,
            "shapes": bench_shape_label("rms_norm_matmul", quick=True),
            "mlir_jit": True,
        },
    ]


def _sample_mlir_profile_report():
    return {
        "backend": "cpu",
        "mode": "mlir_bench_profile",
        "ok": True,
        "stage": {
            "ok": True,
            "rows": _sample_bench_rows(),
        },
        "profile": {
            "bench_quick": True,
            "shape_contract_ok": True,
            "shape_validation_errors": [],
            "profile_ok": True,
            "concat_matmul": {"contract_ok": True},
        },
    }


def test_validate_mlir_bench_profile_archive_accepts_sample():
    assert validate_mlir_bench_profile_archive(_sample_mlir_profile_report()) == []


def test_validate_mlir_bench_profile_archive_rejects_shape_mismatch():
    report = _sample_mlir_profile_report()
    report["stage"]["rows"][0]["shapes"] = "wrong"
    errs = validate_mlir_bench_profile_archive(report)
    assert any("concat_matmul" in e for e in errs)


def _write_manifest(bundle: Path, profile_path: Path, **overrides):
    profile_report = json.loads(profile_path.read_text(encoding="utf-8"))
    shape_summary = mlir_ci_bundle_shape_summary(profile_report)
    manifest = {
        "schema": MLIR_CI_BUNDLE_MANIFEST_SCHEMA,
        "workflow": "cpu-mlir-jit-contract",
        "bench_quick": True,
        "profile_sha256": mlir_profile_file_sha256(str(profile_path)),
        **shape_summary,
        "files": ["mlir-dialect-smoke.log", "mlir-bench-profile.json"],
    }
    manifest.update(overrides)
    (bundle / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def test_validate_mlir_ci_bundle_manifest_schema_v3_accepts_sample():
    manifest = {
        "schema": MLIR_CI_BUNDLE_MANIFEST_SCHEMA,
        "workflow": "cpu-mlir-jit-contract",
        "bench_quick": True,
        "profile_sha256": "abc123",
        "shape_contract_ok": True,
        "shape_validation_errors_count": 0,
        "shape_validation_errors_summary": [],
        "files": ["mlir-dialect-smoke.log", "mlir-bench-profile.json"],
    }
    assert validate_mlir_ci_bundle_manifest(manifest) == []


def test_validate_mlir_ci_bundle_manifest_accepts_nightly_workflow():
    manifest = {
        "schema": MLIR_CI_BUNDLE_MANIFEST_SCHEMA,
        "workflow": "cpu-mlir-ci-nightly",
        "bench_quick": True,
        "profile_sha256": "abc123",
        "shape_contract_ok": True,
        "shape_validation_errors_count": 0,
        "shape_validation_errors_summary": [],
        "files": ["mlir-dialect-smoke.log", "mlir-bench-profile.json"],
    }
    assert validate_mlir_ci_bundle_manifest(manifest) == []


def test_validate_mlir_ci_bundle_manifest_schema_v3_rejects_missing_schema():
    manifest = {
        "workflow": "cpu-mlir-jit-contract",
        "bench_quick": True,
        "profile_sha256": "abc123",
        "files": ["a"],
    }
    errs = validate_mlir_ci_bundle_manifest(manifest)
    assert any("schema" in e for e in errs)


def test_validate_mlir_ci_bundle_accepts_minimal(tmp_path: Path):
    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    (bundle / "mlir-bench-profile.json").write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    profile_path = bundle / "mlir-bench-profile.json"
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    _write_manifest(bundle, profile_path, run_id="1", sha="abc")
    assert validate_mlir_ci_bundle(str(bundle)) == []


def test_validate_mlir_bench_profile_archive_rejects_shape_errors_inconsistency():
    report = _sample_mlir_profile_report()
    report["profile"]["shape_validation_errors"] = ["plain_matmul: bad shape"]
    report["profile"]["shape_contract_ok"] = True
    errs = validate_mlir_bench_profile_archive(report)
    assert any("shape_contract_ok must not be true" in e for e in errs)


def test_validate_mlir_ci_bundle_rejects_missing_profile_sha256(tmp_path: Path):
    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    (bundle / "mlir-bench-profile.json").write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    manifest = {
        "workflow": "cpu-mlir-jit-contract",
        "bench_quick": True,
        "files": ["mlir-dialect-smoke.log", "mlir-bench-profile.json"],
    }
    (bundle / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    errs = validate_mlir_ci_bundle(str(bundle))
    assert any("profile_sha256" in e for e in errs)


def test_validate_mlir_ci_bundle_rejects_missing_bench_quick(tmp_path: Path):
    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    (bundle / "mlir-bench-profile.json").write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": MLIR_CI_BUNDLE_MANIFEST_SCHEMA,
        "workflow": "cpu-mlir-jit-contract",
        "profile_sha256": mlir_profile_file_sha256(
            str(bundle / "mlir-bench-profile.json")
        ),
        "shape_contract_ok": True,
        "shape_validation_errors_count": 0,
        "shape_validation_errors_summary": [],
        "files": ["mlir-dialect-smoke.log", "mlir-bench-profile.json"],
    }
    (bundle / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    errs = validate_mlir_ci_bundle(str(bundle))
    assert any("bench_quick" in e for e in errs)


def test_validate_mlir_ci_bundle_rejects_bench_quick_mismatch(tmp_path: Path):
    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    (bundle / "mlir-bench-profile.json").write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    _write_manifest(bundle, bundle / "mlir-bench-profile.json", bench_quick=False)
    errs = validate_mlir_ci_bundle(str(bundle))
    assert any("bench_quick" in e for e in errs)


def test_validate_mlir_ci_bundle_rejects_failed_dialect_log(tmp_path: Path):
    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    profile_path = bundle / "mlir-bench-profile.json"
    profile_path.write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "FAILED tests/integration/test_fused_vs_mkl_baseline.py::test_x\n",
        encoding="utf-8",
    )
    _write_manifest(bundle, profile_path)
    errs = validate_mlir_ci_bundle(str(bundle))
    assert any("failed" in e.lower() for e in errs)


def test_mlir_ci_bundle_shape_summary_truncates_to_five():
    report = _sample_mlir_profile_report()
    report["profile"]["shape_validation_errors"] = [f"err{i}" for i in range(7)]
    report["profile"]["shape_contract_ok"] = False
    summary = mlir_ci_bundle_shape_summary(report)
    assert summary["shape_validation_errors_count"] == 7
    assert len(summary["shape_validation_errors_summary"]) == 5
    assert summary["shape_validation_errors_summary"][0] == "err0"


def test_validate_mlir_ci_bundle_rejects_shape_summary_mismatch(tmp_path: Path):
    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    profile_path = bundle / "mlir-bench-profile.json"
    profile_path.write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    _write_manifest(
        bundle,
        profile_path,
        shape_validation_errors_count=1,
        shape_validation_errors_summary=["wrong"],
        shape_contract_ok=False,
    )
    errs = validate_mlir_ci_bundle(str(bundle))
    assert any("shape_validation_errors_count" in e or "shape_validation_errors_summary" in e for e in errs)


def test_validate_mlir_ci_bundle_rejects_non_empty_shape_validation_errors(tmp_path: Path):
    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    profile_report["profile"]["shape_validation_errors"] = ["concat_matmul: shapes mismatch"]
    profile_report["profile"]["shape_contract_ok"] = False
    profile_report["ok"] = False
    profile_path = bundle / "mlir-bench-profile.json"
    profile_path.write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    _write_manifest(bundle, profile_path)
    errs = validate_mlir_ci_bundle(str(bundle))
    assert any("concat_matmul" in e for e in errs)
    assert any("report.ok" in e for e in errs)


def test_validate_mlir_ci_bundle_download_regression_copy(tmp_path: Path):
    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    profile_path = bundle / "mlir-bench-profile.json"
    profile_path.write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    _write_manifest(bundle, profile_path, run_id="1", sha="abc")
    assert validate_mlir_ci_bundle(str(bundle)) == []

    downloaded = tmp_path / "downloaded-regression-mlir"
    assert simulate_downloaded_mlir_ci_bundle_validate(str(bundle), str(downloaded)) == []
    manifest = json.loads((downloaded / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert manifest["shape_validation_errors_count"] == 0
    assert manifest["shape_contract_ok"] is True


def test_build_mlir_ci_bundle_manifest_shared_builder(tmp_path: Path):
    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    profile_path = bundle / "mlir-bench-profile.json"
    profile_path.write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    manifest = build_mlir_ci_bundle_manifest(
        str(bundle),
        workflow="cpu-mlir-ci-nightly",
        run_id="99",
        sha="deadbeef",
    )
    assert manifest["workflow"] == "cpu-mlir-ci-nightly"
    assert manifest["run_id"] == "99"
    assert manifest["shape_validation_errors_count"] == 0
    path = write_mlir_ci_bundle_manifest(
        str(bundle), workflow="cpu-mlir-jit-contract", run_id="1", sha="abc"
    )
    assert path.endswith("bundle-manifest.json")
    assert validate_mlir_ci_bundle(str(bundle)) == []


def test_build_mlir_ci_bundle_manifest_only_smoke(tmp_path: Path):
    import os
    import subprocess
    import sys

    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    (bundle / "mlir-bench-profile.json").write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_mlir_ci_bundle.py",
            "--bundle-dir",
            str(bundle),
            "--workflow",
            "cpu-mlir-jit-contract",
            "--manifest-only",
        ],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert validate_mlir_ci_bundle(str(bundle)) == []


def test_makefile_smoke_build_mlir_ci_bundle_manifest(tmp_path: Path):
    import subprocess

    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    (bundle / "mlir-bench-profile.json").write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            "make",
            "build-mlir-ci-bundle",
            f"BUNDLE={bundle}",
            "WORKFLOW=cpu-mlir-ci-nightly",
            "MANIFEST_ONLY=1",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "cpu-mlir-ci-nightly"

    proc_alias = subprocess.run(
        [
            "make",
            "smoke-build-mlir-ci-bundle-manifest",
            f"BUNDLE={bundle}",
            "WORKFLOW=cpu-mlir-jit-contract",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert proc_alias.returncode == 0, proc_alias.stderr or proc_alias.stdout
    manifest_alias = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert manifest_alias["workflow"] == "cpu-mlir-jit-contract"


def test_makefile_validate_mlir_ci_metadata_download_smoke(tmp_path: Path):
    import os
    import subprocess
    import sys

    bundle = tmp_path / "mlir-ci"
    bundle.mkdir()
    profile_report = _sample_mlir_profile_report()
    profile_path = bundle / "mlir-bench-profile.json"
    profile_path.write_text(
        json.dumps(profile_report, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle / "mlir-dialect-smoke.log").write_text(
        "============================= 2 passed in 1.0s ==============================\n",
        encoding="utf-8",
    )
    _write_manifest(bundle, profile_path, run_id="1", sha="abc")
    assert validate_mlir_ci_bundle(str(bundle)) == []

    dest = tmp_path / "downloaded-regression-mlir"
    proc = subprocess.run(
        [
            "make",
            "validate-mlir-ci-metadata-download",
            f"SRC={bundle}",
            f"DEST={dest}",
        ],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert dest.is_dir()
    assert (dest / "bundle-manifest.json").is_file()
    assert (dest / "mlir-bench-profile.json").is_file()
    assert validate_mlir_ci_bundle(str(dest)) == []
