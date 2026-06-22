# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU infinite-loop close manifest (Loop R66)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.cpu_cert_utils import (  # noqa: E402
    cpu_bench_reference_shape_contract,
    cpu_bench_workload_reference_map,
    cpu_demo_loop_manifest,
    cpu_loop_close_manifest,
    parse_json_marker,
    parse_loop_close_json,
    parse_mlir_bench_profile_json,
    validate_loop_close_archive,
)
from scripts.cpu_bench_shapes import bench_shape_label  # noqa: E402
from scripts.bench_fused_vs_mkl_baseline import (  # noqa: E402
    CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON,
)


def _sample_mlir_bench_profile_stage():
    rows = [
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
    return {
        "ok": True,
        "rows": rows,
        "profile": {
            "bench_quick": True,
            "shape_contract_ok": True,
            "shape_validation_errors": [],
            "profile_ok": True,
            "concat_matmul": {"contract_ok": True},
        },
    }


def test_cpu_loop_close_manifest_make_targets_are_documented():
    allowed = {
        "test-cpu-demos",
        "test-cpu-mlir-bench-contract",
        "test-cpu-cert-e2e-profile",
        "test-cpu-loop-close-archive",
    }
    for entry in cpu_loop_close_manifest():
        assert entry["make"] in allowed


def test_bench_workloads_covered_by_reference_manifest():
    manifest_by_id = {e["id"]: e for e in cpu_demo_loop_manifest()}
    for workload, manifest_id in cpu_bench_workload_reference_map().items():
        assert manifest_id in manifest_by_id, (
            f"bench workload {workload} missing reference manifest id {manifest_id}"
        )
        script = _REPO / manifest_by_id[manifest_id]["script"]
        assert script.is_file(), f"missing reference script for {workload}: {script}"


def test_bench_reference_shape_contract_aligns_with_workload_map():
    shape_contract = cpu_bench_reference_shape_contract()
    ref_map = cpu_bench_workload_reference_map()
    assert set(shape_contract.keys()) == set(ref_map.keys())
    for workload, spec in shape_contract.items():
        fields = spec["dim_fields"]
        bench = spec["bench_quick"]
        reference = spec["reference_quick"]
        bench_full = spec["bench_full"]
        assert len(fields) == len(bench) == len(reference) == len(bench_full)
        for b, r in zip(bench, reference):
            assert r <= b, f"{workload}: reference dim {r} must be <= bench quick {b}"
        for q, f in zip(bench, bench_full):
            assert q <= f, f"{workload}: quick dim {q} must be <= bench full {f}"


def test_parse_json_marker_extracts_loop_close_payload():
    payload = '{"ok": true, "mode": "quick"}'
    text = f"noise\nYIRAGE_CPU_LOOP_CLOSE_JSON_BEGIN\n{payload}\nYIRAGE_CPU_LOOP_CLOSE_JSON_END\n"
    parsed = parse_json_marker(
        text,
        "YIRAGE_CPU_LOOP_CLOSE_JSON_BEGIN",
        "YIRAGE_CPU_LOOP_CLOSE_JSON_END",
    )
    assert parsed is not None
    assert parsed["ok"] is True
    assert parsed["mode"] == "quick"


def test_parse_loop_close_json_helper():
    payload = '{"ok": true, "mode": "full"}'
    text = f"YIRAGE_CPU_LOOP_CLOSE_JSON_BEGIN\n{payload}\nYIRAGE_CPU_LOOP_CLOSE_JSON_END"
    parsed = parse_loop_close_json(text)
    assert parsed is not None
    assert parsed["mode"] == "full"


def test_validate_loop_close_archive_quick_minimal():
    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": _sample_mlir_bench_profile_stage(),
            "mlir_bench_contract": {"ok": True},
        },
        "profile": {"stage_elapsed_s": {"demos": 1.0}},
    }
    assert validate_loop_close_archive(report) == []


def test_validate_loop_close_archive_rejects_bad_bench_shapes():
    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": {
                "ok": True,
                "rows": [
                    {
                        "workload": "plain_matmul",
                        "shapes": "wrong",
                    }
                ],
                "profile": {
                    "bench_quick": True,
                    "shape_contract_ok": False,
                    "shape_validation_errors": ["plain_matmul: shapes 'wrong' != expected ..."],
                },
            },
            "mlir_bench_contract": {"ok": True},
        },
        "profile": {"stage_elapsed_s": {"demos": 1.0}},
    }
    errs = validate_loop_close_archive(report)
    assert any("plain_matmul" in e for e in errs)
    assert any("shape_contract_ok" in e for e in errs)


def test_validate_loop_close_archive_full_mode_uses_full_shape_labels():
    rows = [
        {
            "workload": "concat_matmul",
            "ok": True,
            "shapes": bench_shape_label("concat_matmul", quick=False),
            "mlir_jit_applicable": False,
            "mlir_jit": False,
            "mlir_jit_deferred_reason": CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON,
        },
        {
            "workload": "rms_norm_matmul",
            "ok": True,
            "shapes": bench_shape_label("rms_norm_matmul", quick=False),
            "mlir_jit": True,
        },
    ]
    report = {
        "backend": "cpu",
        "mode": "full",
        "ok": True,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": {
                "ok": True,
                "rows": rows,
                "profile": {
                    "bench_quick": False,
                    "shape_contract_ok": True,
                    "shape_validation_errors": [],
                },
            },
            "mlir_bench_contract": {"ok": True},
            "cert_e2e": {"ok": True},
        },
        "profile": {"stage_elapsed_s": {"demos": 1.0}},
    }
    assert validate_loop_close_archive(report) == []


def test_validate_loop_close_archive_full_rejects_quick_shapes():
    report = {
        "backend": "cpu",
        "mode": "full",
        "ok": True,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": {
                "ok": True,
                "rows": [
                    {
                        "workload": "plain_matmul",
                        "shapes": bench_shape_label("plain_matmul", quick=True),
                    }
                ],
                "profile": {
                    "bench_quick": False,
                    "shape_contract_ok": True,
                    "shape_validation_errors": [],
                },
            },
            "mlir_bench_contract": {"ok": True},
            "cert_e2e": {"ok": True},
        },
        "profile": {"stage_elapsed_s": {}},
    }
    errs = validate_loop_close_archive(report)
    assert any("plain_matmul" in e for e in errs)


def test_validate_loop_close_archive_rejects_ok_false():
    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": False,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": _sample_mlir_bench_profile_stage(),
            "mlir_bench_contract": {"ok": True},
        },
        "profile": {"stage_elapsed_s": {"demos": 1.0}},
    }
    errs = validate_loop_close_archive(report)
    assert any("report.ok" in e for e in errs)


def test_validate_loop_close_archive_full_requires_cert_e2e():
    report = {
        "backend": "cpu",
        "mode": "full",
        "ok": True,
        "stages": {
            "demos": {"ok": True},
            "mlir_bench_profile": _sample_mlir_bench_profile_stage(),
            "mlir_bench_contract": {"ok": True},
        },
        "profile": {"stage_elapsed_s": {}},
    }
    errs = validate_loop_close_archive(report)
    assert any("cert_e2e" in e for e in errs)


def test_loop_close_output_file_matches_marker_parse(tmp_path):
    import json

    report = {
        "backend": "cpu",
        "mode": "quick",
        "stages": {
            "demos": {"ok": True, "elapsed_s": 1.0},
            "mlir_bench_profile": {
                **_sample_mlir_bench_profile_stage(),
                "elapsed_s": 2.0,
            },
            "mlir_bench_contract": {"ok": True, "elapsed_s": 0.5},
        },
        "profile": {"stage_elapsed_s": {"demos": 1.0}},
        "ok": True,
    }
    out_file = tmp_path / "loop-close.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    from_file = json.loads(out_file.read_text(encoding="utf-8"))
    marker_text = (
        "YIRAGE_CPU_LOOP_CLOSE_JSON_BEGIN\n"
        f"{json.dumps(report)}\n"
        "YIRAGE_CPU_LOOP_CLOSE_JSON_END"
    )
    from_marker = parse_loop_close_json(marker_text)
    assert validate_loop_close_archive(from_file) == []
    assert from_marker == from_file


def test_parse_mlir_bench_profile_json_helper():
    payload = '{"ok": true, "mode": "mlir_bench_profile"}'
    text = (
        "YIRAGE_MLIR_BENCH_PROFILE_JSON_BEGIN\n"
        f"{payload}\n"
        "YIRAGE_MLIR_BENCH_PROFILE_JSON_END"
    )
    parsed = parse_mlir_bench_profile_json(text)
    assert parsed is not None
    assert parsed["mode"] == "mlir_bench_profile"


def test_reference_demos_import_shared_bench_workload_constants():
    from scripts.cpu_bench_shapes import REFERENCE_DEMO_WORKLOADS

    for script_name, workload in REFERENCE_DEMO_WORKLOADS.items():
        text = (_REPO / "demo/reference_mugraphs" / script_name).read_text()
        assert f'BENCH_WORKLOAD = "{workload}"' in text
        assert "reference_quick_dims" in text


def test_loop_close_archive_metadata_quick_sample():
    from scripts.cpu_cert_utils import loop_close_archive_metadata

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {
            "mlir_bench_profile": _sample_mlir_bench_profile_stage(),
        },
        "profile": {
            "stage_elapsed_s": {"demos": 30.0, "mlir_bench_profile": 2.0},
            "total_elapsed_s": 32.0,
            "stages_ok": 3,
            "demos_passed": 29,
            "mlir_bench_profile_ok": True,
        },
    }
    meta = loop_close_archive_metadata(report, archive_path="x.json", validation_ok=True)
    assert meta["schema"] == "loop_close_artifact_metadata_v2"
    assert meta["bench_quick"] is True
    assert meta["validation_ok"] is True
    assert meta["shape_contract_ok"] is True
    assert meta["stage_timeout_warning_count"] == 0
    assert meta["timeout_alert_pending"] is False
    assert meta["archive_path"] == "x.json"


def test_loop_close_archive_metadata_full_bench_quick_false():
    from scripts.cpu_cert_utils import loop_close_archive_metadata

    stage = _sample_mlir_bench_profile_stage()
    stage["profile"]["bench_quick"] = False
    report = {
        "backend": "cpu",
        "mode": "full",
        "ok": True,
        "stages": {"mlir_bench_profile": stage},
        "profile": {
            "stage_elapsed_s": {
                "demos": 35.0,
                "mlir_bench_profile": 25.0,
                "cert_e2e": 70.0,
            },
            "total_elapsed_s": 130.0,
            "cert_e2e_ok": True,
            "value_verify_aligned": True,
        },
    }
    meta = loop_close_archive_metadata(report, validation_ok=True)
    assert meta["bench_quick"] is False
    assert meta["mode"] == "full"
    assert meta["cert_e2e_ok"] is True


def test_validate_loop_close_archive_metadata_schema_accepts_sample():
    from scripts.cpu_cert_utils import (
        loop_close_archive_metadata,
        validate_loop_close_archive_metadata,
    )

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {"mlir_bench_profile": _sample_mlir_bench_profile_stage()},
        "profile": {
            "stage_elapsed_s": {"demos": 30.0},
            "total_elapsed_s": 30.0,
        },
    }
    meta = loop_close_archive_metadata(report, validation_ok=True)
    assert validate_loop_close_archive_metadata(meta, archive=report) == []


def test_validate_loop_close_archive_metadata_rejects_schema_mismatch():
    from scripts.cpu_cert_utils import validate_loop_close_archive_metadata

    errs = validate_loop_close_archive_metadata({"schema": "wrong"})
    assert any("schema" in e for e in errs)


def test_validate_loop_close_archive_stage_timeouts_full_archive():
    from scripts.cpu_cert_utils import validate_loop_close_archive_stage_timeouts

    report = {
        "mode": "full",
        "profile": {
            "stage_elapsed_s": {
                "demos": 40.0,
                "mlir_bench_profile": 20.0,
                "cert_e2e": 70.0,
            },
            "total_elapsed_s": 130.0,
        },
    }
    assert validate_loop_close_archive_stage_timeouts(report) == []

    slow = {
        "mode": "full",
        "profile": {
            "stage_elapsed_s": {"cert_e2e": 200.0},
            "total_elapsed_s": 200.0,
        },
    }
    errs = validate_loop_close_archive_stage_timeouts(slow)
    assert any("cert_e2e" in e for e in errs)


def test_loop_close_stage_timeout_warnings_soft_only(tmp_path):
    from scripts.cpu_cert_utils import loop_close_stage_timeout_warnings

    report = {
        "mode": "quick",
        "profile": {
            "stage_elapsed_s": {"demos": 55.0, "mlir_bench_profile": 2.0},
            "total_elapsed_s": 58.0,
        },
    }
    warnings = loop_close_stage_timeout_warnings(report)
    assert any(w["stage"] == "demos" for w in warnings)
    assert all(w["level"] == "soft" for w in warnings)


def test_loop_close_archive_hash_roundtrip(tmp_path):
    import json

    from scripts.cpu_cert_utils import (
        loop_close_archive_file_sha256,
        loop_close_archive_metadata,
        validate_loop_close_archive_hash,
    )

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {"mlir_bench_profile": _sample_mlir_bench_profile_stage()},
        "profile": {"stage_elapsed_s": {"demos": 30.0}, "total_elapsed_s": 30.0},
    }
    archive = tmp_path / "archive.json"
    archive.write_text(json.dumps(report) + "\n", encoding="utf-8")
    meta = loop_close_archive_metadata(
        report, archive_path=str(archive), validation_ok=True
    )
    assert meta["archive_sha256"] == loop_close_archive_file_sha256(str(archive))
    assert validate_loop_close_archive_hash(meta, archive_path=str(archive)) == []


def test_loop_close_timing_contract_single_source():
    from scripts.cpu_cert_utils import (
        CERT_TIMING_BASELINE_E2E_FULL_S,
        CERT_TIMING_BASELINE_LOOP_CLOSE_PROFILE_QUICK_S,
        LOOP_CLOSE_STAGE_SOFT_LIMIT_S,
        loop_close_timing_contract,
    )

    contract = loop_close_timing_contract()
    assert contract["stage_soft_limit_s"] == LOOP_CLOSE_STAGE_SOFT_LIMIT_S
    assert (
        contract["stage_soft_limit_s"]["quick"]["demos"]
        == CERT_TIMING_BASELINE_LOOP_CLOSE_PROFILE_QUICK_S
    )
    assert (
        contract["stage_soft_limit_s"]["full"]["cert_e2e"]
        == CERT_TIMING_BASELINE_E2E_FULL_S
    )


def test_loop_close_timeout_alert_payload_none_when_no_warnings():
    from scripts.cpu_cert_utils import loop_close_timeout_alert_payload

    meta = {"stage_timeout_warnings": [], "mode": "quick"}
    assert loop_close_timeout_alert_payload(meta) is None


def test_loop_close_timeout_alert_payload_when_warnings():
    from scripts.cpu_cert_utils import loop_close_timeout_alert_payload

    meta = {
        "mode": "full",
        "bench_quick": False,
        "stage_timeout_warnings": [
            {"stage": "demos", "elapsed_s": 55.0, "soft_limit_s": 50.0, "level": "soft"}
        ],
    }
    alert = loop_close_timeout_alert_payload(meta)
    assert alert is not None
    assert alert["schema"] == "loop_close_timeout_alert_v1"
    assert "demos" in alert["summary"]
    assert alert["channels"]["slack"].startswith("disabled")


def test_loop_close_archive_metadata_warning_count_matches_warnings():
    from scripts.cpu_cert_utils import loop_close_archive_metadata

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {"mlir_bench_profile": _sample_mlir_bench_profile_stage()},
        "profile": {
            "stage_elapsed_s": {"demos": 55.0, "mlir_bench_profile": 2.0},
            "total_elapsed_s": 58.0,
        },
    }
    meta = loop_close_archive_metadata(report, validation_ok=True)
    assert meta["stage_timeout_warning_count"] == len(meta["stage_timeout_warnings"])
    assert meta["stage_timeout_warning_count"] >= 1
    assert meta["timeout_alert_pending"] is True


def test_validate_loop_close_archive_metadata_requires_alert_when_warnings_post_alert():
    from scripts.cpu_cert_utils import (
        loop_close_archive_metadata,
        validate_loop_close_archive_metadata,
    )

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {"mlir_bench_profile": _sample_mlir_bench_profile_stage()},
        "profile": {
            "stage_elapsed_s": {"demos": 55.0, "mlir_bench_profile": 2.0},
            "total_elapsed_s": 58.0,
        },
    }
    meta = loop_close_archive_metadata(report, validation_ok=True)
    assert validate_loop_close_archive_metadata(meta, archive=report) == []
    errs = validate_loop_close_archive_metadata(
        meta, archive=report, require_alert_annotation=True
    )
    assert any("timeout_alert_emitted required" in e for e in errs)

    meta["timeout_alert_emitted"] = True
    meta["timeout_alert_pending"] = False
    assert (
        validate_loop_close_archive_metadata(
            meta, archive=report, require_alert_annotation=True
        )
        == []
    )


def test_emit_loop_close_timeout_alert_annotate_metadata(tmp_path):
    import json
    import subprocess

    from scripts.cpu_cert_utils import loop_close_archive_metadata

    report = {
        "backend": "cpu",
        "mode": "quick",
        "ok": True,
        "stages": {"mlir_bench_profile": _sample_mlir_bench_profile_stage()},
        "profile": {
            "stage_elapsed_s": {"demos": 55.0, "mlir_bench_profile": 2.0},
            "total_elapsed_s": 58.0,
        },
    }
    meta_path = tmp_path / "meta.json"
    meta = loop_close_archive_metadata(report, validation_ok=True)
    meta_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/emit_loop_close_timeout_alert.py",
            str(meta_path),
            "--annotate-metadata",
        ],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    updated = json.loads(meta_path.read_text(encoding="utf-8"))
    assert updated.get("timeout_alert_emitted") is True
    assert updated.get("timeout_alert_pending") is False
