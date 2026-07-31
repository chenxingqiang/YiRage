# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S31: Qwen decode-step bench archive validate + nightly CI helpers."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL
from .qwen_decode_bench import run_qwen_decode_bench


def load_serving_qwen_decode_bench_archive(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("decode bench archive root must be a JSON object")
    return payload


def validate_serving_qwen_decode_bench_archive(payload: Mapping[str, Any]) -> List[str]:
    """Return validation errors (empty list means OK)."""
    errors: List[str] = []
    if not payload.get("serving_qwen_decode_bench"):
        errors.append("missing serving_qwen_decode_bench=true marker")
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        errors.append("missing or empty version string")
    if payload.get("parity_ok") is not True:
        errors.append("parity_ok must be true for archive merge")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        errors.append("rows must contain native + yirage decode bench entries")
    else:
        names = {r.get("name") for r in rows if isinstance(r, dict)}
        if "native_decode_step" not in names:
            errors.append("rows missing native_decode_step")
        if "yirage_rf_decode_step" not in names:
            errors.append("rows missing yirage_rf_decode_step")
    speedup = payload.get("speedup_yirage_vs_native")
    if not isinstance(speedup, (int, float)) or speedup <= 0:
        errors.append("speedup_yirage_vs_native must be a positive number")
    return errors


def serving_qwen_decode_bench_archive_metadata(
    payload: Mapping[str, Any],
    *,
    archive_path: str,
    validation_ok: bool,
    quick: bool = False,
) -> Dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return {
        "serving_qwen_decode_bench_archive_metadata": True,
        "archive_path": archive_path,
        "validation_ok": validation_ok,
        "quick": quick,
        "version": payload.get("version"),
        "parity_ok": payload.get("parity_ok"),
        "speedup_yirage_vs_native": payload.get("speedup_yirage_vs_native"),
        "max_rf_mlp_layers": payload.get("max_rf_mlp_layers"),
        "serving_search_tier": payload.get("serving_search_tier"),
        "archive_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "created_unix": time.time(),
    }


def run_serving_qwen_decode_bench_archive(
    *,
    model_id: str = DEFAULT_QWEN05B_MODEL,
    prompt: str = "The capital of France is",
    max_rf_mlp_layers: int = 1,
    all_rf_layers: bool = False,
    quick: bool = True,
    version: str = "s31",
) -> Dict[str, Any]:
    """Run decode bench and return JSON-serializable archive payload."""
    report = run_qwen_decode_bench(
        model_id=model_id,
        prompt=prompt,
        max_rf_mlp_layers=max_rf_mlp_layers,
        all_rf_layers=all_rf_layers,
        quick=quick,
        version=version,
    )
    payload = report.to_dict()
    errors = validate_serving_qwen_decode_bench_archive(payload)
    if errors:
        raise RuntimeError(f"decode bench archive validation failed: {errors}")
    return payload
