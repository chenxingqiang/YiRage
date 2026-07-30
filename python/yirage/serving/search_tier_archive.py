# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S26: multi-tier ServingBenchArchive compare + CI archive helpers."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .bench_archive import ServingBenchArchive
from .yirage_exec import inspect_serving_search_tier

# Tier presets for nightly multi-tier archive compare (seed_verify vs full_tb_ray).
SERVING_SEARCH_TIER_PRESETS: Dict[str, Dict[str, Optional[str]]] = {
    "seed_verify": {
        "YIRAGE_SERVING_FULL_TB_SEARCH": None,
        "YIRAGE_SERVING_USE_RAY": None,
        "YIRAGE_SERVING_USE_COORDINATOR": None,
        "YIRAGE_SERVING_ACCELFORGE_PRESCREEN": None,
        "YIRAGE_SERVING_KN_MATMUL_ONLY": "1",
    },
    "full_tb_ray": {
        "YIRAGE_SERVING_FULL_TB_SEARCH": "1",
        "YIRAGE_SERVING_USE_RAY": "1",
        "YIRAGE_SERVING_USE_COORDINATOR": "1",
        "YIRAGE_SERVING_ACCELFORGE_PRESCREEN": None,
        "YIRAGE_SERVING_KN_MATMUL_ONLY": None,
    },
}

_DEFAULT_COMPARE_BASELINE = "seed_verify"
_DEFAULT_COMPARE_CANDIDATE = "full_tb_ray"


def serving_search_tier_preset_names() -> Tuple[str, ...]:
    return tuple(SERVING_SEARCH_TIER_PRESETS.keys())


@contextmanager
def serving_search_tier_preset(tier_name: str) -> Iterator[Dict[str, Any]]:
    """Apply a search-tier env preset; restore previous env on exit."""
    if tier_name not in SERVING_SEARCH_TIER_PRESETS:
        raise KeyError(f"unknown serving search tier preset: {tier_name}")
    preset = SERVING_SEARCH_TIER_PRESETS[tier_name]
    saved = {key: os.environ.get(key) for key in preset}
    try:
        for key, value in preset.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield inspect_serving_search_tier()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_serving_bench_archive(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("serving bench archive root must be a JSON object")
    return payload


def validate_serving_bench_archive(payload: Mapping[str, Any]) -> List[str]:
    """Return validation errors (empty list means OK)."""
    errors: List[str] = []
    if not payload.get("serving_bench_archive"):
        errors.append("missing serving_bench_archive=true marker")
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        errors.append("missing version string")
    device = payload.get("device")
    if not isinstance(device, str) or not device:
        errors.append("missing device string")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows must be a non-empty list")
        return errors
    e2e_rows = [r for r in rows if isinstance(r, dict) and r.get("name") == "qwen05b_yirage_e2e"]
    if not e2e_rows:
        errors.append("rows must include qwen05b_yirage_e2e summary row")
    search_tier = payload.get("search_tier")
    if search_tier is not None:
        if not isinstance(search_tier, dict):
            errors.append("search_tier must be an object when present")
        elif "tier" not in search_tier:
            errors.append("search_tier.tier required when search_tier present")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"rows[{idx}] must be an object")
            continue
        for key in ("name", "mean_ms", "iters", "device", "parity_ok"):
            if key not in row:
                errors.append(f"rows[{idx}] missing {key}")
    return errors


def validate_serving_multi_tier_bench_archive(payload: Mapping[str, Any]) -> List[str]:
    """Return validation errors for a combined multi-tier archive (empty list means OK)."""
    errors: List[str] = []
    if not payload.get("serving_multi_tier_bench_archive"):
        errors.append("missing serving_multi_tier_bench_archive=true marker")
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        errors.append("missing version string")
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict) or len(tiers) < 2:
        errors.append("tiers must be a dict with at least 2 preset archives")
        return errors
    for tier_name, tier_payload in tiers.items():
        if not isinstance(tier_payload, dict):
            errors.append(f"tiers[{tier_name}] must be an object")
            continue
        tier_errors = validate_serving_bench_archive(tier_payload)
        errors.extend(f"tiers[{tier_name}]: {err}" for err in tier_errors)
    compare = payload.get("compare")
    if not isinstance(compare, dict):
        errors.append("compare block required for multi-tier archive")
    else:
        if "baseline" not in compare or "candidate" not in compare:
            errors.append("compare must include baseline and candidate summaries")
        if compare.get("ok") is not True:
            errors.append("compare.ok must be true")
    return errors


def validate_serving_search_tier_archive(payload: Mapping[str, Any]) -> List[str]:
    """Validate single-tier or multi-tier serving search archive JSON."""
    if payload.get("serving_multi_tier_bench_archive"):
        return validate_serving_multi_tier_bench_archive(payload)
    return validate_serving_bench_archive(payload)


def is_serving_multi_tier_bench_archive(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("serving_multi_tier_bench_archive"))


def extract_tier_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize one archive for nightly tier compare."""
    rows = [r for r in payload.get("rows", []) if isinstance(r, dict)]
    e2e = next((r for r in rows if r.get("name") == "qwen05b_yirage_e2e"), {})
    layer_rows = [r for r in rows if str(r.get("name", "")).startswith("superopt_layer_")]
    superopt_ms = sum(float(r.get("mean_ms", 0.0)) for r in layer_rows)
    tier_info = dict(payload.get("search_tier") or {})
    tier = tier_info.get("tier") or e2e.get("serving_search_tier") or "unknown"
    return {
        "tier": tier,
        "version": payload.get("version"),
        "device": payload.get("device"),
        "parity_ok": bool(e2e.get("parity_ok", False)),
        "used_rf_mlp_layers": int(e2e.get("used_rf_mlp_layers", len(layer_rows))),
        "num_layers": int(e2e.get("num_layers", 0)),
        "all_rf_layers": bool(e2e.get("all_rf_layers", False)),
        "superopt_elapsed_s_total": float(e2e.get("superopt_elapsed_s_total", superopt_ms / 1000.0)),
        "superopt_ms_total": superopt_ms,
        "yirage_generate_ms": float(e2e.get("mean_ms", 0.0)),
    }


def compare_serving_search_tier_archives(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    baseline_label: str = _DEFAULT_COMPARE_BASELINE,
    candidate_label: str = _DEFAULT_COMPARE_CANDIDATE,
) -> Dict[str, Any]:
    """Compare two tier archives (typically seed_verify vs full_tb_ray)."""
    base_summary = extract_tier_summary(baseline)
    cand_summary = extract_tier_summary(candidate)
    base_superopt = max(float(base_summary["superopt_elapsed_s_total"]), 1e-9)
    cand_superopt = float(cand_summary["superopt_elapsed_s_total"])
    return {
        "ok": bool(base_summary["parity_ok"] and cand_summary["parity_ok"]),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline": base_summary,
        "candidate": cand_summary,
        "superopt_slowdown_vs_baseline": cand_superopt / base_superopt,
        "tier_changed": base_summary["tier"] != cand_summary["tier"],
    }


@dataclass
class ServingMultiTierBenchArchive:
    version: str
    device: str
    tiers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compare: Optional[Dict[str, Any]] = None
    created_unix: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "serving_multi_tier_bench_archive": True,
            "version": self.version,
            "device": self.device,
            "created_unix": self.created_unix,
            "tiers": self.tiers,
        }
        if self.compare is not None:
            payload["compare"] = self.compare
        return payload

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def build_multi_tier_bench_archive(
    tier_archives: Mapping[str, ServingBenchArchive | Mapping[str, Any]],
    *,
    version: str = "s29",
    compare_baseline: str = _DEFAULT_COMPARE_BASELINE,
    compare_candidate: str = _DEFAULT_COMPARE_CANDIDATE,
) -> ServingMultiTierBenchArchive:
    tiers: Dict[str, Dict[str, Any]] = {}
    device = "cpu"
    for name, archive in tier_archives.items():
        payload = archive.to_dict() if hasattr(archive, "to_dict") else dict(archive)
        tiers[name] = payload
        device = str(payload.get("device", device))
    multi = ServingMultiTierBenchArchive(version=version, device=device, tiers=tiers)
    if compare_baseline in tiers and compare_candidate in tiers:
        multi.compare = compare_serving_search_tier_archives(
            tiers[compare_baseline],
            tiers[compare_candidate],
            baseline_label=compare_baseline,
            candidate_label=compare_candidate,
        )
    return multi


def serving_bench_archive_metadata(
    payload: Mapping[str, Any],
    *,
    archive_path: Optional[str | Path] = None,
    validation_ok: bool = True,
    quick: bool = False,
) -> Dict[str, Any]:
    """Sidecar metadata for CI artifact upload/regression."""
    summary = extract_tier_summary(payload)
    meta: Dict[str, Any] = {
        "serving_bench_archive_metadata": True,
        "validation_ok": validation_ok,
        "quick": quick,
        "tier": summary["tier"],
        "version": payload.get("version"),
        "device": payload.get("device"),
        "parity_ok": summary["parity_ok"],
        "used_rf_mlp_layers": summary["used_rf_mlp_layers"],
        "superopt_elapsed_s_total": summary["superopt_elapsed_s_total"],
        "created_unix": payload.get("created_unix"),
    }
    if archive_path is not None:
        path = Path(archive_path)
        meta["archive_path"] = str(path)
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            meta["archive_sha256"] = digest
    return meta


def serving_multi_tier_bench_archive_metadata(
    payload: Mapping[str, Any],
    *,
    archive_path: Optional[str | Path] = None,
    validation_ok: bool = True,
    quick: bool = False,
) -> Dict[str, Any]:
    """Sidecar metadata for multi-tier nightly archive artifacts."""
    tiers = payload.get("tiers") if isinstance(payload.get("tiers"), dict) else {}
    tier_names = sorted(str(name) for name in tiers.keys())
    compare = payload.get("compare") if isinstance(payload.get("compare"), dict) else {}
    meta: Dict[str, Any] = {
        "serving_multi_tier_bench_archive_metadata": True,
        "validation_ok": validation_ok,
        "quick": quick,
        "version": payload.get("version"),
        "device": payload.get("device"),
        "tier_names": tier_names,
        "compare_ok": bool(compare.get("ok")),
        "superopt_slowdown_vs_baseline": compare.get("superopt_slowdown_vs_baseline"),
        "created_unix": payload.get("created_unix"),
    }
    if archive_path is not None:
        path = Path(archive_path)
        meta["archive_path"] = str(path)
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            meta["archive_sha256"] = digest
    return meta


def run_serving_search_tier_bench_archive_for_preset(
    tier_name: str,
    *,
    archive_version: str = "s29",
    **run_kwargs: Any,
) -> Tuple[Any, ServingBenchArchive]:
    """Run Qwen search-tier archive under a named tier preset."""
    from .exec_backend import BACKEND_YIRAGE_CPU
    from .hf_qwen_cpu_e2e import run_hf_qwen05b_search_tier_bench_archive

    run_kwargs.setdefault("mlp_backend", BACKEND_YIRAGE_CPU)
    with serving_search_tier_preset(tier_name):
        report, archive = run_hf_qwen05b_search_tier_bench_archive(
            archive_version=archive_version,
            **run_kwargs,
        )
    return report, archive


def run_serving_multi_tier_bench_archive(
    tier_names: Sequence[str] = ("seed_verify",),
    *,
    archive_version: str = "s29",
    compare_baseline: str = _DEFAULT_COMPARE_BASELINE,
    compare_candidate: str = _DEFAULT_COMPARE_CANDIDATE,
    **run_kwargs: Any,
) -> ServingMultiTierBenchArchive:
    """Run one or more tier presets and emit a combined multi-tier archive."""
    tier_archives: Dict[str, ServingBenchArchive] = {}
    device = "cpu"
    for tier_name in tier_names:
        _, archive = run_serving_search_tier_bench_archive_for_preset(
            tier_name,
            archive_version=archive_version,
            **run_kwargs,
        )
        tier_archives[tier_name] = archive
        device = archive.device
    multi = build_multi_tier_bench_archive(
        tier_archives,
        version=archive_version,
        compare_baseline=compare_baseline,
        compare_candidate=compare_candidate,
    )
    multi.device = device
    return multi
