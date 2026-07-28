# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S17: yirage_maca multi-step decode generation latency archive.

Simulates ``decode_steps`` autoregressive decoder passes (batch=1) with engine
reference vs RF hybrid. Uses ``backend=yirage_maca`` when MetaX VM build is
available; otherwise torch hybrid + ``MacaServingRfSpec`` meta (CPU CI gate).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .bench_archive import ServingBenchArchive, ServingBenchArchiveRow
from .exec_backend import BACKEND_TORCH, BACKEND_YIRAGE_MACA
from .hybrid_model import HybridModelOverride
from .maca_exec import is_yirage_maca_available
from .maca_serving_meta import MacaServingRfSpec, maca_serving_present
from .torch_engine import TorchEngineModel
from .torch_exec import bench_forward, require_torch
from .yirage_maca_e2e import _layer_step_meta_has_maca_serving


def resolve_yirage_maca_generation_backend() -> str:
    """Pick execution backend for generation bench (maca when built, else torch)."""
    if is_yirage_maca_available():
        return BACKEND_YIRAGE_MACA
    return BACKEND_TORCH


@dataclass(frozen=True)
class YirageMacaGenerationReport:
    """Multi-step decode loop parity + per-step latency summary."""

    parity_ok: bool
    decode_steps: int
    backend_used: str
    yirage_maca_used: bool
    maca_meta_bridged: bool
    rf_layer_ids: List[int]
    device: str
    num_layers: int
    hidden_size: int
    engine_per_step_ms: float
    hybrid_per_step_ms: float
    engine_loop_ms: float
    hybrid_loop_ms: float
    warp_size: int
    plugin: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "decode_steps": self.decode_steps,
            "backend_used": self.backend_used,
            "yirage_maca_used": self.yirage_maca_used,
            "maca_meta_bridged": self.maca_meta_bridged,
            "rf_layer_ids": list(self.rf_layer_ids),
            "device": self.device,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "engine_per_step_ms": round(self.engine_per_step_ms, 4),
            "hybrid_per_step_ms": round(self.hybrid_per_step_ms, 4),
            "engine_loop_ms": round(self.engine_loop_ms, 4),
            "hybrid_loop_ms": round(self.hybrid_loop_ms, 4),
            "speedup_vs_engine": round(
                self.engine_per_step_ms / max(self.hybrid_per_step_ms, 1e-9), 4
            ),
            "warp_size": self.warp_size,
            "plugin": self.plugin,
        }


def _run_decode_loop(
    *,
    model: TorchEngineModel,
    hybrid: HybridModelOverride,
    x,
    meta: Dict[str, Any],
    decode_steps: int,
    forward_fn,
) -> tuple[Any, float]:
    """Run ``decode_steps`` forwards; return final hidden and total elapsed seconds."""
    require_torch()
    t0 = time.perf_counter()
    h = x
    with __import__("torch").no_grad():
        for _ in range(int(decode_steps)):
            h = forward_fn(h)
    elapsed = time.perf_counter() - t0
    return h, elapsed


def run_yirage_maca_generation_decode_loop(
    *,
    maca_spec: Optional[MacaServingRfSpec] = None,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    decode_steps: int = 4,
    seed: int = 0,
    backend: Optional[str] = None,
    warmup: int = 1,
    bench_iters: int = 1,
) -> YirageMacaGenerationReport:
    """Multi-step decode parity + measured loop latency (batch=1)."""
    require_torch()
    import torch

    spec = maca_spec or MacaServingRfSpec()
    be = backend or resolve_yirage_maca_generation_backend()
    if be == BACKEND_YIRAGE_MACA and not is_yirage_maca_available():
        be = BACKEND_TORCH

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(
        model,
        max_rf_mlp_layers=num_layers,
        mlp_backend=be if be != BACKEND_TORCH else None,
    )
    meta = spec.as_rf_meta(sm_budget=spec.sm_count - spec.reserved_aux_sms)
    x = torch.randn(1, hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        ref_h, eng_elapsed = _run_decode_loop(
            model=model,
            hybrid=hybrid,
            x=x,
            meta=meta,
            decode_steps=decode_steps,
            forward_fn=lambda h: model.forward_engine_full(h),
        )
        got_h, hyb_elapsed = _run_decode_loop(
            model=model,
            hybrid=hybrid,
            x=x,
            meta=meta,
            decode_steps=decode_steps,
            forward_fn=lambda h: hybrid.forward(h, rf_meta=meta).hidden,
        )
        got_once = hybrid.forward(x, rf_meta=meta)

    parity_ok = bool(torch.allclose(got_h, ref_h, rtol=0.05, atol=0.05))
    steps = max(int(decode_steps), 1)
    eng_per = eng_elapsed / steps
    hyb_per = hyb_elapsed / steps

    if warmup > 0 or bench_iters > 1:
        for _ in range(warmup):
            with torch.no_grad():
                model.forward_engine_full(x)
                hybrid.forward(x, rf_meta=meta)
        eng_samples: List[float] = []
        hyb_samples: List[float] = []
        for _ in range(max(bench_iters, 1)):
            _, e = _run_decode_loop(
                model=model,
                hybrid=hybrid,
                x=x,
                meta=meta,
                decode_steps=decode_steps,
                forward_fn=lambda h: model.forward_engine_full(h),
            )
            _, h = _run_decode_loop(
                model=model,
                hybrid=hybrid,
                x=x,
                meta=meta,
                decode_steps=decode_steps,
                forward_fn=lambda h: hybrid.forward(h, rf_meta=meta).hidden,
            )
            eng_samples.append(e / steps)
            hyb_samples.append(h / steps)
        eng_per = sum(eng_samples) / len(eng_samples)
        hyb_per = sum(hyb_samples) / len(hyb_samples)
        eng_elapsed = eng_per * steps
        hyb_elapsed = hyb_per * steps

    plugin = (
        "HybridModelOverride+YirageMacaMlpCapsule+MacaServingMeta"
        if be == BACKEND_YIRAGE_MACA
        else "HybridModelOverride+MacaServingMeta+TorchDecoderMlpRfHook"
    )

    return YirageMacaGenerationReport(
        parity_ok=parity_ok,
        decode_steps=int(decode_steps),
        backend_used=be,
        yirage_maca_used=be == BACKEND_YIRAGE_MACA and is_yirage_maca_available(),
        maca_meta_bridged=maca_serving_present(meta)
        and _layer_step_meta_has_maca_serving(got_once.layer_results),
        rf_layer_ids=list(got_once.rf_layer_ids),
        device=str(model.device),
        num_layers=int(num_layers),
        hidden_size=int(hidden_size),
        engine_per_step_ms=eng_per * 1000.0,
        hybrid_per_step_ms=hyb_per * 1000.0,
        engine_loop_ms=eng_elapsed * 1000.0,
        hybrid_loop_ms=hyb_elapsed * 1000.0,
        warp_size=int(spec.warp_size),
        plugin=plugin,
    )


def run_yirage_maca_generation_bench_archive(
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    decode_steps: int = 4,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    backend: Optional[str] = None,
) -> ServingBenchArchive:
    """S17 latency archive: engine vs hybrid per decode step + full loop."""
    require_torch()
    import torch

    spec = MacaServingRfSpec()
    be = backend or resolve_yirage_maca_generation_backend()
    if be == BACKEND_YIRAGE_MACA and not is_yirage_maca_available():
        be = BACKEND_TORCH

    model = TorchEngineModel(
        num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        seed=seed,
    )
    hybrid = HybridModelOverride(
        model,
        max_rf_mlp_layers=num_layers,
        mlp_backend=be if be != BACKEND_TORCH else None,
    )
    meta = spec.as_rf_meta(sm_budget=spec.sm_count - spec.reserved_aux_sms)
    x = torch.randn(1, hidden_size, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        ref = model.forward_engine_full(x)
        got = hybrid.forward(x, rf_meta=meta)
        parity = bool(torch.allclose(got.hidden, ref, rtol=0.05, atol=0.05))

    def _engine_step():
        with torch.no_grad():
            model.forward_engine_full(x)

    def _hybrid_step():
        with torch.no_grad():
            hybrid.forward(x, rf_meta=meta)

    eng_step = bench_forward(
        _engine_step,
        name="engine_decode_step",
        warmup=warmup,
        iters=iters,
        device=model.device,
    )
    hyb_step = bench_forward(
        _hybrid_step,
        name="hybrid_decode_step",
        warmup=warmup,
        iters=iters,
        device=model.device,
    )

    report = run_yirage_maca_generation_decode_loop(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        decode_steps=decode_steps,
        seed=seed,
        backend=be,
        warmup=0,
        bench_iters=1,
    )

    archive = ServingBenchArchive(version="s18", device=model.device)
    archive.rows.append(
        ServingBenchArchiveRow(
            name=eng_step.name,
            mean_ms=eng_step.mean_ms,
            iters=eng_step.iters,
            device=eng_step.device,
            parity_ok=True,
            extras={"path": "engine", "decode_steps": decode_steps},
        )
    )
    archive.rows.append(
        ServingBenchArchiveRow(
            name=hyb_step.name,
            mean_ms=hyb_step.mean_ms,
            iters=hyb_step.iters,
            device=hyb_step.device,
            parity_ok=parity and report.parity_ok,
            extras={
                "path": "hybrid",
                "backend_used": be,
                "decode_steps": decode_steps,
                "speedup_vs_engine": eng_step.mean_ms / max(hyb_step.mean_ms, 1e-9),
                "warp_size": spec.warp_size,
            },
        )
    )
    archive.rows.append(
        ServingBenchArchiveRow(
            name="generation_loop_engine",
            mean_ms=report.engine_loop_ms,
            iters=decode_steps,
            device=str(model.device),
            parity_ok=True,
            extras={"path": "engine_loop", "decode_steps": decode_steps},
        )
    )
    archive.rows.append(
        ServingBenchArchiveRow(
            name="generation_loop_hybrid",
            mean_ms=report.hybrid_loop_ms,
            iters=decode_steps,
            device=str(model.device),
            parity_ok=report.parity_ok,
            extras={
                "path": "hybrid_loop",
                "backend_used": be,
                "decode_steps": decode_steps,
                "speedup_vs_engine": report.engine_loop_ms
                / max(report.hybrid_loop_ms, 1e-9),
            },
        )
    )
    return archive


def run_yirage_maca_generation_auto(
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    decode_steps: int = 4,
) -> YirageMacaGenerationReport:
    """Cert/demo entry: auto backend + maca meta."""
    return run_yirage_maca_generation_decode_loop(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        decode_steps=decode_steps,
        backend=None,
    )


MCPYTORCH_BASELINE_NAME = "mcPytorch_torch_engine"


@dataclass(frozen=True)
class YirageMacaGenerationBaselineSummary:
    """S18: hybrid decode latency vs mcPytorch (torch engine) baseline."""

    baseline_name: str
    baseline_decode_step_ms: float
    hybrid_decode_step_ms: float
    speedup_vs_baseline: float
    parity_ok: bool
    backend_used: str
    metax_torch: bool
    decode_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "baseline_decode_step_ms": round(self.baseline_decode_step_ms, 4),
            "hybrid_decode_step_ms": round(self.hybrid_decode_step_ms, 4),
            "speedup_vs_baseline": round(self.speedup_vs_baseline, 4),
            "parity_ok": self.parity_ok,
            "backend_used": self.backend_used,
            "metax_torch": self.metax_torch,
            "decode_steps": self.decode_steps,
        }


@dataclass
class YirageMacaGenerationBaselineArchive:
    """JSON-serializable generation bench vs mcPytorch baseline (S18)."""

    version: str
    device: str
    archive: ServingBenchArchive
    summary: YirageMacaGenerationBaselineSummary
    created_unix: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        base = self.archive.to_dict()
        base["generation_baseline_archive"] = True
        base["version"] = self.version
        base["baseline"] = self.summary.baseline_name
        base["summary"] = self.summary.to_dict()
        return base

    def write_json(self, path) -> None:
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _metax_torch_detected() -> bool:
    from .vllm_metax_plugin import is_metax_torch

    return is_metax_torch()


def run_yirage_maca_generation_mcpytorch_baseline_archive(
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    decode_steps: int = 4,
    seed: int = 0,
    warmup: int = 2,
    iters: int = 8,
    backend: Optional[str] = None,
) -> YirageMacaGenerationBaselineArchive:
    """S18 archive: RF hybrid decode step vs mcPytorch torch-engine baseline."""
    archive = run_yirage_maca_generation_bench_archive(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        decode_steps=decode_steps,
        seed=seed,
        warmup=warmup,
        iters=iters,
        backend=backend,
    )
    be = backend or resolve_yirage_maca_generation_backend()
    if be == BACKEND_YIRAGE_MACA and not is_yirage_maca_available():
        be = BACKEND_TORCH

    baseline_row = next(r for r in archive.rows if r.name == "engine_decode_step")
    hybrid_row = next(r for r in archive.rows if r.name == "hybrid_decode_step")
    speedup = baseline_row.mean_ms / max(hybrid_row.mean_ms, 1e-9)

    summary = YirageMacaGenerationBaselineSummary(
        baseline_name=MCPYTORCH_BASELINE_NAME,
        baseline_decode_step_ms=baseline_row.mean_ms,
        hybrid_decode_step_ms=hybrid_row.mean_ms,
        speedup_vs_baseline=speedup,
        parity_ok=bool(baseline_row.parity_ok and hybrid_row.parity_ok),
        backend_used=be,
        metax_torch=_metax_torch_detected(),
        decode_steps=int(decode_steps),
    )

    baseline_archive = YirageMacaGenerationBaselineArchive(
        version="s18",
        device=archive.device,
        archive=archive,
        summary=summary,
    )
    baseline_archive.archive.version = "s18"
    for row in baseline_archive.archive.rows:
        row.extras.setdefault("baseline", MCPYTORCH_BASELINE_NAME)
        if row.name == "hybrid_decode_step":
            row.extras["speedup_vs_mcpytorch"] = speedup
    return baseline_archive
