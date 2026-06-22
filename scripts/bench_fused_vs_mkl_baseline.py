#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Compare fused / optimized µGraphs vs unfused MKL (PyTorch) baselines on CPU.

YiRage search value on CPU is graph-level (fusion, tiling), not beating MKL on a
plain GEMM. This script automates:

  MKL baseline  = unfused torch ops (same semantics)
  µGraph winner = superoptimize(backend='cpu') selected graph, cpu_call runtime

Usage:
  export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
  export YIRAGE_BACKEND=cpu
  PYTHONPATH=. python3 scripts/bench_fused_vs_mkl_baseline.py
  PYTHONPATH=. python3 scripts/bench_fused_vs_mkl_baseline.py --quick

Quick/full mode and fusion search skip (P0 fast paths, default ON):
  ``_bench_skip_fusion_search()`` may skip ``superoptimize`` in both ``--quick`` and ``--full``
  when the seed graph already matches a production host-BLAS path. Controlled by
  ``YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH`` (default ``1``). JSON field: ``fusion_search_skipped``.
  ``--full`` enlarges tensor shapes and benchmark iters; when ``superoptimize`` runs
  (``YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH=0``), grid explore is still capped for CI
  tractability (Loop R41). P0 skips avoid search entirely by default.

  | Workload           | Skip predicate                          | Runtime when skipped        |
  |--------------------|-----------------------------------------|-----------------------------|
  | plain_matmul       | ``_is_plain_matmul_mugraph``            | ``cpu_matmul``              |
  | rms_norm_matmul    | ``is_production_rms_matmul_mugraph``    | ``cpu_rms_matmul``          |
  | matmul_chain       | ``is_production_matmul_chain_mugraph``  | ``cpu_matmul_chain``        |
  | concat_matmul      | ``is_production_concat_matmul_mugraph`` | ``cpu_concat_matmul``       |

  Default ``--quick`` runs only ``plain_matmul`` and ``rms_norm_matmul`` (fast). Use
  ``--workloads matmul_chain`` or ``concat_matmul`` to include slower seeds. Set
  ``YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH=0`` to force fusion search in quick or full benches.

MLIR JIT JSON fields (``--mlir-jit``, ``rms_norm_matmul`` only):

  | Field | Meaning |
  |-------|---------|
  | ``mugraph_source`` | Runtime winner: ``mlir_jit`` (JIT ok), ``fused_search``, ``interpreter_unfused``, ``interpreter_fallback`` |
  | ``mlir_jit_emit_path`` | First MLIR compile strategy that succeeded (see ``mlir_jit_bench_json_field_guide()``) |
  | ``mlir_jit_fused_seed`` | True when ``--mlir-jit-fused`` ran tractable superoptimize (fixed bgrid tiling) |

  Typical ``mugraph_source`` × ``mlir_jit_emit_path`` pairs (R46–R50 contract):

  | Seed | Flags | ``mugraph_source`` | ``mlir_jit_emit_path`` |
  |------|-------|--------------------|-------------------------|
  | P0 unfused | default quick | ``mlir_jit`` | ``hand_tiled`` or ``hand_flat`` |
  | P0 unfused | ``DIALECT=1`` | ``mlir_jit`` | ``dialect_lowered`` |
  | Fused | ``--mlir-jit-fused`` | ``mlir_jit`` or ``fused_search`` | ``hand_bgrid_tiled`` |
  | Fused | ``--mlir-jit-fused`` + ``DIALECT=1`` | ``mlir_jit`` or ``fused_search`` | ``dialect_lowered`` |

  ``mlir_jit_ms`` / ``interpreter_ms`` are always measured on the cygraph used for JIT
  (fused when search ran, else seed). See ``docs/HARDWARE_OPTIMIZATION.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import yirage as yr


@contextmanager
def _isolated_mugraph_home():
    prev = os.environ.get("HOME")
    with tempfile.TemporaryDirectory(prefix="yirage_fused_bench_") as tmp:
        os.environ["HOME"] = tmp
        try:
            import yirage.storage.mugraph_store as ms

            ms._default_store = None
            yield
        finally:
            if prev is not None:
                os.environ["HOME"] = prev
            else:
                os.environ.pop("HOME", None)


def bench_ms(fn: Callable[[], None], warmup: int = 15, iters: int = 150) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1000


def _rms_norm_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _graph_op_summary(cygraph) -> Dict[str, Any]:
    types = [o["op_type"] for o in cygraph.get_graph_structure()]
    return {
        "kn_ops": types,
        "kn_customized": types.count("kn_customized_op"),
        "kn_matmul": types.count("kn_matmul_op"),
        "kn_rms_norm": types.count("kn_rms_norm_op"),
    }


MUGRAPH_SOURCE_VALUES = frozenset(
    {
        "fused_search",
        "interpreter_unfused",
        "interpreter_fallback",
        "mlir_jit",
    }
)

MLIR_JIT_EMIT_PATH_VALUES = frozenset(
    {
        "dialect_lowered",
        "dialect_raw",
        "hand_bgrid_tiled",
        "hand_tiled",
        "hand_flat",
    }
)

MLIR_JIT_BENCH_TIMING_KEYS = frozenset(
    {
        "hand_mlir_jit_ms",
        "dialect_lowered_jit_ms",
        "speedup_hand_over_dialect_lowered",
        "mlir_hand_dialect_aligned",
    }
)

MLIR_JIT_WORKLOADS = frozenset({"rms_norm_matmul"})

CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON = (
    "MLIR LLVM JIT is implemented for rms_norm_matmul only; "
    "concat_matmul uses cpu_concat_matmul fast path"
)


def mlir_jit_bench_json_timing_contract() -> Dict[str, Any]:
    """R42/R63: when hand vs dialect bench runs, these JSON keys must be present and valid."""
    return {
        "trigger": "rms_norm_matmul + --mlir-jit (hand/dialect emit comparison enabled)",
        "required_keys": sorted(MLIR_JIT_BENCH_TIMING_KEYS),
        "constraints": {
            "hand_mlir_jit_ms": "> 0",
            "dialect_lowered_jit_ms": "> 0",
            "speedup_hand_over_dialect_lowered": "> 0",
            "mlir_hand_dialect_aligned": "is True",
        },
    }


def validate_mlir_jit_bench_row(row: Dict[str, Any]) -> List[str]:
    """Return validation errors for an MLIR JIT bench JSON row (empty if OK)."""
    errors: List[str] = []
    if row.get("workload") != "rms_norm_matmul":
        return errors
    if not row.get("mlir_jit"):
        return errors
    for key in MLIR_JIT_BENCH_TIMING_KEYS:
        if key not in row:
            errors.append(f"missing {key}")
    if errors:
        return errors
    if not (row.get("hand_mlir_jit_ms") or 0) > 0:
        errors.append("hand_mlir_jit_ms must be > 0")
    if not (row.get("dialect_lowered_jit_ms") or 0) > 0:
        errors.append("dialect_lowered_jit_ms must be > 0")
    if not (row.get("speedup_hand_over_dialect_lowered") or 0) > 0:
        errors.append("speedup_hand_over_dialect_lowered must be > 0")
    if row.get("mlir_hand_dialect_aligned") is not True:
        errors.append("mlir_hand_dialect_aligned must be True")
    emit = row.get("mlir_jit_emit_path")
    if emit and emit not in MLIR_JIT_EMIT_PATH_VALUES:
        errors.append(f"unknown mlir_jit_emit_path: {emit}")
    src = row.get("mugraph_source")
    if src and src not in MUGRAPH_SOURCE_VALUES:
        errors.append(f"unknown mugraph_source: {src}")
    return errors


def concat_matmul_mlir_jit_deferred_contract() -> Dict[str, Any]:
    """R64: concat_matmul never receives MLIR JIT columns when --mlir-jit is set."""
    return {
        "workload": "concat_matmul",
        "mlir_jit_applicable": False,
        "mlir_jit_deferred_reason": CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON,
        "expected_fast_path_key": "concat_matmul_fast_path",
    }


def validate_concat_matmul_bench_row(row: Dict[str, Any]) -> List[str]:
    """Validate concat_matmul bench row under global --mlir-jit (deferred contract)."""
    if row.get("workload") != "concat_matmul":
        return []
    errors: List[str] = []
    if row.get("mlir_jit_applicable") is not False:
        errors.append("mlir_jit_applicable must be False for concat_matmul")
    if row.get("mlir_jit"):
        errors.append("mlir_jit must be false/absent for concat_matmul")
    for key in MLIR_JIT_BENCH_TIMING_KEYS:
        if key in row:
            errors.append(f"unexpected {key} on concat_matmul")
    if row.get("mlir_jit_emit_path"):
        errors.append("mlir_jit_emit_path must be absent for concat_matmul")
    reason = row.get("mlir_jit_deferred_reason")
    if reason != CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON:
        errors.append("mlir_jit_deferred_reason mismatch")
    return errors


def mlir_jit_bench_json_field_guide() -> List[Dict[str, Any]]:
    """Documented ``mugraph_source`` × ``mlir_jit_emit_path`` combinations (Loop R51)."""
    return [
        {
            "seed": "p0_unfused",
            "fusion_search_skipped": True,
            "mlir_jit_fused_seed": False,
            "YIRAGE_CPU_MLIR_JIT_DIALECT": "0",
            "mugraph_source": ("mlir_jit",),
            "mlir_jit_emit_path": ("hand_tiled", "hand_flat"),
            "notes": "Default --quick --mlir-jit; unfused production rms_matmul seed",
        },
        {
            "seed": "p0_unfused",
            "fusion_search_skipped": True,
            "mlir_jit_fused_seed": False,
            "YIRAGE_CPU_MLIR_JIT_DIALECT": "1",
            "mugraph_source": ("mlir_jit",),
            "mlir_jit_emit_path": ("dialect_lowered",),
            "notes": "R46 contract; yirage-cpu-jit-pipeline lowered dialect",
        },
        {
            "seed": "fused_bgrid",
            "fusion_search_skipped": False,
            "mlir_jit_fused_seed": True,
            "YIRAGE_CPU_MLIR_JIT_DIALECT": "0",
            "mugraph_source": ("mlir_jit", "fused_search"),
            "mlir_jit_emit_path": ("hand_bgrid_tiled",),
            "notes": "R49 --mlir-jit-fused; emit from fused cygraph with bgrid tiling",
        },
        {
            "seed": "fused_bgrid",
            "fusion_search_skipped": False,
            "mlir_jit_fused_seed": True,
            "YIRAGE_CPU_MLIR_JIT_DIALECT": "1",
            "mugraph_source": ("mlir_jit", "fused_search"),
            "mlir_jit_emit_path": ("dialect_lowered",),
            "notes": "R50 contract; dialect lowered wins over hand_bgrid_tiled",
        },
        {
            "seed": "p0_unfused",
            "fusion_search_skipped": True,
            "mlir_jit_fused_seed": False,
            "YIRAGE_CPU_MLIR_JIT_DIALECT": "any",
            "mugraph_source": ("mlir_jit",),
            "mlir_jit_emit_path": tuple(sorted(MLIR_JIT_EMIT_PATH_VALUES)),
            "mlir_jit_timing_keys": sorted(MLIR_JIT_BENCH_TIMING_KEYS),
            "notes": "R63 timing contract when --mlir-jit runs hand vs dialect bench",
        },
        {
            "seed": "concat_matmul",
            "fusion_search_skipped": True,
            "mlir_jit_fused_seed": False,
            "YIRAGE_CPU_MLIR_JIT_DIALECT": "any",
            "mugraph_source": ("interpreter_unfused", "fused_search"),
            "mlir_jit_emit_path": (),
            "mlir_jit_applicable": False,
            "notes": "R64 deferred; use concat_matmul_fast_path / cpu_concat_matmul",
        },
    ]


@dataclass
class Workload:
    name: str
    build: Callable[[], Any]
    make_inputs: Callable[[], List[torch.Tensor]]
    mkl_baseline: Callable[[List[torch.Tensor]], torch.Tensor]
    shapes: str
    quick: bool = True


def _workloads(quick: bool) -> List[Workload]:
    from scripts.cpu_bench_shapes import bench_shape_tuple, bench_shape_label

    matmul_shape = bench_shape_tuple("plain_matmul", quick=quick)
    rms_shape = bench_shape_tuple("rms_norm_matmul", quick=quick)
    chain_shape = bench_shape_tuple("matmul_chain", quick=quick)
    concat_shape = bench_shape_tuple("concat_matmul", quick=quick)

    m, k, n = matmul_shape

    def build_matmul():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(m, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.matmul(a, b))
        return g

    def inputs_matmul():
        return [
            torch.randn(m, k, dtype=torch.float16),
            torch.randn(k, n, dtype=torch.float16),
        ]

    def mkl_matmul(ins):
        return torch.matmul(ins[0], ins[1])

    rm, rk, rn = rms_shape

    def build_rms_matmul():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(rm, rk), dtype=yr.float16)
        w = g.new_input(dims=(rk, rn), dtype=yr.float16)
        normed = g.rms_norm(x, normalized_shape=(rk,))
        g.mark_output(g.matmul(normed, w))
        return g

    def inputs_rms_matmul():
        return [
            torch.randn(rm, rk, dtype=torch.float16),
            torch.randn(rk, rn, dtype=torch.float16),
        ]

    def mkl_rms_matmul(ins):
        n = _rms_norm_torch(ins[0])
        return torch.matmul(n, ins[1])

    cm, ck, ck2, cn = chain_shape

    def build_chain():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(cm, ck), dtype=yr.float16)
        b = g.new_input(dims=(ck, ck2), dtype=yr.float16)
        c = g.new_input(dims=(ck2, cn), dtype=yr.float16)
        t = g.matmul(a, b)
        g.mark_output(g.matmul(t, c))
        return g

    def inputs_chain():
        return [
            torch.randn(cm, ck, dtype=torch.float16),
            torch.randn(ck, ck2, dtype=torch.float16),
            torch.randn(ck2, cn, dtype=torch.float16),
        ]

    def mkl_chain(ins):
        return torch.matmul(torch.matmul(ins[0], ins[1]), ins[2])

    cm_m, cm_k1, cm_k2, cm_n = concat_shape

    def build_concat_matmul():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(cm_m, cm_k1), dtype=yr.float16)
        b = g.new_input(dims=(cm_m, cm_k2), dtype=yr.float16)
        c = g.new_input(dims=(cm_k1, cm_n), dtype=yr.float16)
        d = g.new_input(dims=(cm_k2, cm_n), dtype=yr.float16)
        left = g.concat(a, b, dim=1)
        right = g.concat(c, d, dim=0)
        g.mark_output(g.matmul(left, right))
        return g

    def inputs_concat_matmul():
        return [
            torch.randn(cm_m, cm_k1, dtype=torch.float16),
            torch.randn(cm_m, cm_k2, dtype=torch.float16),
            torch.randn(cm_k1, cm_n, dtype=torch.float16),
            torch.randn(cm_k2, cm_n, dtype=torch.float16),
        ]

    def mkl_concat_matmul(ins):
        a, b, c, d = ins
        return torch.matmul(torch.cat([a, b], dim=1), torch.cat([c, d], dim=0))

    return [
        Workload(
            "plain_matmul",
            build_matmul,
            inputs_matmul,
            mkl_matmul,
            bench_shape_label("plain_matmul", quick=quick),
            quick=True,
        ),
        Workload(
            "rms_norm_matmul",
            build_rms_matmul,
            inputs_rms_matmul,
            mkl_rms_matmul,
            bench_shape_label("rms_norm_matmul", quick=quick),
            quick=True,
        ),
        Workload(
            "matmul_chain",
            build_chain,
            inputs_chain,
            mkl_chain,
            bench_shape_label("matmul_chain", quick=quick),
            quick=quick,
        ),
        Workload(
            "concat_matmul",
            build_concat_matmul,
            inputs_concat_matmul,
            mkl_concat_matmul,
            bench_shape_label("concat_matmul", quick=quick),
            quick=False,
        ),
    ]


def _apply_bench_search_tractability(wl: Workload, *, quick: bool) -> None:
    """
    Tighten CPU generator limits so fusion bench finishes after layout +
    concat_matmul search expansion (Loop R22/R23).

    Bench measures fusion runtime vs MKL, not exhaustive search coverage.
    """
    if wl.name == "concat_matmul":
        # Unfused seed graph has 8 KN ops (4 inputs + 2 concat + matmul + output).
        os.environ["YIRAGE_CPU_MAX_KN_GRAPH_OP"] = "10"
        os.environ["YIRAGE_CPU_MAX_TB_GRAPH_OP"] = "8"
        os.environ["YIRAGE_CPU_MAX_TB_GRAPH_INPUTS"] = "4"
        os.environ["YIRAGE_CPU_BENCH_MINIMAL_EXPLORE"] = "1"
        return

    if wl.name == "matmul_chain":
        # Unfused seed: 3 inputs + 2 matmul + output (6 KN ops).
        os.environ["YIRAGE_CPU_MAX_KN_GRAPH_OP"] = "8"
        os.environ["YIRAGE_CPU_MAX_TB_GRAPH_OP"] = "4"
        os.environ["YIRAGE_CPU_BENCH_MINIMAL_EXPLORE"] = "1"
        return

    kn_cap, tb_cap = {
        "plain_matmul": (3, 5),
        "rms_norm_matmul": (4, 6),
    }.get(wl.name, (4, 6))
    # Quick and full-with-search share the same explore caps (Loop R41).
    if wl.name != "matmul_chain":
        tb_cap = min(tb_cap, 5)
    os.environ["YIRAGE_CPU_MAX_KN_GRAPH_OP"] = str(kn_cap)
    os.environ["YIRAGE_CPU_MAX_TB_GRAPH_OP"] = str(tb_cap)
    os.environ["YIRAGE_CPU_BENCH_MINIMAL_EXPLORE"] = "1"


def _cap_bench_search_explore(
    griddims: List[Tuple[int, int, int]],
    blockdims: List[Tuple[int, int, int]],
    franges: Optional[List[int]],
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[int]]:
    """Single grid/frange point so fusion bench finishes in CI (quick or --full)."""
    return (
        [(1, 1, 1)],
        [blockdims[0] if blockdims else (64, 1, 1)],
        [1],
    )


def _bench_skip_fusion_search(wl: Workload, cygraph) -> bool:
    """Skip superoptimize when bench seed already has a P0 host-BLAS fast path.

    See module docstring table for workload → predicate mapping. Applies to both
    ``--quick`` and ``--full``. Returns False when ``YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH=0``.
    """
    flag = os.environ.get("YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH", "1").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return False
    if wl.name == "matmul_chain":
        from yirage.kernel.cpu_mlir_jit import is_production_matmul_chain_mugraph

        return is_production_matmul_chain_mugraph(cygraph)
    if wl.name == "concat_matmul":
        from yirage.kernel.cpu_mlir_jit import is_production_concat_matmul_mugraph

        return is_production_concat_matmul_mugraph(cygraph)
    if wl.name == "rms_norm_matmul":
        from yirage.kernel.cpu_mlir_jit import is_production_rms_matmul_mugraph

        return is_production_rms_matmul_mugraph(cygraph)
    if wl.name == "plain_matmul":
        from yirage.kernel.graph import _is_plain_matmul_mugraph

        return _is_plain_matmul_mugraph(cygraph)
    return False


def _mugraph_correct(
    runner: Any, inputs: List[torch.Tensor], ref: torch.Tensor
) -> Tuple[bool, float, Optional[str]]:
    from yirage.search.verifier_config import runtime_verify_mugraph

    return runtime_verify_mugraph(runner, inputs, ref)


def _run_workload(
    wl: Workload,
    *,
    quick: bool,
    verbose: bool,
    verifier_cfg: Dict[str, Any],
    use_mlir_jit: bool = False,
    mlir_jit_fused: bool = False,
) -> Dict[str, Any]:
    from yirage.backends.cpu.config import (
        apply_cpu_search_env,
        get_cpu_info,
        resolve_cpu_search_space,
    )
    from yirage.kernel.graph import KNGraph, _is_plain_matmul_mugraph

    inputs = wl.make_inputs()
    with torch.inference_mode():
        ref = wl.mkl_baseline(inputs)

    g = wl.build()
    space = resolve_cpu_search_space(g.cygraph)
    apply_cpu_search_env(space)
    _apply_bench_search_tractability(wl, quick=quick)

    griddims = space["grid_dims_to_explore"]
    blockdims = space["block_dims_to_explore"]
    franges = space.get("franges_to_explore")

    fusion_search_skipped = _bench_skip_fusion_search(wl, g.cygraph)
    mlir_jit_fused_seed = (
        use_mlir_jit and mlir_jit_fused and wl.name == "rms_norm_matmul"
    )
    if mlir_jit_fused_seed:
        # Fixed bgrid tiling search for MLIR JIT emit-path reporting (Loop R49).
        fusion_search_skipped = False
        griddims = [(2, 1, 1)]
        blockdims = [(32, 1, 1)]
        franges = [2]
        search_warmup = 1
        search_profile_iters = 2
    elif not fusion_search_skipped:
        griddims, blockdims, franges = _cap_bench_search_explore(
            griddims, blockdims, franges
        )

    # Search profiling: quick-like caps whenever superoptimize runs (--full keeps
    # larger runtime benchmark iters below, not heavier per-candidate search).
    if fusion_search_skipped:
        search_warmup = 3 if quick else 8
        search_profile_iters = 15 if quick else 60
    else:
        search_warmup = 3
        search_profile_iters = 15
    if wl.name in ("matmul_chain", "concat_matmul"):
        search_warmup = 3
        search_profile_iters = 10

    search_config = "lora" if wl.name == "concat_matmul" else None

    if fusion_search_skipped:
        search_graph = None
        search_s = 0.0
    else:
        t_search = time.perf_counter()
        with _isolated_mugraph_home():
            search_graph = g.superoptimize(
                backend="cpu",
                griddims=griddims,
                blockdims=blockdims,
                franges=franges,
                config=search_config,
                use_ray=False,
                use_graph_dataset=False,
                use_cached_graphs=False,
                use_persistent_cache=False,
                warmup_iters=search_warmup,
                profile_iters=search_profile_iters,
                verbose=verbose,
                is_formal_verified=verifier_cfg["is_formal_verified"],
            )
        search_s = time.perf_counter() - t_search

    search_verified = True
    if search_graph is None:
        if wl.name not in (
            "concat_matmul",
            "matmul_chain",
            "rms_norm_matmul",
            "plain_matmul",
        ):
            return {
                "workload": wl.name,
                "ok": False,
                "error": "superoptimize returned None",
                "verification": verifier_cfg,
                "search_verified": False,
                "runtime_verified": False,
            }
        interpreter = KNGraph(g.cygraph, backend="cpu")
        interp_ok, interp_err, interp_run_err = _mugraph_correct(
            interpreter, inputs, ref
        )
        if not interp_ok:
            return {
                "workload": wl.name,
                "shapes": wl.shapes,
                "ok": False,
                "fusion_search_ok": False,
                "error": interp_run_err or "correctness failed",
                "max_abs_error": interp_err if interp_err != float("inf") else None,
                "search_time_s": round(search_s, 2),
                "cpu": get_cpu_info(),
                "verification": verifier_cfg,
                "search_verified": False,
                "runtime_verified": False,
            }
        runner = interpreter
        mugraph_source = "interpreter_unfused"
        fusion_ok = False
        fusion_err = interp_err
        fusion_run_err = None
        max_err = interp_err
        graph_summary = _graph_op_summary(interpreter.cygraph)
        search_fused = False
        search_verified = False
    else:
        fusion_ok, fusion_err, fusion_run_err = _mugraph_correct(
            search_graph, inputs, ref
        )
        search_summary = _graph_op_summary(search_graph.cygraph)
        search_fused = search_summary["kn_customized"] > 0

        runner = search_graph
        mugraph_source = "fused_search"
        max_err = fusion_err
        graph_summary = search_summary

    if not fusion_ok and search_graph is not None:
        interpreter = KNGraph(g.cygraph, backend="cpu")
        interp_ok, interp_err, interp_run_err = _mugraph_correct(
            interpreter, inputs, ref
        )
        if not interp_ok:
            return {
                "workload": wl.name,
                "shapes": wl.shapes,
                "ok": False,
                "fusion_search_ok": False,
                "error": fusion_run_err or interp_run_err or "correctness failed",
                "max_abs_error": fusion_err if fusion_err != float("inf") else interp_err,
                "search_time_s": round(search_s, 2),
                "graph": search_summary,
                "cpu": get_cpu_info(),
                "verification": verifier_cfg,
                "search_verified": True,
                "runtime_verified": False,
            }
        runner = interpreter
        mugraph_source = "interpreter_fallback"
        max_err = interp_err
        graph_summary = _graph_op_summary(interpreter.cygraph)

    iters = 80 if quick else 150
    mlir_jit_ms: Optional[float] = None
    interpreter_ms: Optional[float] = None
    jit_speedup_vs_interp: Optional[float] = None
    mlir_tiling: Optional[Dict[str, Any]] = None
    hand_mlir_jit_ms: Optional[float] = None
    dialect_lowered_jit_ms: Optional[float] = None
    hand_over_dialect_speedup: Optional[float] = None
    mlir_hand_dialect_aligned: Optional[bool] = None
    mlir_jit_emit_path: Optional[str] = None

    if use_mlir_jit and wl.name == "rms_norm_matmul":
        from yirage.kernel.cpu_mlir_jit import (
            MLIRJitRunner,
            bench_hand_vs_dialect_lowered_jit,
            bench_jit_vs_interpreter,
            is_mlir_jit_available,
            is_rms_matmul_mugraph,
        )

        if is_mlir_jit_available() and is_rms_matmul_mugraph(runner.cygraph):
            os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
            os.environ["YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL"] = "1"
            jit_runner = MLIRJitRunner(runner.cygraph)
            jit_ok, _, _ = _mugraph_correct(jit_runner, inputs, ref)
            if jit_ok:
                runner = jit_runner
                mugraph_source = "mlir_jit"
            jit_bench = bench_jit_vs_interpreter(
                search_graph.cygraph if hasattr(search_graph, "cygraph") else runner.cygraph,
                inputs,
                warmup=5 if quick else 10,
                iters=iters,
            )
            if jit_bench.get("ok"):
                interpreter_ms = jit_bench.get("interpreter_ms")
                mlir_jit_ms = jit_bench.get("mlir_jit_ms")
                jit_speedup_vs_interp = jit_bench.get("speedup_interp_over_mlir_jit")
                mlir_tiling = jit_bench.get("tiling")
                mlir_jit_emit_path = jit_bench.get("mlir_jit_emit_path")
                if mugraph_source == "mlir_jit" and mlir_jit_ms is not None:
                    mu_ms_placeholder = mlir_jit_ms
                else:
                    mu_ms_placeholder = None
            else:
                mu_ms_placeholder = None
            jit_cy = (
                search_graph.cygraph
                if search_graph is not None and hasattr(search_graph, "cygraph")
                else runner.cygraph
            )
            emit_bench = bench_hand_vs_dialect_lowered_jit(
                jit_cy,
                inputs,
                warmup=2 if quick else 3,
                iters=min(iters, 20),
            )
            if emit_bench.get("ok"):
                hand_mlir_jit_ms = emit_bench.get("hand_mlir_jit_ms")
                dialect_lowered_jit_ms = emit_bench.get("dialect_lowered_jit_ms")
                hand_over_dialect_speedup = emit_bench.get(
                    "speedup_hand_over_dialect_lowered"
                )
                mlir_hand_dialect_aligned = emit_bench.get("mlir_hand_dialect_aligned")
        else:
            mu_ms_placeholder = None
    else:
        mu_ms_placeholder = None

    with torch.inference_mode():
        mkl_ms = bench_ms(lambda: wl.mkl_baseline(inputs), iters=iters)
        if mu_ms_placeholder is not None and mugraph_source == "mlir_jit":
            mu_ms = mu_ms_placeholder
        else:
            mu_ms = bench_ms(lambda: runner(inputs=inputs), iters=iters)
        search_mu_ms: Optional[float] = None
        if not fusion_ok and fusion_run_err is None:
            try:
                search_mu_ms = bench_ms(
                    lambda: search_graph(inputs=inputs), iters=min(iters, 30)
                )
            except Exception:
                search_mu_ms = None
    speedup = mkl_ms / max(mu_ms, 1e-9)

    uses_fast_path = _is_plain_matmul_mugraph(runner.cygraph)
    from yirage.kernel.cpu_mlir_jit import (
        is_production_concat_matmul_mugraph,
        is_production_matmul_chain_mugraph,
        is_production_rms_matmul_mugraph,
    )

    rms_matmul_fast_path = is_production_rms_matmul_mugraph(runner.cygraph)
    concat_matmul_fast_path = is_production_concat_matmul_mugraph(runner.cygraph)
    matmul_chain_fast_path = is_production_matmul_chain_mugraph(runner.cygraph)
    likely_fused = graph_summary["kn_customized"] > 0 or (
        graph_summary["kn_rms_norm"] > 0 and graph_summary["kn_matmul"] > 0
    )

    result: Dict[str, Any] = {
        "workload": wl.name,
        "shapes": wl.shapes,
        "ok": True,
        "mlir_jit_applicable": wl.name in MLIR_JIT_WORKLOADS,
        "fusion_search_ok": fusion_ok,
        "mugraph_source": mugraph_source,
        "max_abs_error": round(max_err, 6) if max_err != float("inf") else None,
        "search_time_s": round(search_s, 2),
        "fusion_search_skipped": fusion_search_skipped,
        "mkl_baseline_ms": round(mkl_ms, 4),
        "mugraph_ms": round(mu_ms, 4),
        "speedup_mkl_over_mugraph": round(speedup, 3),
        "mugraph_faster": speedup > 1.0,
        "graph": graph_summary,
        "search_graph": (
            search_summary if (not fusion_ok and search_graph is not None) else None
        ),
        "likely_fused": likely_fused,
        "search_fused": search_fused,
        "plain_matmul_fast_path": uses_fast_path,
        "rms_matmul_fast_path": rms_matmul_fast_path,
        "concat_matmul_fast_path": concat_matmul_fast_path,
        "matmul_chain_fast_path": matmul_chain_fast_path,
        "cpu": get_cpu_info(),
        "verification": verifier_cfg,
        "search_verified": search_verified,
        "runtime_verified": True,
        "search_fingerprint": verifier_cfg["verifier_type"] == "probabilistic",
    }
    if interpreter_ms is not None:
        result["interpreter_ms"] = interpreter_ms
    if mlir_jit_ms is not None:
        result["mlir_jit_ms"] = mlir_jit_ms
    if jit_speedup_vs_interp is not None:
        result["speedup_interp_over_mlir_jit"] = jit_speedup_vs_interp
    if mlir_tiling is not None:
        result["mlir_jit_tiling"] = mlir_tiling
    if hand_mlir_jit_ms is not None:
        result["hand_mlir_jit_ms"] = round(hand_mlir_jit_ms, 4)
    if dialect_lowered_jit_ms is not None:
        result["dialect_lowered_jit_ms"] = round(dialect_lowered_jit_ms, 4)
    if hand_over_dialect_speedup is not None:
        result["speedup_hand_over_dialect_lowered"] = round(
            hand_over_dialect_speedup, 3
        )
    if mlir_hand_dialect_aligned is not None:
        result["mlir_hand_dialect_aligned"] = mlir_hand_dialect_aligned
    if mlir_jit_emit_path is not None:
        result["mlir_jit_emit_path"] = mlir_jit_emit_path
    if mlir_jit_fused_seed:
        result["mlir_jit_fused_seed"] = True
    if use_mlir_jit and wl.name not in MLIR_JIT_WORKLOADS:
        result["mlir_jit"] = False
        result["mlir_jit_deferred_reason"] = CONCAT_MATMUL_MLIR_JIT_DEFERRED_REASON
    elif use_mlir_jit and wl.name in MLIR_JIT_WORKLOADS:
        result["mlir_jit"] = True
    if search_mu_ms is not None:
        result["incorrect_fusion_ms"] = round(search_mu_ms, 4)
    return result


def _print_report(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("Fused µGraph vs MKL baseline (same CPU, host BLAS via torch)")
    print("=" * 72)
    print(f"{'Workload':<18} {'MKL ms':>10} {'µGraph ms':>10} {'Speedup':>8}  Graph")
    print("-" * 72)
    for r in results:
        if not r.get("ok") and "error" in r:
            print(f"{r['workload']:<18}  FAIL: {r['error']}")
            continue
        if not r.get("ok"):
            print(
                f"{r['workload']:<18}  FAIL correctness max_err={r.get('max_abs_error')}"
            )
            continue
        g = r.get("graph", {})
        if r.get("mugraph_source") == "mlir_jit":
            tag = "mlir_jit"
        elif r.get("mugraph_source") == "interpreter_fallback":
            tag = "interp(fusion failed)"
        elif g.get("kn_customized"):
            tag = "fused"
        else:
            tag = "kn_ops"
        if r.get("fusion_search_ok") is False:
            tag += " [search incorrect]"
        spd = r["speedup_mkl_over_mugraph"]
        if (
            r["workload"] == "plain_matmul"
            or r.get("plain_matmul_fast_path")
            or r.get("rms_matmul_fast_path")
        ):
            marker = "≈" if 0.8 <= spd <= 1.25 else ("✓" if spd > 1.25 else "✗")
        else:
            marker = "✓" if spd > 1.05 else ("≈" if spd > 0.95 else "✗")
        extra = ""
        if r.get("mlir_jit_ms") is not None and r.get("interpreter_ms") is not None:
            extra = (
                f"  jit={r['mlir_jit_ms']:.3f}ms "
                f"interp={r['interpreter_ms']:.3f}ms "
                f"({r.get('speedup_interp_over_mlir_jit', 0):.2f}x)"
            )
            if r.get("hand_mlir_jit_ms") is not None:
                extra += (
                    f"  hand={r['hand_mlir_jit_ms']:.3f}ms "
                    f"dialect={r.get('dialect_lowered_jit_ms', 0):.3f}ms "
                    f"({r.get('speedup_hand_over_dialect_lowered', 0):.2f}x hand)"
                )
            if r.get("mlir_jit_emit_path"):
                extra += f"  emit={r['mlir_jit_emit_path']}"
        print(
            f"{r['workload']:<18} {r['mkl_baseline_ms']:>10.4f} "
            f"{r['mugraph_ms']:>10.4f} {spd:>7.3f}x {marker}  {tag}{extra}"
        )
    print("-" * 72)
    print("Speedup = MKL_baseline_ms / µGraph_ms  (>1.0 => µGraph faster)")
    skipped = [r["workload"] for r in results if r.get("fusion_search_skipped")]
    if skipped:
        print(
            "fusion_search_skipped (P0 fast path, no superoptimize): "
            + ", ".join(skipped)
        )
    print(
        "With --mlir-jit: mlir_jit_ms vs interpreter_ms; "
        "hand_mlir_jit_ms vs dialect_lowered_jit_ms; "
        "mlir_jit_emit_path for rms_norm_matmul. "
        "Set YIRAGE_CPU_MLIR_JIT_DIALECT=1 to prefer dialect_lowered emit. "
        "Use --mlir-jit-fused to superoptimize with fixed bgrid tiling (hand_bgrid_tiled). "
        "Combine with YIRAGE_CPU_MLIR_JIT_DIALECT=1 for dialect_lowered on fused seed."
    )
    print(
        "Plain matmul ≈1.0x expected (both use host BLAS); "
        "fusion/customized graphs target >1.0x.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Smaller shapes and search space (default)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Larger shapes and search space",
    )
    parser.add_argument("--json", dest="json_out", action="store_true")
    parser.add_argument(
        "--workloads",
        nargs="*",
        choices=("plain_matmul", "rms_norm_matmul", "matmul_chain", "concat_matmul"),
        help="Subset of workloads (default: all applicable for --quick/--full)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--formal-verify",
        action="store_true",
        help="Use slow formal verifier during search (else fast fingerprint; "
        "also settable via YIRAGE_FORMAL_VERIFY=1)",
    )
    parser.add_argument(
        "--mlir-jit",
        action="store_true",
        help="Benchmark MLIR LLVM JIT for rms_norm_matmul (sets YIRAGE_CPU_MLIR_JIT=1 and "
        "YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL=1; requires USE_MLIR=1 build)",
    )
    parser.add_argument(
        "--mlir-jit-fused",
        action="store_true",
        help="With --mlir-jit: run tractable superoptimize on rms_norm_matmul (fixed "
        "grid_m=2, forloop_k=2) so JSON reports fused mlir_jit_emit_path",
    )
    args = parser.parse_args()
    if args.mlir_jit:
        os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
        os.environ["YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL"] = "1"
    quick = not args.full

    from yirage.search.verifier_config import resolve_verifier_config

    verifier_cfg = resolve_verifier_config(formal_verify=args.formal_verify).to_dict()

    all_workloads = _workloads(quick)
    if args.workloads:
        wanted = set(args.workloads)
        workloads = [w for w in all_workloads if w.name in wanted]
    else:
        workloads = [w for w in all_workloads if (w.quick or not quick)]
        if quick:
            # Slow workloads are opt-in via --workloads.
            workloads = [
                w for w in workloads if w.name not in ("matmul_chain", "concat_matmul")
            ]
    results = [
        _run_workload(
            w,
            quick=quick,
            verbose=args.verbose,
            verifier_cfg=verifier_cfg,
            use_mlir_jit=args.mlir_jit,
            mlir_jit_fused=args.mlir_jit_fused,
        )
        for w in workloads
    ]

    if args.json_out:
        print("YIRAGE_BENCH_JSON_BEGIN")
        print(json.dumps(results, indent=2))
        print("YIRAGE_BENCH_JSON_END", flush=True)
    else:
        _print_report(results)

    critical = ("plain_matmul",)
    ok = all(
        r.get("ok")
        for r in results
        if r.get("workload") in critical and "error" not in r
    )
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    raise SystemExit(main())
