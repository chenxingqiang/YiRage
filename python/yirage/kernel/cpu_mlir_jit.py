# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU execution via YiRage MLIR → LLVM JIT (optional ``USE_MLIR=1`` build).

Semantic path for ``rms_norm + matmul`` (unfused or fused ``kn_customized_op``):
emit fused RMS (scf + math.rsqrt) + linalg matmul → LLVM JIT.

Fused ``bgraph`` M-grid tiling is lowered when present (M-tile loop on memref).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

_MLIR_JIT: Optional[Any] = None
_MLIR_IMPORT_ERROR: Optional[str] = None


@dataclass(frozen=True)
class RmsMatmulTiling:
    """Thread-block tiling extracted from fused ``kn_customized_op`` bgraph."""

    grid_m: int
    grid_y: int
    grid_z: int
    forloop_k: int
    m_tile: int
    k_tile: int
    n_dim: int

    @property
    def cache_key(self) -> Tuple[int, ...]:
        return (
            self.grid_m,
            self.grid_y,
            self.grid_z,
            self.forloop_k,
            self.m_tile,
            self.k_tile,
            self.n_dim,
        )

    @property
    def uses_loops(self) -> bool:
        return self.grid_m > 1 or self.forloop_k > 1


def _load_mlir_jit():
    global _MLIR_JIT, _MLIR_IMPORT_ERROR
    if _MLIR_JIT is not None:
        return _MLIR_JIT
    if _MLIR_IMPORT_ERROR is not None:
        return None
    try:
        from yirage import _yirage_mlir as mlir  # type: ignore

        _MLIR_JIT = mlir
        return mlir
    except Exception as exc:  # pragma: no cover - optional native ext
        _MLIR_IMPORT_ERROR = str(exc)
        return None


def is_mlir_jit_available() -> bool:
    return _load_mlir_jit() is not None


def mlir_jit_enabled() -> bool:
    return os.environ.get("YIRAGE_CPU_MLIR_JIT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def mlir_jit_experimental_enabled() -> bool:
    """LLVM JIT in ``cpu_call`` requires explicit experimental opt-in (P0 production path: TB+BLAS)."""
    if not mlir_jit_enabled():
        return False
    return os.environ.get("YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _tensor_shape_dtype(cygraph, dt) -> Tuple[Tuple[int, ...], str]:
    dims, _ = cygraph.get_input_dtensor_shape_and_stride(dt)
    return tuple(int(d) for d in dims), str(dt.dtype)


def rms_matmul_shapes_from_cygraph(cygraph) -> Optional[Tuple[int, int, int]]:
    """Return (M, K, N) for a 2-input rms_norm+matmul graph, or None."""
    inputs = cygraph.get_input_dtensors()
    if len(inputs) != 2:
        return None
    shape0, _ = _tensor_shape_dtype(cygraph, inputs[0])
    shape1, _ = _tensor_shape_dtype(cygraph, inputs[1])
    if len(shape0) != 2 or len(shape1) != 2:
        return None
    m, k = shape0
    k2, n = shape1
    if k != k2:
        return None
    return m, k, n


def _customized_bgraph(cygraph) -> Optional[Dict[str, Any]]:
    for op in cygraph.get_graph_structure():
        if op.get("op_type") == "kn_customized_op":
            return op.get("bgraph") or op.get("customized_graph")
    return None


def extract_rms_matmul_tiling(cygraph) -> Optional[RmsMatmulTiling]:
    """Read M-grid and K-forloop tiling from fused bgraph (if any)."""
    shapes = rms_matmul_shapes_from_cygraph(cygraph)
    if shapes is None:
        return None
    m, k, n = shapes
    bgraph = _customized_bgraph(cygraph)
    if not bgraph:
        return RmsMatmulTiling(1, 1, 1, 1, m, k, n)

    grid = bgraph.get("grid_dim", {})
    gx = max(1, int(grid.get("x", 1)))
    gy = max(1, int(grid.get("y", 1)))
    gz = max(1, int(grid.get("z", 1)))
    forloop_k = max(1, int(bgraph.get("forloop_range", 1)))
    m_tile = max(1, (m + gx - 1) // gx)
    k_tile = max(1, (k + forloop_k - 1) // forloop_k)
    return RmsMatmulTiling(gx, gy, gz, forloop_k, m_tile, k_tile, n)


def is_rms_matmul_mugraph(cygraph) -> bool:
    """True when cygraph is unfused or fused rms_norm+matmul."""
    if rms_matmul_shapes_from_cygraph(cygraph) is not None:
        ops = {o.get("op_type") for o in cygraph.get_graph_structure()}
        if "kn_matmul_op" in ops and (
            "kn_rms_norm_op" in ops or "kn_customized_op" in ops
        ):
            return True
    bgraph = _customized_bgraph(cygraph)
    if not bgraph:
        return False
    tb_types = {o.get("op_type") for o in bgraph.get("operators", [])}
    has_rms = (
        "tb_forloop_accum_red_ld_rms_op" in tb_types
        or "tb_rms_norm_op" in tb_types
    )
    has_matmul = "tb_matmul_op" in tb_types
    return has_rms and has_matmul and rms_matmul_shapes_from_cygraph(cygraph) is not None


_PRODUCTION_RMS_MATMUL_TB_OPS = frozenset(
    {
        "tb_input_op",
        "tb_output_op",
        "tb_matmul_op",
        "tb_rms_norm_op",
        "tb_forloop_accum_red_ld_rms_op",
        "tb_forloop_accum_no_red_op",
        "tb_div_op",
    }
)


def production_rms_matmul_fast_enabled() -> bool:
    """Host-BLAS fused rms+matmul for unfused and fused µGraphs (P1 production path)."""
    return os.environ.get("YIRAGE_CPU_RMS_MATMUL_FAST", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_KN_CONCAT_OP_TYPES = frozenset(
    {
        "kn_concat_0_op",
        "kn_concat_1_op",
        "kn_concat_2_op",
        "kn_concat_first_op_id",
    }
)


def concat_matmul_shapes_from_cygraph(
    cygraph,
) -> Optional[Tuple[int, int, int, int]]:
    """Return (M, K1, K2, N) for 4-input LoRA blocked GEMM, or None."""
    inputs = cygraph.get_input_dtensors()
    if len(inputs) != 4:
        return None
    (m, k1), _ = _tensor_shape_dtype(cygraph, inputs[0])
    (m2, k2), _ = _tensor_shape_dtype(cygraph, inputs[1])
    (k1b, n), _ = _tensor_shape_dtype(cygraph, inputs[2])
    (k2b, n2), _ = _tensor_shape_dtype(cygraph, inputs[3])
    if m != m2 or k1 != k1b or k2 != k2b or n != n2:
        return None
    return m, k1, k2, n


def _kn_concat_dim(op: Dict[str, Any]) -> Optional[int]:
    if "concat_dim" in op:
        return int(op["concat_dim"])
    ot = op.get("op_type", "")
    if ot in ("kn_concat_0_op", "kn_concat_first_op_id"):
        return 0
    if ot.startswith("kn_concat_") and ot.endswith("_op"):
        try:
            return int(ot[len("kn_concat_") : -len("_op")])
        except ValueError:
            return None
    return None


def _is_unfused_concat_matmul_mugraph(cygraph) -> bool:
    shapes = concat_matmul_shapes_from_cygraph(cygraph)
    if shapes is None:
        return False
    ops = cygraph.get_graph_structure()
    compute = [
        o
        for o in ops
        if o.get("op_type") not in ("kn_input_op", "kn_output_op")
    ]
    if len(compute) != 3:
        return False
    concats = [o for o in compute if o.get("op_type") in _KN_CONCAT_OP_TYPES]
    matmuls = [o for o in compute if o.get("op_type") == "kn_matmul_op"]
    if len(concats) != 2 or len(matmuls) != 1:
        return False
    inputs = cygraph.get_input_dtensors()
    num_dims = len(_tensor_shape_dtype(cygraph, inputs[0])[0])
    hi_dim = num_dims - 1
    lo_dim = num_dims - 2
    dims = {_kn_concat_dim(c) for c in concats}
    if dims != {hi_dim, lo_dim}:
        return False
    mat_in_guids = {t["guid"] for t in matmuls[0].get("input_tensors", [])}
    concat_out_guids = {c["output_tensors"][0]["guid"] for c in concats}
    return mat_in_guids == concat_out_guids


_PRODUCTION_CONCAT_MATMUL_TB_OPS = frozenset(
    {
        "tb_input_op",
        "tb_output_op",
        "tb_matmul_op",
        "tb_mul_op",
        "tb_add_op",
        "tb_forloop_accum_no_red_op",
        "tb_forloop_accum_red_ld_sum_op",
        "tb_concat_0_op",
        "tb_concat_1_op",
        "tb_concat_2_op",
        "tb_concat_first_op_id",
    }
)


def production_concat_matmul_fast_enabled() -> bool:
    """Host-BLAS blocked GEMM for unfused and fused LoRA concat_matmul µGraphs."""
    return os.environ.get("YIRAGE_CPU_CONCAT_MATMUL_FAST", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def is_production_concat_matmul_mugraph(cygraph) -> bool:
    """True when graph is semantically ``matmul(cat(a,b), cat(c,d))`` (4 inputs)."""
    if concat_matmul_shapes_from_cygraph(cygraph) is None:
        return False
    if _is_unfused_concat_matmul_mugraph(cygraph):
        return True
    types = [o.get("op_type") for o in cygraph.get_graph_structure()]
    compute_ops = [t for t in types if t not in ("kn_input_op", "kn_output_op")]
    if compute_ops != ["kn_customized_op"]:
        return False
    bgraph = _customized_bgraph(cygraph)
    if not bgraph:
        return False
    tb_types = {o.get("op_type") for o in bgraph.get("operators", [])}
    if not tb_types.issubset(_PRODUCTION_CONCAT_MATMUL_TB_OPS):
        return False
    return "tb_matmul_op" in tb_types


def matmul_chain_shapes_from_cygraph(
    cygraph,
) -> Optional[Tuple[int, int, int, int]]:
    """Return (M, K, K2, N) for ``matmul(matmul(a,b), c)`` with 3 inputs, or None."""
    inputs = cygraph.get_input_dtensors()
    if len(inputs) != 3:
        return None
    shape0, _ = _tensor_shape_dtype(cygraph, inputs[0])
    shape1, _ = _tensor_shape_dtype(cygraph, inputs[1])
    shape2, _ = _tensor_shape_dtype(cygraph, inputs[2])
    if len(shape0) != 2 or len(shape1) != 2 or len(shape2) != 2:
        return None
    m, k = shape0
    k2, kmid = shape1
    kmid2, n = shape2
    if k != k2 or kmid != kmid2:
        return None
    return m, k, kmid, n


def _kn_input_guids_in_order(cygraph) -> List[Any]:
    return [
        op["output_tensors"][0]["guid"]
        for op in cygraph.get_graph_structure()
        if op.get("op_type") == "kn_input_op"
    ]


def _is_unfused_matmul_chain_mugraph(cygraph) -> bool:
    if matmul_chain_shapes_from_cygraph(cygraph) is None:
        return False
    ops = cygraph.get_graph_structure()
    compute = [
        o
        for o in ops
        if o.get("op_type") not in ("kn_input_op", "kn_output_op")
    ]
    matmuls = [o for o in compute if o.get("op_type") == "kn_matmul_op"]
    if len(compute) != 2 or len(matmuls) != 2:
        return False
    in_guids = _kn_input_guids_in_order(cygraph)
    if len(in_guids) != 3:
        return False
    m0, m1 = matmuls[0], matmuls[1]
    m0_in = {t["guid"] for t in m0.get("input_tensors", [])}
    m1_in = {t["guid"] for t in m1.get("input_tensors", [])}
    m0_out = m0["output_tensors"][0]["guid"]
    return m0_in == {in_guids[0], in_guids[1]} and m1_in == {m0_out, in_guids[2]}


_PRODUCTION_MATMUL_CHAIN_TB_OPS = frozenset(
    {
        "tb_input_op",
        "tb_output_op",
        "tb_matmul_op",
        "tb_forloop_accum_no_red_op",
        "tb_forloop_accum_red_ld_sum_op",
    }
)


def production_matmul_chain_fast_enabled() -> bool:
    """Host-BLAS chained GEMM for unfused and fused 3-input matmul_chain µGraphs."""
    return os.environ.get("YIRAGE_CPU_MATMUL_CHAIN_FAST", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def is_production_matmul_chain_mugraph(cygraph) -> bool:
    """True when graph is semantically ``matmul(matmul(a,b), c)`` (3 inputs)."""
    if matmul_chain_shapes_from_cygraph(cygraph) is None:
        return False
    if _is_unfused_matmul_chain_mugraph(cygraph):
        return True
    compute_ops = [
        o.get("op_type")
        for o in cygraph.get_graph_structure()
        if o.get("op_type") not in ("kn_input_op", "kn_output_op")
    ]
    if not compute_ops or not all(
        t in ("kn_matmul_op", "kn_customized_op") for t in compute_ops
    ):
        return False
    has_kn_matmul = "kn_matmul_op" in compute_ops
    bgraph = _customized_bgraph(cygraph)
    if bgraph is not None:
        tb_types = {o.get("op_type") for o in bgraph.get("operators", [])}
        if not tb_types.issubset(_PRODUCTION_MATMUL_CHAIN_TB_OPS):
            return False
        if "tb_matmul_op" not in tb_types and not has_kn_matmul:
            return False
        return True
    return False


def is_production_rms_matmul_mugraph(cygraph) -> bool:
    """True when graph is semantically ``rms_norm(x) @ w`` (unfused or fused customized)."""
    if rms_matmul_shapes_from_cygraph(cygraph) is None:
        return False
    if not is_rms_matmul_mugraph(cygraph):
        return False
    types = [o["op_type"] for o in cygraph.get_graph_structure()]
    compute_ops = [t for t in types if t not in ("kn_input_op", "kn_output_op")]
    if compute_ops == ["kn_rms_norm_op", "kn_matmul_op"]:
        return True
    if compute_ops != ["kn_customized_op"]:
        return False
    bgraph = _customized_bgraph(cygraph)
    if not bgraph:
        return False
    tb_types = {o.get("op_type") for o in bgraph.get("operators", [])}
    if not tb_types.issubset(_PRODUCTION_RMS_MATMUL_TB_OPS):
        return False
    has_matmul = "tb_matmul_op" in tb_types
    has_rms = (
        "tb_rms_norm_op" in tb_types
        or "tb_forloop_accum_red_ld_rms_op" in tb_types
    )
    return has_matmul and has_rms


def blas_fast_path_enabled() -> bool:
    """Use host BLAS (``torch.matmul``) for large rms+matmul when JIT loops are slow."""
    return os.environ.get("YIRAGE_CPU_MLIR_JIT_BLAS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _blas_elem_threshold() -> int:
    raw = os.environ.get("YIRAGE_CPU_MLIR_JIT_BLAS_ELEMS", "262144")
    try:
        return max(1, int(raw))
    except ValueError:
        return 262144


def should_use_blas_fast_path(m: int, k: int, n: int) -> bool:
    if not blas_fast_path_enabled():
        return False
    return m * k * n >= _blas_elem_threshold()


def preserve_bgrid_tiling() -> bool:
    """When set (default), fused bgraph M/K tiling lowers to tiled hand MLIR."""
    return os.environ.get("YIRAGE_CPU_MLIR_JIT_PRESERVE_TILING", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _torch_rms_matmul(
    x: torch.Tensor, w: torch.Tensor, *, epsilon: float = 1e-6
) -> torch.Tensor:
    """Reference-quality rms_norm+matmul via host BLAS (fp32 reduction)."""
    x32 = x.float()
    scale = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + epsilon)
    normed = (x32 * scale).to(x.dtype)
    return torch.matmul(normed, w)


def _mlir_float_const(value: float, dtype: str) -> str:
    """MLIR float literal (must include '.' before 'e' exponent marker)."""
    text = f"{value:.12g}"
    if "e" in text and "." not in text.split("e", 1)[0]:
        text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text


def _use_fp32_accum(dtype: str) -> bool:
    return dtype == "f16"


def _rms_row_loop(
    m: int,
    k: int,
    *,
    dtype: str,
    epsilon: float,
    x_ref: str = "%x",
    normed_ref: str = "%normed",
    forloop_k: int = 1,
    k_tile: int = 0,
) -> str:
    """Per-row RMS norm into ``normed_ref`` (unit gamma, last-dim reduction).

    When ``forloop_k > 1``, accumulate sum-of-squares in K chunks (bgraph tiling).
    fp16 inputs use f32 accumulators for the reduction and scale.
    """
    accum_dtype = "f32" if _use_fp32_accum(dtype) else dtype
    eps_lit = _mlir_float_const(epsilon, accum_dtype)
    fk = max(1, forloop_k)
    kt = k_tile if k_tile > 0 else max(1, (k + fk - 1) // fk)

    zero = "%c0acc"
    if fk <= 1:
        vf_line = (
            f"%vf = arith.extf %v : {dtype} to {accum_dtype}"
            if accum_dtype != dtype
            else "%vf = %v"
        )
        load_store = f"""      %acc = memref.load %sum_buf[] : memref<{accum_dtype}>
        %v = memref.load {x_ref}[%i, %j] : memref<{m}x{k}x{dtype}>
        {vf_line}
        %sq = arith.mulf %vf, %vf : {accum_dtype}
        %nacc = arith.addf %acc, %sq : {accum_dtype}"""
        norm_body = ""
        if accum_dtype != dtype:
            norm_body = f"""        %v2 = memref.load {x_ref}[%i, %j2] : memref<{m}x{k}x{dtype}>
        %v2f = arith.extf %v2 : {dtype} to {accum_dtype}
        %nf = arith.mulf %v2f, %scale : {accum_dtype}
        %n = arith.truncf %nf : {accum_dtype} to {dtype}"""
        else:
            norm_body = f"""        %v2 = memref.load {x_ref}[%i, %j2] : memref<{m}x{k}x{dtype}>
        %n = arith.mulf %v2, %scale : {dtype}"""
        return f"""    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cM = arith.constant {m} : index
    %cK = arith.constant {k} : index
    {zero} = arith.constant 0.0 : {accum_dtype}
    %cKf = arith.constant {float(k)} : {accum_dtype}
    %ceps = arith.constant {eps_lit} : {accum_dtype}
    %sum_buf = memref.alloca() : memref<{accum_dtype}>
    scf.for %i = %c0 to %cM step %c1 {{
      memref.store {zero}, %sum_buf[] : memref<{accum_dtype}>
      scf.for %j = %c0 to %cK step %c1 {{
{load_store}
        memref.store %nacc, %sum_buf[] : memref<{accum_dtype}>
      }}
      %sum = memref.load %sum_buf[] : memref<{accum_dtype}>
      %mean = arith.divf %sum, %cKf : {accum_dtype}
      %me = arith.addf %mean, %ceps : {accum_dtype}
      %scale = math.rsqrt %me : {accum_dtype}
      scf.for %j2 = %c0 to %cK step %c1 {{
{norm_body}
        memref.store %n, {normed_ref}[%i, %j2] : memref<{m}x{k}x{dtype}>
      }}
    }}"""

    # K-chunked RMS: one f32 sum per row, accumulated across forloop_k tiles.
    vf_line_k = (
        f"%vf = arith.extf %v : {dtype} to {accum_dtype}"
        if accum_dtype != dtype
        else "%vf = %v"
    )
    if accum_dtype != dtype:
        norm_ext = f"""        %v2f = arith.extf %v2 : {dtype} to {accum_dtype}
        %nf = arith.mulf %v2f, %scale : {accum_dtype}
        %n = arith.truncf %nf : {accum_dtype} to {dtype}"""
    else:
        norm_ext = f"        %n = arith.mulf %v2, %scale : {dtype}"
    return f"""    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cM = arith.constant {m} : index
    %cK = arith.constant {k} : index
    %cFk = arith.constant {fk} : index
    %cKT = arith.constant {kt} : index
    {zero} = arith.constant 0.0 : {accum_dtype}
    %cKf = arith.constant {float(k)} : {accum_dtype}
    %ceps = arith.constant {eps_lit} : {accum_dtype}
    %row_sums = memref.alloca() : memref<{m}x{accum_dtype}>
    scf.for %i0 = %c0 to %cM step %c1 {{
      memref.store {zero}, %row_sums[%i0] : memref<{m}x{accum_dtype}>
    }}
    scf.for %bk = %c0 to %cFk step %c1 {{
      %k0 = arith.muli %bk, %cKT : index
      scf.for %i = %c0 to %cM step %c1 {{
        %k_end = arith.addi %k0, %cKT : index
        %k_lim = arith.minsi %k_end, %cK : index
        scf.for %j = %k0 to %k_lim step %c1 {{
          %acc = memref.load %row_sums[%i] : memref<{m}x{accum_dtype}>
          %v = memref.load {x_ref}[%i, %j] : memref<{m}x{k}x{dtype}>
          {vf_line_k}
          %sq = arith.mulf %vf, %vf : {accum_dtype}
          %nacc = arith.addf %acc, %sq : {accum_dtype}
          memref.store %nacc, %row_sums[%i] : memref<{m}x{accum_dtype}>
        }}
      }}
    }}
    scf.for %i = %c0 to %cM step %c1 {{
      %sum = memref.load %row_sums[%i] : memref<{m}x{accum_dtype}>
      %mean = arith.divf %sum, %cKf : {accum_dtype}
      %me = arith.addf %mean, %ceps : {accum_dtype}
      %scale = math.rsqrt %me : {accum_dtype}
      scf.for %j2 = %c0 to %cK step %c1 {{
        %v2 = memref.load {x_ref}[%i, %j2] : memref<{m}x{k}x{dtype}>
        {norm_ext}
        memref.store %n, {normed_ref}[%i, %j2] : memref<{m}x{k}x{dtype}>
      }}
    }}"""


def _emit_matmul_mlir(
    m: int,
    k: int,
    n: int,
    *,
    dtype: str,
    lhs_ref: str,
    rhs_ref: str,
    out_ref: str,
    m_dyn: bool = False,
    m_len_ref: str = "%m_use",
    m_row_offset_ref: Optional[str] = None,
) -> str:
    """Emit matmul; fp16 uses f32 tile accumulators.

    When ``m_row_offset_ref`` is set (bgrid M tiling), rows are indexed in the
    parent memref via ``offset + %mi`` instead of dynamic ``memref.subview``.
    """
    lhs_type = f"{m}x{k}x{dtype}"
    out_type = f"{m}x{n}x{dtype}"
    if m_dyn and m_row_offset_ref:
        row_load = f"""          %row = arith.addi {m_row_offset_ref}, %mi : index
          %a = memref.load {lhs_ref}[%row, %kk] : memref<{lhs_type}>
          %b = memref.load {rhs_ref}[%kk, %nj] : memref<{k}x{n}x{dtype}>"""
        row_store = f"""        %row = arith.addi {m_row_offset_ref}, %mi : index
        memref.store %oh, {out_ref}[%row, %nj] : memref<{out_type}>"""
    elif m_dyn:
        row_load = f"""          %a = memref.load {lhs_ref}[%mi, %kk] : memref<?x?x{dtype}, strided<[?, ?], offset: ?>>
          %b = memref.load {rhs_ref}[%kk, %nj] : memref<{k}x{n}x{dtype}>"""
        row_store = (
            f"        memref.store %oh, {out_ref}[%mi, %nj]"
            f" : memref<?x?x{dtype}, strided<[?, ?], offset: ?>>"
        )
    else:
        row_load = f"""          %a = memref.load {lhs_ref}[%mi, %kk] : memref<{lhs_type}>
          %b = memref.load {rhs_ref}[%kk, %nj] : memref<{k}x{n}x{dtype}>"""
        row_store = f"        memref.store %oh, {out_ref}[%mi, %nj] : memref<{out_type}>"

    if not _use_fp32_accum(dtype):
        if m_dyn:
            return f"""      linalg.matmul
          ins({lhs_ref}, {rhs_ref} : memref<?x?x{dtype}, strided<[?, ?], offset: ?>>, memref<{k}x{n}x{dtype}>)
          outs({out_ref} : memref<?x?x{dtype}, strided<[?, ?], offset: ?>>)"""
        return f"""    linalg.matmul
        ins({lhs_ref}, {rhs_ref} : memref<{lhs_type}>, memref<{k}x{n}x{dtype}>)
        outs({out_ref} : memref<{out_type}>)"""

    m_bound = m_len_ref if m_dyn else "%mmcM"
    return f"""    %mmc0 = arith.constant 0 : index
    %mmc1 = arith.constant 1 : index
    %mmcM = arith.constant {m} : index
    %mmcK = arith.constant {k} : index
    %mmcN = arith.constant {n} : index
    %mmc0f32 = arith.constant 0.0 : f32
    scf.for %mi = %mmc0 to {m_bound} step %mmc1 {{
      scf.for %nj = %mmc0 to %mmcN step %mmc1 {{
        %acc_final = scf.for %kk = %mmc0 to %mmcK step %mmc1 iter_args(%acc = %mmc0f32) -> (f32) {{
{row_load}
          %af = arith.extf %a : {dtype} to f32
          %bf = arith.extf %b : {dtype} to f32
          %p = arith.mulf %af, %bf : f32
          %acc2 = arith.addf %acc, %p : f32
          scf.yield %acc2 : f32
        }}
        %oh = arith.truncf %acc_final : f32 to {dtype}
{row_store}
      }}
    }}"""


def emit_rms_matmul_mlir(
    m: int,
    k: int,
    n: int,
    *,
    dtype: str = "f16",
    epsilon: float = 1e-6,
    tiling: Optional[RmsMatmulTiling] = None,
) -> str:
    """Emit memref MLIR for ``out = matmul(rms_norm(x), w)`` (LLVM JIT friendly)."""
    if tiling is None or not tiling.uses_loops:
        return _emit_flat_rms_matmul_mlir(m, k, n, dtype=dtype, epsilon=epsilon)
    if tiling.grid_m > 1:
        return _emit_tiled_rms_matmul_mlir(m, k, n, dtype=dtype, epsilon=epsilon, tiling=tiling)
    return _emit_k_tiled_rms_matmul_mlir(m, k, n, dtype=dtype, epsilon=epsilon, tiling=tiling)


def _emit_flat_rms_matmul_mlir(
    m: int, k: int, n: int, *, dtype: str, epsilon: float
) -> str:
    rms = _rms_row_loop(m, k, dtype=dtype, epsilon=epsilon)
    mm = _emit_matmul_mlir(
        m, k, n, dtype=dtype, lhs_ref="%normed", rhs_ref="%w", out_ref="%out"
    )
    return f"""module {{
  func.func @mugraph(
      %x: memref<{m}x{k}x{dtype}>,
      %w: memref<{k}x{n}x{dtype}>,
      %out: memref<{m}x{n}x{dtype}>) {{
    %normed = memref.alloca() : memref<{m}x{k}x{dtype}>
{rms}
{mm}
    return
  }}
}}
"""


def _emit_k_tiled_rms_matmul_mlir(
    m: int,
    k: int,
    n: int,
    *,
    dtype: str,
    epsilon: float,
    tiling: RmsMatmulTiling,
) -> str:
    """K-forloop tiling from bgraph (partial RMS accum across K chunks)."""
    rms = _rms_row_loop(
        m,
        k,
        dtype=dtype,
        epsilon=epsilon,
        forloop_k=tiling.forloop_k,
        k_tile=tiling.k_tile,
    )
    mm = _emit_matmul_mlir(
        m, k, n, dtype=dtype, lhs_ref="%normed", rhs_ref="%w", out_ref="%out"
    )
    attrs = _yirage_tiling_func_attrs(tiling)
    return f"""module {{
  func.func @mugraph(
      %x: memref<{m}x{k}x{dtype}>,
      %w: memref<{k}x{n}x{dtype}>,
      %out: memref<{m}x{n}x{dtype}>) {attrs} {{
    %normed = memref.alloca() : memref<{m}x{k}x{dtype}>
{rms}
{mm}
    return
  }}
}}
"""


def _yirage_tiling_func_attrs(tiling: RmsMatmulTiling) -> str:
    """Func-level metadata for bgrid tiling (must not attach to memref args)."""
    parts = [f"yirage.grid_m = {tiling.grid_m} : i64"]
    if tiling.grid_m > 1:
        parts.append(f"yirage.m_tile = {tiling.m_tile} : i64")
    if tiling.forloop_k > 1:
        parts.extend(
            [
                f"yirage.forloop_k = {tiling.forloop_k} : i64",
                f"yirage.k_tile = {tiling.k_tile} : i64",
            ]
        )
    return f" attributes {{{', '.join(parts)}}}"


def _emit_tiled_rms_matmul_mlir(
    m: int,
    k: int,
    n: int,
    *,
    dtype: str,
    epsilon: float,
    tiling: RmsMatmulTiling,
) -> str:
    gm = tiling.grid_m
    m_tile = tiling.m_tile
    rms = _rms_row_loop(
        m,
        k,
        dtype=dtype,
        epsilon=epsilon,
        forloop_k=tiling.forloop_k,
        k_tile=tiling.k_tile,
    )
    mm_inner = _emit_matmul_mlir(
        m,
        k,
        n,
        dtype=dtype,
        lhs_ref="%normed",
        rhs_ref="%w",
        out_ref="%out",
        m_dyn=True,
        m_len_ref="%m_use",
        m_row_offset_ref="%m0",
    )
    attrs = _yirage_tiling_func_attrs(tiling)
    return f"""module {{
  func.func @mugraph(
      %x: memref<{m}x{k}x{dtype}>,
      %w: memref<{k}x{n}x{dtype}>,
      %out: memref<{m}x{n}x{dtype}>) {attrs} {{
    %normed = memref.alloca() : memref<{m}x{k}x{dtype}>
{rms}
    %cGm = arith.constant {gm} : index
    %cMT = arith.constant {m_tile} : index
    scf.for %bx = %c0 to %cGm step %c1 {{
      %m0 = arith.muli %bx, %cMT : index
      %m_len = arith.subi %cM, %m0 : index
      %m_use = arith.minsi %m_len, %cMT : index
{mm_inner}
    }}
    return
  }}
}}
"""


@dataclass
class _JitCacheEntry:
    kernel: Any
    m: int
    k: int
    n: int
    tiling_key: Tuple[int, ...]
    emit_path: str


_JIT_CACHE: Dict[Tuple[Any, ...], _JitCacheEntry] = {}


def _dtype_mlir(dtype: str) -> str:
    mapping = {"fp16": "f16", "bf16": "bf16", "fp32": "f32", "fp64": "f64"}
    return mapping.get(str(dtype), "f16")


def mlir_dialect_jit_enabled() -> bool:
    """When set, try cygraph → lowered dialect → raw dialect before hand emit."""
    return os.environ.get("YIRAGE_CPU_MLIR_JIT_DIALECT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _yirage_cpu_opt_path() -> Optional[str]:
    for candidate in (
        os.environ.get("YIRAGE_CPU_OPT"),
        os.path.join(os.getcwd(), "build", "mlir", "yirage-cpu-opt"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "build",
            "mlir",
            "yirage-cpu-opt",
        ),
    ):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def emit_dialect_mlir_from_cygraph(cygraph) -> Optional[str]:
    """Export unfused/fused cygraph to YiRage dialect MLIR text."""
    try:
        import sys
        from pathlib import Path

        mlir_py = Path(__file__).resolve().parents[3] / "mlir" / "python"
        if str(mlir_py) not in sys.path:
            sys.path.insert(0, str(mlir_py))
        from mugraph_to_mlir import MuGraphToMLIR
    except Exception:
        return None

    class _GraphWrap:
        def __init__(self, cg):
            self.cygraph = cg

    try:
        return MuGraphToMLIR().convert(_GraphWrap(cygraph))
    except Exception:
        return None


def lower_dialect_mlir_via_cpu_opt(mlir_text: str) -> Optional[str]:
    """Run ``yirage-cpu-jit-pipeline`` out-of-process; returns lowered MLIR or None."""
    import subprocess
    import tempfile

    opt = _yirage_cpu_opt_path()
    if opt is None:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as fin:
        fin.write(mlir_text)
        in_path = fin.name
    out_path = in_path + ".lowered.mlir"
    try:
        proc = subprocess.run(
            [opt, in_path, "-yirage-cpu-jit-pipeline", "-o", out_path],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            return None
        with open(out_path, encoding="utf-8") as fout:
            return fout.read()
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        for path in (in_path, out_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _compile_mlir_text(kernel: Any, mlir_text: str) -> bool:
    return bool(kernel.compile_mlir(mlir_text, "mugraph"))


def invoke_rms_matmul_mlir_text(
    mlir_text: str,
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    m: Optional[int] = None,
    k: Optional[int] = None,
    n: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Compile and run hand/dialect lowered MLIR via ``CPUJITKernel``."""
    mlir = _load_mlir_jit()
    if mlir is None:
        return None
    m = m if m is not None else x.shape[0]
    k = k if k is not None else x.shape[1]
    n = n if n is not None else w.shape[1]
    kernel = mlir.CPUJITKernel()
    if not _compile_mlir_text(kernel, mlir_text):
        return None
    out = torch.zeros((m, n), dtype=torch.float16, device=x.device)
    ok = kernel.invoke_rms_matmul_f16(
        x.data_ptr(), w.data_ptr(), out.data_ptr(), m, k, n
    )
    return out if ok else None


def _hand_and_dialect_lowered_mlir_texts(
    cygraph,
    *,
    hand_tiling: Optional[RmsMatmulTiling] = None,
) -> Dict[str, Any]:
    """Resolve hand emit + dialect lowered MLIR for bench/compare helpers."""
    shapes = rms_matmul_shapes_from_cygraph(cygraph)
    if shapes is None:
        return {"ok": False, "error": "invalid_rms_matmul_graph"}
    m, k, n = shapes
    inputs_dt = cygraph.get_input_dtensors()
    mlir_dtype = "f16"
    if inputs_dt:
        _, dtype_name = _tensor_shape_dtype(cygraph, inputs_dt[0])
        mlir_dtype = _dtype_mlir(dtype_name)

    if hand_tiling is not None:
        hand_text = emit_rms_matmul_mlir(
            m, k, n, dtype=mlir_dtype, tiling=hand_tiling
        )
        hand_label = "hand_synthetic_tiling"
    else:
        hand_text = emit_bgrid_tiled_mlir_from_cygraph(cygraph)
        hand_label = "hand_bgrid_tiled"
        if hand_text is None:
            tiling = extract_rms_matmul_tiling(cygraph)
            hand_text = emit_rms_matmul_mlir(
                m, k, n, dtype=mlir_dtype, tiling=tiling
            )
            hand_label = "hand_flat_or_k_tiled"

    dialect_text = emit_dialect_mlir_from_cygraph(cygraph)
    if dialect_text is None:
        return {"ok": False, "error": "dialect_emit_failed", "hand_path": hand_label}
    lowered_text = lower_dialect_mlir_via_cpu_opt(dialect_text)
    if lowered_text is None:
        return {
            "ok": False,
            "error": "dialect_lower_failed",
            "hand_path": hand_label,
        }
    return {
        "ok": True,
        "hand_text": hand_text,
        "lowered_text": lowered_text,
        "hand_path": hand_label,
        "shapes": (m, k, n),
    }


def _make_mlir_invoke_fn(
    mlir_text: str,
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    m: int,
    k: int,
    n: int,
) -> Optional[Any]:
    """Compile MLIR once; return a zero-arg invoke callable for benchmarking."""
    mlir = _load_mlir_jit()
    if mlir is None:
        return None
    kernel = mlir.CPUJITKernel()
    if not _compile_mlir_text(kernel, mlir_text):
        return None
    out = torch.zeros((m, n), dtype=torch.float16, device=x.device)
    x_ptr, w_ptr, out_ptr = x.data_ptr(), w.data_ptr(), out.data_ptr()

    def _invoke() -> None:
        if not kernel.invoke_rms_matmul_f16(x_ptr, w_ptr, out_ptr, m, k, n):
            raise RuntimeError(kernel.last_error())

    return _invoke


def compare_hand_tiled_vs_dialect_lowered_jit(
    cygraph,
    inputs: List[torch.Tensor],
    *,
    hand_tiling: Optional[RmsMatmulTiling] = None,
    rtol: float = 0.02,
    atol: float = 0.15,
) -> Dict[str, Any]:
    """Numerical alignment between hand emit and ``yirage-cpu-jit-pipeline`` lowered IR.

    Returns a dict with ``aligned``, ``max_abs_diff``, and per-path compile status.
    When ``hand_tiling`` is set, hand emit uses that tiling (for unfused M/K grid tests).
    Otherwise uses ``emit_bgrid_tiled_mlir_from_cygraph`` / extracted bgraph tiling.
    """
    if not is_mlir_jit_available():
        return {"ok": False, "error": "jit_unavailable"}
    if len(inputs) < 2:
        return {"ok": False, "error": "invalid_rms_matmul_graph"}
    texts = _hand_and_dialect_lowered_mlir_texts(cygraph, hand_tiling=hand_tiling)
    if not texts.get("ok"):
        return texts
    m, k, n = texts["shapes"]
    x, w = inputs[0], inputs[1]
    hand_text = texts["hand_text"]
    lowered_text = texts["lowered_text"]
    hand_label = texts["hand_path"]

    hand_out = invoke_rms_matmul_mlir_text(hand_text, x, w, m=m, k=k, n=n)
    if hand_out is None:
        return {
            "ok": False,
            "error": "hand_compile_or_invoke_failed",
            "hand_path": hand_label,
        }
    dialect_out = invoke_rms_matmul_mlir_text(
        lowered_text, x, w, m=m, k=k, n=n
    )
    if dialect_out is None:
        return {
            "ok": False,
            "error": "dialect_compile_or_invoke_failed",
            "hand_path": hand_label,
        }

    max_abs_diff = float((hand_out.float() - dialect_out.float()).abs().max().item())
    aligned = bool(torch.allclose(hand_out, dialect_out, rtol=rtol, atol=atol))
    return {
        "ok": aligned,
        "aligned": aligned,
        "max_abs_diff": max_abs_diff,
        "hand_path": hand_label,
        "dialect_path": "yirage-cpu-jit-pipeline",
        "shapes": {"m": m, "k": k, "n": n},
    }


def bench_hand_vs_dialect_lowered_jit(
    cygraph,
    inputs: List[torch.Tensor],
    *,
    warmup: int = 10,
    iters: int = 80,
    hand_tiling: Optional[RmsMatmulTiling] = None,
) -> Dict[str, Any]:
    """Benchmark hand emit vs dialect ``yirage-cpu-jit-pipeline`` lowered JIT.

    Timing includes compile+invoke per iteration (CPUJITKernel is single-tenant).
    """
    import time

    if not is_mlir_jit_available():
        return {"ok": False, "error": "jit_unavailable"}
    if len(inputs) < 2:
        return {"ok": False, "error": "invalid_rms_matmul_graph"}

    texts = _hand_and_dialect_lowered_mlir_texts(cygraph, hand_tiling=hand_tiling)
    if not texts.get("ok"):
        return texts
    m, k, n = texts["shapes"]
    x, w = inputs[0], inputs[1]
    _JIT_CACHE.clear()
    align = compare_hand_tiled_vs_dialect_lowered_jit(
        cygraph, inputs, hand_tiling=hand_tiling
    )
    if not align.get("aligned"):
        return {
            "ok": False,
            "error": "hand_dialect_misaligned",
            "mlir_hand_dialect_max_abs_diff": align.get("max_abs_diff"),
        }

    def _ms_invoke(mlir_text: str) -> float:
        def _run() -> None:
            if invoke_rms_matmul_mlir_text(mlir_text, x, w, m=m, k=k, n=n) is None:
                raise RuntimeError("mlir invoke failed")

        for _ in range(warmup):
            _run()
        t0 = time.perf_counter()
        for _ in range(iters):
            _run()
        return (time.perf_counter() - t0) / iters * 1000.0

    # CPUJITKernel cannot host two compiled modules; timing includes compile+invoke.
    hand_ms = _ms_invoke(texts["hand_text"])
    dialect_ms = _ms_invoke(texts["lowered_text"])
    return {
        "ok": True,
        "hand_mlir_jit_ms": hand_ms,
        "dialect_lowered_jit_ms": dialect_ms,
        "speedup_hand_over_dialect_lowered": dialect_ms / max(hand_ms, 1e-9),
        "hand_path": texts["hand_path"],
        "dialect_path": "yirage-cpu-jit-pipeline",
        "mlir_hand_dialect_aligned": align.get("aligned"),
        "mlir_hand_dialect_max_abs_diff": align.get("max_abs_diff"),
    }


def emit_bgrid_tiled_mlir_from_cygraph(cygraph) -> Optional[str]:
    """Hand memref MLIR with bgraph M/K tiling (preserves search tiling in IR)."""
    shapes = rms_matmul_shapes_from_cygraph(cygraph)
    if shapes is None:
        return None
    m, k, n = shapes
    tiling = extract_rms_matmul_tiling(cygraph)
    inputs = cygraph.get_input_dtensors()
    dtype = "fp16"
    if inputs:
        _, dtype = _tensor_shape_dtype(cygraph, inputs[0])
    return emit_rms_matmul_mlir(
        m, k, n, dtype=_dtype_mlir(dtype), tiling=tiling
    )


def _rms_matmul_mlir_compile_candidates(
    m: int,
    k: int,
    n: int,
    mlir_dtype: str,
    tiling: Optional[RmsMatmulTiling],
    cygraph,
) -> List[Tuple[str, str]]:
    """Ordered MLIR emit strategies as ``(path_label, mlir_text)`` pairs."""
    candidates: List[Tuple[str, str]] = []
    seen: set[str] = set()

    def _add(path: str, text: Optional[str]) -> None:
        if text and text not in seen:
            seen.add(text)
            candidates.append((path, text))

    use_tiled_hand = (
        preserve_bgrid_tiling()
        and tiling is not None
        and tiling.uses_loops
        and cygraph is not None
    )
    if mlir_dialect_jit_enabled() and cygraph is not None:
        dialect_text = emit_dialect_mlir_from_cygraph(cygraph)
        if dialect_text:
            _add(
                "dialect_lowered",
                lower_dialect_mlir_via_cpu_opt(dialect_text),
            )
        _add("dialect_raw", dialect_text)
    if use_tiled_hand:
        _add("hand_bgrid_tiled", emit_bgrid_tiled_mlir_from_cygraph(cygraph))
    _add(
        "hand_tiled",
        emit_rms_matmul_mlir(m, k, n, dtype=mlir_dtype, tiling=tiling),
    )
    # Flat memref fallback when tiled/dialect emit fails to compile.
    _add(
        "hand_flat",
        emit_rms_matmul_mlir(m, k, n, dtype=mlir_dtype, tiling=None),
    )
    return candidates


def _get_or_compile_kernel(
    m: int,
    k: int,
    n: int,
    dtype: str,
    tiling: Optional[RmsMatmulTiling],
    cygraph=None,
) -> Optional[_JitCacheEntry]:
    mlir = _load_mlir_jit()
    if mlir is None:
        return None
    mlir_dtype = _dtype_mlir(dtype)
    tkey = tiling.cache_key if tiling else (0,)
    key = (m, k, n, mlir_dtype, tkey)
    if key in _JIT_CACHE:
        return _JIT_CACHE[key]

    last_error = ""
    for emit_path, mlir_text in _rms_matmul_mlir_compile_candidates(
        m, k, n, mlir_dtype, tiling, cygraph
    ):
        kernel = mlir.CPUJITKernel()
        if _compile_mlir_text(kernel, mlir_text):
            entry = _JitCacheEntry(
                kernel=kernel,
                m=m,
                k=k,
                n=n,
                tiling_key=tkey,
                emit_path=emit_path,
            )
            _JIT_CACHE[key] = entry
            return entry
        last_error = str(kernel.last_error())

    warnings.warn(
        f"CPU MLIR JIT compile failed: {last_error}",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def rms_matmul_mlir_emit_path(cygraph) -> Optional[str]:
    """Return the emit strategy label that compiled for ``cygraph``, or None."""
    if not is_mlir_jit_available() or not is_rms_matmul_mugraph(cygraph):
        return None
    shapes = rms_matmul_shapes_from_cygraph(cygraph)
    if shapes is None:
        return None
    m, k, n = shapes
    tiling = extract_rms_matmul_tiling(cygraph)
    inputs = cygraph.get_input_dtensors()
    dtype = "fp16"
    if inputs:
        _, dtype = _tensor_shape_dtype(cygraph, inputs[0])
    entry = _get_or_compile_kernel(m, k, n, dtype, tiling, cygraph)
    return entry.emit_path if entry else None


def try_rms_matmul_jit(
    cygraph,
    input_tensors: List[torch.Tensor],
    *,
    require_experimental: bool = False,
) -> Optional[List[torch.Tensor]]:
    """Run rms_norm+matmul via LLVM JIT when MLIR is built and env is enabled.

    When ``require_experimental`` is True (``cpu_call`` path), LLVM JIT runs only if
    ``YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL=1``. Benchmarks/tests may pass False to force JIT.
    """
    if not mlir_jit_enabled() or not is_mlir_jit_available():
        return None
    if require_experimental and not mlir_jit_experimental_enabled():
        return None
    if not is_rms_matmul_mugraph(cygraph):
        return None
    shapes = rms_matmul_shapes_from_cygraph(cygraph)
    if shapes is None:
        return None
    m, k, n = shapes
    if len(input_tensors) != 2:
        return None

    x, w = input_tensors
    if x.dtype != torch.float16 or w.dtype != torch.float16:
        return None
    if not x.is_contiguous() or not w.is_contiguous():
        x = x.contiguous()
        w = w.contiguous()

    if should_use_blas_fast_path(m, k, n):
        return [_torch_rms_matmul(x, w)]

    tiling = extract_rms_matmul_tiling(cygraph)
    entry = _get_or_compile_kernel(m, k, n, "fp16", tiling, cygraph)
    if entry is None or not entry.kernel.is_ready():
        return None

    out = torch.zeros((m, n), dtype=torch.float16, device=x.device)
    ok = entry.kernel.invoke_rms_matmul_f16(
        x.data_ptr(), w.data_ptr(), out.data_ptr(), m, k, n
    )
    if not ok:
        warnings.warn(
            f"CPU MLIR JIT invoke failed: {entry.kernel.last_error()}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return [out]


class MLIRJitRunner:
    """KNGraph-like runner that always targets the MLIR JIT path (for benchmarks)."""

    def __init__(self, cygraph):
        self.cygraph = cygraph

    def __call__(self, *, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        prev = os.environ.get("YIRAGE_CPU_MLIR_JIT")
        os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
        try:
            out = try_rms_matmul_jit(self.cygraph, inputs)
            if out is None:
                raise RuntimeError("MLIR JIT path unavailable or failed")
            return out
        finally:
            if prev is None:
                os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)
            else:
                os.environ["YIRAGE_CPU_MLIR_JIT"] = prev


def bench_jit_vs_interpreter(
    cygraph,
    inputs: List[torch.Tensor],
    *,
    warmup: int = 10,
    iters: int = 80,
) -> Dict[str, Any]:
    """Benchmark interpreter vs MLIR JIT; requires ``is_mlir_jit_available()``."""
    from yirage.kernel.graph import _interpret_mugraph_on_cpu_impl

    import time

    def _ms(fn):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1000.0

    if not is_mlir_jit_available():
        return {"ok": False, "error": "jit_unavailable"}

    os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
    try:
        jit_out = try_rms_matmul_jit(cygraph, inputs)
        if jit_out is None:
            return {"ok": False, "error": "jit_failed"}
        jit_ms = _ms(lambda: try_rms_matmul_jit(cygraph, inputs))
        interp_ms = _ms(lambda: _interpret_mugraph_on_cpu_impl(cygraph, inputs))
        tiling = extract_rms_matmul_tiling(cygraph)
        return {
            "ok": True,
            "mlir_jit_ms": jit_ms,
            "interpreter_ms": interp_ms,
            "speedup_interp_over_mlir_jit": interp_ms / max(jit_ms, 1e-9),
            "mlir_jit_emit_path": rms_matmul_mlir_emit_path(cygraph),
            "tiling": (
                {
                    "grid_m": tiling.grid_m,
                    "forloop_k": tiling.forloop_k,
                    "m_tile": tiling.m_tile,
                    "k_tile": tiling.k_tile,
                }
                if tiling is not None
                else None
            ),
        }
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)
