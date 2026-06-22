# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AccelForge Bridge — Integration layer between YiRage and AccelForge.

AccelForge is a framework to model and design tensor algebra accelerators,
providing area, energy, latency, and leak power modeling of hardware components.
See: https://github.com/Accelergy-Project/accelforge

Real AccelForge API used here:
    spec = Spec.from_yaml(arch_yaml_path, workload_yaml_path)
    results = spec.map_workload_to_arch(print_progress=False)
    energy_pj  = results.energy()          # total energy (user-defined units)
    latency_cy = results.latency()         # cycles (MAC latency = 1 cycle)
    n_computes = results.n_computes()      # total MAC operations
    spec_area  = spec.calculate_component_area_energy_latency_leak()
    # spec_area.arch.nodes[i].total_area / .total_leak_power

This bridge enables:
1. Using AccelForge as a performance oracle for RL-guided kernel search
2. Hardware-software co-design by modeling custom accelerators
3. Multi-objective reward computation (latency + energy + area + power)
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover - packaging is normally available with build tools
    InvalidVersion = None
    Version = None

if TYPE_CHECKING:
    from .profile import HardwareProfile

logger = logging.getLogger(__name__)

# Shared encoding constants for AccelForge design parameters
DATAFLOW_ENCODING = {
    "output_stationary": 0.25,
    "weight_stationary": 0.5,
    "row_stationary": 0.75,
}

NOC_TOPOLOGY_ENCODING = {
    "mesh": 0.33,
    "ring": 0.66,
    "tree": 1.0,
}

DATA_PRECISION_ENCODING = {
    "int8": 0.25,
    "fp16": 0.5,
    "bf16": 0.75,
    "fp32": 1.0,
}

PRECISION_OPS_PER_PE = {
    "int8": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp32": 1.0,
}

# log2(128) = 7, used to normalize PE array dimensions
MAX_PE_ARRAY_LOG2 = 7.0

# Check AccelForge availability — import the Spec class we actually use
ACCELFORGE_MIN_VERSION = "1.0.355"
ACCELFORGE_MAX_VERSION_EXCLUSIVE = "2.0.0"
DEFAULT_ACCELFORGE_PARALLEL_JOBS = 2
ACCELFORGE_AVAILABLE = False
ACCELFORGE_IMPORT_ERROR = ""
ACCELFORGE_VERSION: Optional[str] = None
try:
    from accelforge import Spec as _AccelForgeSpec  # noqa: F401 — presence test
    ACCELFORGE_AVAILABLE = True
    try:
        ACCELFORGE_VERSION = metadata.version("accelforge")
    except metadata.PackageNotFoundError:
        ACCELFORGE_VERSION = None
except ImportError as e:
    ACCELFORGE_IMPORT_ERROR = str(e)


def _version_tuple(version: Optional[str]) -> tuple:
    """Fallback version parser for environments without ``packaging``."""
    if not version:
        return ()
    parts = []
    for token in version.split("+", 1)[0].split("-", 1)[0].split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _is_supported_accelforge_version(version: Optional[str]) -> bool:
    """Return whether the installed AccelForge version is in YiRage's tested range."""
    if version and Version is not None:
        try:
            return Version(ACCELFORGE_MIN_VERSION) <= Version(version) < Version(
                ACCELFORGE_MAX_VERSION_EXCLUSIVE
            )
        except InvalidVersion:
            logger.warning("Unable to parse AccelForge version: %s", version)
            return False

    parsed = _version_tuple(version)
    if not parsed:
        # Editable/local AccelForge installs may not expose package metadata.
        return True
    return _version_tuple(ACCELFORGE_MIN_VERSION) <= parsed < _version_tuple(
        ACCELFORGE_MAX_VERSION_EXCLUSIVE
    )


def _compute_batch_dim(shape: List[int]) -> int:
    """Return the product of leading batch dimensions for rank >= 3 tensors."""
    return int(math.prod(shape[:-2])) if len(shape) > 2 else 1


_CY_OP_TYPE_TO_RL: Dict[str, str] = {
    "kn_matmul_op": "matmul",
    "kn_reduction_0_op": "reduction",
    "kn_reduction_1_op": "reduction",
    "kn_reduction_2_op": "reduction",
    "kn_relu_op": "relu",
    "kn_gelu_op": "gelu",
    "kn_silu_op": "silu",
    "kn_exp_op": "elementwise",
    "kn_square_op": "elementwise",
    "kn_sqrt_op": "elementwise",
    "kn_sigmoid_op": "elementwise",
    "kn_log_op": "elementwise",
    "kn_add_op": "add",
    "kn_mul_op": "mul",
    "kn_div_op": "mul",
    "kn_pow_op": "mul",
    "kn_rms_norm_op": "reduction",
    "kn_customized_op": "matmul",
}

_CY_SKIP_OPS = frozenset({"kn_input_op", "kn_output_op"})


def _dims_from_cy_tensor(tensor: Dict[str, Any]) -> List[int]:
    """Extract logical shape from a C++ ``cy_to_json`` tensor dict."""
    if not isinstance(tensor, dict):
        return []

    dims = tensor.get("dims", tensor.get("shape"))
    if isinstance(dims, list) and dims:
        return [int(d) for d in dims]

    num_dims = int(tensor.get("num_dims", 0) or 0)
    raw = tensor.get("dim", [])
    if num_dims > 0 and isinstance(raw, list):
        return [int(raw[i]) for i in range(min(num_dims, len(raw)))]
    if isinstance(raw, list) and raw:
        return [int(d) for d in raw if int(d) != 0 or len(raw) <= 2]
    return []


def _cy_op_type_to_rl(op_type: str) -> str:
    """Map native KN operator names to RL-style operator labels."""
    lowered = str(op_type).lower()
    if lowered in _CY_OP_TYPE_TO_RL:
        return _CY_OP_TYPE_TO_RL[lowered]
    if lowered.startswith("kn_") and lowered.endswith("_op"):
        return lowered[3:-3]
    return lowered


def _bgraph_has_matmul(op: Dict[str, Any]) -> bool:
    bgraph = op.get("bgraph")
    if not isinstance(bgraph, dict):
        return False
    for tb_op in bgraph.get("operators", []):
        if str(tb_op.get("op_type", "")).lower() == "tb_matmul_op":
            return True
    return False


def _select_cy_graph_variant(variants: List[Any]) -> List[Dict[str, Any]]:
    """Pick the most compute-heavy graph variant from a MuGraph cache entry."""
    best_ops: List[Dict[str, Any]] = []
    best_score = -1
    for variant in variants:
        if not isinstance(variant, list):
            continue
        score = 0
        for op in variant:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("op_type", "")).lower()
            if op_type == "kn_matmul_op":
                score += 4
            elif op_type == "kn_customized_op" and _bgraph_has_matmul(op):
                score += 3
            elif op_type.startswith("kn_reduction"):
                score += 1
        if score > best_score:
            best_score = score
            best_ops = variant
    if best_ops:
        return best_ops
    for variant in variants:
        if isinstance(variant, list) and variant and isinstance(variant[0], dict):
            return variant
    return []


def _normalize_cy_graph_ops(ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert ``cy_to_json`` operator list into RL ``operators``/``tensors`` dict."""
    tensors_by_guid: Dict[int, Dict[str, Any]] = {}
    operators: List[Dict[str, Any]] = []
    reduction_dimx = 0
    forloop_range = 1

    def _register_tensor(tensor: Dict[str, Any]) -> int:
        guid = int(tensor.get("guid", tensor.get("tensor_id", tensor.get("id", -1))))
        if guid < 0:
            return guid
        dims = _dims_from_cy_tensor(tensor)
        if guid not in tensors_by_guid:
            tensors_by_guid[guid] = {
                "tensor_id": guid,
                "id": guid,
                "dims": dims,
                "shape": dims,
            }
        elif dims and not tensors_by_guid[guid].get("dims"):
            tensors_by_guid[guid]["dims"] = dims
            tensors_by_guid[guid]["shape"] = dims
        return guid

    for op_id, op in enumerate(ops):
        if not isinstance(op, dict):
            continue
        raw_type = str(op.get("op_type", op.get("type", ""))).lower()
        if raw_type in _CY_SKIP_OPS:
            for tensor in op.get("output_tensors", []):
                _register_tensor(tensor)
            for tensor in op.get("input_tensors", []):
                _register_tensor(tensor)
            continue

        rl_type = _cy_op_type_to_rl(raw_type)
        if raw_type == "kn_customized_op" and not _bgraph_has_matmul(op):
            rl_type = "customized"

        input_ids = [
            tid
            for tid in (_register_tensor(t) for t in op.get("input_tensors", []))
            if tid >= 0
        ]
        output_ids = [
            tid
            for tid in (_register_tensor(t) for t in op.get("output_tensors", []))
            if tid >= 0
        ]

        bgraph = op.get("bgraph")
        if isinstance(bgraph, dict):
            reduction_dimx = max(
                reduction_dimx, int(bgraph.get("reduction_dimx", 0) or 0)
            )
            forloop_range = max(forloop_range, int(bgraph.get("forloop_range", 1) or 1))

        operators.append(
            {
                "op_id": op_id,
                "op_type": rl_type,
                "type": rl_type,
                "input_tensor_ids": input_ids,
                "output_tensor_ids": output_ids,
            }
        )

    graph: Dict[str, Any] = {
        "operators": operators,
        "tensors": list(tensors_by_guid.values()),
    }
    if reduction_dimx > 0:
        graph["reduction_dimx"] = reduction_dimx
    if forloop_range > 1:
        graph["forloop_range"] = forloop_range
    return graph


def normalize_mugraph_json(data: Any) -> Dict[str, Any]:
    """
    Normalize YiRage graph JSON into the RL ``operators``/``tensors`` layout.

    Accepts:
      - RL search graphs: ``{"operators": [...], "tensors": [...]}``
      - ``cy_to_json`` output: ``[{"op_type": "kn_matmul_op", ...}, ...]``
      - MuGraph cache entries: ``[[variant0_ops...], [variant1_ops...], ...]``
    """
    if isinstance(data, dict) and ("operators" in data or "tensors" in data):
        return data

    if isinstance(data, list):
        if not data:
            return {"operators": [], "tensors": []}
        if isinstance(data[0], dict) and "op_type" in data[0]:
            return _normalize_cy_graph_ops(data)
        if isinstance(data[0], list):
            return _normalize_cy_graph_ops(_select_cy_graph_variant(data))

    return {"operators": [], "tensors": []}


def kngraph_to_workload(graph: Any) -> Dict[str, Any]:
    """
    Build an AccelForge workload from a live ``KNGraph`` / ``CyKNGraph``.

    Uses ``serialize_optimized_graph`` (``cy_to_json``) when available, then
    delegates to ``mugraph_to_workload``.
    """
    from yirage.storage.graph_serde import serialize_optimized_graph

    graph_json = serialize_optimized_graph(graph)
    if graph_json:
        return mugraph_to_workload(graph_json)

    import json

    cygraph = getattr(graph, "cygraph", graph)
    get_structure = getattr(cygraph, "get_graph_structure", None)
    if callable(get_structure):
        return mugraph_to_workload(json.dumps(get_structure()))

    return {"estimated_flops": 1e9}


def mugraph_to_workload(graph_json: str) -> Dict[str, Any]:
    """
    Translate a YiRage µGraph JSON into an AccelForge workload descriptor.

    YiRage generates µGraphs with two operator levels:
      - Kernel-level (kn) operators: coarse-grained tensor operations
        such as matmul, batch_matmul, attention, convolution, reduction.
      - Threadblock-level (tb) operators: fine-grained CUDA tile operations.

    This function inspects the dominant kn-level operator type and extracts
    the actual M×K×N (or equivalent) dimensions from the tensor shapes stored
    in the graph.  The result is a workload dict with ``m_dim``, ``k_dim``, and
    ``n_dim`` keys that ``AccelForgeBridge._workload_to_yaml()`` will use
    directly (priority-0 path) so AccelForge models the real computation, not
    a synthetic proxy.

    Operator-type mapping:
      matmul / batch_matmul → M=rows of A, K=cols of A / rows of B, N=cols of B
      attention             → M=seq_len, K=head_dim (QKᵀ), N=head_dim (AV)
      conv                  → M=batch×H_out×W_out, K=C_in×R×S, N=C_out
      reduction             → M=outer_dims, K=reduction_dimx, N=1
      elementwise / unknown → fall back to estimated_flops

    If the graph JSON is invalid or carries no useful operator information the
    function returns ``{"estimated_flops": 1e9}`` so downstream code falls
    through to the cube-root approximation.

    Args:
        graph_json: JSON string of a YiRage µGraph (from C++ core or
                    ``kernel_graph_json`` stored in ``YiRageSearchEnv``).

    Returns:
        Workload dict with one of:
          {"m_dim": int, "k_dim": int, "n_dim": int,
           "op_type": str, "estimated_flops": float}        — dominant operator
          {"estimated_flops": float}                         — fallback
    """
    import json

    try:
        raw = json.loads(graph_json) if graph_json else {}
    except (json.JSONDecodeError, TypeError):
        return {"estimated_flops": 1e9}

    data = normalize_mugraph_json(raw)
    operators = data.get("operators", [])
    tensors = data.get("tensors", [])

    # Build a tensor-id → shape map for quick lookup
    tensor_map: Dict[int, List[int]] = {}
    batch_dim_cache: Dict[tuple, int] = {}
    for t in tensors:
        tid = t.get("tensor_id", t.get("id", -1))
        dims = t.get("dims", t.get("shape", []))
        if tid >= 0 and dims:
            tensor_map[tid] = dims

    def _cached_batch_dim(shape: List[int]) -> int:
        key = tuple(shape)
        if key not in batch_dim_cache:
            batch_dim_cache[key] = _compute_batch_dim(shape)
        return batch_dim_cache[key]

    # -----------------------------------------------------------------------
    # Score each operator by its matmul/compute weight and pick the dominant one
    # Weight: matmul/attention > conv > reduction > elementwise
    # -----------------------------------------------------------------------
    _OP_WEIGHT: Dict[str, int] = {
        "matmul": 4,
        "batch_matmul": 4,
        "bmm": 4,
        "attention": 3,
        "softmax_attention": 3,
        "conv": 2,
        "convolution": 2,
        "reduction": 1,
        "reduce": 1,
        "elementwise": 0,
        "elu": 0,
        "relu": 0,
        "gelu": 0,
        "add": 0,
        "mul": 0,
    }

    best_op: Optional[Dict[str, Any]] = None
    best_weight = -1
    for op in operators:
        op_type = str(op.get("op_type", op.get("type", ""))).lower()
        weight = _OP_WEIGHT.get(op_type, 0)
        # Prefer higher weight; on tie prefer larger flops
        op_flops = float(op.get("flops", 0.0))
        best_op_flops = float(best_op.get("flops", 0)) if best_op is not None else 0.0
        if weight > best_weight or (weight == best_weight and op_flops > best_op_flops):
            best_weight = weight
            best_op = op

    if best_op is None:
        # No operators — fall back
        total_flops = sum(float(op.get("flops", 0)) for op in operators)
        return {"estimated_flops": max(total_flops, 1e9)}

    op_type = str(best_op.get("op_type", best_op.get("type", ""))).lower()
    op_flops = float(best_op.get("flops", 0.0))

    # -----------------------------------------------------------------------
    # Extract M/K/N from tensor shapes
    # -----------------------------------------------------------------------
    input_ids: List[int] = best_op.get("input_tensor_ids", [])
    output_ids: List[int] = best_op.get("output_tensor_ids", [])

    # Helper: get shape of first/second input tensor
    def _shape(ids: List[int], idx: int = 0) -> List[int]:
        if idx < len(ids):
            return tensor_map.get(ids[idx], [])
        return []

    if op_type in ("matmul", "batch_matmul", "bmm"):
        # A: [..., M, K]  B: [..., K, N]  → Output: [..., M, N]
        a_shape = _shape(input_ids, 0)
        b_shape = _shape(input_ids, 1)
        if len(a_shape) >= 2 and len(b_shape) >= 2:
            m_dim = a_shape[-2]
            k_dim = a_shape[-1]
            n_dim = b_shape[-1]
            batch_dim = _cached_batch_dim(a_shape)
        else:
            # Fall back to flops if shapes not available
            side = max(1, round(math.pow(max(op_flops / 2.0, 1.0), 1.0 / 3.0)))
            m_dim = k_dim = n_dim = side
            batch_dim = 1

    elif op_type in ("attention", "softmax_attention"):
        # Q: [B, H, seq, head_dim]  K: [B, H, head_dim, seq]  V: [B, H, seq, head_dim]
        # Model as two matmuls: QKᵀ (M=seq, K=head_dim, N=seq) then AV (M=seq, K=seq, N=head_dim)
        # Use the first matmul (QKᵀ) as the dominant one
        q_shape = _shape(input_ids, 0)
        if len(q_shape) >= 2:
            m_dim = q_shape[-2]  # seq_len
            k_dim = q_shape[-1]  # head_dim
            n_dim = q_shape[-2]  # seq_len (QKᵀ output width)
            batch_dim = _cached_batch_dim(q_shape)
        else:
            side = max(1, round(math.pow(max(op_flops / 2.0, 1.0), 1.0 / 3.0)))
            m_dim = k_dim = n_dim = side
            batch_dim = 1

    elif op_type in ("conv", "convolution"):
        # Input: [N, C, H, W]  Weight: [C_out, C_in, R, S]
        # Linearised matmul: M = N × H_out × W_out,  K = C_in × R × S,  N = C_out
        inp_shape = _shape(input_ids, 0)  # [N, C, H, W]
        w_shape = _shape(input_ids, 1)    # [C_out, C_in, R, S]
        if len(inp_shape) == 4 and len(w_shape) == 4:
            batch, c_in, h, w = inp_shape
            c_out, _, r, s = w_shape
            h_out = max(1, h - r + 1)
            w_out = max(1, w - s + 1)
            m_dim = batch * h_out * w_out
            k_dim = c_in * r * s
            n_dim = c_out
            batch_dim = 1
        else:
            side = max(1, round(math.pow(max(op_flops / 2.0, 1.0), 1.0 / 3.0)))
            m_dim = k_dim = n_dim = side
            batch_dim = 1

    elif op_type in ("reduction", "reduce"):
        # Use forloop_range and reduction_dimx from the graph if available
        reduction_dimx = data.get("reduction_dimx", best_op.get("reduction_dimx", 0))
        forloop_range = data.get("forloop_range", best_op.get("forloop_range", 1))
        inp_shape = _shape(input_ids, 0)
        if reduction_dimx > 0 and inp_shape:
            outer = max(1, int(math.prod(inp_shape)) // max(reduction_dimx, 1))
            m_dim = outer
            k_dim = reduction_dimx
            n_dim = 1
            batch_dim = 1
        elif inp_shape:
            # Reduce the last dimension
            k_dim = inp_shape[-1]
            m_dim = max(1, int(math.prod(inp_shape[:-1])))
            n_dim = 1
            batch_dim = 1
        else:
            m_dim, k_dim, n_dim = 128, 256, 1
            batch_dim = 1

    else:
        # Elementwise or unknown: represent as vector matmul (1×K×1)
        inp_shape = _shape(input_ids, 0)
        total_elems = max(int(math.prod(inp_shape)), 1) if inp_shape else 1
        m_dim = 1
        k_dim = total_elems
        n_dim = 1
        batch_dim = 1

    def _fallback_op_flops(op: Dict[str, Any]) -> float:
        explicit = float(op.get("flops", 0.0))
        if explicit > 0:
            return explicit

        cur_type = str(op.get("op_type", op.get("type", ""))).lower()
        ids = op.get("input_tensor_ids", [])
        first = tensor_map.get(ids[0], []) if ids else []
        second = tensor_map.get(ids[1], []) if len(ids) > 1 else []

        if cur_type in ("matmul", "batch_matmul", "bmm") and len(first) >= 2 and len(second) >= 2:
            cur_batch = _cached_batch_dim(first)
            return float(2 * cur_batch * first[-2] * first[-1] * second[-1])
        if cur_type in ("attention", "softmax_attention") and len(first) >= 2:
            cur_batch = _cached_batch_dim(first)
            seq = first[-2]
            head_dim = first[-1]
            # QKᵀ and AV matmuls.
            return float(4 * cur_batch * seq * seq * head_dim)
        if cur_type in ("conv", "convolution") and len(first) == 4 and len(second) == 4:
            batch, c_in, h, w = first
            c_out, _, r, s = second
            h_out = max(1, h - r + 1)
            w_out = max(1, w - s + 1)
            return float(2 * batch * h_out * w_out * c_in * r * s * c_out)
        if cur_type in ("reduction", "reduce", "elementwise", "elu", "relu", "gelu", "add", "mul"):
            return float(max(int(math.prod(first)), 1)) if first else 1.0
        return 1.0

    total_flops = max(sum(_fallback_op_flops(op) for op in operators), op_flops, 1.0)
    op_counts: Dict[str, int] = {}
    for op in operators:
        cur_type = str(op.get("op_type", op.get("type", "unknown"))).lower() or "unknown"
        op_counts[cur_type] = op_counts.get(cur_type, 0) + 1

    return {
        "m_dim": m_dim,
        "k_dim": k_dim,
        "n_dim": n_dim,
        "batch_dim": batch_dim,
        "effective_m_dim": m_dim * batch_dim,
        "op_type": op_type,
        "dominant_op_type": op_type,
        "num_operators": len(operators),
        "operator_counts": op_counts,
        "estimated_flops": total_flops,
    }


@dataclass
class AccelForgeDesignPoint:
    """
    A single accelerator design point in AccelForge's design space.

    Represents one possible hardware configuration to evaluate.
    """

    # PE (Processing Element) array
    pe_array_rows: int = 16
    pe_array_cols: int = 16

    # Memory hierarchy (buffer sizes in KB)
    l0_buffer_kb: float = 1.0
    l1_buffer_kb: float = 64.0
    l2_buffer_kb: float = 512.0

    # Dataflow
    dataflow: str = "output_stationary"  # output_stationary, weight_stationary, row_stationary

    # Network-on-Chip
    noc_topology: str = "mesh"  # mesh, ring, tree

    # Data precision
    data_precision: str = "fp16"  # int8, fp16, bf16, fp32

    # Clock frequency (MHz)
    clock_mhz: float = 1000.0

    # Technology node (nm)
    tech_node_nm: int = 7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pe_array_rows": self.pe_array_rows,
            "pe_array_cols": self.pe_array_cols,
            "l0_buffer_kb": self.l0_buffer_kb,
            "l1_buffer_kb": self.l1_buffer_kb,
            "l2_buffer_kb": self.l2_buffer_kb,
            "dataflow": self.dataflow,
            "noc_topology": self.noc_topology,
            "data_precision": self.data_precision,
            "clock_mhz": self.clock_mhz,
            "tech_node_nm": self.tech_node_nm,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AccelForgeDesignPoint":
        """Create from dictionary."""
        return cls(
            pe_array_rows=d.get("pe_array_rows", 16),
            pe_array_cols=d.get("pe_array_cols", 16),
            l0_buffer_kb=d.get("l0_buffer_kb", 1.0),
            l1_buffer_kb=d.get("l1_buffer_kb", 64.0),
            l2_buffer_kb=d.get("l2_buffer_kb", 512.0),
            dataflow=d.get("dataflow", "output_stationary"),
            noc_topology=d.get("noc_topology", "mesh"),
            data_precision=d.get("data_precision", "fp16"),
            clock_mhz=d.get("clock_mhz", 1000.0),
            tech_node_nm=d.get("tech_node_nm", 7),
        )

    @property
    def total_pes(self) -> int:
        """Total number of processing elements."""
        return self.pe_array_rows * self.pe_array_cols

    @property
    def total_buffer_kb(self) -> float:
        """Total buffer size in KB."""
        return self.l0_buffer_kb * self.total_pes + self.l1_buffer_kb + self.l2_buffer_kb


@dataclass
class AccelForgeMetrics:
    """
    Hardware design metrics from AccelForge evaluation.

    Contains area, energy, latency, and power estimates for a design point.
    """

    # Area (mm²)
    area_mm2: float = 0.0
    pe_area_mm2: float = 0.0
    buffer_area_mm2: float = 0.0
    noc_area_mm2: float = 0.0

    # Energy per operation (pJ)
    energy_per_op_pj: float = 0.0
    compute_energy_pj: float = 0.0
    memory_energy_pj: float = 0.0
    noc_energy_pj: float = 0.0

    # Latency (cycles and ms)
    latency_cycles: int = 0
    latency_ms: float = 0.0

    # Power (mW)
    total_power_mw: float = 0.0
    dynamic_power_mw: float = 0.0
    leak_power_mw: float = 0.0

    # Throughput
    peak_tops: float = 0.0  # Tera operations per second
    achieved_tops: float = 0.0

    # Utilization
    pe_utilization: float = 0.0
    buffer_utilization: float = 0.0
    noc_bandwidth_utilization: float = 0.0

    # Confidence of the estimate
    confidence: float = 0.85  # AccelForge analytical model

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "area_mm2": self.area_mm2,
            "pe_area_mm2": self.pe_area_mm2,
            "buffer_area_mm2": self.buffer_area_mm2,
            "noc_area_mm2": self.noc_area_mm2,
            "energy_per_op_pj": self.energy_per_op_pj,
            "compute_energy_pj": self.compute_energy_pj,
            "memory_energy_pj": self.memory_energy_pj,
            "noc_energy_pj": self.noc_energy_pj,
            "latency_cycles": self.latency_cycles,
            "latency_ms": self.latency_ms,
            "total_power_mw": self.total_power_mw,
            "dynamic_power_mw": self.dynamic_power_mw,
            "leak_power_mw": self.leak_power_mw,
            "peak_tops": self.peak_tops,
            "achieved_tops": self.achieved_tops,
            "pe_utilization": self.pe_utilization,
            "buffer_utilization": self.buffer_utilization,
            "noc_bandwidth_utilization": self.noc_bandwidth_utilization,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AccelForgeMetrics":
        """Create from dictionary."""
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class AccelForgeBridge:
    """
    Bridge between YiRage RL search and AccelForge hardware modeling.

    Provides:
    1. Design space enumeration for Level 0 RL policy
    2. Performance estimation (latency, energy, area, power) for reward computation
    3. Hardware profile generation for Level 1/2 constraint propagation

    Usage:
        bridge = AccelForgeBridge()

        # Evaluate a design point
        design = AccelForgeDesignPoint(pe_array_rows=32, pe_array_cols=32)
        metrics = bridge.evaluate(design, workload_spec)

        # Generate HardwareProfile
        profile = bridge.to_hardware_profile(design, metrics)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._af_model = None
        self._cache: Dict[str, AccelForgeMetrics] = {}

        if ACCELFORGE_AVAILABLE and _is_supported_accelforge_version(ACCELFORGE_VERSION):
            self._init_accelforge()
        elif ACCELFORGE_AVAILABLE:
            logger.warning("AccelForge is installed but outside YiRage's tested range: %s", ACCELFORGE_VERSION)

    def _init_accelforge(self):
        """Initialize AccelForge — store the Spec class for later use."""
        try:
            from accelforge import Spec, set_n_parallel_jobs

            self._af_model = Spec
            configured_parallel_jobs = self._config.get(
                "n_parallel_jobs", DEFAULT_ACCELFORGE_PARALLEL_JOBS
            )
            try:
                n_parallel_jobs = max(1, int(configured_parallel_jobs))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid AccelForge n_parallel_jobs=%r; using %d",
                    configured_parallel_jobs,
                    DEFAULT_ACCELFORGE_PARALLEL_JOBS,
                )
                n_parallel_jobs = DEFAULT_ACCELFORGE_PARALLEL_JOBS
            set_n_parallel_jobs(n_parallel_jobs)
            logger.info("AccelForge initialized successfully")
        except Exception as e:
            logger.warning("AccelForge initialization failed: %s", e)
            self._af_model = None

    @property
    def is_available(self) -> bool:
        """Check if AccelForge is available."""
        return ACCELFORGE_AVAILABLE and _is_supported_accelforge_version(ACCELFORGE_VERSION)

    def availability(self) -> Dict[str, Any]:
        """Return detailed AccelForge availability and compatibility diagnostics."""
        return get_accelforge_availability()

    def evaluate(
        self,
        design: AccelForgeDesignPoint,
        workload: Optional[Dict[str, Any]] = None,
    ) -> AccelForgeMetrics:
        """
        Evaluate a hardware design point.

        Uses AccelForge if available, otherwise falls back to analytical model.

        Args:
            design: Accelerator design point to evaluate
            workload: Optional workload specification for workload-specific metrics

        Returns:
            AccelForgeMetrics with area, energy, latency, power estimates
        """
        # Check cache
        cache_key = str(design.to_dict()) + str(workload or {})
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._af_model is not None:
            metrics = self._evaluate_with_accelforge(design, workload)
        else:
            metrics = self._evaluate_analytical(design, workload)

        self._cache[cache_key] = metrics
        return metrics

    def _evaluate_with_accelforge(
        self,
        design: AccelForgeDesignPoint,
        workload: Optional[Dict[str, Any]],
    ) -> AccelForgeMetrics:
        """
        Evaluate using the real AccelForge library.

        Workflow:
          1. Generate architecture YAML from AccelForgeDesignPoint (physics-based
             energy / area / latency models parameterised by tech node, PE array,
             buffer sizes, dataflow, NoC topology, and data precision).
          2. Generate workload YAML (Einsum matmul) from the workload dict.
          3. Write both to temp files and call Spec.from_yaml() + map_workload_to_arch().
          4. Extract AccelForgeMetrics from Mappings + component area/leak results.
        """
        arch_yaml = self._design_to_arch_yaml(design)
        workload_yaml = self._workload_to_yaml(design, workload)

        arch_path: Optional[str] = None
        workload_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as af:
                af.write(arch_yaml)
                arch_path = af.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as wf:
                wf.write(workload_yaml)
                workload_path = wf.name

            Spec = self._af_model
            spec = Spec.from_yaml(arch_path, workload_path)
            results = spec.map_workload_to_arch(print_progress=False)
            spec_with_area = spec.calculate_component_area_energy_latency_leak()
            return self._extract_metrics(results, spec_with_area, design)

        except Exception as e:
            logger.warning(
                "AccelForge evaluation failed, falling back to analytical: %s", e
            )
            return self._evaluate_analytical(design, workload)

        finally:
            for path in (arch_path, workload_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    # ------------------------------------------------------------------
    # AccelForge YAML generation helpers
    # ------------------------------------------------------------------

    def _design_to_arch_yaml(self, design: AccelForgeDesignPoint) -> str:
        """
        Build an AccelForge architecture YAML string from a design point.

        Energy / area / latency values use technology-node-scaled analytical
        models derived from published SRAM and MAC characterisation data:

        Energy (pJ/bit):
          DRAM  ~10 pJ/bit  (roughly node-independent)
          L2    ~0.5 × (tech/7) pJ/bit
          L1    ~0.1 × (tech/7) pJ/bit
          L0    ~0.01 × (tech/7) pJ/bit  (register file, per PE)

        Compute energy (pJ/MAC, tech-scaled):
          int8  0.05 pJ,  fp16/bf16  0.2 pJ,  fp32  0.5 pJ  @ 7 nm

        Area (mm²/KB SRAM, scales as (tech/7)²):
          L2 / L1: ~0.002 mm²/KB;  L0 (register file): ~0.0002 mm²/KB per PE
          PE compute:  ~0.001 mm² per PE

        Leak power: ~50 mW/mm² at 7 nm, scales linearly with node.

        NoC topology introduces a data-movement energy multiplier applied to
        the L1 action energy (mesh 1×, tree 1.2×, ring 1.5×).
        """
        tech = design.tech_node_nm
        tech_scale = tech / 7.0
        area_scale = tech_scale ** 2
        leak_per_mm2 = 50.0 * tech_scale  # mW/mm²

        # ---- precision ----
        bpe_bits = {"int8": 8, "fp16": 16, "bf16": 16, "fp32": 32}.get(
            design.data_precision, 16
        )

        # ---- energy (pJ/bit) ----
        # DRAM: Micron characterisation ~10 pJ/bit (DDR4/LPDDR5, node-independent)
        # SRAM: Cacti-based model, scales ~linearly with node
        #   [Muralimanohar et al., CACTI 6.0, HP Labs, 2007;
        #    Shao et al., ISCA 2016 — Aladdin models for 40/28/16/7 nm]
        dram_e = 10.0          # pJ/bit — off-chip DRAM
        l2_e = 0.5 * tech_scale    # pJ/bit — large on-chip SRAM (L2)
        l1_e = 0.1 * tech_scale    # pJ/bit — small on-chip SRAM (L1)
        # NoC topology adds extra hop cost on the L1 → PE path
        # [Balfour & Dally, ISCA 2006 — mesh vs ring bandwidth/energy trade-offs]
        noc_scale = {"mesh": 1.0, "ring": 1.5, "tree": 1.2}.get(
            design.noc_topology, 1.0
        )
        l1_eff_e = l1_e * noc_scale
        l0_e = 0.01 * tech_scale  # pJ/bit — register file, per-PE

        # ---- compute energy (pJ/MAC) ----
        # [Horowitz, ISSCC 2014 — "Computing's energy problem (and what we can do)"]
        # int8 ~0.05 pJ, fp16 ~0.2 pJ, fp32 ~0.5 pJ at 7 nm, scales linearly
        mac_e = {
            "int8": 0.05,
            "fp16": 0.2,
            "bf16": 0.2,
            "fp32": 0.5,
        }.get(design.data_precision, 0.2) * tech_scale

        # ---- buffer sizes (bits) ----
        l2_bits = int(design.l2_buffer_kb * 1024 * 8)
        l1_bits = int(design.l1_buffer_kb * 1024 * 8)
        # L0 is per-PE; we expose it as a single aggregate buffer at compute level
        l0_bits_total = int(design.l0_buffer_kb * design.total_pes * 1024 * 8)

        # ---- access latencies (cycles) ----
        clock_ghz = design.clock_mhz / 1000.0
        l2_lat = max(1, round(10.0 / clock_ghz))  # ~10 ns L2
        l1_lat = max(1, round(3.0 / clock_ghz))   # ~3 ns L1

        # ---- area (mm²) ----
        l2_area = design.l2_buffer_kb * 0.002 * area_scale
        l1_area = design.l1_buffer_kb * 0.002 * area_scale
        # Per-PE area: AccelForge multiplies by n_parallel_instances internally,
        # so we provide the single-instance (per-PE) value here.
        l0_area_per_pe = design.l0_buffer_kb * 0.0002 * area_scale
        pe_compute_area_per_pe = 0.001 * area_scale
        mac_area_per_pe = l0_area_per_pe + pe_compute_area_per_pe

        # ---- leak power (mW) — same per-PE convention for MAC node ----
        l2_leak = l2_area * leak_per_mm2
        l1_leak = l1_area * leak_per_mm2
        mac_leak_per_pe = mac_area_per_pe * leak_per_mm2

        total_pes = design.total_pes

        # Mapper time limit — keep evaluation bounded for RL training.
        # max_loops_minus_ranks=1 limits tiling depth, time_limit caps wall time.
        mapper_time = self._config.get("mapper_time_limit", 30.0)

        return (
            "arch:\n"
            "  nodes:\n"
            "\n"
            "  - !Memory\n"
            "    name: MainMemory\n"
            "    size: inf\n"
            "    leak_power: 0\n"
            "    area: 0\n"
            "    tensors: {keep: ~Intermediates, may_keep: All}\n"
            "    actions:\n"
            f"    - {{name: read,  energy: {dram_e:.6f}, latency: 0}}\n"
            f"    - {{name: write, energy: {dram_e:.6f}, latency: 0}}\n"
            "\n"
            "  - !Memory\n"
            "    name: L2Buffer\n"
            f"    size: {l2_bits}\n"
            f"    leak_power: {l2_leak:.6f}\n"
            f"    area: {l2_area:.6f}\n"
            "    tensors: {keep: ~MainMemory, may_keep: All}\n"
            "    actions:\n"
            f"    - {{name: read,  energy: {l2_e:.6f}, latency: {l2_lat}}}\n"
            f"    - {{name: write, energy: {l2_e:.6f}, latency: {l2_lat}}}\n"
            "\n"
            "  - !Memory\n"
            "    name: L1Buffer\n"
            f"    size: {l1_bits}\n"
            f"    leak_power: {l1_leak:.6f}\n"
            f"    area: {l1_area:.6f}\n"
            "    tensors: {keep: ~(MainMemory | L2Buffer), may_keep: All}\n"
            "    actions:\n"
            f"    - {{name: read,  energy: {l1_eff_e:.6f}, latency: {l1_lat}}}\n"
            f"    - {{name: write, energy: {l1_eff_e:.6f}, latency: {l1_lat}}}\n"
            "\n"
            "  - !Compute\n"
            "    name: MAC\n"
            # area / leak_power are per-instance; AccelForge multiplies by
            # n_parallel_instances to produce the total component values.
            f"    area: {mac_area_per_pe:.6f}\n"
            f"    leak_power: {mac_leak_per_pe:.6f}\n"
            f"    n_parallel_instances: {total_pes}\n"
            "    actions:\n"
            f"    - {{name: compute, energy: {mac_e:.6f}, latency: 1}}\n"
            "\n"
            "mapper:\n"
            "  max_loops_minus_ranks: 1\n"
            f"  time_limit: {mapper_time}\n"
        )

    def _workload_to_yaml(
        self,
        design: AccelForgeDesignPoint,
        workload: Optional[Dict[str, Any]],
    ) -> str:
        """
        Build an AccelForge workload YAML string (single-Einsum matmul).

        Dimension derivation (priority order):
          0. Direct m_dim/k_dim/n_dim keys — set by mugraph_to_workload()
             from actual YiRage µGraph tensor shapes (highest fidelity)
          1. Explicit shape keys: batch_size × sequence_length, hidden_dim, output_dim
          2. estimated_flops only: cube-root approximation for M=K=N
          3. Default 128×256×256

        Dimensions are capped at 4096 to keep mapper runtime bounded during RL.
        """
        bpe_bits = {"int8": 8, "fp16": 16, "bf16": 16, "fp32": 32}.get(
            design.data_precision, 16
        )
        _MAX = 4096

        def _clamp(val: int, lo: int = 1, hi: int = _MAX) -> int:
            return min(max(int(val), lo), hi)

        if workload:
            if "m_dim" in workload and "k_dim" in workload and "n_dim" in workload:
                # Priority 0: direct dimensions from mugraph_to_workload()
                m_dim = int(workload.get("effective_m_dim", workload.get("m_dim", 128)))
                k_dim = int(workload["k_dim"])
                n_dim = int(workload["n_dim"])
            elif "batch_size" in workload or "hidden_dim" in workload:
                m_dim = int(
                    workload.get("batch_size", 1)
                    * workload.get("sequence_length", 512)
                )
                k_dim = int(workload.get("hidden_dim", 1024))
                n_dim = int(workload.get("output_dim", k_dim))
            elif "estimated_flops" in workload:
                # matmul flops ≈ 2 × M × K × N; use cube root for square case
                macs = max(workload["estimated_flops"] / 2.0, 1.0)
                side = max(1, round(macs ** (1.0 / 3.0)))
                m_dim = k_dim = n_dim = side
            else:
                m_dim, k_dim, n_dim = 128, 256, 256
        else:
            m_dim, k_dim, n_dim = 128, 256, 256

        m_dim, k_dim, n_dim = _clamp(m_dim), _clamp(k_dim), _clamp(n_dim)

        return (
            "workload:\n"
            "  rank_sizes:\n"
            f"    M: {m_dim}\n"
            f"    K: {k_dim}\n"
            f"    N: {n_dim}\n"
            f"  bits_per_value: {{All: {bpe_bits}}}\n"
            "  einsums:\n"
            "  - name: MatMul\n"
            "    tensor_accesses:\n"
            "    - {name: Input,  projection: [m, k]}\n"
            "    - {name: Weight, projection: [k, n]}\n"
            "    - {name: Output, projection: [m, n], output: True}\n"
        )

    def _extract_metrics(
        self,
        results: Any,
        spec_with_area: Any,
        design: AccelForgeDesignPoint,
    ) -> AccelForgeMetrics:
        """
        Convert AccelForge Mappings + annotated Spec into AccelForgeMetrics.

        Unit conventions (set by _design_to_arch_yaml):
          Energy  → pJ   (arch action energies are in pJ/bit)
          Latency → cycles  (MAC latency = 1 cycle)
          Area    → mm²
          Leak    → mW
        """
        # ---- energy (pJ) ----
        total_energy_pj = float(results.energy())
        per_comp = results.energy(per_component=True)

        mac_energy_pj = float(per_comp.get("MAC", 0.0))
        l1_energy_pj = float(per_comp.get("L1Buffer", 0.0))
        l2_energy_pj = float(per_comp.get("L2Buffer", 0.0))
        main_energy_pj = float(per_comp.get("MainMemory", 0.0))

        compute_energy_pj = mac_energy_pj
        memory_energy_pj = l1_energy_pj + l2_energy_pj + main_energy_pj
        noc_energy_pj = 0.0  # NoC is folded into L1 action energy

        n_computes = max(float(results.n_computes()), 1.0)
        energy_per_op_pj = total_energy_pj / n_computes

        # ---- latency ----
        latency_cycles = max(int(results.latency()), 1)
        latency_ms = latency_cycles / (design.clock_mhz * 1_000.0)

        # ---- throughput ----
        ops_per_mac = PRECISION_OPS_PER_PE.get(design.data_precision, 2.0)
        peak_tops = design.total_pes * ops_per_mac * design.clock_mhz / 1_000_000.0
        achieved_tops = (
            (n_computes * ops_per_mac) / max(latency_ms * 1e9, 1e-12)
        )
        pe_utilization = (
            min(achieved_tops / peak_tops, 1.0) if peak_tops > 0 else 0.0
        )

        # ---- area & leak (mm², mW) from annotated Spec ----
        total_area_mm2 = 0.0
        total_leak_mw = 0.0
        pe_area_mm2 = 0.0
        buffer_area_mm2 = 0.0

        for node in spec_with_area.arch.nodes:
            area = float(node.total_area or node.area or 0.0)
            leak = float(node.total_leak_power or node.leak_power or 0.0)
            total_area_mm2 += area
            total_leak_mw += leak
            if node.name == "MAC":
                pe_area_mm2 = area
            else:
                buffer_area_mm2 += area

        # ---- dynamic power (mW) ----
        latency_s = latency_ms / 1_000.0
        dynamic_power_mw = (
            (total_energy_pj * 1e-12) / latency_s * 1e3 if latency_s > 0 else 0.0
        )
        total_power_mw = dynamic_power_mw + total_leak_mw

        # ---- buffer utilisation — from AccelForge resource_usage() ----
        # resource_usage() returns {component_name: fractional_occupancy (0–1)}
        resource = results.resource_usage()
        buffer_utilization = float(resource.get("L1Buffer", 0.5))

        return AccelForgeMetrics(
            area_mm2=total_area_mm2,
            pe_area_mm2=pe_area_mm2,
            buffer_area_mm2=buffer_area_mm2,
            noc_area_mm2=0.0,
            energy_per_op_pj=energy_per_op_pj,
            compute_energy_pj=compute_energy_pj,
            memory_energy_pj=memory_energy_pj,
            noc_energy_pj=noc_energy_pj,
            latency_cycles=latency_cycles,
            latency_ms=latency_ms,
            total_power_mw=total_power_mw,
            dynamic_power_mw=dynamic_power_mw,
            leak_power_mw=total_leak_mw,
            peak_tops=peak_tops,
            achieved_tops=achieved_tops,
            pe_utilization=pe_utilization,
            buffer_utilization=buffer_utilization,
            noc_bandwidth_utilization=0.0,  # not directly available from AccelForge
            confidence=0.90,  # higher confidence: real AccelForge mapper
        )

    def _evaluate_analytical(
        self,
        design: AccelForgeDesignPoint,
        workload: Optional[Dict[str, Any]],
    ) -> AccelForgeMetrics:
        """
        Analytical model fallback when AccelForge is not available.

        Provides reasonable estimates based on hardware parameters.
        """
        total_pes = design.total_pes
        clock_ghz = design.clock_mhz / 1000.0

        # Precision multiplier (ops per cycle per PE)
        precision_ops = PRECISION_OPS_PER_PE
        ops_per_pe = precision_ops.get(design.data_precision, 2.0)

        # Peak throughput (TOPS)
        peak_tops = total_pes * ops_per_pe * clock_ghz / 1000.0

        # Area model (simplified)
        pe_area = total_pes * 0.01 * (design.tech_node_nm / 7.0)  # mm² per PE
        buffer_area = design.total_buffer_kb * 0.001 * (design.tech_node_nm / 7.0)
        noc_area = total_pes * 0.002 * (design.tech_node_nm / 7.0)
        total_area = pe_area + buffer_area + noc_area

        # Energy model (simplified pJ per op)
        compute_energy = 0.5 * (design.tech_node_nm / 7.0)
        memory_energy = 1.0 * (design.tech_node_nm / 7.0)
        noc_energy = 0.3 * (design.tech_node_nm / 7.0)
        total_energy = compute_energy + memory_energy + noc_energy

        # Power model
        dynamic_power = peak_tops * total_energy * 1000.0  # mW
        leak_power = total_area * 50.0 * (design.tech_node_nm / 7.0)  # mW per mm²
        total_power = dynamic_power + leak_power

        # Latency (workload-dependent)
        if workload:
            flops = workload.get("estimated_flops", 1e9)
            # Assuming some utilization
            utilization = 0.6
            achieved_tops = peak_tops * utilization
            latency_s = flops / (achieved_tops * 1e12) if achieved_tops > 0 else 1.0
            latency_ms = latency_s * 1000.0
            latency_cycles = int(latency_ms * design.clock_mhz * 1000.0)
        else:
            latency_ms = 1.0
            latency_cycles = int(latency_ms * design.clock_mhz * 1000.0)
            utilization = 0.6
            achieved_tops = peak_tops * utilization

        return AccelForgeMetrics(
            area_mm2=total_area,
            pe_area_mm2=pe_area,
            buffer_area_mm2=buffer_area,
            noc_area_mm2=noc_area,
            energy_per_op_pj=total_energy,
            compute_energy_pj=compute_energy,
            memory_energy_pj=memory_energy,
            noc_energy_pj=noc_energy,
            latency_cycles=latency_cycles,
            latency_ms=latency_ms,
            total_power_mw=total_power,
            dynamic_power_mw=dynamic_power,
            leak_power_mw=leak_power,
            peak_tops=peak_tops,
            achieved_tops=achieved_tops,
            pe_utilization=utilization,
            buffer_utilization=0.5,
            noc_bandwidth_utilization=0.4,
            confidence=0.85 if self._af_model is not None else 0.6,
        )

    def to_hardware_profile(
        self,
        design: AccelForgeDesignPoint,
        metrics: Optional[AccelForgeMetrics] = None,
    ) -> "HardwareProfile":
        """
        Convert AccelForge design + metrics to YiRage HardwareProfile.

        Args:
            design: Accelerator design point
            metrics: Pre-computed metrics (evaluated if None)

        Returns:
            HardwareProfile compatible with YiRage RL pipeline
        """
        from .profile import HardwareProfile

        if metrics is None:
            metrics = self.evaluate(design)

        # Map AccelForge PE array to "cores"
        total_cores = design.total_pes

        # Map buffer to shared memory
        shared_memory_kb = design.l1_buffer_kb

        # Map L2 to global memory (approximate)
        global_memory_gb = design.l2_buffer_kb / (1024.0 * 1024.0)

        # Map precision to peak TFLOPS
        peak_tflops_fp16 = metrics.peak_tops if design.data_precision == "fp16" else 0.0
        peak_tflops_fp32 = metrics.peak_tops if design.data_precision == "fp32" else 0.0
        peak_tflops_int8 = metrics.peak_tops if design.data_precision == "int8" else 0.0

        # Build extensions with AccelForge-specific data
        extensions = {
            "accelforge_design": design.to_dict(),
            "accelforge_metrics": metrics.to_dict(),
            "area_mm2": metrics.area_mm2,
            "energy_per_op_pj": metrics.energy_per_op_pj,
            "total_power_mw": metrics.total_power_mw,
            "leak_power_mw": metrics.leak_power_mw,
            "pe_utilization": metrics.pe_utilization,
            "noc_bandwidth_utilization": metrics.noc_bandwidth_utilization,
        }

        return HardwareProfile(
            backend="accelforge",
            device_name=f"AccelForge-{design.pe_array_rows}x{design.pe_array_cols}-{design.dataflow}",
            device_id=0,
            device_count=1,
            compute_capability=(0, 0),
            total_cores=total_cores,
            tensor_core_count=total_cores,  # All PEs can do tensor ops
            warp_size=design.pe_array_cols,  # Row-parallel execution
            global_memory_gb=global_memory_gb,
            shared_memory_kb=shared_memory_kb,
            l1_cache_kb=design.l0_buffer_kb * design.total_pes,
            l2_cache_mb=design.l2_buffer_kb / 1024.0,
            memory_bandwidth_gbps=0.0,  # Determined by NoC
            max_threads_per_block=design.pe_array_cols,
            max_blocks_per_sm=design.pe_array_rows,
            max_shared_memory_per_block=int(design.l1_buffer_kb * 1024),
            max_registers_per_thread=64,
            peak_tflops_fp16=peak_tflops_fp16,
            peak_tflops_fp32=peak_tflops_fp32,
            peak_tflops_int8=peak_tflops_int8,
            supports_tensor_cores=True,
            supports_async_copy=False,
            supports_cooperative_groups=False,
            supports_unified_memory=False,
            extensions=extensions,
        )

    def get_design_space(self) -> List[AccelForgeDesignPoint]:
        """
        Get the default design space for exploration.

        Returns a list of representative design points covering
        the common accelerator design space.
        """
        designs = []

        pe_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]
        dataflows = ["output_stationary", "weight_stationary", "row_stationary"]
        precisions = ["int8", "fp16", "bf16", "fp32"]

        for rows, cols in pe_sizes:
            for dataflow in dataflows:
                for precision in precisions:
                    designs.append(
                        AccelForgeDesignPoint(
                            pe_array_rows=rows,
                            pe_array_cols=cols,
                            dataflow=dataflow,
                            data_precision=precision,
                        )
                    )

        return designs

    def clear_cache(self):
        """Clear evaluation cache."""
        self._cache.clear()


def get_accelforge_availability() -> Dict[str, Any]:
    """Return detailed AccelForge import/version diagnostics."""
    supported_version = _is_supported_accelforge_version(ACCELFORGE_VERSION)
    if not ACCELFORGE_AVAILABLE:
        reason = ACCELFORGE_IMPORT_ERROR or "accelforge package is not installed"
    elif not supported_version:
        reason = (
            f"unsupported accelforge version {ACCELFORGE_VERSION}; expected "
            f">={ACCELFORGE_MIN_VERSION},<{ACCELFORGE_MAX_VERSION_EXCLUSIVE}"
        )
    else:
        reason = ""

    return {
        "available": ACCELFORGE_AVAILABLE and supported_version,
        "installed": ACCELFORGE_AVAILABLE,
        "version": ACCELFORGE_VERSION,
        "minimum_version": ACCELFORGE_MIN_VERSION,
        "maximum_version_exclusive": ACCELFORGE_MAX_VERSION_EXCLUSIVE,
        "supported_version": supported_version,
        "has_spec_api": ACCELFORGE_AVAILABLE,
        "reason": reason,
        "import_error": ACCELFORGE_IMPORT_ERROR,
    }


def is_accelforge_available() -> bool:
    """Check if AccelForge is installed and in YiRage's tested version range."""
    return get_accelforge_availability()["available"]
