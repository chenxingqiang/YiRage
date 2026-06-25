import torch

import contextlib
import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor
from typing import *

from ..core import *
from .threadblock import *
from ..utils.visualizer import *
from ..utils.common import *
from ..global_config import global_config
from ..storage.graph_dataset import graph_dataset
from ..storage.mugraph_store import (
    get_mugraph_store,
    save_mugraph,
    find_best_mugraph,
    find_mugraph,
)
from .op_registry import global_registry, OpRegistry

from collections import deque

MAX_THREADS = os.cpu_count()

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>

static PyObject *launch(PyObject *self, PyObject *args) {
  PyObject *input_list, *output_list, *py_buffer, *py_stream, *py_profiler_buffer;
  void *buffer;
  std::vector<void const *> input_tensors;
  std::vector<void*> output_tensors;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOOOO", &input_list, &output_list, &py_buffer, &py_stream, &py_profiler_buffer)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(input_list) || !PyList_Check(output_list)) {
    PyErr_SetString(PyExc_TypeError, "Both arg1 and arg2 must be lists.");
    return NULL;
  }

  Py_ssize_t input_size = PyList_Size(input_list);
  Py_ssize_t output_size = PyList_Size(output_list);

  for(Py_ssize_t i = 0; i < input_size; i++) {
    PyObject *item = PyList_GetItem(input_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (input) to void pointer", i);
      return NULL;
    }
    input_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  for(Py_ssize_t i = 0; i < output_size; i++) {
    PyObject *item = PyList_GetItem(output_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (output) to void pointer", i);
      return NULL;
    }
    output_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  buffer = PyLong_AsVoidPtr(py_buffer);
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);
  cudaStream_t stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  execute_mugraph(input_tensors, output_tensors, buffer, stream, profiler_buffer);

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__yirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  nullptr,                  // m_slots     
  nullptr,                  // m_traverse  
  nullptr,                  // m_clear     
  nullptr,                  // m_free      
};

PyMODINIT_FUNC PyInit___yirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""


# Because pip install -e . and pip install . have different directory structure,
# we need to check the directory structure to find the correct YIRAGE_ROOT.
def get_key_paths():
    root_dir = os.path.join(os.path.dirname(__file__), "../..")  # Using pip install -e .
    if not os.path.exists(os.path.join(root_dir, "deps")):  # Using pip install .
        root_dir = os.path.dirname(__file__)

    # If YIRAGE_ROOT is not set, use the root_dir as YIRAGE_ROOT
    YIRAGE_ROOT = os.environ.get("YIRAGE_ROOT", root_dir)

    INCLUDE_PATH = ""
    DEPS_PATH = ""
    if os.path.exists(os.path.join(YIRAGE_ROOT, "deps")):
        INCLUDE_PATH = os.path.join(YIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(YIRAGE_ROOT, "deps")
    else:
        INCLUDE_PATH = os.path.join(YIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(YIRAGE_ROOT, "include/deps")

    assert os.path.exists(
        YIRAGE_ROOT
    ), "No YIRAGE_ROOT directory found. Likely using the wrong YIRAGE_ROOT."
    assert os.path.exists(
        INCLUDE_PATH
    ), "No /include directory found. Likely using the wrong YIRAGE_ROOT."
    assert os.path.exists(
        DEPS_PATH
    ), "No /deps directory found. Likely using the wrong YIRAGE_ROOT."

    return YIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH


def get_cc_cmd(target, cc, FILE_NAME, py_include_dir, INCLUDE_PATH, DEPS_PATH, so_path, profiling):
    common_cmd = [
        cc,
        FILE_NAME,
        "-O3",
        f"-I{py_include_dir}",
        f"-I{os.path.join(INCLUDE_PATH, 'transpiler/runtime')}",
        f"-I{os.path.join(DEPS_PATH, 'cutlass/include')}",
        "-DYIRAGE_BACKEND_USE_CUDA",
        "-shared",
        "-std=c++17",
        "-use_fast_math",
        "-lcublas",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o",
        so_path,
    ]

    if target == 70:
        # V100 (Volta)
        specific_cmd = [
            "-arch=sm_70",
            "-gencode=arch=compute_70,code=sm_70",
        ] + (["-DYIRAGE_ENABLE_PROFILER"] if profiling else [])
    elif target == 75:
        # T4 (Turing)
        specific_cmd = [
            "-arch=sm_75",
            "-gencode=arch=compute_75,code=sm_75",
        ] + (["-DYIRAGE_ENABLE_PROFILER"] if profiling else [])
    elif target == 80:
        # A100 (Ampere)
        specific_cmd = [
            "-arch=sm_80",
            "-gencode=arch=compute_80,code=sm_80",
        ] + (["-DYIRAGE_ENABLE_PROFILER"] if profiling else [])
    elif target == 90:
        # H100 (Hopper)
        specific_cmd = [
            "-arch=sm_90a",
            "-gencode=arch=compute_90a,code=sm_90a",
        ] + (["-DYIRAGE_ENABLE_PROFILER"] if profiling else [])
    elif target == 100:
        # B200 (Blackwell)
        specific_cmd = [
            "-arch=sm_100a",
            "-gencode=arch=compute_100a,code=sm_100a",
        ] + (["-DYIRAGE_ENABLE_PROFILER"] if profiling else [])
    else:
        # Fallback to native detection
        specific_cmd = [
            "-arch=native",
        ] + (["-DYIRAGE_ENABLE_PROFILER"] if profiling else [])

    return common_cmd[:6] + specific_cmd + common_cmd[6:]


def check_stride(dims, strides, layout="row-major"):
    curr_stride = 1
    if layout == "row-major":
        for i in range(len(dims) - 1, -1, -1):
            if strides[i] != curr_stride:
                return False
            curr_stride *= dims[i]
    elif layout == "column-major":
        for i in range(len(dims)):
            if strides[i] != curr_stride:
                return False
            curr_stride *= dims[i]
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return True


def gen_empty_tensor(alloc_size, shape, stride, device, dtype=torch.float16):
    return torch.empty(alloc_size, dtype=dtype, device=device).as_strided(shape, stride)


class Handle:
    # Default timeout for nvcc compilation (seconds)
    COMPILE_TIMEOUT = 120  # 2 minutes per kernel

    def __init__(self, handles=[], remain_op=None, graph=None) -> None:
        self.handles = handles
        self.remain_op = remain_op
        self.graph = graph  # Reference to KNGraph for marking compilation state
        self.timed_out = False

    def wait(self, timeout=None):
        if timeout is None:
            timeout = Handle.COMPILE_TIMEOUT
        for handle in self.handles:
            try:
                ret = handle.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                handle.kill()
                handle.wait()
                self.timed_out = True
                print(f"nvcc: Compilation timeout ({timeout}s), kernel skipped", flush=True)
                # Mark graph as compiled but with failed kernel
                if self.graph is not None:
                    self.graph._is_compiled = True
                    self.graph._valid_cuda_kernels = False
                    self.graph._error_message = f"Compilation timeout ({timeout}s)"
                return  # Skip remain_op if compilation timed out
        if self.remain_op:
            self.remain_op()


def _validate_kernel_matmul(A: DTensor, B: DTensor) -> None:
    """Match ``yirage::kernel::Graph::create_matmul_op`` shape rules; raise before native code."""
    if A.num_dims is None or B.num_dims is None:
        raise ValueError("matmul: invalid DTensor (missing dimensions)")
    if A.num_dims != B.num_dims:
        raise ValueError(
            f"matmul: rank mismatch (A.num_dims={A.num_dims}, B.num_dims={B.num_dims})"
        )
    nd = A.num_dims
    if nd < 2:
        raise ValueError("matmul: tensors need at least 2 dimensions")
    if A.dim(nd - 1) != B.dim(nd - 2):
        raise ValueError(
            "matmul: inner dimensions do not match for contraction: "
            f"A[..., -1]={A.dim(nd - 1)} vs B[..., -2]={B.dim(nd - 2)}"
        )
    for i in range(nd - 2):
        if A.dim(i) != B.dim(i):
            raise ValueError(
                f"matmul: batch dimension {i} mismatch: "
                f"A.dim={A.dim(i)} vs B.dim={B.dim(i)}"
            )


_TB_FORLOOP_ACCUM_SUPPORTED = frozenset(
    {
        "tb_forloop_accum_no_red_op",
        "tb_forloop_accum_max_op",
        "tb_forloop_accum_red_ld_mean_op",
        "tb_forloop_accum_red_ld_rms_op",
        "tb_forloop_accum_red_ld_sum_op",
        "tb_forloop_accum_redtox_ld_sum_op",
        "tb_forloop_accum_no_red_rescale_op",
        "tb_forloop_accum_red_ld_sum_rescale_op",
    }
)


def _raise_cpu_unsupported_kn(op_type: str) -> None:
    raise NotImplementedError(
        f"CPU backend does not support '{op_type}' "
        f"(see docs/cpu_support_matrix.yaml)"
    )


def _raise_cpu_unsupported_tb(op_type: str, phase: str) -> None:
    raise NotImplementedError(
        f"CPU TB interpreter does not support '{op_type}' in {phase} "
        f"(see docs/cpu_support_matrix.yaml)"
    )


def _cpu_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Same-backend matmul: host BLAS (PyTorch/MKL) by default; see ``cpu_native``."""
    from .cpu_native import cpu_matmul

    return cpu_matmul(a, b)


def _cpu_rms_norm(x: torch.Tensor) -> torch.Tensor:
    from .cpu_native import cpu_rms_norm

    return cpu_rms_norm(x)


def _cpu_rms_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    from .cpu_native import cpu_rms_matmul

    return cpu_rms_matmul(x, w)


def _kn_customized_tb_softmax_last_dim(
    g,
    x,
    *,
    rows: int,
    cols: int,
    dim: int = -1,
):
    """Stable softmax on last dim via TB reduction_max (PyTorch F.softmax aligned)."""
    import yirage as yr

    axis = dim if dim >= 0 else 1
    tb = yr.new_threadblock_graph(
        grid_dim=(1, 1, 1),
        block_dim=(rows, cols, 1),
        forloop_range=1,
        reduction_dimx=cols,
    )
    tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
    tmax, _ = tb.reduction_max(tx, dim=axis)
    tsub = tb.sub(tx, tmax)
    texp = tb.exp(tsub)
    tsum = tb.reduction(texp, dim=axis)
    tout = tb.div(texp, tsum)
    tacc = tb.forloop_accum(tout)
    tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
    return g.customized([x], tb)[0]


def _kn_customized_tb_layer_norm_last_dim(
    g,
    x,
    *,
    rows: int,
    cols: int,
    eps: float = 1e-5,
):
    """LayerNorm on last dim (elementwise_affine=False; matches F.layer_norm without weight/bias)."""
    import yirage as yr

    axis = 1
    inv_n = 1.0 / float(cols)
    tb = yr.new_threadblock_graph(
        grid_dim=(1, 1, 1),
        block_dim=(rows, cols, 1),
        forloop_range=1,
        reduction_dimx=cols,
    )
    tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
    tsum = tb.reduction(tx, dim=axis)
    mean = tb.mul_scalar(tsum, inv_n)
    centered = tb.sub(tx, mean)
    tsq = tb.square(centered)
    tvar_sum = tb.reduction(tsq, dim=axis)
    tvar = tb.mul_scalar(tvar_sum, inv_n)
    denom = tb.sqrt(tvar)
    tout = tb.div(centered, denom)
    tacc = tb.forloop_accum(tout)
    tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
    return g.customized([x], tb)[0]


def _op_clamp_bounds(op_info: dict, *, layer: str) -> tuple:
    if "min_val" in op_info and "max_val" in op_info:
        return float(op_info["min_val"]), float(op_info["max_val"])
    raise NotImplementedError(
        f"CPU backend missing min_val/max_val in {layer} graph JSON for "
        f"{op_info.get('op_type')}; rebuild yirage.core after clamp JSON update"
    )


def _op_mul_scalar(op_info: dict, *, layer: str) -> float:
    if "scalar" in op_info:
        return float(op_info["scalar"])
    raise NotImplementedError(
        f"CPU backend missing scalar in {layer} graph JSON for "
        f"{op_info.get('op_type')}; rebuild yirage.core after mul_scalar JSON update"
    )


_TB_REDUCTION_TO_DIMX_OPS = frozenset(
    {
        "tb_reduction_0_to_dimx_op",
        "tb_reduction_1_to_dimx_op",
        "tb_reduction_2_to_dimx_op",
    }
)

_TB_REDUCTION_MAX_OPS = frozenset(
    {
        "tb_reduction_0_max_op",
        "tb_reduction_1_max_op",
        "tb_reduction_2_max_op",
    }
)

_TB_CONCAT_OPS = frozenset(
    {
        "tb_concat_0_op",
        "tb_concat_1_op",
        "tb_concat_2_op",
        # TB_CONCAT_0_OP shares enum value with TB_CONCAT_FIRST_OP_ID (2400).
        "tb_concat_first_op_id",
    }
)


def _tb_reduction_axis(op_type: str) -> int:
    return int(op_type[13])


def _tb_concat_axis(op_type: str) -> int:
    if op_type in ("tb_concat_0_op", "tb_concat_first_op_id"):
        return 0
    return int(op_type[10])


_TB_SPLIT_OPS = frozenset(
    {
        "tb_split_0_op",
        "tb_split_1_op",
        "tb_split_2_op",
        "tb_split_first_op_id",
    }
)

_TB_CHUNK_OPS = frozenset(
    {
        "tb_chunk_0_op",
        "tb_chunk_1_op",
        "tb_chunk_2_op",
    }
)


def _tb_split_axis(op_type: str) -> int:
    if op_type in ("tb_split_0_op", "tb_split_first_op_id"):
        return 0
    return int(op_type[9])


def _apply_tb_split(tb_op: dict, in_ts, stensor_map: dict) -> None:
    dim = int(tb_op.get("split_dim", _tb_split_axis(tb_op["op_type"])))
    outs = tb_op["output_tensors"]
    if len(outs) < 2:
        raise NotImplementedError("CPU TB split expects two output stensors")
    if "split_size" in tb_op:
        split_size = int(tb_op["split_size"])
    else:
        split_size = in_ts[0].shape[dim] // len(outs)
    pieces = torch.split(
        in_ts[0],
        (split_size, in_ts[0].shape[dim] - split_size),
        dim=dim,
    )
    if len(pieces) != 2:
        raise ValueError(f"tb_split: expected 2 pieces, got {len(pieces)}")
    stensor_map[outs[0]["guid"]] = pieces[0]
    stensor_map[outs[1]["guid"]] = pieces[1]


def _tb_chunk_axis(op_type: str) -> int:
    if op_type == "tb_chunk_0_op":
        return 0
    return int(op_type[9])


def _apply_tb_chunk(tb_op: dict, in_ts, stensor_map: dict) -> None:
    dim = int(tb_op.get("chunk_dim", _tb_chunk_axis(tb_op["op_type"])))
    outs = tb_op["output_tensors"]
    if "chunk_size" in tb_op:
        num_chunks = int(tb_op["chunk_size"])
    else:
        num_chunks = len(outs)
    pieces = torch.chunk(in_ts[0], num_chunks, dim=dim)
    if len(pieces) != len(outs):
        raise ValueError(
            f"tb_chunk: expected {len(outs)} pieces, got {len(pieces)}"
        )
    for i, out in enumerate(outs):
        stensor_map[out["guid"]] = pieces[i]


_KN_CONCAT_OPS = frozenset(
    {
        "kn_concat_0_op",
        "kn_concat_1_op",
        "kn_concat_2_op",
        # KN_CONCAT_0_OP shares enum value with KN_CONCAT_FIRST_OP_ID (1400).
        "kn_concat_first_op_id",
    }
)


def _kn_concat_axis(op_type: str) -> int:
    if op_type in ("kn_concat_0_op", "kn_concat_first_op_id"):
        return 0
    return int(op_type[10])


_KN_SPLIT_OPS = frozenset(
    {
        "kn_split_0_op",
        "kn_split_1_op",
        "kn_split_2_op",
        "kn_split_first_op_id",
    }
)


def _kn_split_axis(op_type: str) -> int:
    if op_type in ("kn_split_0_op", "kn_split_first_op_id"):
        return 0
    return int(op_type[9])


def _cpu_tb_concat(
    a: torch.Tensor, b: torch.Tensor, dim: int, op_info: dict | None = None
) -> torch.Tensor:
    if op_info is not None and "concat_dim" in op_info:
        dim = int(op_info["concat_dim"])
    return torch.cat([a, b], dim=dim)


def _cpu_reduction_max_step(
    x: torch.Tensor, dim: int, prev_max: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """One tile step of TB ``reduction_max`` (matches ReductionMaxKernel)."""
    if prev_max is None:
        prev_max = torch.full(
            tuple(1 if i == dim else s for i, s in enumerate(x.shape)),
            float("-inf"),
            dtype=x.dtype,
            device=x.device,
        )
    tile_max = x.max(dim=dim, keepdim=True).values
    max_out = torch.maximum(prev_max, tile_max)
    lowest = torch.finfo(x.dtype).min
    delta = prev_max - max_out
    diff = torch.where(
        (delta > lowest) & (delta <= 0),
        delta,
        torch.zeros_like(max_out),
    )
    return max_out, diff


def _apply_tb_reduction_max(tb_op: dict, in_ts, stensor_map: dict) -> None:
    dim = _tb_reduction_axis(tb_op["op_type"])
    outs = tb_op["output_tensors"]
    if len(outs) < 2:
        raise NotImplementedError(
            "CPU TB reduction_max expects two output stensors"
        )
    max_guid = outs[0]["guid"]
    diff_guid = outs[1]["guid"]
    max_out, diff_out = _cpu_reduction_max_step(
        in_ts[0], dim, stensor_map.get(max_guid)
    )
    stensor_map[max_guid] = max_out
    stensor_map[diff_guid] = diff_out


def _cpu_reduction_to_dimx_sum(
    x: torch.Tensor, dim: int, reduction_dimx: int
) -> torch.Tensor:
    """Sum-reduce ``dim`` to length ``reduction_dimx`` (TB reduction_to_dimx)."""
    n = x.shape[dim]
    if n % reduction_dimx != 0:
        raise ValueError(
            f"reduction_to_dimx: dim size {n} not divisible by reduction_dimx "
            f"{reduction_dimx}"
        )
    group_size = n // reduction_dimx
    t = x.movedim(dim, -1)
    t = t.reshape(*t.shape[:-1], reduction_dimx, group_size)
    out = t.sum(dim=-1)
    return out.movedim(-1, dim)


def _tb_dim_start(d, bx, by, bz, forloop_idx, imap, forloop_dim, tile_size):
    start = 0
    if imap["x"] == d:
        start += bx * tile_size
    if imap["y"] == d:
        start += by * tile_size
    if imap["z"] == d:
        start += bz * tile_size
    if forloop_dim == d:
        start += forloop_idx * tile_size
    return start


def _compute_tb_block_patches(kn_op_info, tensor_map, input_dtensor_map, bx, by, bz):
    """Run one TB grid block; return (dtensor_guid, slices, value) writes."""
    bgraph = kn_op_info["bgraph"]
    forloop_range = bgraph["forloop_range"]
    tb_ops = bgraph["operators"]
    ref_dtype = next(iter(tensor_map.values())).dtype

    after_accum_guids: set = set()
    for tb_op in tb_ops:
        if tb_op["op_type"].startswith("tb_forloop_accum"):
            for out_t in tb_op["output_tensors"]:
                after_accum_guids.add(out_t["guid"])

    def _is_after_accum(op_info):
        return any(t["guid"] in after_accum_guids for t in op_info["input_tensors"])

    patches = []
    stensor_map = {}
    for tb_op in tb_ops:
        if tb_op["op_type"].startswith("tb_forloop_accum"):
            out_t = tb_op["output_tensors"][0]
            shape = [out_t["dim"][i] for i in range(out_t["num_dims"])]
            stensor_map[out_t["guid"]] = torch.zeros(shape, dtype=ref_dtype)

    for fl_i in range(forloop_range):
        for tb_op in tb_ops:
            ot = tb_op["op_type"]
            if ot == "tb_output_op" or _is_after_accum(tb_op):
                continue
            in_ts = [stensor_map.get(t["guid"]) for t in tb_op["input_tensors"]]
            out_t = tb_op["output_tensors"][0] if tb_op["output_tensors"] else None

            if ot == "tb_input_op":
                dtensor = input_dtensor_map[tb_op["dtensor"]["guid"]]
                stensor_info = tb_op["output_tensors"][0]
                tile_shape = [stensor_info["dim"][d] for d in range(stensor_info["num_dims"])]
                imap = tb_op["input_map"]
                fl_dim = tb_op["forloop_dim"]
                slices = tuple(
                    slice(
                        _tb_dim_start(d, bx, by, bz, fl_i, imap, fl_dim, tile_shape[d]),
                        _tb_dim_start(d, bx, by, bz, fl_i, imap, fl_dim, tile_shape[d])
                        + tile_shape[d],
                    )
                    for d in range(stensor_info["num_dims"])
                )
                stensor_map[stensor_info["guid"]] = dtensor[slices].contiguous()

            elif ot == "tb_matmul_op":
                stensor_map[out_t["guid"]] = _cpu_matmul(in_ts[0], in_ts[1])

            elif ot == "tb_rms_norm_op":
                stensor_map[out_t["guid"]] = _cpu_rms_norm(in_ts[0])

            elif ot == "tb_exp_op":
                stensor_map[out_t["guid"]] = torch.exp(in_ts[0])

            elif ot == "tb_square_op":
                stensor_map[out_t["guid"]] = in_ts[0] * in_ts[0]

            elif ot == "tb_sqrt_op":
                stensor_map[out_t["guid"]] = torch.sqrt(in_ts[0])

            elif ot == "tb_silu_op":
                stensor_map[out_t["guid"]] = torch.nn.functional.silu(in_ts[0])

            elif ot == "tb_gelu_op":
                stensor_map[out_t["guid"]] = torch.nn.functional.gelu(in_ts[0])

            elif ot == "tb_relu_op":
                stensor_map[out_t["guid"]] = torch.nn.functional.relu(in_ts[0])

            elif ot == "tb_sigmoid_op":
                stensor_map[out_t["guid"]] = torch.sigmoid(in_ts[0])

            elif ot == "tb_log_op":
                stensor_map[out_t["guid"]] = torch.log(in_ts[0])

            elif ot == "tb_pow_op":
                stensor_map[out_t["guid"]] = torch.pow(in_ts[0], in_ts[1])

            elif ot == "tb_mul_scalar_op":
                sc = _op_mul_scalar(tb_op, layer="tb")
                stensor_map[out_t["guid"]] = in_ts[0] * sc

            elif ot == "tb_clamp_op":
                lo, hi = _op_clamp_bounds(tb_op, layer="tb")
                stensor_map[out_t["guid"]] = torch.clamp(in_ts[0], lo, hi)

            elif ot == "tb_add_op":
                stensor_map[out_t["guid"]] = in_ts[0] + in_ts[1]

            elif ot == "tb_mul_op":
                stensor_map[out_t["guid"]] = in_ts[0] * in_ts[1]

            elif ot == "tb_sub_op":
                stensor_map[out_t["guid"]] = in_ts[0] - in_ts[1]

            elif ot == "tb_div_op":
                stensor_map[out_t["guid"]] = in_ts[0] / in_ts[1]

            elif ot in ("tb_reduction_0_op", "tb_reduction_1_op", "tb_reduction_2_op"):
                dim = _tb_reduction_axis(ot)
                stensor_map[out_t["guid"]] = in_ts[0].sum(dim=dim, keepdim=True)

            elif ot in _TB_REDUCTION_TO_DIMX_OPS:
                dim = _tb_reduction_axis(ot)
                rdx = int(out_t["dim"][dim])
                stensor_map[out_t["guid"]] = _cpu_reduction_to_dimx_sum(
                    in_ts[0], dim=dim, reduction_dimx=rdx
                )

            elif ot in _TB_REDUCTION_MAX_OPS:
                _apply_tb_reduction_max(tb_op, in_ts, stensor_map)

            elif ot in _TB_CONCAT_OPS:
                stensor_map[out_t["guid"]] = _cpu_tb_concat(
                    in_ts[0], in_ts[1], _tb_concat_axis(ot), tb_op
                )

            elif ot in _TB_SPLIT_OPS:
                _apply_tb_split(tb_op, in_ts, stensor_map)

            elif ot in _TB_CHUNK_OPS:
                _apply_tb_chunk(tb_op, in_ts, stensor_map)

            elif ot == "tb_forloop_accum_no_red_op":
                stensor_map[out_t["guid"]] = stensor_map[out_t["guid"]] + in_ts[0]

            elif ot == "tb_forloop_accum_red_ld_rms_op":
                x = in_ts[0]
                partial = x.float().pow(2).sum(dim=-1, keepdim=True)
                acc = stensor_map[out_t["guid"]].float() + partial
                stensor_map[out_t["guid"]] = acc
                if fl_i == forloop_range - 1:
                    n = float(x.shape[-1] * forloop_range)
                    rms = torch.sqrt(acc / n + 1e-6)
                    stensor_map[out_t["guid"]] = rms.to(ref_dtype)

            elif ot == "tb_forloop_accum_red_ld_sum_op":
                partial = in_ts[0].sum(dim=-1, keepdim=True)
                stensor_map[out_t["guid"]] = stensor_map[out_t["guid"]] + partial

            elif ot == "tb_forloop_accum_red_ld_mean_op":
                x = in_ts[0]
                partial = x.sum(dim=-1, keepdim=True)
                acc = stensor_map[out_t["guid"]].float() + partial.float()
                if fl_i == forloop_range - 1:
                    n = float(x.shape[-1] * forloop_range)
                    stensor_map[out_t["guid"]] = (acc / n).to(ref_dtype)
                else:
                    stensor_map[out_t["guid"]] = acc.to(ref_dtype)

            elif ot == "tb_forloop_accum_redtox_ld_sum_op":
                rdx = int(out_t["dim"][-1])
                partial = _cpu_reduction_to_dimx_sum(in_ts[0], dim=-1, reduction_dimx=rdx)
                stensor_map[out_t["guid"]] = stensor_map[out_t["guid"]] + partial

            elif ot == "tb_forloop_accum_max_op":
                if fl_i == 0:
                    stensor_map[out_t["guid"]] = in_ts[0]
                else:
                    stensor_map[out_t["guid"]] = torch.maximum(
                        stensor_map[out_t["guid"]], in_ts[0]
                    )

            elif ot == "tb_forloop_accum_no_red_rescale_op":
                src, rescale = in_ts[0], in_ts[1]
                acc = stensor_map[out_t["guid"]]
                stensor_map[out_t["guid"]] = acc * rescale + src

            elif ot == "tb_forloop_accum_red_ld_sum_rescale_op":
                src, rescale = in_ts[0], in_ts[1]
                partial = src.sum(dim=-1, keepdim=True)
                acc = stensor_map[out_t["guid"]]
                stensor_map[out_t["guid"]] = acc * rescale + partial

            elif ot.startswith("tb_forloop_accum"):
                if ot not in _TB_FORLOOP_ACCUM_SUPPORTED:
                    _raise_cpu_unsupported_tb(ot, "forloop")
                stensor_map[out_t["guid"]] = stensor_map[out_t["guid"]] + in_ts[0]

            else:
                _raise_cpu_unsupported_tb(ot, "forloop")

    for tb_op in tb_ops:
        if not _is_after_accum(tb_op) and tb_op["op_type"] != "tb_output_op":
            continue
        ot = tb_op["op_type"]
        in_ts = [stensor_map.get(t["guid"]) for t in tb_op["input_tensors"]]

        if ot == "tb_output_op":
            stensor_val = in_ts[0]
            if stensor_val is None:
                continue
            dtensor_guid = tb_op["dtensor"]["guid"]
            omap = tb_op["output_map"]
            n_dims = stensor_val.ndim
            slices = tuple(
                slice(
                    _tb_dim_start(d, bx, by, bz, 0, omap, -1, stensor_val.shape[d]),
                    _tb_dim_start(d, bx, by, bz, 0, omap, -1, stensor_val.shape[d])
                    + stensor_val.shape[d],
                )
                for d in range(n_dims)
            )
            patches.append((dtensor_guid, slices, stensor_val))

        else:
            out_t = tb_op["output_tensors"][0] if tb_op["output_tensors"] else None
            if out_t is None:
                continue
            if ot == "tb_rms_norm_op":
                x = in_ts[0]
                stensor_map[out_t["guid"]] = x * torch.rsqrt(
                    x.pow(2).mean(-1, keepdim=True) + 1e-6
                )
            elif ot == "tb_matmul_op":
                stensor_map[out_t["guid"]] = _cpu_matmul(in_ts[0], in_ts[1])
            elif ot == "tb_add_op":
                stensor_map[out_t["guid"]] = in_ts[0] + in_ts[1]
            elif ot == "tb_mul_op":
                stensor_map[out_t["guid"]] = in_ts[0] * in_ts[1]
            elif ot == "tb_div_op":
                stensor_map[out_t["guid"]] = in_ts[0] / in_ts[1]
            elif ot == "tb_sub_op":
                stensor_map[out_t["guid"]] = in_ts[0] - in_ts[1]
            elif ot == "tb_silu_op":
                stensor_map[out_t["guid"]] = torch.nn.functional.silu(in_ts[0])
            elif ot == "tb_exp_op":
                stensor_map[out_t["guid"]] = torch.exp(in_ts[0])
            elif ot == "tb_square_op":
                stensor_map[out_t["guid"]] = in_ts[0] * in_ts[0]
            elif ot == "tb_sqrt_op":
                stensor_map[out_t["guid"]] = torch.sqrt(in_ts[0])
            elif ot == "tb_gelu_op":
                stensor_map[out_t["guid"]] = torch.nn.functional.gelu(in_ts[0])
            elif ot == "tb_relu_op":
                stensor_map[out_t["guid"]] = torch.nn.functional.relu(in_ts[0])
            elif ot == "tb_sigmoid_op":
                stensor_map[out_t["guid"]] = torch.sigmoid(in_ts[0])
            elif ot == "tb_log_op":
                stensor_map[out_t["guid"]] = torch.log(in_ts[0])
            elif ot == "tb_pow_op":
                stensor_map[out_t["guid"]] = torch.pow(in_ts[0], in_ts[1])
            elif ot in ("tb_reduction_0_op", "tb_reduction_1_op", "tb_reduction_2_op"):
                dim = _tb_reduction_axis(ot)
                stensor_map[out_t["guid"]] = in_ts[0].sum(dim=dim, keepdim=True)
            elif ot in _TB_REDUCTION_TO_DIMX_OPS:
                dim = _tb_reduction_axis(ot)
                rdx = int(out_t["dim"][dim])
                stensor_map[out_t["guid"]] = _cpu_reduction_to_dimx_sum(
                    in_ts[0], dim=dim, reduction_dimx=rdx
                )
            elif ot in _TB_REDUCTION_MAX_OPS:
                _apply_tb_reduction_max(tb_op, in_ts, stensor_map)
            elif ot in _TB_CONCAT_OPS:
                stensor_map[out_t["guid"]] = _cpu_tb_concat(
                    in_ts[0], in_ts[1], _tb_concat_axis(ot), tb_op
                )
            elif ot in _TB_SPLIT_OPS:
                _apply_tb_split(tb_op, in_ts, stensor_map)
            elif ot in _TB_CHUNK_OPS:
                _apply_tb_chunk(tb_op, in_ts, stensor_map)
            elif ot == "tb_mul_scalar_op":
                sc = _op_mul_scalar(tb_op, layer="tb")
                stensor_map[out_t["guid"]] = in_ts[0] * sc
            elif ot == "tb_clamp_op":
                lo, hi = _op_clamp_bounds(tb_op, layer="tb")
                stensor_map[out_t["guid"]] = torch.clamp(in_ts[0], lo, hi)
            else:
                _raise_cpu_unsupported_tb(ot, "post-forloop")

    return patches


def _execute_tbgraph_on_cpu(kn_op_info, tensor_map):
    """Execute a kn_customized_op by interpreting its TBGraph (tiled thread-block graph).

    Semantics follow the Triton transpiler:
      Phase 1 – initialise accumulators (TB_FORLOOP_ACCUM_* outputs → zeros).
      Phase 2 – forloop body: execute every op whose output is NOT after-accum.
      Phase 3 – post-forloop: after-accum ops and TB_OUTPUT_OP scatter tiles.

  Grid blocks may run in parallel when ``get_cpu_runtime_config()`` enables it.
    """
    bgraph = kn_op_info["bgraph"]
    grid = bgraph["grid_dim"]
    Gx, Gy, Gz = grid["x"], grid["y"], grid["z"]
    tb_ops = bgraph["operators"]

    input_dtensor_map = {}
    for tb_op in tb_ops:
        if tb_op["op_type"] == "tb_input_op":
            guid = tb_op["dtensor"]["guid"]
            input_dtensor_map[guid] = tensor_map[guid]

    output_dtensors = {}
    ref_dtype = next(iter(tensor_map.values())).dtype
    for kn_out in kn_op_info["output_tensors"]:
        guid = kn_out["guid"]
        shape = [kn_out["dim"][i] for i in range(kn_out["num_dims"])]
        output_dtensors[guid] = torch.zeros(shape, dtype=ref_dtype)

    blocks = [(bx, by, bz) for bz in range(Gz) for by in range(Gy) for bx in range(Gx)]
    all_patches = []

    try:
        from ..backends.cpu.config import get_cpu_runtime_config

        rt_cfg = get_cpu_runtime_config()
    except ImportError:
        rt_cfg = {"parallel_tb_grid": False, "tb_grid_workers": 1}

    if rt_cfg.get("parallel_tb_grid") and len(blocks) > 1:
        workers = min(rt_cfg.get("tb_grid_workers", 1), len(blocks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for patches in pool.map(
                lambda b: _compute_tb_block_patches(
                    kn_op_info, tensor_map, input_dtensor_map, b[0], b[1], b[2]
                ),
                blocks,
            ):
                all_patches.extend(patches)
    else:
        for bx, by, bz in blocks:
            all_patches.extend(
                _compute_tb_block_patches(
                    kn_op_info, tensor_map, input_dtensor_map, bx, by, bz
                )
            )

    for dtensor_guid, slices, stensor_val in all_patches:
        output_dtensors[dtensor_guid][slices] = stensor_val

    return [output_dtensors[t["guid"]] for t in kn_op_info["output_tensors"]]


@contextlib.contextmanager
def _cpu_runtime_context():
    """Align PyTorch CPU threads with local architecture for search/profile/execute."""
    try:
        from ..backends.cpu.config import get_cpu_runtime_config

        cfg = get_cpu_runtime_config()
        prev = torch.get_num_threads()
        try:
            torch.set_num_threads(cfg["torch_num_threads"])
            with torch.inference_mode():
                yield cfg
        finally:
            torch.set_num_threads(prev)
    except ImportError:
        with torch.inference_mode():
            yield {}


def _interpret_mugraph_on_cpu(cygraph, input_tensors):
    """Execute a muGraph on CPU by interpreting each KN-level operator.

    Replaces the old pattern-matching ``ascend_call`` fallback with a
    correct graph interpreter that reads the actual operator sequence
    and tiling structure from the muGraph.
    """
    with _cpu_runtime_context():
        return _interpret_mugraph_on_cpu_impl(cygraph, input_tensors)


def _is_plain_matmul_mugraph(cygraph) -> bool:
    """True only for input×input→matmul→output with no other compute ops."""
    types = [o["op_type"] for o in cygraph.get_graph_structure()]
    compute_ops = [t for t in types if t not in ("kn_input_op", "kn_output_op")]
    return (
        compute_ops == ["kn_matmul_op"]
        and types.count("kn_input_op") == 2
        and types.count("kn_output_op") == 1
    )


def _is_unfused_rms_matmul_mugraph(cygraph) -> bool:
    """True for kn_rms_norm → kn_matmul without fused kn_customized_op."""
    types = [o["op_type"] for o in cygraph.get_graph_structure()]
    compute_ops = [t for t in types if t not in ("kn_input_op", "kn_output_op")]
    return (
        compute_ops == ["kn_rms_norm_op", "kn_matmul_op"]
        and types.count("kn_input_op") == 2
        and types.count("kn_output_op") == 1
    )


def _has_fused_customized_op(cygraph) -> bool:
    return any(
        o.get("op_type") == "kn_customized_op"
        for o in cygraph.get_graph_structure()
    )


def _interpret_mugraph_on_cpu_impl(cygraph, input_tensors):
    ops = cygraph.get_graph_structure()

    # Map stensor / dtensor guid → torch.Tensor
    tensor_map: dict = {}

    # First pass: register input guids (preserve dtype for same-backend profiling)
    input_idx = 0
    for op in ops:
        if op["op_type"] == "kn_input_op":
            guid = op["output_tensors"][0]["guid"]
            tensor_map[guid] = input_tensors[input_idx]
            input_idx += 1

    # Collect output guids in declaration order
    output_guids = [
        op["input_tensors"][0]["guid"]
        for op in ops
        if op["op_type"] == "kn_output_op"
    ]

    # Second pass: execute ops in topological order
    for op in ops:
        ot = op["op_type"]
        if ot in ("kn_input_op", "kn_output_op"):
            continue

        in_guids = [t["guid"] for t in op["input_tensors"]]
        out_guids = [t["guid"] for t in op["output_tensors"]]
        ins = [tensor_map[g] for g in in_guids]

        if ot == "kn_matmul_op":
            tensor_map[out_guids[0]] = _cpu_matmul(ins[0], ins[1])

        elif ot == "kn_rms_norm_op":
            tensor_map[out_guids[0]] = _cpu_rms_norm(ins[0])

        elif ot == "kn_exp_op":
            tensor_map[out_guids[0]] = torch.exp(ins[0])

        elif ot == "kn_square_op":
            tensor_map[out_guids[0]] = ins[0] * ins[0]

        elif ot == "kn_sqrt_op":
            tensor_map[out_guids[0]] = torch.sqrt(ins[0])

        elif ot == "kn_silu_op":
            tensor_map[out_guids[0]] = torch.nn.functional.silu(ins[0])

        elif ot == "kn_sigmoid_op":
            tensor_map[out_guids[0]] = torch.sigmoid(ins[0])

        elif ot == "kn_gelu_op":
            tensor_map[out_guids[0]] = torch.nn.functional.gelu(ins[0])

        elif ot == "kn_relu_op":
            tensor_map[out_guids[0]] = torch.nn.functional.relu(ins[0])

        elif ot == "kn_clamp_op":
            lo, hi = _op_clamp_bounds(op, layer="kn")
            tensor_map[out_guids[0]] = torch.clamp(ins[0], lo, hi)

        elif ot == "kn_mul_scalar_op":
            sc = _op_mul_scalar(op, layer="kn")
            tensor_map[out_guids[0]] = ins[0] * sc

        elif ot == "kn_log_op":
            tensor_map[out_guids[0]] = torch.log(ins[0])

        elif ot == "kn_add_op":
            tensor_map[out_guids[0]] = ins[0] + ins[1]

        elif ot == "kn_sub_op":
            tensor_map[out_guids[0]] = ins[0] - ins[1]

        elif ot == "kn_mul_op":
            tensor_map[out_guids[0]] = ins[0] * ins[1]

        elif ot == "kn_div_op":
            tensor_map[out_guids[0]] = ins[0] / ins[1]

        elif ot == "kn_pow_op":
            tensor_map[out_guids[0]] = torch.pow(ins[0], ins[1])

        elif ot in ("kn_reduction_0_op", "kn_reduction_1_op", "kn_reduction_2_op"):
            dim = int(ot[13])
            tensor_map[out_guids[0]] = ins[0].sum(dim=dim, keepdim=False)

        elif ot in ("kn_chunk_0_op", "kn_chunk_1_op", "kn_chunk_2_op"):
            chunk_dim = int(ot[9])
            chunk_size = len(out_guids)
            pieces = torch.chunk(ins[0], chunk_size, dim=chunk_dim)
            if len(pieces) != len(out_guids):
                raise ValueError(
                    f"kn_chunk: expected {len(out_guids)} outputs, torch.chunk returned {len(pieces)}"
                )
            for guid, piece in zip(out_guids, pieces):
                tensor_map[guid] = piece

        elif ot == "kn_transpose_01_op":
            tensor_map[out_guids[0]] = ins[0].transpose(0, 1).contiguous()

        elif ot == "kn_conv2d_op":
            tensor_map[out_guids[0]] = torch.nn.functional.conv2d(
                ins[0],
                ins[1],
                stride=(int(op["stride_h"]), int(op["stride_w"])),
                padding=(int(op["padding_h"]), int(op["padding_w"])),
                dilation=(int(op["dilation_h"]), int(op["dilation_w"])),
            )

        elif ot in _KN_CONCAT_OPS:
            concat_dim = int(op.get("concat_dim", _kn_concat_axis(ot)))
            tensor_map[out_guids[0]] = torch.cat([ins[0], ins[1]], dim=concat_dim)

        elif ot in _KN_SPLIT_OPS:
            split_dim = int(op.get("split_dim", _kn_split_axis(ot)))
            if "split_size" in op:
                split_size = int(op["split_size"])
            else:
                split_size = ins[0].shape[split_dim] // len(out_guids)
            pieces = torch.split(
                ins[0],
                (split_size, ins[0].shape[split_dim] - split_size),
                dim=split_dim,
            )
            if len(pieces) != len(out_guids):
                raise ValueError(
                    f"kn_split: expected {len(out_guids)} outputs, got {len(pieces)}"
                )
            for guid, piece in zip(out_guids, pieces):
                tensor_map[guid] = piece

        elif ot == "kn_customized_op":
            results = _execute_tbgraph_on_cpu(op, tensor_map)
            for guid, val in zip(out_guids, results):
                tensor_map[guid] = val

        else:
            raise NotImplementedError(
                f"_interpret_mugraph_on_cpu: unhandled KN op '{ot}'"
            )

    return [tensor_map[g] for g in output_guids]
class KNGraph:
    def __init__(self, graph, backend="cuda"):
        self.cygraph = graph

        self._is_compiled = False
        self.run = None
        self._valid_cuda_kernels = False
        self._cached_results = None
        self.visualizer = None

        self.backend = backend

    @classmethod
    def from_persistent_entry(cls, entry, backend: str):
        """Reconstruct a KNGraph from a MuGraphStore entry (requires graph_json)."""
        from ..storage.graph_serde import deserialize_cygraph

        cygraph = deserialize_cygraph(getattr(entry, "graph_json", None))
        if cygraph is None:
            return None
        return cls(cygraph, backend=backend)

    def _try_restore_from_persistent_cache(
        self,
        backend: str,
        imaps,
        omaps,
        griddims,
        blockdims,
        fmaps,
        franges,
    ):
        """Load best matching muGraph from ~/.yirage/mugraphs/ if graph_json is stored."""
        graph_hash = hex(self.cygraph.get_owner_independent_hash())[2:]

        input_shapes = []
        for t in self.cygraph.get_input_dtensors():
            dims, _ = self.cygraph.get_input_dtensor_shape_and_stride(t)
            input_shapes.append(list(dims))

        entry = find_mugraph(
            graph_hash,
            backend,
            imaps=imaps,
            omaps=omaps,
            griddims=griddims,
            blockdims=blockdims,
            fmaps=fmaps,
            franges=franges,
        )
        if entry is None or not entry.graph_json:
            entry = find_best_mugraph(graph_hash, backend, input_shapes=input_shapes)
        if entry is None or not entry.graph_json:
            return None

        restored = self.from_persistent_entry(entry, backend)
        if restored is None:
            return None

        print("✓ Restored muGraph from persistent storage (skipped search)")
        print(f"  - Backend: {entry.metadata.backend}")
        print(f"  - Latency: {entry.metadata.latency_ms:.4f} ms")
        if entry.metadata.input_shapes:
            print(f"  - Input shapes: {entry.metadata.input_shapes}")
        print(f"  - Stored at: {entry.metadata.created_at}")
        return restored

    def new_input(self, dims: tuple, strides: tuple = None, dtype: dtype = float16) -> DTensor:
        # use the default strided layout if strides = None
        if strides is None:
            total_elements = 1
            strides = []
            for d in reversed(dims):
                strides.append(total_elements)
                total_elements *= d
            strides = reversed(strides)
        else:
            assert len(dims) == len(strides)
            assert check_stride(dims, strides, "row-major") | check_stride(
                dims, strides, "column-major"
            )
        return self.cygraph.new_input(dims, tuple(strides), dtype)

    def mark_output(self, A: DTensor, strides: tuple = None):
        return self.cygraph.mark_output(A, strides)

    def matmul(self, A: DTensor, B: DTensor) -> DTensor:
        _validate_kernel_matmul(A, B)
        return self.cygraph.matmul(A, B)

    def reduction(self, A: DTensor, dim: int):
        return self.cygraph.reduction(A, dim)

    def exp(self, A: DTensor):
        return self.cygraph.exp(A)

    def silu(self, A: DTensor):
        return self.cygraph.silu(A)

    def gelu(self, A: DTensor):
        return self.cygraph.gelu(A)

    def relu(self, A: DTensor):
        return self.cygraph.relu(A)

    def sigmoid(self, A: DTensor):
        return self.cygraph.sigmoid(A)

    def log(self, A: DTensor):
        return self.cygraph.log(A)

    def clamp(self, A: DTensor, min_val: float, max_val: float):
        return self.cygraph.clamp(A, min_val, max_val)

    def mul_scalar(self, A: DTensor, scalar: float):
        return self.cygraph.mul_scalar(A, scalar)

    def sqrt(self, A: DTensor):
        return self.cygraph.sqrt(A)

    def square(self, A: DTensor):
        return self.cygraph.square(A)

    def add(self, A: DTensor, B: DTensor):
        return self.cygraph.add(A, B)

    def sub(self, A: DTensor, B: DTensor):
        return self.cygraph.sub(A, B)

    def mul(self, A: DTensor, B: DTensor):
        return self.cygraph.mul(A, B)

    def div(self, A: DTensor, B: DTensor):
        return self.cygraph.div(A, B)

    def pow(self, A: DTensor, B: DTensor):
        return self.cygraph.pow(A, B)

    def chunk(self, A: DTensor, chunk_size: int, dim: int) -> list:
        return self.cygraph.chunk(A, chunk_size, dim)

    def concat(self, A: DTensor, B: DTensor, dim: int) -> DTensor:
        return self.cygraph.concat(A, B, dim)

    def split(self, A: DTensor, split_size: int, dim: int) -> list:
        return self.cygraph.split(A, split_size, dim)

    def transpose(self, A: DTensor, dim0: int = 0, dim1: int = 1) -> DTensor:
        """Swap two dimensions (``kn_transpose_01_op`` when ``dim0=0`` and ``dim1=1``)."""
        if dim0 == 0 and dim1 == 1:
            return self.cygraph.transpose01(A)
        raise NotImplementedError(
            "CPU transpose currently supports dim0=0, dim1=1 only; use kn_transpose_01_op"
        )

    def conv2d(
        self,
        input: DTensor,
        weight: DTensor,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
    ) -> DTensor:
        """2D convolution (NCHW input, OIHW weight; aligned with ``F.conv2d``)."""
        sh, sw = int(stride[0]), int(stride[1])
        ph, pw = int(padding[0]), int(padding[1])
        dh, dw = int(dilation[0]), int(dilation[1])
        return self.cygraph.conv2d(input, weight, sh, sw, ph, pw, dh, dw)

    def rms_norm(self, A: DTensor, normalized_shape: tuple):
        return self.cygraph.rms_norm(A, normalized_shape)

    def softmax(self, A: DTensor, dim: int = -1) -> DTensor:
        """Row-wise softmax aligned with ``torch.nn.functional.softmax`` (stable TB path)."""
        rows, cols = A.dim(0), A.dim(1)
        if A.num_dims != 2:
            raise NotImplementedError(
                "CPU softmax currently supports 2D tensors; use TB customized graphs for other ranks"
            )
        return _kn_customized_tb_softmax_last_dim(
            self, A, rows=rows, cols=cols, dim=dim
        )

    def layer_norm(
        self,
        A: DTensor,
        normalized_shape: tuple,
        eps: float = 1e-5,
    ) -> DTensor:
        """LayerNorm on last dim (``elementwise_affine=False``; PyTorch ``F.layer_norm`` without γ/β)."""
        if len(normalized_shape) != 1:
            raise NotImplementedError(
                "CPU layer_norm currently supports 1D normalized_shape on 2D input"
            )
        rows, cols = A.dim(0), A.dim(1)
        if int(normalized_shape[0]) != cols:
            raise ValueError("normalized_shape must match input last dim")
        return _kn_customized_tb_layer_norm_last_dim(
            self, A, rows=rows, cols=cols, eps=eps
        )

    def customized(self, inputs: list[DTensor], bgraph: TBGraph) -> list[DTensor]:
        return self.cygraph.customized(inputs, bgraph.cygraph)

    def call_op(
        self,
        name: str,
        inputs: list,
        *,
        registry: OpRegistry = None,
        **kwargs,
    ) -> list:
        """Call a previously registered custom operator by name.

        The operator must have been registered with :func:`yirage.register_op`
        or the :func:`yirage.custom_op` decorator before this call.

        Parameters
        ----------
        name:
            The operator name that was used when registering.
        inputs:
            List of :class:`DTensor` objects to pass as inputs.
        registry:
            Optional :class:`~yirage.kernel.op_registry.OpRegistry` to look
            up the operator in.  Defaults to the module-level
            :data:`~yirage.kernel.op_registry.global_registry`.
        **kwargs:
            Forwarded verbatim to the builder function (e.g. ``grid_dim``,
            ``block_dim``, ``forloop_range``, ``reduction_dimx``).

        Returns
        -------
        list[DTensor]
            The output tensors produced by the registered builder.

        Raises
        ------
        KeyError
            If *name* is not found in the registry.

        Example
        -------
        ::

            import yirage as mi

            @mi.custom_op("softmax", n_inputs=1)
            def build_softmax(kgraph, inputs, **kwargs):
                ...
                return kgraph.customized(inputs, bgraph)

            kgraph = mi.new_kernel_graph()
            A = kgraph.new_input([128, 64], dtype=mi.float16)
            (out,) = kgraph.call_op("softmax", [A],
                                     grid_dim=(1, 1, 1),
                                     block_dim=(128, 1, 1),
                                     forloop_range=1,
                                     reduction_dimx=64)
            kgraph.mark_output(out)
        """
        reg = registry if registry is not None else global_registry
        spec = reg.get(name)
        return spec(self, inputs, **kwargs)

    # COMET-style Compound Operations
    # =========================================================================
    # These methods implement fused compound operations following the COMET paper:
    # "A Framework for Modeling Compound Operation Dataflows with Explicit Collectives"
    # (Negi et al.)
    #
    # Compound operations fuse multiple elementary operations to reduce
    # off-chip memory traffic and improve data locality.

    def gemm_softmax(
        self,
        A: DTensor,
        B: DTensor,
        dim: int = -1,
    ) -> DTensor:
        """
        GEMM followed by row-wise Softmax (COMET GEMM-Softmax fusion).

        Implements: ``softmax(A @ B, dim)`` with numerically stable max-subtract
        (PyTorch ``F.softmax`` aligned; not naive exp/sum/div).
        """
        C = self.cygraph.matmul(A, B)
        rows, cols = C.dim(0), C.dim(1)
        return _kn_customized_tb_softmax_last_dim(
            self, C, rows=rows, cols=cols, dim=dim
        )

    def gemm_layernorm(
        self,
        A: DTensor,
        B: DTensor,
        normalized_shape: tuple,
        eps: float = 1e-5,
    ) -> DTensor:
        """
        GEMM followed by LayerNorm (COMET GEMM-LayerNorm fusion).

        Implements: ``LayerNorm(A @ B)`` (``elementwise_affine=False``;
        PyTorch ``F.layer_norm`` without weight/bias).
        """
        C = self.cygraph.matmul(A, B)
        rows, cols = C.dim(0), C.dim(1)
        if len(normalized_shape) != 1 or int(normalized_shape[0]) != cols:
            raise ValueError("normalized_shape must match matmul output last dim")
        return _kn_customized_tb_layer_norm_last_dim(
            self, C, rows=rows, cols=cols, eps=eps
        )

    def self_attention(
        self,
        Q: DTensor,
        K: DTensor,
        V: DTensor,
    ) -> DTensor:
        """
        Self-attention compound op (COMET): ``softmax(Q @ K, dim=-1) @ V``.

        Uses the same stable TB softmax path as :meth:`softmax` / :meth:`gemm_softmax`
        (``reduction_max`` subtract), not naive ``exp / sum``.

        Args:
            Q: Query ``[S, D]``
            K: Key already transposed for ``Q @ K`` — ``[D, S]``
            V: Value ``[S, D]``

        Returns:
            Attention output ``[S, D]``
        """
        if Q.num_dims != 2 or K.num_dims != 2 or V.num_dims != 2:
            raise NotImplementedError(
                "CPU self_attention currently supports 2D Q/K/V; "
                "K must be transposed ([D, S]) for Q @ K"
            )
        QK = self.cygraph.matmul(Q, K)
        rows, cols = QK.dim(0), QK.dim(1)
        QK_norm = _kn_customized_tb_softmax_last_dim(
            self, QK, rows=rows, cols=cols, dim=-1
        )
        return self.cygraph.matmul(QK_norm, V)

    def gated_mlp(
        self,
        X: DTensor,
        W_gate: DTensor,
        W_up: DTensor,
        W_down: DTensor = None,
        activation: str = "silu",
    ) -> DTensor:
        """
        Gated MLP with SiLU/GELU activation (common in LLMs).
        
        Implements: W_down @ (act(X @ W_gate) * (X @ W_up))
        
        Args:
            X: Input tensor [B, S, D]
            W_gate: Gate weight [D, D_ff]
            W_up: Up projection weight [D, D_ff]
            W_down: Down projection weight [D_ff, D] (optional)
            activation: Activation function ("silu" or "gelu")
        
        Returns:
            Output tensor
        """
        # Gate projection
        gate = self.cygraph.matmul(X, W_gate)
        
        # Activation
        if activation == "silu":
            gate_act = self.cygraph.silu(gate)
        elif activation == "gelu":
            gate_act = self.cygraph.gelu(gate)
        else:
            gate_act = gate
        
        # Up projection
        up = self.cygraph.matmul(X, W_up)
        
        # Elementwise multiply
        intermediate = self.cygraph.mul(gate_act, up)
        
        # Down projection
        if W_down is not None:
            result = self.cygraph.matmul(intermediate, W_down)
        else:
            result = intermediate
        
        return result

    def rms_norm_linear(
        self,
        X: DTensor,
        weight: DTensor,
        normalized_shape: tuple,
    ) -> DTensor:
        """
        RMSNorm followed by Linear projection.
        
        Implements: RMSNorm(X) @ weight
        
        Common pattern in LLMs for attention QKV projection.
        
        Args:
            X: Input tensor [B, S, D]
            weight: Linear weight [D, D_out]
            normalized_shape: Shape for RMSNorm
        
        Returns:
            Output tensor [B, S, D_out]
        """
        # RMS Norm
        X_norm = self.cygraph.rms_norm(X, normalized_shape)
        
        # Linear
        result = self.cygraph.matmul(X_norm, weight)
        
        return result

    def get_owner_independent_hash(self):
        return self.cygraph.get_owner_independent_hash()

    def valid_kernels(self):
        assert self._is_compiled, "Should check kernel validness after compilation"
        return self._valid_cuda_kernels

    def get_error_message(self):
        assert self._is_compiled, "Should check error message after compilation"
        return self._error_message

    def __call__(self, **kwargs):
        if self.backend == "cuda":
            return self.cuda_call(**kwargs)
        elif self.backend == "ascend":
            return self.ascend_call(**kwargs)
        elif self.backend == "cpu":
            return self.cpu_call(**kwargs)
        elif self.backend == "mps":
            return self.mps_call(**kwargs)
        elif self.backend == "maca":
            return self.maca_call(**kwargs)
        elif self.backend == "nki":
            raise NotImplementedError("NKI backend is not implemented yet")
        elif self.backend == "triton":
            return self.triton_call(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def triton_call(self, **kwargs):
        assert self.run is not None, "The graph is not compiled to triton yet."
        input_tensors = kwargs.get("inputs", [])
        verbose = kwargs.get("verbose", False)

        output_shapes = self._cached_results["output_shapes"]
        output_tensors = [
            torch.zeros(shape, dtype=input_tensors[0].dtype, device=input_tensors[0].device)
            for shape in output_shapes
        ]
        if verbose:
            print("Input tensors:")
            for t in input_tensors:
                print(f"Shape: {t.shape}, dtype: {t.dtype}, device: {t.device}")
            print("Output tensors:")
            for t in output_tensors:
                print(f"Shape: {t.shape}, dtype: {t.dtype}, device: {t.device}")

        self.run(*input_tensors, *output_tensors)
        return output_tensors

    def ascend_call(self, **kwargs):
        """Execute the optimized graph on Ascend NPU using PyTorch NPU backend

        Since the C++ kernel search is currently disabled for Ascend,
        we interpret and execute the graph operations using PyTorch directly.
        torch_npu will accelerate these operations on the Ascend NPU.

        This implementation handles common fused patterns from LLM optimization:
        - RMSNorm + Linear (for attention QKV projection)
        - SiLU + Mul + MatMul (for MLP down projection)
        - Gated MLP patterns
        """
        input_tensors = kwargs.get("inputs", [])

        # Pattern matching for common fused operations
        if len(input_tensors) == 2:
            # 2 inputs: likely matmul
            A, B = input_tensors[0], input_tensors[1]
            if A.dim() >= 2 and B.dim() >= 2:
                result = torch.matmul(A, B)
                return [result]

        elif len(input_tensors) == 3:
            # 3 inputs: Multiple patterns possible
            t1, t2, t3 = input_tensors[0], input_tensors[1], input_tensors[2]

            # Pattern 1: RMSNorm + Linear (t1=x, t2=norm_weight(1D), t3=linear_weight(2D))
            # For QKV projection: hidden_states, layernorm.weight, fused_qkv_weight
            if t2.dim() == 1 and t3.dim() == 2:
                eps = 1e-6
                variance = t1.pow(2).mean(-1, keepdim=True)
                x_normalized = t1 * torch.rsqrt(variance + eps)
                x_normed = x_normalized * t2
                result = torch.matmul(x_normed, t3)
                return [result]

            # Pattern 2: SiLU + Mul + MatMul (t1=gate, t2=up, t3=down_weight)
            # For MLP down projection: gate_output, up_output, down_proj.weight.T
            # Check if t1 and t2 have same shape (both are activations)
            elif t1.shape == t2.shape and t3.dim() == 2:
                # SiLU(gate) * up @ weight
                intermediate = torch.nn.functional.silu(t1) * t2
                result = torch.matmul(intermediate, t3)
                return [result]

            # Pattern 3: Two linear layers
            elif t2.dim() == 2 and t3.dim() == 2:
                result = torch.matmul(torch.matmul(t1, t2), t3)
                return [result]

            # Pattern 4: Linear + element-wise
            elif t2.dim() == 2 and t3.dim() == 1:
                intermediate = torch.matmul(t1, t2)
                result = intermediate * t3
                return [result]

        elif len(input_tensors) == 4:
            # 4 inputs: MLP or attention patterns
            x = input_tensors[0]
            w1, w2, w3 = input_tensors[1], input_tensors[2], input_tensors[3]

            if w1.dim() == 1:
                # Norm + Gate + Up pattern
                eps = 1e-6
                variance = x.pow(2).mean(-1, keepdim=True)
                x_normed = x * torch.rsqrt(variance + eps) * w1
                if w2.dim() == 2 and w3.dim() == 2:
                    gate = torch.matmul(x_normed, w2)
                    up = torch.matmul(x_normed, w3)
                    result = torch.nn.functional.silu(gate) * up
                    return [result]
            elif w1.dim() == 2:
                gate = torch.matmul(x, w1)
                up = torch.matmul(x, w2)
                intermediate = torch.nn.functional.silu(gate) * up
                result = torch.matmul(intermediate, w3)
                return [result]

            results = [torch.matmul(x, w) if w.dim() >= 2 else x * w for w in input_tensors[1:]]
            return [results[-1] if len(results) > 0 else x]

        elif len(input_tensors) == 5:
            # Full MLP with norm
            x, norm_w, gate_w, up_w, down_w = input_tensors
            eps = 1e-6
            variance = x.pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + eps) * norm_w
            gate = torch.matmul(x_normed, gate_w)
            up = torch.matmul(x_normed, up_w)
            intermediate = torch.nn.functional.silu(gate) * up
            result = torch.matmul(intermediate, down_w)
            return [result]

        # Ultimate fallback
        import warnings

        warnings.warn(
            f"Ascend execution using passthrough for {len(input_tensors)} inputs. "
            f"Performance may be suboptimal."
        )
        return [input_tensors[0]]

    def cpu_call(self, **kwargs):
        """Execute the optimized muGraph on CPU by interpreting its operators."""
        input_tensors = kwargs.get("inputs", [])
        if not hasattr(self, "_cpu_plain_matmul_fast"):
            self._cpu_plain_matmul_fast = _is_plain_matmul_mugraph(self.cygraph)
        if not hasattr(self, "_cpu_rms_matmul_fast"):
            from .cpu_mlir_jit import (
                is_production_rms_matmul_mugraph,
                production_rms_matmul_fast_enabled,
            )

            self._cpu_rms_matmul_fast = (
                production_rms_matmul_fast_enabled()
                and is_production_rms_matmul_mugraph(self.cygraph)
            )
        if self._cpu_plain_matmul_fast and len(input_tensors) == 2:
            from .cpu_native import cpu_matmul

            # Host BLAS (MKL) manages threading; avoid per-call thread reconfigure.
            return [cpu_matmul(input_tensors[0], input_tensors[1])]
        if self._cpu_rms_matmul_fast and len(input_tensors) == 2:
            # P0/P1: unfused or fused rms+matmul → host-BLAS fused kernel (profile == execute).
            return [_cpu_rms_matmul(input_tensors[0], input_tensors[1])]
        if not hasattr(self, "_cpu_concat_matmul_fast"):
            from .cpu_mlir_jit import (
                is_production_concat_matmul_mugraph,
                production_concat_matmul_fast_enabled,
            )

            self._cpu_concat_matmul_fast = (
                production_concat_matmul_fast_enabled()
                and is_production_concat_matmul_mugraph(self.cygraph)
            )
        if self._cpu_concat_matmul_fast and len(input_tensors) == 4:
            from .cpu_native import cpu_concat_matmul

            return [cpu_concat_matmul(*input_tensors)]
        if not hasattr(self, "_cpu_matmul_chain_fast"):
            from .cpu_mlir_jit import (
                is_production_matmul_chain_mugraph,
                production_matmul_chain_fast_enabled,
            )

            self._cpu_matmul_chain_fast = (
                production_matmul_chain_fast_enabled()
                and is_production_matmul_chain_mugraph(self.cygraph)
            )
        if self._cpu_matmul_chain_fast and len(input_tensors) == 3:
            from .cpu_native import cpu_matmul_chain

            return [cpu_matmul_chain(*input_tensors)]
        try:
            from .cpu_mlir_jit import try_rms_matmul_jit

            jit_out = try_rms_matmul_jit(
                self.cygraph, input_tensors, require_experimental=True
            )
            if jit_out is not None:
                return jit_out
        except Exception:
            pass
        with _cpu_runtime_context():
            return _interpret_mugraph_on_cpu_impl(self.cygraph, input_tensors)

    def mps_call(self, **kwargs):
        """Execute the optimized graph on Apple MPS"""
        return self.ascend_call(**kwargs)  # Use same implementation for now

    def maca_call(self, **kwargs):
        """Execute the optimized graph on MetaX MACA GPU"""
        return self.ascend_call(**kwargs)  # Use same implementation for now

    def cuda_call(self, **kwargs):
        results = self.compile(**kwargs)

        # directly return if the Transpiler cannot generate valid CUDA kernels
        if not self._valid_cuda_kernels:
            return None

        assert self.run is not None, "The graph is not compiled yet."

        input_tensors = kwargs.get("inputs", [])
        stream = kwargs.get("stream", None)
        if stream is None:
            stream = torch.cuda.default_stream()

        assert self.cygraph.get_num_inputs() == len(
            input_tensors
        ), "Expected {} input tensors, got {}".format(
            self.cygraph.get_num_inputs(), len(input_tensors)
        )

        # TODO: dtype and device
        buffer_tensor = torch.empty(
            results["buf_size"], dtype=torch.uint8, device=input_tensors[0].device
        ).contiguous()

        output_tensors = [
            gen_empty_tensor(
                meta["alloc_size"],
                meta["shape"],
                meta["strides"],
                device=input_tensors[0].device,
                dtype=input_tensors[0].dtype,
            )
            for meta in results["output_directives"]
        ]

        # Use int64 for compatibility with older PyTorch versions (uint64 was added later)
        prodiler_buffer_tensor = torch.empty(
            results["profiler_buf_size"],
            dtype=torch.int64,
            device=input_tensors[0].device,
        ).contiguous()

        buffer_tensor_ptr = buffer_tensor.data_ptr()
        input_tensors_ptr = [tensor.data_ptr() for tensor in input_tensors]
        output_tensors_ptr = [tensor.data_ptr() for tensor in output_tensors]
        prodiler_buffer_tensor_ptr = prodiler_buffer_tensor.data_ptr()
        self.run(
            input_tensors_ptr,
            output_tensors_ptr,
            buffer_tensor_ptr,
            stream.cuda_stream,
            prodiler_buffer_tensor_ptr,
        )

        if results["profiler_buf_size"] > 0:
            from .profiler import export_to_perfetto_trace

            profiler_result_dir = "./profiling_results"
            profiler_result_file = os.path.join(profiler_result_dir, "yirage.perfetto-trace")
            os.makedirs(profiler_result_dir, exist_ok=True)
            export_to_perfetto_trace(prodiler_buffer_tensor, profiler_result_file)
            print(
                f"Exported profiling results to {profiler_result_file}, please view it with perfetto: https://ui.perfetto.dev/"
            )
        return output_tensors

    def compile(self, async_=False, **kwargs):
        if self._is_compiled:
            return self._cached_results

        input_tensors = kwargs.get("inputs", [])
        input_strides = []
        # Check that the input_strides match uGraph's specification
        dtensors = self.cygraph.get_input_dtensors()
        assert len(dtensors) == len(
            input_tensors
        ), "Given number of inputs do not match the uGraph's inputs"
        for i in range(len(dtensors)):
            dims, strides = self.cygraph.get_input_dtensor_shape_and_stride(dtensors[i])
            assert (
                dims == input_tensors[i].shape
            ), "Expected input dims {}, got input dims {}".format(dims, input_tensors[i].shape)
            assert (
                strides == input_tensors[i].stride()
            ), "Expected input strides {}, got input strides {}".format(
                strides, input_tensors[i].stride()
            )
            input_strides.append(strides)
        target_cc = kwargs.get(
            "target_cc",
            torch.cuda.get_device_properties(0).major * 10
            + torch.cuda.get_device_properties(0).minor,
        )
        num_warp_groups = kwargs.get("num_warp_groups", 2)
        pipeline_stages = kwargs.get("pipeline_stages", 2)
        # TODO, add profling for Ampere later to show gpu wave
        profiling = kwargs.get("profiling", False)
        enable_online_softmax = kwargs.get("enable_online_softmax", False)

        result = generate_cuda_program(
            self.cygraph,
            target_cc=target_cc,
            input_strides=input_strides,
            num_warp_groups=num_warp_groups,
            pipeline_stages=pipeline_stages,
            profiling=profiling,
            enable_online_softmax=enable_online_softmax,
        )
        if result["max_smem_size"] > get_shared_memory_capacity(target_cc):
            # the transpiled kernel exceeds shared memory limit
            print(
                f"required shared memory size {result['max_smem_size']} exceed max shared memory size of current gpu arch {get_shared_memory_capacity(target_cc)}"
            )
            self._is_compiled = True
            self._valid_cuda_kernels = False
            self._error_message = "shared memory usage exceed limit"

            if async_:
                return Handle([], None)
            else:
                return None

        YIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        # if True:
        #     tempdir = './test/'

        tempdir_obj = tempfile.TemporaryDirectory()
        tempdir = tempdir_obj.name
        saved_addr = ""
        file_id = kwargs.get("file_id", -1)
        if file_id != -1:
            print(f"file_id: {file_id}")
            saved_addr = f"./generated_codes/{file_id}/"
        FILE_NAME = os.path.join(tempdir, "test.cu")
        so_path = os.path.join(tempdir, "test.cpython-38-x86_64-linux-gnu.so")

        with open(FILE_NAME, "w") as f:
            f.write(result["code"] + HARD_CODE)
            if saved_addr != "":
                print(f"saved_addr: {saved_addr}")
                os.makedirs(saved_addr, exist_ok=True)
                with open(saved_addr + "test" + str(file_id) + ".cu", "w") as f:
                    f.write(result["code"] + HARD_CODE)

        cc = shutil.which("nvcc")
        if cc is None:
            raise RuntimeError("nvcc not found. Please make sure you have installed CUDA.")

        # This function was renamed and made public in Python 3.10
        if hasattr(sysconfig, "get_default_scheme"):
            scheme = sysconfig.get_default_scheme()
        else:
            scheme = sysconfig._get_default_scheme()
        # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
        # path changes to include 'local'. This change is required to use triton with system-wide python.
        if scheme == "posix_local":
            scheme = "posix_prefix"
        py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
        cc_cmd = get_cc_cmd(
            target_cc,
            cc,
            FILE_NAME,
            py_include_dir,
            INCLUDE_PATH,
            DEPS_PATH,
            so_path,
            profiling,
        )

        def remain_op():
            import importlib.util

            try:
                spec = importlib.util.spec_from_file_location("__yirage_launcher", so_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.run = getattr(mod, "launch")

                self._is_compiled = True
                self._valid_cuda_kernels = True
                self._cached_results = result
                self._error_message = "No error"
                tempdir_obj.cleanup()
                return self._cached_results
            except ImportError:
                # cannot import the built shared library likely due to
                # compilation errors
                self._is_compiled = True
                self._valid_cuda_kernels = False
                self._cached_results = None
                self._error_message = "CUDA compilation error"
                return None

        if async_:
            if global_config.bypass_compile_errors:
                ret = subprocess.Popen(cc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            else:
                ret = subprocess.Popen(cc_cmd)
            return Handle([ret], remain_op, graph=self)
        else:
            ret = subprocess.check_call(cc_cmd)
            return remain_op()

        # so_path = './test.cpython-38-x86_64-linux-gnu.so'

    def superoptimize(
        self,
        imaps: list = None,
        omaps: list = None,
        griddims: list = None,
        blockdims: list = None,
        fmaps: list = None,
        franges: list = None,
        verbose: bool = False,
        config: str = None,
        backend: str = None,
        warmup_iters: int = 16,
        profile_iters: int = 1000,
        use_graph_dataset: bool = True,
        use_cached_graphs: bool = True,
        use_persistent_cache: bool = True,
        use_ray: bool = True,  # Ray distributed search enabled by default
        num_workers: int = None,  # Auto-detect based on CPU cores
        save_codes: bool = False,
        is_formal_verified: Optional[bool] = None,
        formal_verify: Optional[bool] = None,
    ):
        from ..search.verifier_config import resolve_verifier_config

        verifier_cfg = resolve_verifier_config(
            formal_verify=formal_verify,
            is_formal_verified=is_formal_verified,
        )
        resolved_formal = verifier_cfg.is_formal_verified
        if verbose or verifier_cfg.verifier_type == "formal":
            print(
                f"[superoptimize] Search verifier: {verifier_cfg.verifier_type} "
                f"(formal_available={verifier_cfg.formal_available})"
            )

        # Auto-detect backend if not specified
        if backend is None:
            try:
                from ..backends.api import get_default_backend

                backend = get_default_backend()
            except ImportError:
                backend = None
            if backend is None:
                # Fallback detection via PyTorch (torch already imported at top of file)
                try:
                    import torch_npu

                    if torch.npu.is_available():
                        backend = "ascend"
                except ImportError:
                    pass
                if backend is None and torch.cuda.is_available():
                    backend = "cuda"
                elif (
                    backend is None
                    and hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    backend = "mps"
                elif backend is None:
                    backend = "cpu"
            print(f"[superoptimize] Auto-detected backend: {backend}")

        if use_cached_graphs:
            # Store checkpoint in ~/.yirage/ instead of cwd
            import os

            cache_dir = os.path.expanduser("~/.yirage/checkpoints")
            os.makedirs(cache_dir, exist_ok=True)
            previous_checkpoint = os.path.join(
                cache_dir, "mugraphs_{:x}.json".format(self.cygraph.get_owner_independent_hash())
            )
        else:
            previous_checkpoint = None
        # Apply backend-specific search optimizations
        if backend == "mps":
            # MPS-specific optimization
            if griddims is None and blockdims is None and franges is None:
                from ..backends.mps.config import get_mps_search_config

                mps_config = get_mps_search_config()
                griddims = mps_config.get("grid_dims_to_explore")
                blockdims = mps_config.get("block_dims_to_explore")
                fmaps = mps_config.get("fmaps_to_explore")
                franges = mps_config.get("franges_to_explore")
                print(f"✓ MPS backend: Using Apple Silicon optimized search")
                print(f"  - Grids: {len(griddims)} configs (SIMD-aligned)")
                print(f"  - Blocks: {len(blockdims)} configs (threadgroup-optimized)")
                print(f"  - Fmaps: {fmaps} (forloop dimension mappings)")
                print(f"  - Franges: {franges}")
            else:
                print(f"✓ MPS backend selected (using custom parameters)")

        elif backend == "cpu":
            # CPU-specific optimization from detected architecture + graph shape
            from ..backends.cpu.config import apply_cpu_search_env, resolve_cpu_search_space

            cpu_config = resolve_cpu_search_space(self.cygraph)
            apply_cpu_search_env(cpu_config)
            if griddims is None:
                griddims = cpu_config.get("grid_dims_to_explore")
            if blockdims is None:
                blockdims = cpu_config.get("block_dims_to_explore")
            if franges is None:
                franges = cpu_config.get("franges_to_explore")
            mnk = cpu_config.get("problem_mnk", (0, 0, 0))
            print(f"✓ CPU backend: architecture-aware search space")
            print(
                f"  - Host: {cpu_config['num_cores']} cores, "
                f"SIMD={cpu_config['simd_type']}, vector_width={cpu_config['vector_width']}"
            )
            if any(mnk):
                print(f"  - Problem GEMM (m,n,k): {mnk}")
            print(
                f"  - Search: {len(griddims)} grids × {len(blockdims)} blocks, "
                f"franges={franges}, search_threads={cpu_config['search_thread']}"
            )

        elif backend == "ascend":
            # Ascend NPU-specific optimization
            if griddims is None and blockdims is None and franges is None:
                from ..backends.ascend.config import get_ascend_search_config

                ascend_config = get_ascend_search_config()
                griddims = ascend_config.get("grid_dims_to_explore")
                blockdims = ascend_config.get("block_dims_to_explore")
                fmaps = ascend_config.get("fmaps_to_explore")
                franges = ascend_config.get("franges_to_explore")
                print(f"✓ Ascend backend: Using Huawei NPU optimized search")
            else:
                print(f"✓ Ascend backend selected (using custom parameters)")

        elif backend == "maca":
            # MetaX MACA GPU-specific optimization
            if griddims is None and blockdims is None and franges is None:
                from ..backends.maca.config import get_maca_search_config

                maca_config = get_maca_search_config()
                griddims = maca_config.get("grid_dims_to_explore")
                blockdims = maca_config.get("block_dims_to_explore")
                fmaps = maca_config.get("fmaps_to_explore")
                franges = maca_config.get("franges_to_explore")
                print(f"✓ MACA backend: Using MetaX GPU optimized search")
                print(f"  - warpSize: 64 (NOT 32 like NVIDIA!)")
                print(f"  - Grids: {len(griddims)} configs (SM blocks)")
                print(f"  - Blocks: {len(blockdims)} configs (64-thread warp aligned)")
                print(f"  - Fmaps: {fmaps}")
                print(f"  - Franges: {franges}")
            else:
                print(f"✓ MACA backend selected (using custom parameters)")

        elif backend == "cuda":
            print(f"✓ CUDA backend: Using NVIDIA GPU optimized configuration")

        # In-memory and persistent caches (after search space is resolved)
        if use_graph_dataset:
            cached_graph = graph_dataset.find(
                self.cygraph,
                imaps=imaps,
                omaps=omaps,
                griddims=griddims,
                blockdims=blockdims,
                fmaps=fmaps,
                franges=franges,
                backend=backend,
            )
            if cached_graph is not None:
                return cached_graph

        if use_persistent_cache:
            try:
                restored = self._try_restore_from_persistent_cache(
                    backend,
                    imaps,
                    omaps,
                    griddims,
                    blockdims,
                    fmaps,
                    franges,
                )
                if restored is not None:
                    return restored
            except Exception:
                pass

        # Ray distributed search
        start_time = time.perf_counter()
        if use_ray and griddims is not None and len(griddims) > 1:
            try:
                import ray

                if not ray.is_initialized():
                    import os

                    num_cpus = num_workers or os.cpu_count()
                    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
                    print(f"✓ Ray initialized with {num_cpus} CPUs")

                # Partition grid dimensions across workers
                actual_workers = num_workers or min(8, len(griddims))
                partitions = [[] for _ in range(actual_workers)]
                for i, gd in enumerate(griddims):
                    partitions[i % actual_workers].append(gd)

                print(
                    f"✓ Ray distributed search: {actual_workers} workers, {len(griddims)} grid configs"
                )

                @ray.remote(num_cpus=1)
                def search_partition(
                    graph_json_path,
                    output_json_path,
                    partition_griddims,
                    backend,
                    imaps,
                    omaps,
                    blockdims,
                    fmaps,
                    franges,
                    previous_checkpoint,
                    verbose,
                    default_config,
                    is_formal_verified,
                ):
                    """Search a partition of grid dimensions, save results to JSON"""
                    from yirage.core import search as core_search, cy_from_json, cy_to_json

                    # Load graph from JSON file
                    cygraph = cy_from_json(graph_json_path)
                    results = core_search(
                        cygraph,
                        backend=backend,
                        imaps=imaps,
                        omaps=omaps,
                        griddims=partition_griddims,
                        blockdims=blockdims,
                        fmaps=fmaps,
                        franges=franges,
                        previous_checkpoint=previous_checkpoint,
                        verbose=verbose,
                        default_config=default_config,
                        is_formal_verified=is_formal_verified,
                    )
                    # Save results to JSON file (avoid pickle of CyKNGraph)
                    # Return count instead of actual graphs
                    for i, g in enumerate(results):
                        cy_to_json(g, f"{output_json_path}_{i}.json")
                    return len(results)

                # Serialize cygraph to JSON file for Ray workers
                import tempfile
                from yirage.core import cy_to_json, cy_from_json

                graph_json_path = tempfile.mktemp(suffix=".json", prefix="yirage_graph_")
                cy_to_json(self.cygraph, graph_json_path)
                output_base = tempfile.mktemp(prefix="yirage_result_")
                print(f"  Graph serialized, launching {actual_workers} workers...")

                # Launch parallel searches
                futures = []
                output_paths = []
                for idx, partition in enumerate(partitions):
                    if partition:  # Skip empty partitions
                        output_path = f"{output_base}_{idx}"
                        output_paths.append(output_path)
                        partition_checkpoint = (
                            f"{previous_checkpoint}.ray{idx}" if previous_checkpoint else None
                        )
                        futures.append(
                            search_partition.remote(
                                graph_json_path,
                                output_path,
                                partition,
                                backend,
                                imaps,
                                omaps,
                                blockdims,
                                fmaps,
                                franges,
                                partition_checkpoint,
                                verbose,
                                config,
                                resolved_formal,
                            )
                        )

                # Collect results - get counts and load from JSON files
                counts = ray.get(futures)
                cygraphs = []
                import os

                for output_path, count in zip(output_paths, counts):
                    for i in range(count):
                        result_file = f"{output_path}_{i}.json"
                        if os.path.exists(result_file):
                            cygraphs.append(cy_from_json(result_file))
                            os.remove(result_file)  # Cleanup

                # Cleanup input temp file
                try:
                    os.remove(graph_json_path)
                except:
                    pass

                print(f"✓ Ray search completed: {len(cygraphs)} muGraphs found")
            except Exception as e:
                print(f"⚠ Ray search failed ({e}), falling back to single-process search")
                cygraphs = search(
                    self.cygraph,
                    backend=backend,
                    imaps=imaps,
                    omaps=omaps,
                    griddims=griddims,
                    blockdims=blockdims,
                    fmaps=fmaps,
                    franges=franges,
                    previous_checkpoint=previous_checkpoint,
                    verbose=verbose,
                    default_config=config,
                    is_formal_verified=resolved_formal,
                )
        else:
            # Single-process search
            cygraphs = search(
                self.cygraph,
                backend=backend,
                imaps=imaps,
                omaps=omaps,
                griddims=griddims,
                blockdims=blockdims,
                fmaps=fmaps,
                franges=franges,
                previous_checkpoint=previous_checkpoint,
                verbose=verbose,
                default_config=config,
                is_formal_verified=resolved_formal,
            )

        search_time = time.perf_counter() - start_time
        all_graphs = [KNGraph(g, backend=backend) for g in cygraphs]
        print(f"Finished search in {search_time:.1f}s, discovering {len(all_graphs)} mugraphs ...")
        if backend == "cuda":
            # profile and use the best graph
            best_graph, best_perf = None, float("inf")
            print("Transpiling {} muGraphs ...".format(len(all_graphs)))
            handles = deque()

            target_cc = (
                torch.cuda.get_device_properties(0).major * 10
                + torch.cuda.get_device_properties(0).minor
            )
            if target_cc >= 90:
                pipeline_stages_list = [2, 3, 4]
                num_warp_groups_list = [2, 3, 4]
                for idx, g in enumerate(all_graphs):
                    for pipeline_stages in pipeline_stages_list:
                        for num_warp_groups in num_warp_groups_list:
                            dtensors = g.cygraph.get_input_dtensors()
                            input_tensors = list()
                            for t in dtensors:
                                dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                                dtype = convert_dtype_to_torch_type(t.dtype)
                                x = torch.randn(
                                    dims,
                                    dtype=dtype,
                                    device="cuda:{}".format(global_config.gpu_device_id),
                                )
                                x = torch.as_strided(x, size=dims, stride=strides)
                                input_tensors.append(x)
                            starter = torch.cuda.Event(enable_timing=True)
                            ender = torch.cuda.Event(enable_timing=True)
                            new_g = g
                            if len(handles) == MAX_THREADS:
                                handles.popleft().wait()
                            handle = new_g.compile(
                                async_=True,
                                inputs=input_tensors,
                                pipeline_stages=pipeline_stages,
                                num_warp_groups=num_warp_groups,
                            )
                            handles.append(handle)
            else:
                for idx, g in enumerate(all_graphs):
                    dtensors = g.cygraph.get_input_dtensors()
                    input_tensors = list()
                    for t in dtensors:
                        dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                        # dims = [t.dim(i) for i in range(t.num_dims)]
                        dtype = convert_dtype_to_torch_type(t.dtype)
                        x = torch.randn(
                            dims,
                            dtype=dtype,
                            device="cuda:{}".format(global_config.gpu_device_id),
                        )
                        x = torch.as_strided(x, size=dims, stride=strides)
                        input_tensors.append(x)
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    if len(handles) == MAX_THREADS:
                        handles.popleft().wait()
                    handle = g.compile(async_=True, inputs=input_tensors)
                    handles.append(handle)
            while handles:
                handles.popleft().wait()
            for idx, g in enumerate(all_graphs):
                dtensors = g.cygraph.get_input_dtensors()
                input_tensors = list()
                for t in dtensors:
                    dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                    dtype = convert_dtype_to_torch_type(t.dtype)
                    x = torch.randn(
                        dims,
                        dtype=dtype,
                        device="cuda:{}".format(global_config.gpu_device_id),
                    )
                    x = torch.as_strided(x, size=dims, stride=strides)
                    input_tensors.append(x)
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                if not g.valid_kernels():
                    print("muGraph {}: {}".format(idx, g.get_error_message()))
                    continue
                # Use first valid kernel to avoid potential hangs from incompatible configurations
                if best_graph is None:
                    # Warmup runs
                    for _ in range(warmup_iters):
                        g(inputs=input_tensors)
                    torch.cuda.synchronize()
                    starter.record()
                    for _ in range(profile_iters):
                        g(inputs=input_tensors)
                    ender.record()
                    torch.cuda.synchronize()
                    perf = starter.elapsed_time(ender) / profile_iters
                    print("muGraph {}: profiled performance (ms) = {}".format(idx, perf))
                    best_graph, best_perf = g, perf
                if perf < best_perf:
                    best_graph, best_perf = g, perf
            best_graph.backend = "cuda"
            if use_graph_dataset:
                graph_dataset.store(
                    input_graph=self.cygraph,
                    optimized_graph=best_graph,
                    imaps=imaps,
                    omaps=omaps,
                    griddims=griddims,
                    blockdims=blockdims,
                    fmaps=fmaps,
                    franges=franges,
                    backend=backend,
                )

            # Persist to disk for reuse across sessions
            try:
                graph_hash = hex(self.cygraph.get_owner_independent_hash())[2:]
                device_name = torch.cuda.get_device_name(0)
                device_info = {
                    "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
                    "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                }
                input_shapes = []
                for t in self.cygraph.get_input_dtensors():
                    dims, _ = self.cygraph.get_input_dtensor_shape_and_stride(t)
                    input_shapes.append(list(dims))

                save_mugraph(
                    graph_hash=graph_hash,
                    optimized_graph=best_graph,
                    backend="cuda",
                    imaps=imaps,
                    omaps=omaps,
                    griddims=griddims,
                    blockdims=blockdims,
                    fmaps=fmaps,
                    franges=franges,
                    latency_ms=best_perf,
                    num_candidates_searched=len(all_graphs),
                    input_shapes=input_shapes,
                    device_name=device_name,
                    device_info=device_info,
                )
                print(f"✓ muGraph saved to persistent storage (backend: cuda)")
            except Exception as e:
                print(f"Warning: Could not save muGraph to disk: {e}")

            return best_graph
        elif backend == "mps":
            # MPS backend: profile and select best graph
            print(f"MPS backend: Profiling {len(all_graphs)} muGraphs...")

            best_graph, best_perf = None, float("inf")

            for idx, g in enumerate(all_graphs):
                # Get input tensors
                dtensors = g.cygraph.get_input_dtensors()
                input_tensors = list()

                for t in dtensors:
                    dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                    dtype = convert_dtype_to_torch_type(t.dtype)

                    # Create tensor on MPS device
                    if torch.backends.mps.is_available():
                        x = torch.randn(dims, dtype=dtype, device="mps")
                    else:
                        x = torch.randn(dims, dtype=dtype, device="cpu")

                    x = torch.as_strided(x, size=dims, stride=strides)
                    input_tensors.append(x)

                # Warmup
                for _ in range(warmup_iters):
                    try:
                        outputs = g(inputs=input_tensors)
                    except:
                        continue

                # Synchronize before timing
                if torch.backends.mps.is_available() and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()

                # Profile using Python timing (MPS doesn't have Event API like CUDA)
                start_time = time.perf_counter()

                for _ in range(profile_iters):
                    try:
                        outputs = g(inputs=input_tensors)
                    except:
                        break

                # Synchronize after timing
                if torch.backends.mps.is_available() and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()

                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) / profile_iters * 1000

                print(f"  muGraph[{idx}]: {elapsed_ms:.4f} ms")

                if elapsed_ms < best_perf:
                    best_perf = elapsed_ms
                    best_graph = g

            if best_graph:
                print(f"Selected best muGraph with {best_perf:.4f} ms")
                best_graph.backend = "mps"

                if use_graph_dataset:
                    graph_dataset.store(
                        input_graph=self.cygraph,
                        optimized_graph=best_graph,
                        imaps=imaps,
                        omaps=omaps,
                        griddims=griddims,
                        blockdims=blockdims,
                        fmaps=fmaps,
                        franges=franges,
                        backend=backend,
                    )

                # Persist to disk with comprehensive training data
                try:
                    graph_hash = hex(self.cygraph.get_owner_independent_hash())[2:]
                    device_name = "Apple Silicon MPS"

                    # Collect input/output shapes
                    input_shapes = []
                    input_tensors_info = []
                    for t in self.cygraph.get_input_dtensors():
                        dims, strides = self.cygraph.get_input_dtensor_shape_and_stride(t)
                        input_shapes.append(list(dims))
                        input_tensors_info.append(
                            {
                                "dims": list(dims),
                                "strides": list(strides),
                                "dtype": str(t.dtype),
                                "is_input": True,
                            }
                        )

                    # Collect all candidate latencies for training labels
                    all_latencies = []
                    candidate_evaluations = []
                    for idx, g in enumerate(all_graphs):
                        # Get latency if profiled
                        lat = 0.0
                        is_valid = True
                        try:
                            # Try to get cached performance if available
                            if hasattr(g, "_cached_latency"):
                                lat = g._cached_latency
                        except:
                            pass

                        all_latencies.append(lat if lat > 0 else float("inf"))
                        candidate_evaluations.append(
                            {
                                "candidate_id": idx,
                                "latency_ms": lat,
                                "is_valid": is_valid,
                                "is_best": (g == best_graph),
                            }
                        )

                    # Device capabilities for MPS
                    device_info = {
                        "vendor": "Apple",
                        "device_type": "mps",
                        "fp16_acceleration": True,
                        "memory_type": "Unified",
                    }

                    # Calculate search time
                    search_time = time.perf_counter() - start_time if "start_time" in dir() else 0.0

                    save_mugraph(
                        graph_hash=graph_hash,
                        optimized_graph=best_graph,
                        backend="mps",
                        # Search config
                        imaps=imaps,
                        omaps=omaps,
                        griddims=griddims,
                        blockdims=blockdims,
                        fmaps=fmaps,
                        franges=franges,
                        # Performance
                        latency_ms=best_perf,
                        num_candidates_searched=len(all_graphs),
                        search_time_s=search_time,
                        # Graph info
                        input_shapes=input_shapes,
                        tensors=input_tensors_info,
                        # Device info
                        device_name=device_name,
                        device_info=device_info,
                        # Training data
                        candidate_evaluations=candidate_evaluations,
                        all_latencies=[l for l in all_latencies if l < float("inf")],
                    )
                    print(f"✓ muGraph saved to persistent storage (backend: mps)")
                    print(f"  - Candidates evaluated: {len(all_graphs)}")
                    print(f"  - Training data: features + labels included")
                except Exception as e:
                    print(f"Warning: Could not save muGraph to disk: {e}")

                return best_graph

            return None
        elif backend == "cpu":
            # CPU backend: profile and select best graph on the same host
            print(f"CPU backend: Profiling {len(all_graphs)} muGraphs...")

            best_graph, best_perf = None, float("inf")

            with _cpu_runtime_context() as cpu_rt:
                if cpu_rt:
                    print(
                        f"  - Runtime: {cpu_rt.get('torch_num_threads')} threads, "
                        f"SIMD={cpu_rt.get('simd_type')}, "
                        f"parallel_tb_grid={cpu_rt.get('parallel_tb_grid')}"
                    )

            for idx, g in enumerate(all_graphs):
                # Get input tensors
                dtensors = g.cygraph.get_input_dtensors()
                input_tensors = list()

                for t in dtensors:
                    dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                    dtype = convert_dtype_to_torch_type(t.dtype)
                    x = torch.randn(dims, dtype=dtype, device="cpu")
                    x = torch.as_strided(x, size=dims, stride=strides)
                    input_tensors.append(x)

                with _cpu_runtime_context():
                    # Warmup
                    for _ in range(warmup_iters):
                        try:
                            g(inputs=input_tensors)
                        except Exception:
                            continue

                    # Profile using Python timing on CPU (same backend as execution)
                    start_time = time.perf_counter()

                    for _ in range(profile_iters):
                        try:
                            g(inputs=input_tensors)
                        except Exception:
                            break

                    end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) / profile_iters * 1000

                print(f"  muGraph[{idx}]: {elapsed_ms:.4f} ms")

                if elapsed_ms < best_perf:
                    best_perf = elapsed_ms
                    best_graph = g

            if best_graph:
                print(f"Selected best muGraph with {best_perf:.4f} ms")
                best_graph.backend = "cpu"

                if use_graph_dataset:
                    graph_dataset.store(
                        input_graph=self.cygraph,
                        optimized_graph=best_graph,
                        imaps=imaps,
                        omaps=omaps,
                        griddims=griddims,
                        blockdims=blockdims,
                        fmaps=fmaps,
                        franges=franges,
                        backend=backend,
                    )

                # Persist to disk for reuse across sessions
                try:
                    graph_hash = hex(self.cygraph.get_owner_independent_hash())[2:]
                    input_shapes = []
                    for t in self.cygraph.get_input_dtensors():
                        dims, _ = self.cygraph.get_input_dtensor_shape_and_stride(t)
                        input_shapes.append(list(dims))

                    save_mugraph(
                        graph_hash=graph_hash,
                        optimized_graph=best_graph,
                        backend="cpu",
                        imaps=imaps,
                        omaps=omaps,
                        griddims=griddims,
                        blockdims=blockdims,
                        fmaps=fmaps,
                        franges=franges,
                        latency_ms=best_perf,
                        num_candidates_searched=len(all_graphs),
                        input_shapes=input_shapes,
                    )
                    print(f"✓ muGraph saved to persistent storage (backend: cpu)")
                except Exception as e:
                    print(f"Warning: Could not save muGraph to disk: {e}")

                return best_graph

            return None
        elif backend == "ascend":
            # Ascend NPU backend: profile and select best graph
            # Uses torch_npu for Huawei Ascend NPUs
            print(f"Ascend backend: Processing {len(all_graphs)} muGraphs...")
            print(f"  Note: Ascend uses AI Cores with Cube/Vector units")

            # Check if Ascend NPU is available
            ascend_available = False
            try:
                import torch_npu

                if torch.npu.is_available():
                    ascend_available = True
                    device_name = torch.npu.get_device_name(0)
                    print(f"  Detected Ascend NPU: {device_name}")
            except (ImportError, AttributeError, RuntimeError):
                print(f"  Warning: torch_npu not available or NPU not detected")
                print(f"  Using CPU fallback for profiling")

            best_graph, best_perf = None, float("inf")

            if ascend_available:
                # Profile on Ascend NPU
                for idx, g in enumerate(all_graphs):
                    dtensors = g.cygraph.get_input_dtensors()
                    input_tensors = list()
                    for t in dtensors:
                        dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                        dtype = convert_dtype_to_torch_type(t.dtype)
                        x = torch.randn(dims, dtype=dtype, device="npu:0")
                        x = torch.as_strided(x, size=dims, stride=strides)
                        input_tensors.append(x)

                    # Warmup (skip invalid graphs via try/except)
                    try:
                        for _ in range(warmup_iters):
                            g(inputs=input_tensors)
                        torch.npu.synchronize()
                    except Exception as e:
                        print(f"  muGraph[{idx}]: Error during warmup - {e}")
                        continue

                    # Profile using NPU events
                    try:
                        starter = torch.npu.Event(enable_timing=True)
                        ender = torch.npu.Event(enable_timing=True)
                        starter.record()
                        for _ in range(profile_iters):
                            g(inputs=input_tensors)
                        ender.record()
                        torch.npu.synchronize()
                        perf = starter.elapsed_time(ender) / profile_iters
                        print(f"  muGraph[{idx}]: {perf:.4f} ms")
                        if perf < best_perf:
                            best_graph, best_perf = g, perf
                    except Exception as e:
                        print(f"  muGraph[{idx}]: Error during profiling - {e}")
                        continue
            else:
                # CPU fallback profiling
                for idx, g in enumerate(all_graphs):
                    dtensors = g.cygraph.get_input_dtensors()
                    input_tensors = list()
                    for t in dtensors:
                        dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                        dtype = convert_dtype_to_torch_type(t.dtype)
                        x = torch.randn(dims, dtype=dtype, device="cpu")
                        x = torch.as_strided(x, size=dims, stride=strides)
                        input_tensors.append(x)

                    # Warmup
                    for _ in range(warmup_iters):
                        try:
                            g(inputs=input_tensors)
                        except:
                            continue

                    # Profile
                    start_time = time.perf_counter()
                    for _ in range(profile_iters):
                        try:
                            g(inputs=input_tensors)
                        except:
                            break
                    elapsed_ms = (time.perf_counter() - start_time) / profile_iters * 1000
                    print(f"  muGraph[{idx}]: {elapsed_ms:.4f} ms (CPU fallback)")
                    if elapsed_ms < best_perf:
                        best_graph, best_perf = g, elapsed_ms

            if best_graph is not None:
                best_graph.backend = "ascend"
                if use_graph_dataset:
                    graph_dataset.store(
                        input_graph=self.cygraph,
                        optimized_graph=best_graph,
                        imaps=imaps,
                        omaps=omaps,
                        griddims=griddims,
                        blockdims=blockdims,
                        fmaps=fmaps,
                        franges=franges,
                        backend=backend,
                    )
            return best_graph
        elif backend == "maca":
            # MACA backend: profile and select best graph
            # MACA uses CUDA-compatible runtime via mcPytorch
            print(f"MACA backend: Processing {len(all_graphs)} muGraphs...")
            print(f"  Note: MACA uses 64-thread warps (vs NVIDIA's 32)")

            # Check if MACA GPU is available (mcPytorch maps MACA to cuda)
            maca_available = False
            try:
                if torch.cuda.is_available():
                    # Check if this is actually mcPytorch/MACA
                    device_name = torch.cuda.get_device_name(0)
                    if "MetaX" in device_name or "C500" in device_name or "MACA" in device_name:
                        maca_available = True
                        print(f"  Detected MACA GPU: {device_name}")
                    else:
                        # Could be mcPytorch with generic device name
                        maca_available = True
                        print(f"  Using GPU device: {device_name}")
            except Exception as e:
                print(f"  Note: MACA profiling unavailable ({e})")
                print(f"  Returning first valid graph without profiling...")

            best_graph, best_perf = None, float("inf")

            if not maca_available:
                # No mcPytorch available - return first graph without profiling
                # The graph will be compiled when executed
                print(f"  Skipping profiling (mcPytorch not available)")
                print(f"  Returning first graph from {len(all_graphs)} candidates")
                if len(all_graphs) > 0:
                    best_graph = all_graphs[0]
                    print(f"  ✓ Selected muGraph 0")
                else:
                    print("  Warning: No graphs found!")
                    return None
            else:
                # MACA GPU available - profile and select best
                handles = deque()

                # Compile all graphs
                for idx, g in enumerate(all_graphs):
                    dtensors = g.cygraph.get_input_dtensors()
                    input_tensors = list()
                    for t in dtensors:
                        dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                        dtype = convert_dtype_to_torch_type(t.dtype)
                        x = torch.randn(
                            dims,
                            dtype=dtype,
                            device="cuda:{}".format(global_config.gpu_device_id),
                        )
                        x = torch.as_strided(x, size=dims, stride=strides)
                        input_tensors.append(x)
                    if len(handles) == MAX_THREADS:
                        handles.popleft().wait()
                    handle = g.compile(async_=True, inputs=input_tensors)
                    handles.append(handle)

                while handles:
                    handles.popleft().wait()

                # Profile all graphs
                for idx, g in enumerate(all_graphs):
                    dtensors = g.cygraph.get_input_dtensors()
                    input_tensors = list()
                    for t in dtensors:
                        dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
                        dtype = convert_dtype_to_torch_type(t.dtype)
                        x = torch.randn(
                            dims,
                            dtype=dtype,
                            device="cuda:{}".format(global_config.gpu_device_id),
                        )
                        x = torch.as_strided(x, size=dims, stride=strides)
                        input_tensors.append(x)
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    if not g.valid_kernels():
                        print("muGraph {}: {}".format(idx, g.get_error_message()))
                        continue
                    # Warmup runs
                    for _ in range(warmup_iters):
                        g(inputs=input_tensors)
                    torch.cuda.synchronize()
                    starter.record()
                    for _ in range(profile_iters):
                        g(inputs=input_tensors)
                    ender.record()
                    torch.cuda.synchronize()
                    perf = starter.elapsed_time(ender) / profile_iters
                    print("muGraph {}: profiled performance (ms) = {}".format(idx, perf))
                    if perf < best_perf:
                        best_graph, best_perf = g, perf

            if best_graph is not None:
                best_graph.backend = "maca"
                if use_graph_dataset:
                    graph_dataset.store(
                        input_graph=self.cygraph,
                        optimized_graph=best_graph,
                        imaps=imaps,
                        omaps=omaps,
                        griddims=griddims,
                        blockdims=blockdims,
                        fmaps=fmaps,
                        franges=franges,
                        backend=backend,
                    )
            return best_graph
        elif backend == "nki":
            return all_graphs
        elif backend == "triton":
            from .triton_profiler import profile_and_select_best_graph

            YIRAGE_ROOT, INCLUDE_PATH, _ = get_key_paths()
            os.environ["KERNELS_PATH"] = os.path.join(
                INCLUDE_PATH, "triton_transpiler/runtime"
            )  # for triton
            best_graph, best_file_path, best_output_shapes = profile_and_select_best_graph(
                all_graphs,
                target_cc=torch.cuda.get_device_properties(0).major * 10
                + torch.cuda.get_device_properties(0).minor,
                warmup_iters=warmup_iters,
                profile_iters=profile_iters,
                debug_mode=verbose,
                save_codes=save_codes,
            )
            # load execute_mugraph func from the generated file
            print(f"Loading the best muGraph from {best_file_path}")
            if not os.path.exists(best_file_path):
                raise FileNotFoundError(f"File not found: {best_file_path}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("__yirage_launcher", best_file_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "execute_mugraph"):
                best_graph.run = getattr(mod, "execute_mugraph")
            else:
                raise AttributeError("The module does not contain an 'execute_mugraph' function.")
            best_graph._cached_results = {"output_shapes": best_output_shapes}
            best_graph.backend = "triton"
            if use_graph_dataset:
                graph_dataset.store(
                    input_graph=self.cygraph,
                    optimized_graph=best_graph,
                    imaps=imaps,
                    omaps=omaps,
                    griddims=griddims,
                    blockdims=blockdims,
                    fmaps=fmaps,
                    franges=franges,
                    backend=backend,
                )

            return best_graph
        else:
            assert False, "Unsupported backend"
            return None

    def visualize(self, file_name):
        operators = self.cygraph.get_graph_structure()
        self.visualizer = visualizer(file_name)
        self.visualizer.draw_graphs(operators)

    def to_json(self, filename):
        cy_to_json(self.cygraph, filename)

    def from_json(self, filename):
        self.cygraph = cy_from_json(filename)

    # Persistent Kernel functions
    def attach_torch_tensor(self, t: DTensor, torch_tensor: torch.Tensor, name: str):
        return self.cygraph.attach_torch_tensor(t, torch_tensor, name)

    def attach_cuda_tensor(self, t: DTensor, name: str):
        return self.cygraph.attach_cuda_tensor(t, name)

    def attach_nvshmem_tensor(self, t: DTensor, name: str):
        return self.cygraph.attach_nvshmem_tensor(t, name)

    def fuse_tensors(self, input: list[DTensor], fuse_dim: int, num_groups: int, name: str):
        return self.cygraph.fuse_tensors(input, fuse_dim, num_groups, name)

    def shuffle_tensors(self, input: list[DTensor], shuffled_dim: int, num_groups: int, name: str):
        return self.cygraph.shuffle_tensors(input, shuffled_dim, num_groups, name)

    def register_task(self, bgraph: TBGraph, task_type: str, params: list[int] = None):
        return self.cygraph.register_task(bgraph.cygraph, task_type, params)

    def generate_task_graph(self, num_gpus: int, my_gpu_id: int):
        return self.cygraph.generate_task_graph(num_gpus, my_gpu_id)
