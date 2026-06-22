"""
CPU-specific search configuration
Optimized for multi-threaded CPU execution with SIMD

Supports:
- SSE/AVX/AVX2/AVX-512 instruction sets
- OpenMP parallelization
- Cache-aware tiling
"""

import os
import subprocess
import multiprocessing
import platform
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class SIMDType(str, Enum):
    """SIMD instruction set types."""

    NONE = "none"
    SSE = "sse"
    SSE2 = "sse2"
    SSE4 = "sse4"
    AVX = "avx"
    AVX2 = "avx2"
    AVX512 = "avx512"
    NEON = "neon"  # ARM


def detect_simd_support() -> SIMDType:
    """Detect highest supported SIMD instruction set."""
    system = platform.system()
    machine = platform.machine()

    # ARM (Apple Silicon, etc.)
    if machine in ("arm64", "aarch64"):
        return SIMDType.NEON

    # x86/x64
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"], capture_output=True, text=True, timeout=2
            )
            features = result.stdout.upper()
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                features = f.read().upper()
        else:
            features = ""

        if "AVX512" in features or "AVX-512" in features:
            return SIMDType.AVX512
        elif "AVX2" in features:
            return SIMDType.AVX2
        elif "AVX" in features:
            return SIMDType.AVX
        elif "SSE4" in features:
            return SIMDType.SSE4
        elif "SSE2" in features:
            return SIMDType.SSE2
        elif "SSE" in features:
            return SIMDType.SSE
    except:
        pass

    return SIMDType.NONE


def _extract_gemm_dims_from_cygraph(cygraph) -> tuple:
    """Best-effort (m, n, k) from the first matmul-like KN graph."""
    try:
        dtensors = cygraph.get_input_dtensors()
        if len(dtensors) < 2:
            return 0, 0, 0
        shapes = []
        for t in dtensors:
            dims, _ = cygraph.get_input_dtensor_shape_and_stride(t)
            shapes.append(list(dims))
        if len(shapes[0]) >= 2 and len(shapes[1]) >= 2:
            m = shapes[0][-2]
            k = shapes[0][-1]
            n = shapes[1][-1]
            return int(m), int(n), int(k)
    except Exception:
        pass
    return 0, 0, 0


def build_arch_aware_block_dims(
    vector_width: int, simd_type: SIMDType, dtype_bytes: int = 2
) -> List[tuple]:
    """
    Block sizes aligned to SIMD vector width (fp16 default).

    ``block_dim.x`` is chosen in multiples of ``vector_width`` so TB tiles
    match the detected instruction set (AVX2/AVX-512/NEON).
    """
    # Elements per SIMD register for fp16 (2 bytes)
    elems_per_reg = max(1, (vector_width * 4) // max(dtype_bytes, 1))
    bases = sorted(
        {
            max(elems_per_reg, 8),
            elems_per_reg * 2,
            elems_per_reg * 4,
            elems_per_reg * 8,
        }
    )
    block_dims = []
    for b in bases:
        if b <= 256:
            block_dims.append((b, 1, 1))
    return block_dims or [(16, 1, 1), (32, 1, 1)]


def build_arch_aware_grid_dims(
    num_cores: int, m_dim: int, max_grids: int = 6
) -> List[tuple]:
    """Grid parallelism along M, capped by physical cores and divisibility."""
    candidates = [1]
    for c in (2, 4, num_cores):
        if c <= num_cores and c not in candidates:
            if m_dim <= 0 or m_dim % c == 0 or m_dim >= c:
                candidates.append(c)
    grids = [(c, 1, 1) for c in sorted(set(candidates))[:max_grids]]
    return grids


def build_arch_aware_franges(tile_k: int, vector_width: int) -> List[int]:
    """For-loop ranges sized for cache lines and SIMD K-unroll."""
    base = max(4, vector_width)
    sizes = sorted({base, base * 2, min(64, tile_k or 64)})
    return [s for s in sizes if s > 0]


def resolve_cpu_search_space(
    cygraph=None, dtype_bytes: int = 2
) -> Dict[str, Any]:
    """
  Build a CPU search space from **detected** architecture + optional graph shape.

  This is the entry point ``superoptimize(backend='cpu')`` should use instead of
  generic GPU-oriented grids or hand-picked ``griddims``.
    """
    cfg = get_cpu_search_config()
    simd = detect_simd_support()
    vw = cfg["vector_width"]
    cores = cfg["num_cores"]
    m, n, k = _extract_gemm_dims_from_cygraph(cygraph) if cygraph is not None else (0, 0, 0)

    tile = cfg.get("tile_m", 64)
    if m > 0 and n > 0 and k > 0:
        tile = min(tile, m, n, k)

    cfg["grid_dims_to_explore"] = build_arch_aware_grid_dims(cores, m)
    cfg["block_dims_to_explore"] = build_arch_aware_block_dims(vw, simd, dtype_bytes)
    cfg["franges_to_explore"] = build_arch_aware_franges(tile, vw)
    cfg["problem_mnk"] = (m, n, k)
    cfg["search_thread"] = min(cfg["search_thread"], cores)
    return cfg


def apply_cpu_search_env(cfg: Dict[str, Any]) -> None:
    """Publish CPU generator limits for the C++ search (read in search_c.cc)."""
    os.environ["YIRAGE_CPU_SEARCH_THREADS"] = str(cfg.get("search_thread", 4))
    if "YIRAGE_CPU_MAX_TB_GRAPH_OP" not in os.environ:
        os.environ["YIRAGE_CPU_MAX_TB_GRAPH_OP"] = str(
            cfg.get("max_num_threadblock_graph_op", 6)
        )
    if "YIRAGE_CPU_MAX_KN_GRAPH_OP" not in os.environ:
        os.environ["YIRAGE_CPU_MAX_KN_GRAPH_OP"] = str(
            cfg.get("max_num_kernel_graph_op", 4)
        )
    if "YIRAGE_CPU_MAX_TB_GRAPH_INPUTS" not in os.environ:
        os.environ["YIRAGE_CPU_MAX_TB_GRAPH_INPUTS"] = str(
            cfg.get("max_num_threadblock_graph_inputs", 3)
        )
    os.environ["YIRAGE_CPU_SIMD"] = str(cfg.get("simd_type", "none"))


def get_cpu_search_config() -> Dict[str, Any]:
    """
    Get optimized search configuration for CPU backend.

    CPU characteristics:
    - Multi-threaded via OpenMP
    - SIMD vectorization (AVX2/AVX-512)
    - Cache hierarchy optimization
    - MoE expert-parallel token batching

    Returns:
        dict: Search configuration optimized for CPU
    """
    cpu_count = multiprocessing.cpu_count()
    # Match OpenMP / physical cores on this host (do not oversubscribe search)
    search_threads = max(1, min(cpu_count, int(cpu_count * 0.75) or cpu_count))
    simd_type = detect_simd_support()

    # Get vector width
    vector_width = {
        SIMDType.AVX512: 16,  # 512-bit / 32-bit
        SIMDType.AVX2: 8,  # 256-bit / 32-bit
        SIMDType.AVX: 8,
        SIMDType.SSE4: 4,
        SIMDType.SSE2: 4,
        SIMDType.SSE: 4,
        SIMDType.NEON: 4,
        SIMDType.NONE: 1,
    }.get(simd_type, 1)

    return {
        # CPU info
        "num_cores": cpu_count,
        "simd_type": simd_type.value,
        "vector_width": vector_width,
        # Search parameters
        "max_num_threadblock_graph_op": 6,  # +1 for MoE gate/linear ops
        "max_num_kernel_graph_op": 4,       # 4 distinct kernel types: gate, moe_linear, rms_norm, attn
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        # Search space — filled by resolve_cpu_search_space() when graph is known
        "grid_dims_to_explore": build_arch_aware_grid_dims(cpu_count, m_dim=0),
        "block_dims_to_explore": build_arch_aware_block_dims(
            vector_width, simd_type
        ),
        "fmaps_to_explore": [-1, 0, 1],
        "franges_to_explore": build_arch_aware_franges(64, vector_width),
        # Tiling for cache (GEMM tile sizes)
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        # MoE-specific search dimensions
        # num_experts_to_explore: try the standard MoE expert counts
        "num_experts_to_explore": [4, 8, 16, 64],
        # expert_batch_sizes: tokens to process together per expert
        # smaller = less scratch memory; larger = better cache locality
        "expert_batch_sizes": [1, 4, 8, 16, 32],
        # top_k values to cover during search
        "top_k_to_explore": [1, 2, 4],
    }


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    return {
        "num_cores": multiprocessing.cpu_count(),
        "simd_type": detect_simd_support().value,
        "platform": platform.processor(),
        "machine": platform.machine(),
    }


def get_cpu_runtime_config() -> Dict[str, Any]:
    """
    Runtime execution settings aligned with the local CPU architecture.

    Used by ``cpu_call`` and CPU profiling in ``superoptimize`` so execution
    thread count and TB-grid parallelism match the same machine that defined
    the search configuration.
    """
    search = get_cpu_search_config()
    cpu_count = search["num_cores"]
    torch_threads = min(search["search_thread"], cpu_count)
    parallel_grid = cpu_count > 1 and os.environ.get("YIRAGE_CPU_PARALLEL_GRID", "1") != "0"
    mlir_jit = os.environ.get("YIRAGE_CPU_MLIR_JIT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return {
        "torch_num_threads": torch_threads,
        "parallel_tb_grid": parallel_grid,
        "tb_grid_workers": min(cpu_count, 8),
        "simd_type": search["simd_type"],
        "vector_width": search["vector_width"],
        "num_cores": cpu_count,
        "mlir_jit_enabled": mlir_jit,
    }
