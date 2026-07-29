# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""YiRage core execution + CPU superoptimize helpers for RuntimeFusion serving.

Serving MLP on CPU uses a **split kernel** strategy (aligned with Qwen MACA demos):

1. **gate_up** — seed graph execute via ``yirage.core`` (rms_norm + mul + matmul)
2. **mid** — ``silu(gate) * up`` in PyTorch (tiny epilogue)
3. **down** — ``superoptimize(backend="cpu")`` on plain matmul ``(1,I) @ (I,H)``
4. **residual** — PyTorch add

Full-graph superoptimize for the entire MLP may yield 0 valid µGraphs under
tractable CPU search caps. Down matmul uses ``superoptimize(backend=\"cpu\")``.

**Search tiers** (no seed fallback):
- Default (``YIRAGE_SERVING_USE_RAY`` unset): seed fingerprint verify — fast smoke
- ``YIRAGE_SERVING_FULL_TB_SEARCH=1``: tractable TB-customized matmul search (no seed verify)
- ``YIRAGE_SERVING_USE_RAY=1`` / ``YIRAGE_SERVING_USE_COORDINATOR=1``:
  ``DistributedSearchCoordinator.parallel_search`` with CPU search space;
  partitions ``blockdims`` when decode ``m=1`` (griddims=1)
- Full TB + Ray: combine both; ``serving_env`` propagated to workers via coordinator config
- ``YIRAGE_SERVING_ACCELFORGE_PRESCREEN=1``: optional AccelForge prescreen before profiling
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .torch_exec import require_torch, to_torch


def is_yirage_core_available() -> bool:
    if os.environ.get("YIRAGE_SKIP_NATIVE") == "1":
        return False
    try:
        import yirage as yr
        import yirage.core  # noqa: F401

        return hasattr(yr, "float32")
    except ImportError:
        return False


def require_yirage_core() -> None:
    if not is_yirage_core_available():
        raise RuntimeError(
            "yirage.core is not built. Run scripts/setup_serving_yirage_core.sh "
            "or pip install -e . with YIRAGE_BACKEND=cpu."
        )


def _yr_dtype(name: str):
    import yirage as yr

    if name in ("float32", "fp32"):
        return yr.float32
    if name in ("float16", "fp16"):
        return yr.float16
    if name in ("bfloat16", "bf16"):
        return yr.bfloat16
    raise ValueError(f"unsupported yirage dtype name: {name!r}")


def resolve_serving_use_ray(*, default: bool = False) -> bool:
    """Opt-in Ray for serving CPU superoptimize (``YIRAGE_SERVING_USE_RAY=1``).

    Ray partitions ``griddims`` when ``m>1``, or ``blockdims`` for decode ``m=1``.
    """
    raw = os.environ.get("YIRAGE_SERVING_USE_RAY", "")
    if raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def resolve_serving_num_workers(*, default: int = 2) -> int:
    raw = os.environ.get("YIRAGE_SERVING_RAY_WORKERS", "")
    if raw.strip():
        return max(1, int(raw))
    return max(1, default)


def resolve_serving_use_coordinator(*, default: bool = False) -> bool:
    """Use ``DistributedSearchCoordinator`` (default on when ``YIRAGE_SERVING_USE_RAY=1``)."""
    raw = os.environ.get("YIRAGE_SERVING_USE_COORDINATOR", "")
    if raw == "":
        return resolve_serving_use_ray(default=default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def serving_superoptimize_ray_kwargs(*, default: bool = False) -> Dict[str, Any]:
    """``use_ray`` / ``num_workers`` kwargs for serving ``superoptimize``."""
    use_ray = resolve_serving_use_ray(default=default)
    if not use_ray:
        return {"use_ray": False}
    workers_raw = os.environ.get("YIRAGE_SERVING_RAY_WORKERS", "")
    kwargs: Dict[str, Any] = {"use_ray": True}
    if workers_raw.strip():
        kwargs["num_workers"] = max(1, int(workers_raw))
    return kwargs


def resolve_serving_full_tb_search(*, default: bool = False) -> bool:
    """Opt-in tractable TB-customized down matmul search (no seed-verify shortcut)."""
    raw = os.environ.get("YIRAGE_SERVING_FULL_TB_SEARCH", "")
    if raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def resolve_serving_accelforge_prescreen(*, default: bool = False) -> bool:
    """Opt-in AccelForge prescreen before coordinator result profiling."""
    raw = os.environ.get("YIRAGE_SERVING_ACCELFORGE_PRESCREEN", "")
    if raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def resolve_serving_accelforge_latency_budget_ms() -> Optional[float]:
    """Optional AccelForge latency budget (ms) for prescreen reject-path."""
    raw = os.environ.get("YIRAGE_SERVING_ACCELFORGE_LATENCY_BUDGET_MS", "")
    if not raw.strip():
        return None
    return float(raw)


def _accelforge_prescreen_kwargs() -> Dict[str, Any]:
    budget = resolve_serving_accelforge_latency_budget_ms()
    if budget is None:
        return {}
    return {"latency_budget_ms": budget}


def resolve_serving_search_tier() -> str:
    """Active serving down-matmul search tier label for e2e/cert reporting."""
    full_tb = resolve_serving_full_tb_search()
    use_ray = resolve_serving_use_ray()
    prescreen = resolve_serving_accelforge_prescreen()
    if full_tb and use_ray:
        return "full_tb_ray_accelforge" if prescreen else "full_tb_ray"
    if use_ray:
        return "ray_coordinator" if resolve_serving_use_coordinator() else "ray"
    if full_tb:
        return "full_tb"
    return "seed_verify"


def inspect_serving_search_tier() -> Dict[str, Any]:
    """JSON-serializable serving search tier snapshot."""
    return {
        "tier": resolve_serving_search_tier(),
        "full_tb_search": resolve_serving_full_tb_search(),
        "use_ray": resolve_serving_use_ray(),
        "use_coordinator": resolve_serving_use_coordinator(),
        "accelforge_prescreen": resolve_serving_accelforge_prescreen(),
        "accelforge_latency_budget_ms": resolve_serving_accelforge_latency_budget_ms(),
        "ray_workers": resolve_serving_num_workers(),
    }


_LAST_ACCELFORGE_PRESCREEN_STATS: Optional[Dict[str, Any]] = None


def _load_accelforge_verifier_class():
    """Import AccelForgeVerifier without executing broken ``yirage.rl`` package init."""
    import importlib
    import sys
    import types
    from pathlib import Path

    rl_root = Path(__file__).resolve().parents[1] / "rl"
    for name, sub in (
        ("yirage.rl", rl_root),
        ("yirage.rl.verifier", rl_root / "verifier"),
        ("yirage.rl.hardware", rl_root / "hardware"),
    ):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(sub)]
            sys.modules[name] = pkg

    mod = importlib.import_module("yirage.rl.verifier.accelforge_verifier")
    return mod.AccelForgeVerifier


# Env keys propagated to Ray workers via coordinator ``serving_env`` payload.
_SERVING_ENV_KEYS: Tuple[str, ...] = (
    "YIRAGE_SERVING_FULL_TB_SEARCH",
    "YIRAGE_SERVING_KN_MATMUL_ONLY",
    "YIRAGE_CPU_MAX_KN_GRAPH_OP",
    "YIRAGE_CPU_MAX_TB_GRAPH_OP",
    "YIRAGE_CPU_MAX_TB_GRAPH_INPUTS",
    "YIRAGE_CPU_BENCH_MINIMAL_EXPLORE",
    "YIRAGE_SERVING_USE_RAY",
)


def snapshot_serving_env() -> Dict[str, Optional[str]]:
    """Capture serving search env for Ray worker replay (``None`` → unset on worker)."""
    return {key: os.environ.get(key) for key in _SERVING_ENV_KEYS}


def apply_serving_env(env: Optional[Dict[str, Optional[str]]]) -> None:
    """Apply ``serving_env`` snapshot on a search worker process."""
    if not env:
        return
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def apply_serving_full_tb_search_tractability(*, use_ray: Optional[bool] = None) -> None:
    """Enable tractable TB matmul explore for serving (Qwen-scale K with capped search)."""
    os.environ.pop("YIRAGE_SERVING_KN_MATMUL_ONLY", None)
    os.environ["YIRAGE_SERVING_FULL_TB_SEARCH"] = "1"
    os.environ["YIRAGE_CPU_MAX_KN_GRAPH_OP"] = "4"
    os.environ["YIRAGE_CPU_MAX_TB_GRAPH_OP"] = "3"
    os.environ["YIRAGE_CPU_MAX_TB_GRAPH_INPUTS"] = "2"
    os.environ["YIRAGE_CPU_BENCH_MINIMAL_EXPLORE"] = "1"
    ray = resolve_serving_use_ray() if use_ray is None else use_ray
    if ray:
        os.environ["YIRAGE_SERVING_USE_RAY"] = "1"
    else:
        os.environ.pop("YIRAGE_SERVING_USE_RAY", None)


def apply_serving_cpu_search_tractability(*, use_ray: Optional[bool] = None) -> None:
    """Cap CPU search for serving plain-matmul superoptimize smoke."""
    from scripts.cpu_cert_utils import apply_plain_matmul_search_tractability

    apply_plain_matmul_search_tractability()
    ray = resolve_serving_use_ray() if use_ray is None else use_ray
    if ray:
        os.environ["YIRAGE_SERVING_USE_RAY"] = "1"
        os.environ.pop("YIRAGE_SERVING_KN_MATMUL_ONLY", None)
    else:
        os.environ["YIRAGE_SERVING_KN_MATMUL_ONLY"] = "1"
        os.environ.pop("YIRAGE_SERVING_USE_RAY", None)


def apply_serving_kn_down_matmul_tractability(*, use_ray: Optional[bool] = None) -> None:
    """Serving down matmul search tractability (seed verify, full TB, or Ray)."""
    if resolve_serving_full_tb_search():
        apply_serving_full_tb_search_tractability(use_ray=use_ray)
    else:
        apply_serving_cpu_search_tractability(use_ray=use_ray)


def superoptimize_kwargs(*, quick: bool = True) -> Dict[str, Any]:
    use_ray = resolve_serving_use_ray()
    full_tb = resolve_serving_full_tb_search()
    kwargs: Dict[str, Any] = {
        "backend": "cpu",
        "use_graph_dataset": False,
        "use_cached_graphs": False,
        "use_persistent_cache": True,
        "warmup_iters": 1,
        "profile_iters": 5 if quick else 20,
        "verbose": False,
        **serving_superoptimize_ray_kwargs(),
    }
    if use_ray:
        return kwargs
    if full_tb:
        kwargs.update(
            {
                "griddims": [(1, 1, 1)],
                "blockdims": [(128, 1, 1)],
                "franges": [1],
            }
        )
        return kwargs
    kwargs.update(
        {
            "griddims": [(1, 1, 1)],
            "blockdims": [(32, 1, 1)],
            "franges": [1],
        }
    )
    return kwargs


def build_gate_up_seed_graph(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
):
    """RMSNorm + mul + matmul gate/up (decode shape ``[1, H]``)."""
    import yirage as yr

    dtype = _yr_dtype(dtype_name)
    graph = yr.new_kernel_graph()
    x = graph.new_input(dims=(1, hidden_size), dtype=dtype)
    g = graph.new_input(dims=(1, hidden_size), dtype=dtype)
    w = graph.new_input(
        dims=(hidden_size, 2 * intermediate_size),
        strides=(1, hidden_size),
        dtype=dtype,
    )
    d = graph.rms_norm(x, normalized_shape=(hidden_size,))
    d = graph.mul(d, g)
    o = graph.matmul(d, w)
    graph.mark_output(o)
    return graph


def build_mlp_down_seed_graph(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
):
    """SiLU(gate) * up + matmul down (decode shape)."""
    import yirage as yr

    dtype = _yr_dtype(dtype_name)
    graph = yr.new_kernel_graph()
    x = graph.new_input(dims=(1, intermediate_size), dtype=dtype)
    y = graph.new_input(dims=(1, intermediate_size), dtype=dtype)
    w = graph.new_input(
        dims=(intermediate_size, hidden_size),
        strides=(1, intermediate_size),
        dtype=dtype,
    )
    d = graph.mul(graph.silu(x), y)
    o = graph.matmul(d, w)
    graph.mark_output(o)
    return graph


def build_down_matmul_seed_graph(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
):
    """Plain matmul ``(1,I) @ (I,H)`` for superoptimize."""
    import yirage as yr

    dtype = _yr_dtype(dtype_name)
    graph = yr.new_kernel_graph()
    mid = graph.new_input(dims=(1, intermediate_size), dtype=dtype)
    w = graph.new_input(
        dims=(intermediate_size, hidden_size),
        strides=(1, intermediate_size),
        dtype=dtype,
    )
    graph.mark_output(graph.matmul(mid, w))
    return graph


def build_serving_cpu_search_config(graph) -> Dict[str, Any]:
    """CPU search config for ``DistributedSearchCoordinator`` (decode m=1 → blockdim partition)."""
    from yirage.backends.cpu.config import apply_cpu_search_env, resolve_cpu_search_space

    cygraph = graph.cygraph if hasattr(graph, "cygraph") else graph
    cpu_config = resolve_cpu_search_space(cygraph)
    apply_cpu_search_env(cpu_config)
    config: Dict[str, Any] = {
        "griddims": list(cpu_config.get("grid_dims_to_explore", [(1, 1, 1)])),
        "blockdims": list(cpu_config.get("block_dims_to_explore", [(32, 1, 1)])),
        "franges": list(cpu_config.get("franges_to_explore", [1])),
        "fmaps": [-1],
        "verbose": False,
        "serving_env": snapshot_serving_env(),
    }
    if resolve_serving_full_tb_search():
        config["griddims"] = [(1, 1, 1)]
        config["blockdims"] = [(128, 1, 1)]
        config["franges"] = [1]
    return config


def last_serving_accelforge_prescreen_stats() -> Optional[Dict[str, Any]]:
    """Stats from the most recent coordinator AccelForge prescreen (if any)."""
    return _LAST_ACCELFORGE_PRESCREEN_STATS


def bench_serving_accelforge_prescreen(
    entries: Sequence[Dict[str, Any]],
    *,
    enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Bench AccelForge prescreen on coordinator graph entries; returns accept/reject stats."""
    global _LAST_ACCELFORGE_PRESCREEN_STATS

    prescreen_on = (
        resolve_serving_accelforge_prescreen()
        if enabled is None
        else enabled
    )
    input_count = sum(1 for e in entries if e.get("graph_json"))
    stats: Dict[str, Any] = {
        "enabled": prescreen_on,
        "input_count": input_count,
        "accepted_count": input_count,
        "rejected_count": 0,
        "verifier_available": False,
    }
    if not prescreen_on or input_count == 0:
        _LAST_ACCELFORGE_PRESCREEN_STATS = stats
        return stats

    try:
        AccelForgeVerifier = _load_accelforge_verifier_class()
    except Exception:
        _LAST_ACCELFORGE_PRESCREEN_STATS = stats
        return stats

    stats["verifier_available"] = True
    verifier = AccelForgeVerifier()
    prescreen_kwargs = _accelforge_prescreen_kwargs()
    accepted = 0
    rejected = 0
    sample: List[Dict[str, Any]] = []
    for entry in entries:
        graph_json = entry.get("graph_json")
        if not graph_json:
            continue
        result = verifier.prescreen_kernel(graph_json, **prescreen_kwargs)
        row = {
            "accepted": bool(result.get("accepted", True)),
            "rejections": list(result.get("rejections") or []),
        }
        if len(sample) < 4:
            sample.append(row)
        if row["accepted"]:
            accepted += 1
        else:
            rejected += 1
    stats["accepted_count"] = accepted
    stats["rejected_count"] = rejected
    stats["sample"] = sample
    _LAST_ACCELFORGE_PRESCREEN_STATS = stats
    return stats


def _prescreen_coordinator_graph_entries(
    entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Optional AccelForge prescreen; returns accepted entries (no-op when disabled)."""
    if not resolve_serving_accelforge_prescreen():
        bench_serving_accelforge_prescreen(entries, enabled=False)
        return list(entries)

    stats = bench_serving_accelforge_prescreen(entries, enabled=True)
    if not stats.get("verifier_available"):
        return list(entries)

    try:
        AccelForgeVerifier = _load_accelforge_verifier_class()
    except Exception:
        return list(entries)

    verifier = AccelForgeVerifier()
    prescreen_kwargs = _accelforge_prescreen_kwargs()
    accepted: List[Dict[str, Any]] = []
    for entry in entries:
        graph_json = entry.get("graph_json")
        if not graph_json:
            continue
        result = verifier.prescreen_kernel(graph_json, **prescreen_kwargs)
        if result.get("accepted", True):
            accepted.append(entry)
    return accepted


def _kngraph_from_graph_json(graph_json_text: str):
    """Rebuild executable ``KNGraph`` from worker/coordinator JSON payload."""
    import os
    import tempfile

    from yirage.core import cy_from_json
    from yirage.kernel.graph import KNGraph

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(graph_json_text)
        temp_path = f.name
    try:
        cygraph = cy_from_json(temp_path)
        return KNGraph(cygraph, backend="cpu")
    finally:
        os.unlink(temp_path)


def _profile_and_pick_best_kngraph(
    graphs: Sequence[Any],
    *,
    quick: bool = True,
) -> Any:
    """Profile candidate µGraphs on CPU and return the fastest executable graph."""
    import time

    import torch
    from yirage.core import convert_dtype_to_torch_type
    from yirage.kernel.graph import _cpu_runtime_context

    if not graphs:
        raise RuntimeError("no graphs to profile")
    if len(graphs) == 1:
        graphs[0].backend = "cpu"
        return graphs[0]

    warmup_iters = 1
    profile_iters = 5 if quick else 20
    best_graph: Any = None
    best_perf = float("inf")

    for g in graphs:
        dtensors = g.cygraph.get_input_dtensors()
        input_tensors: List[Any] = []
        for t in dtensors:
            dims, strides = g.cygraph.get_input_dtensor_shape_and_stride(t)
            dtype = convert_dtype_to_torch_type(t.dtype)
            x = torch.randn(dims, dtype=dtype, device="cpu")
            x = torch.as_strided(x, size=dims, stride=strides)
            input_tensors.append(x)

        with _cpu_runtime_context():
            for _ in range(warmup_iters):
                try:
                    g(inputs=input_tensors)
                except Exception:
                    continue
            start_time = time.perf_counter()
            for _ in range(profile_iters):
                try:
                    g(inputs=input_tensors)
                except Exception:
                    break
            end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) / profile_iters * 1000
        if elapsed_ms < best_perf:
            best_perf = elapsed_ms
            best_graph = g

    if best_graph is None:
        raise RuntimeError("CPU profile found no executable µGraph")
    best_graph.backend = "cpu"
    return best_graph


def superoptimize_down_matmul_via_coordinator(graph, *, quick: bool = True):
    """Distributed CPU search via ``DistributedSearchCoordinator``; raises on 0 valid µGraphs."""
    from yirage.ray.coordinator import DistributedSearchCoordinator

    apply_serving_kn_down_matmul_tractability(use_ray=True)
    search_config = build_serving_cpu_search_config(graph)
    num_workers = resolve_serving_num_workers()
    use_ray = resolve_serving_use_ray(default=True)

    coord = DistributedSearchCoordinator(num_workers=num_workers, use_ray=use_ray)
    try:
        out = coord.parallel_search(
            computation_graph=graph,
            config=search_config,
            backend="cpu",
            collect_feedback=False,
            verbose=False,
        )
    finally:
        coord.shutdown()

    raw_entries = [e for e in (out.get("graphs") or []) if e.get("graph_json")]
    screened_entries = _prescreen_coordinator_graph_entries(raw_entries)

    kn_graphs: List[Any] = []
    for entry in screened_entries:
        kn_graphs.append(_kngraph_from_graph_json(entry["graph_json"]))

    if not kn_graphs:
        raise RuntimeError(
            "DistributedSearchCoordinator found 0 valid µGraphs for down matmul"
        )
    return _profile_and_pick_best_kngraph(kn_graphs, quick=quick)


def superoptimize_down_matmul_cpu(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
    quick: bool = True,
):
    """Superoptimize down-projection matmul; raises if search finds nothing."""
    require_yirage_core()
    apply_serving_kn_down_matmul_tractability()
    graph = build_down_matmul_seed_graph(
        hidden_size,
        intermediate_size,
        dtype_name=dtype_name,
    )
    if resolve_serving_use_coordinator():
        return superoptimize_down_matmul_via_coordinator(graph, quick=quick)
    optimized = graph.superoptimize(**superoptimize_kwargs(quick=quick))
    if optimized is None:
        raise RuntimeError(
            f"CPU superoptimize found 0 valid µGraphs for down matmul "
            f"(H={hidden_size}, I={intermediate_size})"
        )
    return optimized


@dataclass
class SuperoptimizeTiming:
    elapsed_s: float
    hidden_size: int
    intermediate_size: int
    backend: str = "cpu"


def bench_superoptimize_down_matmul(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
    quick: bool = True,
) -> Tuple[Any, SuperoptimizeTiming]:
    """Run superoptimize once and return (optimized_graph, timing)."""
    t0 = time.perf_counter()
    opt = superoptimize_down_matmul_cpu(
        hidden_size,
        intermediate_size,
        dtype_name=dtype_name,
        quick=quick,
    )
    elapsed = time.perf_counter() - t0
    return opt, SuperoptimizeTiming(
        elapsed_s=elapsed,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


@dataclass
class YirageMlpCompileArtifacts:
    gate_up_graph: Any
    down_optimized: Any
    superopt_elapsed_s: float


class YirageServingMlpRunner:
    """Hybrid MLP: yirage.core seed gate_up + superoptimized down matmul."""

    def __init__(
        self,
        *,
        rms_weight: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        eps: float = 1e-6,
        device: Optional[str] = None,
        dtype_name: str = "float32",
        quick_superopt: bool = True,
    ):
        require_yirage_core()
        require_torch()
        import torch

        self.eps = float(eps)
        self.dtype_name = dtype_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.rms_weight = to_torch(rms_weight, device=self._device, dtype=torch.float32)
        self.w_gate = to_torch(w_gate, device=self._device, dtype=torch.float32)
        self.w_up = to_torch(w_up, device=self._device, dtype=torch.float32)
        self.w_down = to_torch(w_down, device=self._device, dtype=torch.float32)

        h = self.w_gate.shape[0]
        i = self.w_gate.shape[1]
        self.hidden_size = h
        self.intermediate_size = i

        self._w_gate_up = torch.cat([self.w_gate, self.w_up], dim=1)
        self._gate_up_graph = build_gate_up_seed_graph(h, i, dtype_name=dtype_name)
        self._down_optimized, timing = bench_superoptimize_down_matmul(
            h,
            i,
            dtype_name=dtype_name,
            quick=quick_superopt,
        )
        self.superopt_elapsed_s = timing.elapsed_s
        self._yr_dtype = _yr_dtype(dtype_name)

    @property
    def artifacts(self) -> YirageMlpCompileArtifacts:
        return YirageMlpCompileArtifacts(
            gate_up_graph=self._gate_up_graph,
            down_optimized=self._down_optimized,
            superopt_elapsed_s=self.superopt_elapsed_s,
        )

    def _torch_dtype(self):
        import torch

        if self.dtype_name in ("float16", "fp16"):
            return torch.float16
        if self.dtype_name in ("bfloat16", "bf16"):
            return torch.bfloat16
        return torch.float32

    def _gate_up_yirage(self, hidden: Any) -> Any:
        import torch

        h = to_torch(hidden, device=self._device, dtype=torch.float32)
        if h.ndim == 1:
            h = h.unsqueeze(0)
        batch = h.shape[0]
        if batch != 1:
            raise ValueError(
                f"YirageServingMlpRunner gate_up expects batch=1 decode shape, got {batch}"
            )
        rw = self.rms_weight
        if rw.ndim == 1:
            rw = rw.unsqueeze(0)
        yr_dtype = self._torch_dtype()
        yr_h = h.to(dtype=yr_dtype)
        yr_rw = rw.to(dtype=yr_dtype)
        yr_w = self._w_gate_up.to(dtype=yr_dtype)
        out = self._gate_up_graph(inputs=[yr_h, yr_rw, yr_w])
        return out[0]

    def forward(self, hidden: Any) -> Any:
        """Full MLP forward with yirage gate_up + superopt down."""
        import torch
        import torch.nn.functional as F

        residual = to_torch(hidden, device=self._device, dtype=torch.float32)
        if residual.ndim == 1:
            residual = residual.unsqueeze(0)

        gate_up = self._gate_up_yirage(residual)
        gate_up_f = gate_up.float()
        gate, up = torch.chunk(gate_up_f, 2, dim=-1)
        mid = F.silu(gate) * up

        yr_dtype = self._torch_dtype()
        mid_yr = mid.to(dtype=yr_dtype)
        w_down_yr = self.w_down.to(dtype=yr_dtype)
        down_out = self._down_optimized(inputs=[mid_yr, w_down_yr])[0].float()
        return residual + down_out

    def forward_torch_reference(self, hidden: Any) -> Any:
        from .torch_exec import mlp_torch

        return mlp_torch(
            hidden,
            rms_weight=self.rms_weight,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
            eps=self.eps,
        )


def mlp_yirage_cpu(
    hidden: Any,
    *,
    runner: YirageServingMlpRunner,
) -> Any:
    return runner.forward(hidden)
