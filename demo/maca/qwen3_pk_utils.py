"""Shared helpers for MACA Qwen3 PersistentKernel scaffold (CUDA ``demo/qwen3/demo.py`` aligned)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from demo.maca.qwen_hf_utils import DEFAULT_QWEN_MODEL, default_qwen_dims


@dataclass(frozen=True)
class Qwen3PKScaffold:
    """Subset of ``demo/qwen3/demo.py`` PersistentKernel CLI defaults."""

    model: str = DEFAULT_QWEN_MODEL
    max_num_batched_tokens: int = 8
    max_num_batched_requests: int = 4
    page_size: int = 4096
    max_num_pages: int = 16
    max_seq_length: int = 512
    use_cutlass_kernel: bool = True
    mode: str = "offline"


def resolve_maca_pk_workers_schedulers(rank: int = 0) -> Tuple[int, int]:
    """Mirror ``yirage.get_configurations_from_gpu`` used by CUDA qwen3 demo."""
    import importlib.util
    import sys
    from pathlib import Path

    common_path = Path(__file__).resolve().parents[2] / "python" / "yirage" / "utils" / "common.py"
    spec = importlib.util.spec_from_file_location("yirage_utils_common_pk", common_path)
    common = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    return common.get_configurations_from_gpu(rank)


def maca_pk_runtime_smoke(
    *,
    num_workers: int,
    num_schedulers: int,
    device_id: int = 0,
) -> Dict[str, Any]:
    """Initialize MACA PK runtime in offline mode (Python simulation layer)."""
    import importlib.util
    import sys
    from pathlib import Path

    runtime_path = Path(__file__).resolve().parents[2] / "python" / "yirage" / "persistent_kernel" / "runtime.py"
    spec = importlib.util.spec_from_file_location("yirage_pk_runtime_smoke", runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)

    PKBackendType = runtime.PKBackendType
    PKMode = runtime.PKMode
    BACKEND_CAPABILITIES = runtime.BACKEND_CAPABILITIES
    create_runtime = runtime.create_runtime

    pk_runtime = create_runtime(
        backend=PKBackendType.MACA,
        mode=PKMode.OFFLINE,
        device_id=device_id,
        num_workers=num_workers,
        num_schedulers=num_schedulers,
    )
    ok = pk_runtime.initialize()
    caps = BACKEND_CAPABILITIES[PKBackendType.MACA]
    pk_runtime.finalize()
    return {
        "initialized": ok,
        "backend": PKBackendType.MACA.to_name(),
        "mode": PKMode.OFFLINE.name.lower(),
        "num_workers": num_workers,
        "num_schedulers": num_schedulers,
        "max_shared_memory": caps.max_shared_memory,
        "supported_modes": [m.name.lower() for m in caps.supported_modes],
    }


def inspect_qwen3_pk_scaffold(scaffold: Optional[Qwen3PKScaffold] = None) -> Dict[str, Any]:
    """Return inspect-only scaffold report (no GPU / no model weights)."""
    scaffold = scaffold or Qwen3PKScaffold()
    dims = default_qwen_dims()
    return {
        "cuda_reference": "demo/qwen3/demo.py --use-yirage",
        "maca_demo": "demo/maca/qwen3_persistent_kernel_demo.py",
        "model": scaffold.model,
        "mode": scaffold.mode,
        "max_num_batched_tokens": scaffold.max_num_batched_tokens,
        "max_num_batched_requests": scaffold.max_num_batched_requests,
        "page_size": scaffold.page_size,
        "max_num_pages": scaffold.max_num_pages,
        "max_seq_length": scaffold.max_seq_length,
        "hidden_size": dims.hidden_size,
        "intermediate_size": dims.intermediate_size,
        "fused_qkv_outdim": dims.fused_qkv_outdim,
        "compile_path": "mxcc_partial",
        "compile_note": (
            "PersistentKernel.compile() selects mxcc when YIRAGE_BACKEND=maca "
            "(get_maca_pk_compile_command). Full qwen3 task-graph e2e on MetaX VM "
            "remains experimental backlog."
        ),
        "yirage_backend": os.environ.get("YIRAGE_BACKEND", "maca"),
    }


__all__ = [
    "Qwen3PKScaffold",
    "inspect_qwen3_pk_scaffold",
    "maca_pk_runtime_smoke",
    "resolve_maca_pk_workers_schedulers",
]
