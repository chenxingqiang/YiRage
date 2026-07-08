"""Shared helpers for MACA Qwen3 PersistentKernel scaffold (CUDA ``demo/qwen3/demo.py`` aligned)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
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


def build_qwen3_pk_meta_tensors(
    scaffold: Qwen3PKScaffold,
    device: "torch.device",
) -> Dict[str, "torch.Tensor"]:
    """Synthetic meta tensors matching ``demo/qwen3/demo.py`` PersistentKernel contract."""
    import torch

    total = scaffold.max_num_batched_requests
    return {
        "step": torch.zeros(total, dtype=torch.int32, device=device),
        "tokens": torch.zeros(total, scaffold.max_seq_length, dtype=torch.long, device=device),
        "input_tokens": torch.zeros(
            scaffold.max_num_batched_tokens, 1, dtype=torch.long, device=device
        ),
        "output_tokens": torch.zeros(
            scaffold.max_num_batched_tokens, 1, dtype=torch.long, device=device
        ),
        "num_new_tokens": torch.ones(total, dtype=torch.int32, device=device),
        "prompt_lengths": torch.ones(total, dtype=torch.int32, device=device),
        "qo_indptr_buffer": torch.zeros(total + 1, dtype=torch.int32, device=device),
        "paged_kv_indptr_buffer": torch.zeros(total + 1, dtype=torch.int32, device=device),
        "paged_kv_indices_buffer": torch.zeros(
            scaffold.max_num_pages, dtype=torch.int32, device=device
        ),
        "paged_kv_last_page_len_buffer": torch.zeros(total, dtype=torch.int32, device=device),
    }


def inspect_maca_pk_compile_plan(
    scaffold: Optional[Qwen3PKScaffold] = None,
) -> Dict[str, Any]:
    """Cloud-safe compile plan: mxcc contract + minimal embed task prerequisites."""
    scaffold = scaffold or Qwen3PKScaffold()
    dims = default_qwen_dims()
    contract = inspect_maca_pk_compile_contract(scaffold)
    repo_root = Path(__file__).resolve().parents[2]
    return {
        **contract,
        "cuda_reference": "demo/qwen3/demo.py --use-yirage",
        "minimal_task_graph": "embed_layer",
        "hidden_size": dims.hidden_size,
        "vocab_smoke_size": 128,
        "requires": ["yirage.core", "YIRAGE_HOME", "mxcc", "MetaX GPU"],
        "yirage_home_default": str(repo_root),
        "yirage_home_set": "YIRAGE_HOME" in os.environ,
        "compile_plan_ready": contract["compile_ready"],
    }


def maca_pk_minimal_compile_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build minimal embed-only PK task graph and ``ypk.compile()`` via mxcc (MetaX VM)."""
    import tempfile

    import torch
    import yirage as yr

    scaffold = scaffold or Qwen3PKScaffold()
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("YIRAGE_BACKEND", "maca")
    os.environ.setdefault("YIRAGE_HOME", str(repo_root))

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    dims = default_qwen_dims()
    hidden_size = dims.hidden_size
    vocab_smoke = 128

    num_workers, num_schedulers = resolve_maca_pk_workers_schedulers(0)
    meta = build_qwen3_pk_meta_tensors(scaffold, device)

    ypk = yr.PersistentKernel(
        mode=scaffold.mode,
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=scaffold.max_seq_length,
        max_num_batched_requests=scaffold.max_num_batched_requests,
        max_num_batched_tokens=scaffold.max_num_batched_tokens,
        max_num_pages=scaffold.max_num_pages,
        page_size=scaffold.page_size,
        eos_token_id=0,
        meta_tensors=meta,
        profiler_tensor=None,
        trace_name="maca_pk_minimal",
        spec_decode_config=None,
        use_cutlass_kernel=scaffold.use_cutlass_kernel,
    )

    input_t = ypk.attach_input(torch_tensor=meta["input_tokens"], name="input_token")
    w_embed = ypk.attach_input(
        torch_tensor=torch.randn(vocab_smoke, hidden_size, dtype=torch.bfloat16, device=device),
        name="embed_tokens",
    )
    embed_out = ypk.new_tensor(
        dims=(scaffold.max_num_batched_tokens, hidden_size),
        dtype=yr.bfloat16,
        name="embed_out",
        io_category="cuda_tensor",
    )
    ypk.embed_layer(
        input=input_t,
        weight=w_embed,
        output=embed_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        input_source=1,
    )

    out_dir = output_dir or tempfile.mkdtemp(prefix="maca_pk_compile_")
    ypk.compile(output_dir=out_dir)

    cu_path = Path(out_dir) / "test_rank0.cu"
    json_path = Path(out_dir) / "task_graph_rank0.json"
    return {
        "compiled": True,
        "compiler": "mxcc",
        "output_dir": out_dir,
        "cu_artifact": cu_path.is_file(),
        "json_artifact": json_path.is_file(),
        "num_workers": num_workers,
        "num_schedulers": num_schedulers,
        "tasks": ["embedding"],
        "target_cc": ypk.target_cc,
    }


def inspect_maca_pk_compile_contract(
    scaffold: Optional[Qwen3PKScaffold] = None,
) -> Dict[str, Any]:
    """Validate mxcc PK compile flags in kernel.py against qwen3 scaffold (no GPU)."""
    scaffold = scaffold or Qwen3PKScaffold()
    kernel_path = Path(__file__).resolve().parents[2] / "python" / "yirage" / "persistent_kernel" / "kernel.py"
    text = kernel_path.read_text(encoding="utf-8")
    required_tokens = [
        "get_maca_pk_compile_command",
        "_resolve_persistent_kernel_compiler",
        "YIRAGE_BACKEND_MACA_ENABLED",
        "-DMODE_OFFLINE",
        "YPK_MAX_NUM_BATCHED_TOKENS",
        "YPK_MAX_NUM_BATCHED_REQUESTS",
        "YPK_MAX_NUM_PAGES",
        "YPK_PAGE_SIZE",
        "YPK_MAX_SEQ_LENGTH",
        "--maca-path=",
        "-lmcruntime",
    ]
    missing = [tok for tok in required_tokens if tok not in text]
    expected_defines = {
        "max_num_batched_tokens": scaffold.max_num_batched_tokens,
        "max_num_batched_requests": scaffold.max_num_batched_requests,
        "max_num_pages": scaffold.max_num_pages,
        "page_size": scaffold.page_size,
        "max_seq_length": scaffold.max_seq_length,
        "mode": scaffold.mode,
    }
    return {
        "kernel_source": str(kernel_path.relative_to(kernel_path.parents[3])),
        "compiler": "mxcc",
        "required_tokens_ok": len(missing) == 0,
        "missing_tokens": missing,
        "expected_defines": expected_defines,
        "compile_ready": len(missing) == 0 and scaffold.mode == "offline",
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
        "compile_path": "mxcc",
        "compile_note": (
            "PersistentKernel.compile() selects mxcc when YIRAGE_BACKEND=maca "
            "(get_maca_pk_compile_command). Use --compile-plan/--compile-inspect "
            "for Cloud contract; MetaX --compile-only runs minimal embed task-graph."
        ),
        "yirage_backend": os.environ.get("YIRAGE_BACKEND", "maca"),
    }


__all__ = [
    "Qwen3PKScaffold",
    "build_qwen3_pk_meta_tensors",
    "inspect_maca_pk_compile_contract",
    "inspect_maca_pk_compile_plan",
    "inspect_qwen3_pk_scaffold",
    "maca_pk_minimal_compile_smoke",
    "maca_pk_runtime_smoke",
    "resolve_maca_pk_workers_schedulers",
]
