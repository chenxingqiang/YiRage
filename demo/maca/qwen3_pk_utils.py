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


def grid_for_rmsnorm_linear_layer(size: int, use_cutlass_kernel: bool = True) -> int:
    """Mirror ``demo/qwen3/demo.py`` grid selection for rmsnorm+linear PK tasks."""
    if size % 64 == 0 and not use_cutlass_kernel:
        return size // 64
    if size / 96 > 400:
        if size % 256 != 0:
            raise ValueError(f"unsupported rmsnorm linear size {size}")
        return size // 256
    if size % 96 == 0:
        return 96
    if size % 64 == 0:
        return 64
    raise ValueError(f"unsupported rmsnorm linear size {size}")


def _maca_pk_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _init_maca_pk_yirage(
    scaffold: Qwen3PKScaffold,
):
    """Create PersistentKernel + meta tensors on MACA device (MetaX VM)."""
    import torch
    import yirage as yr

    repo_root = _maca_pk_repo_root()
    os.environ.setdefault("YIRAGE_BACKEND", "maca")
    os.environ.setdefault("YIRAGE_HOME", str(repo_root))

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    dims = default_qwen_dims()
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
        trace_name="maca_pk_smoke",
        spec_decode_config=None,
        use_cutlass_kernel=scaffold.use_cutlass_kernel,
    )
    return ypk, meta, dims, device, num_workers, num_schedulers, yr


def _build_maca_pk_embed_graph(
    ypk,
    meta,
    scaffold: Qwen3PKScaffold,
    *,
    hidden_size: int,
    vocab_smoke: int,
    yr,
):
    """Attach embed-only task (returns activation tensor ``x``)."""
    import torch

    input_t = ypk.attach_input(torch_tensor=meta["input_tokens"], name="input_token")
    w_embed = ypk.attach_input(
        torch_tensor=torch.randn(vocab_smoke, hidden_size, dtype=torch.bfloat16, device=meta["tokens"].device),
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
    return embed_out


def _build_maca_pk_one_layer_graph(
    ypk,
    meta,
    scaffold: Qwen3PKScaffold,
    dims: "QwenModelDims",
    *,
    device: "torch.device",
    yr,
    layer_idx: int = 0,
):
    """Attach one Qwen3 decoder block (demo/qwen3/demo.py layer loop, synthetic weights)."""
    import torch

    hidden_size = dims.hidden_size
    intermediate_size = dims.intermediate_size
    head_dim = dims.head_dim
    num_q_heads = dims.num_heads
    num_kv_heads = dims.num_kv_heads
    fused_qkv_out = (num_q_heads + 2 * num_kv_heads) * head_dim
    tokens = scaffold.max_num_batched_tokens

    rmsnorm_out = ypk.new_tensor(
        dims=(tokens, hidden_size), dtype=yr.bfloat16, name="rmsnorm_out", io_category="cuda_tensor"
    )
    attn_in = ypk.new_tensor(
        dims=(tokens, fused_qkv_out), dtype=yr.bfloat16, name="attn_in", io_category="cuda_tensor"
    )
    attn_out = ypk.new_tensor(
        dims=(tokens, num_q_heads * head_dim), dtype=yr.bfloat16, name="attn_out", io_category="cuda_tensor"
    )
    attn_proj_out = ypk.new_tensor(
        dims=(tokens, hidden_size), dtype=yr.bfloat16, name="attn_proj_out", io_category="cuda_tensor"
    )
    mlp_mid = ypk.new_tensor(
        dims=(tokens, 2 * intermediate_size), dtype=yr.bfloat16, name="mlp_mid", io_category="cuda_tensor"
    )
    silu_mul_out = ypk.new_tensor(
        dims=(tokens, intermediate_size), dtype=yr.bfloat16, name="silu_mul_out", io_category="cuda_tensor"
    )
    mlp_out = ypk.new_tensor(
        dims=(tokens, hidden_size), dtype=yr.bfloat16, name="mlp_out", io_category="cuda_tensor"
    )

    cos_pos_embed = ypk.attach_input(
        torch_tensor=torch.randn(scaffold.page_size, head_dim, dtype=torch.bfloat16, device=device),
        name="cos_position_embedding",
    )
    sin_pos_embed = ypk.attach_input(
        torch_tensor=torch.randn(scaffold.page_size, head_dim, dtype=torch.bfloat16, device=device),
        name="sin_position_embedding",
    )

    x = _build_maca_pk_embed_graph(
        ypk, meta, scaffold, hidden_size=hidden_size, vocab_smoke=128, yr=yr
    )

    w_norm = ypk.attach_input(
        torch_tensor=torch.randn(hidden_size, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_input_layernorm",
    )
    w_q = ypk.attach_input(
        torch_tensor=torch.randn(hidden_size, num_q_heads * head_dim, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_q_proj",
    )
    w_k = ypk.attach_input(
        torch_tensor=torch.randn(hidden_size, num_kv_heads * head_dim, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_k_proj",
    )
    w_v = ypk.attach_input(
        torch_tensor=torch.randn(hidden_size, num_kv_heads * head_dim, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_v_proj",
    )
    w_qkv = ypk.shuffle_tensors(
        inputs=[w_q, w_k, w_v],
        shuffled_dim=0,
        num_groups=num_kv_heads,
        name=f"layer_{layer_idx}_qkv_proj",
    )
    ypk.rmsnorm_layer(
        input=x,
        weight=w_norm,
        output=rmsnorm_out,
        grid_dim=(tokens, 1, 1),
        block_dim=(128, 1, 1),
    )
    ypk.linear_layer(
        input=rmsnorm_out,
        weight=w_qkv,
        output=attn_in,
        grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0), scaffold.use_cutlass_kernel), 1, 1),
        block_dim=(128, 1, 1),
    )

    w_q_norm = ypk.attach_input(
        torch_tensor=torch.randn(head_dim, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_q_norm",
    )
    w_k_norm = ypk.attach_input(
        torch_tensor=torch.randn(head_dim, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_k_norm",
    )
    k_cache = ypk.attach_input(
        torch_tensor=torch.randn(
            scaffold.max_num_pages,
            scaffold.page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        name=f"layer_{layer_idx}_k_cache",
    )
    v_cache = ypk.attach_input(
        torch_tensor=torch.randn(
            scaffold.max_num_pages,
            scaffold.page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        name=f"layer_{layer_idx}_v_cache",
    )
    ypk.paged_attention_layer(
        input=attn_in,
        k_cache=k_cache,
        v_cache=v_cache,
        q_norm=w_q_norm,
        k_norm=w_k_norm,
        cos_pos_embed=cos_pos_embed,
        sin_pos_embed=sin_pos_embed,
        output=attn_out,
        grid_dim=(scaffold.max_num_batched_requests, num_kv_heads, 1),
        block_dim=(128, 1, 1),
    )

    w_o = ypk.attach_input(
        torch_tensor=torch.randn(num_q_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_o_proj",
    )
    ypk.linear_with_residual_layer(
        input=attn_out,
        weight=w_o,
        residual=x,
        output=attn_proj_out,
        grid_dim=(hidden_size // 64, 1, 1),
        block_dim=(128, 1, 1),
    )
    x = attn_proj_out

    w_post_norm = ypk.attach_input(
        torch_tensor=torch.randn(hidden_size, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_post_attn_layernorm",
    )
    w_gate = ypk.attach_input(
        torch_tensor=torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_gate_proj",
    )
    w_up = ypk.attach_input(
        torch_tensor=torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_up_proj",
    )
    rmsnorm_tasks = grid_for_rmsnorm_linear_layer(
        w_gate.dim(0) + w_up.dim(0), scaffold.use_cutlass_kernel
    )
    w_gatedup = ypk.shuffle_tensors(
        inputs=[w_gate, w_up],
        shuffled_dim=0,
        num_groups=rmsnorm_tasks // 2,
        name=f"layer_{layer_idx}_gatedup_proj",
    )
    ypk.rmsnorm_layer(
        input=x,
        weight=w_post_norm,
        output=rmsnorm_out,
        grid_dim=(tokens, 1, 1),
        block_dim=(128, 1, 1),
    )
    ypk.linear_layer(
        input=rmsnorm_out,
        weight=w_gatedup,
        output=mlp_mid,
        grid_dim=(rmsnorm_tasks, 1, 1),
        block_dim=(128, 1, 1),
    )
    ypk.silu_mul_layer(
        input=mlp_mid,
        output=silu_mul_out,
        grid_dim=(rmsnorm_tasks // 2, 1, 1),
        block_dim=(128, 1, 1),
    )
    w_down = ypk.attach_input(
        torch_tensor=torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device),
        name=f"layer_{layer_idx}_down_proj",
    )
    ypk.linear_with_residual_layer(
        input=silu_mul_out,
        weight=w_down,
        residual=x,
        output=mlp_out,
        grid_dim=(hidden_size // 64, 1, 1),
        block_dim=(128, 1, 1),
    )
    return mlp_out


def _maca_pk_compile_result(
    ypk,
    *,
    output_dir: Optional[str],
    num_workers: int,
    num_schedulers: int,
    tasks: list[str],
) -> Dict[str, Any]:
    import tempfile

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
        "tasks": tasks,
        "target_cc": ypk.target_cc,
    }


def inspect_maca_pk_compile_plan(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    variant: str = "embed_only",
) -> Dict[str, Any]:
    """Cloud-safe compile plan: mxcc contract + PK task-graph variant prerequisites."""
    scaffold = scaffold or Qwen3PKScaffold()
    dims = default_qwen_dims()
    contract = inspect_maca_pk_compile_contract(scaffold)
    repo_root = _maca_pk_repo_root()
    variants = {
        "embed_only": {
            "tasks": ["embedding"],
            "cuda_slice": "embed_layer",
        },
        "one_layer": {
            "tasks": [
                "embedding",
                "rms_norm",
                "linear",
                "paged_attention",
                "linear_with_residual",
                "silu_mul",
            ],
            "cuda_slice": "demo/qwen3/demo.py layer[0] (embed+attn+mlp)",
        },
    }
    if variant not in variants:
        raise ValueError(f"unknown compile plan variant: {variant}")
    return {
        **contract,
        "cuda_reference": "demo/qwen3/demo.py --use-yirage",
        "variant": variant,
        "minimal_task_graph": variants[variant]["cuda_slice"],
        "tasks": variants[variant]["tasks"],
        "hidden_size": dims.hidden_size,
        "intermediate_size": dims.intermediate_size,
        "fused_qkv_outdim": dims.fused_qkv_outdim,
        "vocab_smoke_size": 128,
        "requires": ["yirage.core", "YIRAGE_HOME", "mxcc", "MetaX GPU"],
        "yirage_home_default": str(repo_root),
        "yirage_home_set": "YIRAGE_HOME" in os.environ,
        "compile_plan_ready": contract["compile_ready"],
        "available_variants": list(variants.keys()),
    }


def maca_pk_minimal_compile_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build minimal embed-only PK task graph and ``ypk.compile()`` via mxcc (MetaX VM)."""
    scaffold = scaffold or Qwen3PKScaffold()
    ypk, meta, dims, _device, num_workers, num_schedulers, yr = _init_maca_pk_yirage(scaffold)
    _build_maca_pk_embed_graph(
        ypk, meta, scaffold, hidden_size=dims.hidden_size, vocab_smoke=128, yr=yr
    )
    return _maca_pk_compile_result(
        ypk,
        output_dir=output_dir,
        num_workers=num_workers,
        num_schedulers=num_schedulers,
        tasks=["embedding"],
    )


def maca_pk_one_layer_compile_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one-layer Qwen3 PK task graph (embed+attn+mlp) and compile via mxcc (MetaX VM)."""
    scaffold = scaffold or Qwen3PKScaffold()
    ypk, meta, dims, device, num_workers, num_schedulers, yr = _init_maca_pk_yirage(scaffold)
    _build_maca_pk_one_layer_graph(ypk, meta, scaffold, dims, device=device, yr=yr)
    return _maca_pk_compile_result(
        ypk,
        output_dir=output_dir,
        num_workers=num_workers,
        num_schedulers=num_schedulers,
        tasks=[
            "embedding",
            "rms_norm",
            "linear",
            "paged_attention",
            "linear_with_residual",
            "silu_mul",
        ],
    )


def inspect_maca_pk_compile_plan_embed_only(
    scaffold: Optional[Qwen3PKScaffold] = None,
) -> Dict[str, Any]:
    return inspect_maca_pk_compile_plan(scaffold, variant="embed_only")


def inspect_maca_pk_one_layer_compile_plan(
    scaffold: Optional[Qwen3PKScaffold] = None,
) -> Dict[str, Any]:
    return inspect_maca_pk_compile_plan(scaffold, variant="one_layer")


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
            "for Cloud contract; MetaX --compile-only (embed) or --compile-one-layer "
            "(decoder block) run mxcc task-graph compile."
        ),
        "yirage_backend": os.environ.get("YIRAGE_BACKEND", "maca"),
    }


__all__ = [
    "Qwen3PKScaffold",
    "build_qwen3_pk_meta_tensors",
    "grid_for_rmsnorm_linear_layer",
    "inspect_maca_pk_compile_contract",
    "inspect_maca_pk_compile_plan",
    "inspect_maca_pk_compile_plan_embed_only",
    "inspect_maca_pk_one_layer_compile_plan",
    "inspect_qwen3_pk_scaffold",
    "maca_pk_minimal_compile_smoke",
    "maca_pk_one_layer_compile_smoke",
    "maca_pk_runtime_smoke",
    "resolve_maca_pk_workers_schedulers",
]
