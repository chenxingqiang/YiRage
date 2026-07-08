"""Shared helpers for MACA Qwen3 PersistentKernel scaffold (CUDA ``demo/qwen3/demo.py`` aligned)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

from demo.maca.qwen_hf_utils import DEFAULT_QWEN_MODEL, default_qwen_dims

if TYPE_CHECKING:
    from demo.maca.qwen3_pk_hf_utils import MacaPkHfWeightBundle


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
    pk_compile_layers: int = 2


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
    hf_bundle: Optional["MacaPkHfWeightBundle"] = None,
):
    """Attach embed-only task (returns activation tensor ``x``)."""
    import torch

    input_t = ypk.attach_input(torch_tensor=meta["input_tokens"], name="input_token")
    if hf_bundle is not None:
        embed_weight = hf_bundle.embed_weight
    else:
        embed_weight = torch.randn(
            vocab_smoke, hidden_size, dtype=torch.bfloat16, device=meta["tokens"].device
        )
    w_embed = ypk.attach_input(torch_tensor=embed_weight, name="embed_tokens")
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


def _maca_pk_work_buffers(
    ypk,
    scaffold: Qwen3PKScaffold,
    dims: "QwenModelDims",
    yr,
    *,
    vocab_smoke: int = 128,
    with_argmax: bool = False,
) -> Dict[str, Any]:
    """Shared intermediate tensors (reused across decoder layers, CUDA demo aligned)."""
    tokens = scaffold.max_num_batched_tokens
    hidden_size = dims.hidden_size
    intermediate_size = dims.intermediate_size
    fused_qkv_out = dims.fused_qkv_outdim
    num_q_heads = dims.num_heads
    head_dim = dims.head_dim

    buffers: Dict[str, Any] = {
        "rmsnorm_out": ypk.new_tensor(
            dims=(tokens, hidden_size), dtype=yr.bfloat16, name="rmsnorm_out", io_category="cuda_tensor"
        ),
        "attn_in": ypk.new_tensor(
            dims=(tokens, fused_qkv_out), dtype=yr.bfloat16, name="attn_in", io_category="cuda_tensor"
        ),
        "attn_out": ypk.new_tensor(
            dims=(tokens, num_q_heads * head_dim), dtype=yr.bfloat16, name="attn_out", io_category="cuda_tensor"
        ),
        "attn_proj_out": ypk.new_tensor(
            dims=(tokens, hidden_size), dtype=yr.bfloat16, name="attn_proj_out", io_category="cuda_tensor"
        ),
        "mlp_mid": ypk.new_tensor(
            dims=(tokens, 2 * intermediate_size), dtype=yr.bfloat16, name="mlp_mid", io_category="cuda_tensor"
        ),
        "silu_mul_out": ypk.new_tensor(
            dims=(tokens, intermediate_size), dtype=yr.bfloat16, name="silu_mul_out", io_category="cuda_tensor"
        ),
        "mlp_out": ypk.new_tensor(
            dims=(tokens, hidden_size), dtype=yr.bfloat16, name="mlp_out", io_category="cuda_tensor"
        ),
    }
    if with_argmax:
        buffers["argmax_in"] = ypk.new_tensor(
            dims=(tokens, vocab_smoke), dtype=yr.bfloat16, name="argmax_in", io_category="cuda_tensor"
        )
        buffers["argmax_part_value"] = ypk.new_tensor(
            dims=(tokens, ypk.num_workers),
            dtype=yr.bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        buffers["argmax_part_index"] = ypk.new_tensor(
            dims=(tokens, ypk.num_workers),
            dtype=yr.int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )
    return buffers


def _append_maca_pk_decoder_layer(
    ypk,
    x,
    buffers: Dict[str, Any],
    scaffold: Qwen3PKScaffold,
    dims: "QwenModelDims",
    *,
    device: "torch.device",
    yr,
    layer_idx: int,
    cos_pos_embed,
    sin_pos_embed,
    hf_bundle: Optional["MacaPkHfWeightBundle"] = None,
):
    """Attach one decoder block; returns activation ``x`` for the next layer."""
    import torch

    hidden_size = dims.hidden_size
    intermediate_size = dims.intermediate_size
    head_dim = dims.head_dim
    num_q_heads = dims.num_heads
    num_kv_heads = dims.num_kv_heads
    tokens = scaffold.max_num_batched_tokens

    rmsnorm_out = buffers["rmsnorm_out"]
    attn_in = buffers["attn_in"]
    attn_out = buffers["attn_out"]
    attn_proj_out = buffers["attn_proj_out"]
    mlp_mid = buffers["mlp_mid"]
    silu_mul_out = buffers["silu_mul_out"]
    mlp_out = buffers["mlp_out"]

    if hf_bundle is not None:
        layer = hf_bundle.layer(layer_idx)
        w_norm = ypk.attach_input(
            torch_tensor=layer.input_layernorm.weight,
            name=f"layer_{layer_idx}_input_layernorm",
        )
        w_q = ypk.attach_input(
            torch_tensor=layer.self_attn.q_proj.weight,
            name=f"layer_{layer_idx}_q_proj",
        )
        w_k = ypk.attach_input(
            torch_tensor=layer.self_attn.k_proj.weight,
            name=f"layer_{layer_idx}_k_proj",
        )
        w_v = ypk.attach_input(
            torch_tensor=layer.self_attn.v_proj.weight,
            name=f"layer_{layer_idx}_v_proj",
        )
        w_q_norm = ypk.attach_input(
            torch_tensor=layer.self_attn.q_norm.weight,
            name=f"layer_{layer_idx}_q_norm",
        )
        w_k_norm = ypk.attach_input(
            torch_tensor=layer.self_attn.k_norm.weight,
            name=f"layer_{layer_idx}_k_norm",
        )
        k_cache = ypk.attach_input(
            torch_tensor=hf_bundle.k_cache(layer_idx),
            name=f"layer_{layer_idx}_k_cache",
        )
        v_cache = ypk.attach_input(
            torch_tensor=hf_bundle.v_cache(layer_idx),
            name=f"layer_{layer_idx}_v_cache",
        )
        w_o = ypk.attach_input(
            torch_tensor=layer.self_attn.o_proj.weight,
            name=f"layer_{layer_idx}_o_proj",
        )
        w_post_norm = ypk.attach_input(
            torch_tensor=layer.post_attention_layernorm.weight,
            name=f"layer_{layer_idx}_post_attn_layernorm",
        )
        w_gate = ypk.attach_input(
            torch_tensor=layer.mlp.gate_proj.weight,
            name=f"layer_{layer_idx}_gate_proj",
        )
        w_up = ypk.attach_input(
            torch_tensor=layer.mlp.up_proj.weight,
            name=f"layer_{layer_idx}_up_proj",
        )
        w_down = ypk.attach_input(
            torch_tensor=layer.mlp.down_proj.weight,
            name=f"layer_{layer_idx}_down_proj",
        )
    else:
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
        w_o = ypk.attach_input(
            torch_tensor=torch.randn(num_q_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device),
            name=f"layer_{layer_idx}_o_proj",
        )
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
        w_down = ypk.attach_input(
            torch_tensor=torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device),
            name=f"layer_{layer_idx}_down_proj",
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
    ypk.linear_with_residual_layer(
        input=attn_out,
        weight=w_o,
        residual=x,
        output=attn_proj_out,
        grid_dim=(hidden_size // 64, 1, 1),
        block_dim=(128, 1, 1),
    )
    x = attn_proj_out

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
    ypk.linear_with_residual_layer(
        input=silu_mul_out,
        weight=w_down,
        residual=x,
        output=mlp_out,
        grid_dim=(hidden_size // 64, 1, 1),
        block_dim=(128, 1, 1),
    )
    return mlp_out


def _append_maca_pk_lm_head_argmax(
    ypk,
    x,
    buffers: Dict[str, Any],
    meta: Dict[str, "torch.Tensor"],
    scaffold: Qwen3PKScaffold,
    dims: "QwenModelDims",
    *,
    device: "torch.device",
    yr,
    vocab_smoke: int = 128,
    hf_bundle: Optional["MacaPkHfWeightBundle"] = None,
):
    """Attach final rmsnorm + lm_head + argmax (CUDA demo epilogue)."""
    import torch

    tokens = scaffold.max_num_batched_tokens
    hidden_size = dims.hidden_size
    rmsnorm_out = buffers["rmsnorm_out"]
    argmax_in = buffers["argmax_in"]
    argmax_part_value = buffers["argmax_part_value"]
    argmax_part_index = buffers["argmax_part_index"]

    if hf_bundle is not None:
        w_norm = ypk.attach_input(
            torch_tensor=hf_bundle.norm_weight,
            name="model_norm_weight",
        )
        w_proj = ypk.attach_input(
            torch_tensor=hf_bundle.lm_head_weight,
            name="lm_head",
        )
        vocab_smoke = hf_bundle.vocab_smoke
    else:
        w_norm = ypk.attach_input(
            torch_tensor=torch.randn(hidden_size, dtype=torch.bfloat16, device=device),
            name="model_norm_weight",
        )
        w_proj = ypk.attach_input(
            torch_tensor=torch.randn(vocab_smoke, hidden_size, dtype=torch.bfloat16, device=device),
            name="lm_head",
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
        weight=w_proj,
        output=argmax_in,
        grid_dim=(ypk.num_workers, 1, 1),
        block_dim=(128, 1, 1),
    )
    ypk.argmax_partial_layer(
        input=argmax_in,
        output=(argmax_part_value, argmax_part_index),
        grid_dim=(ypk.num_workers, 1, 1),
        block_dim=(128, 1, 1),
    )
    argmax_out = ypk.attach_input(torch_tensor=meta["output_tokens"], name="output_token")
    ypk.argmax_reduce_layer(
        input=(argmax_part_value, argmax_part_index),
        output=argmax_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    return argmax_out


def _build_maca_pk_stack_graph(
    ypk,
    meta,
    scaffold: Qwen3PKScaffold,
    dims: "QwenModelDims",
    *,
    device: "torch.device",
    yr,
    num_layers: int,
    lm_head: bool = True,
    vocab_smoke: int = 128,
    hf_bundle: Optional["MacaPkHfWeightBundle"] = None,
):
    """Build embed + N decoder layers + optional lm_head/argmax (CUDA demo aligned)."""
    import torch

    head_dim = dims.head_dim
    hidden_size = dims.hidden_size

    if hf_bundle is not None:
        cos_pos_embed = ypk.attach_input(
            torch_tensor=hf_bundle.cos_pos_embed,
            name="cos_position_embedding",
        )
        sin_pos_embed = ypk.attach_input(
            torch_tensor=hf_bundle.sin_pos_embed,
            name="sin_position_embedding",
        )
        vocab_smoke = hf_bundle.vocab_smoke
    else:
        cos_pos_embed = ypk.attach_input(
            torch_tensor=torch.randn(scaffold.page_size, head_dim, dtype=torch.bfloat16, device=device),
            name="cos_position_embedding",
        )
        sin_pos_embed = ypk.attach_input(
            torch_tensor=torch.randn(scaffold.page_size, head_dim, dtype=torch.bfloat16, device=device),
            name="sin_position_embedding",
        )

    buffers = _maca_pk_work_buffers(
        ypk, scaffold, dims, yr, vocab_smoke=vocab_smoke, with_argmax=lm_head
    )
    x = _build_maca_pk_embed_graph(
        ypk,
        meta,
        scaffold,
        hidden_size=hidden_size,
        vocab_smoke=vocab_smoke,
        yr=yr,
        hf_bundle=hf_bundle,
    )
    for layer_idx in range(num_layers):
        x = _append_maca_pk_decoder_layer(
            ypk,
            x,
            buffers,
            scaffold,
            dims,
            device=device,
            yr=yr,
            layer_idx=layer_idx,
            cos_pos_embed=cos_pos_embed,
            sin_pos_embed=sin_pos_embed,
            hf_bundle=hf_bundle,
        )
    if lm_head:
        x = _append_maca_pk_lm_head_argmax(
            ypk,
            x,
            buffers,
            meta,
            scaffold,
            dims,
            device=device,
            yr=yr,
            vocab_smoke=vocab_smoke,
            hf_bundle=hf_bundle,
        )
    return x


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
    return _build_maca_pk_stack_graph(
        ypk,
        meta,
        scaffold,
        dims,
        device=device,
        yr=yr,
        num_layers=1,
        lm_head=False,
    )


def _maca_pk_stack_tasks(*, lm_head: bool) -> list[str]:
    tasks = [
        "embedding",
        "rms_norm",
        "linear",
        "paged_attention",
        "linear_with_residual",
        "silu_mul",
    ]
    if lm_head:
        tasks.extend(["lm_head", "argmax_partial", "argmax_reduce"])
    return tasks


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
        "stack": {
            "tasks": _maca_pk_stack_tasks(lm_head=True),
            "cuda_slice": "demo/qwen3/demo.py layers[0:N] + lm_head/argmax",
        },
    }
    if variant not in variants:
        raise ValueError(f"unknown compile plan variant: {variant}")
    plan_layers = scaffold.pk_compile_layers if variant == "stack" else 1
    return {
        **contract,
        "cuda_reference": "demo/qwen3/demo.py --use-yirage",
        "variant": variant,
        "minimal_task_graph": variants[variant]["cuda_slice"],
        "tasks": variants[variant]["tasks"],
        "pk_compile_layers": plan_layers,
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
        tasks=_maca_pk_stack_tasks(lm_head=False),
    )


def maca_pk_stack_compile_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
    num_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """Build N-layer Qwen3 PK stack + lm_head/argmax and compile via mxcc (MetaX VM)."""
    scaffold = scaffold or Qwen3PKScaffold()
    layers = num_layers if num_layers is not None else scaffold.pk_compile_layers
    ypk, meta, dims, device, num_workers, num_schedulers, yr = _init_maca_pk_yirage(scaffold)
    _build_maca_pk_stack_graph(
        ypk,
        meta,
        scaffold,
        dims,
        device=device,
        yr=yr,
        num_layers=layers,
        lm_head=True,
    )
    result = _maca_pk_compile_result(
        ypk,
        output_dir=output_dir,
        num_workers=num_workers,
        num_schedulers=num_schedulers,
        tasks=_maca_pk_stack_tasks(lm_head=True),
    )
    result["pk_compile_layers"] = layers
    return result


def inspect_maca_pk_compile_plan_embed_only(
    scaffold: Optional[Qwen3PKScaffold] = None,
) -> Dict[str, Any]:
    return inspect_maca_pk_compile_plan(scaffold, variant="embed_only")


def inspect_maca_pk_one_layer_compile_plan(
    scaffold: Optional[Qwen3PKScaffold] = None,
) -> Dict[str, Any]:
    return inspect_maca_pk_compile_plan(scaffold, variant="one_layer")


def inspect_maca_pk_stack_compile_plan(
    scaffold: Optional[Qwen3PKScaffold] = None,
) -> Dict[str, Any]:
    return inspect_maca_pk_compile_plan(scaffold, variant="stack")


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


def prepare_maca_pk_runtime_meta(
    meta: Dict[str, "torch.Tensor"],
    scaffold: Qwen3PKScaffold,
    *,
    prompt_len: int = 4,
    num_tokens: int = 1,
    active_requests: int = 1,
    base_token_id: int = 100,
) -> Dict[str, Any]:
    """Fill offline PK meta tensors for a minimal single-request decode launch."""
    import torch

    if prompt_len < 1:
        raise ValueError("prompt_len must be >= 1")
    if num_tokens < 1:
        raise ValueError("num_tokens must be >= 1")
    if active_requests < 1:
        raise ValueError("active_requests must be >= 1")
    if active_requests > scaffold.max_num_batched_requests:
        raise ValueError("active_requests exceeds max_num_batched_requests")
    if num_tokens > scaffold.max_num_batched_tokens:
        raise ValueError("num_tokens exceeds max_num_batched_tokens")

    device = meta["tokens"].device
    prompt_ids = torch.arange(base_token_id, base_token_id + prompt_len, device=device, dtype=torch.long)

    meta["tokens"].zero_()
    meta["input_tokens"].zero_()
    meta["output_tokens"].zero_()
    meta["step"].zero_()
    meta["num_new_tokens"].zero_()
    meta["prompt_lengths"].zero_()

    for req in range(active_requests):
        meta["tokens"][req, :prompt_len] = prompt_ids
        meta["prompt_lengths"][req] = prompt_len
        meta["step"][req] = prompt_len - 1
        meta["num_new_tokens"][req] = num_tokens

    meta["input_tokens"][:num_tokens, 0] = prompt_ids[-1]
    meta["qo_indptr_buffer"].zero_()
    meta["paged_kv_indptr_buffer"].zero_()
    meta["paged_kv_indices_buffer"].zero_()
    meta["paged_kv_last_page_len_buffer"].zero_()

    meta["qo_indptr_buffer"][0] = 0
    meta["qo_indptr_buffer"][1] = num_tokens
    meta["paged_kv_indptr_buffer"][0] = 0
    meta["paged_kv_indptr_buffer"][1] = 1
    meta["paged_kv_indices_buffer"][0] = 0
    meta["paged_kv_last_page_len_buffer"][0] = prompt_len

    return {
        "prompt_len": prompt_len,
        "num_tokens": num_tokens,
        "active_requests": active_requests,
        "qo_indptr": [0, num_tokens],
        "paged_kv_indptr": [0, 1],
        "paged_kv_indices_head": [0],
        "paged_kv_last_page_len": [prompt_len],
    }


def inspect_maca_pk_runtime_plan(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    variant: str = "stack",
    num_layers: int = 1,
) -> Dict[str, Any]:
    """Cloud-safe runtime plan: compile + meta fill + ``ypk()`` launch prerequisites."""
    scaffold = scaffold or Qwen3PKScaffold()
    compile_plan = inspect_maca_pk_compile_plan(scaffold, variant=variant)
    layers = num_layers if variant == "stack" else 1
    return {
        **compile_plan,
        "plan_kind": "runtime",
        "variant": variant,
        "pk_runtime_layers": layers,
        "cuda_reference": "demo/qwen3/demo.py --use-yirage (ypk.compile + ypk())",
        "runtime_steps": [
            "build_maca_pk_stack_graph (synthetic weights)",
            "ypk.compile() via mxcc",
            "prepare_maca_pk_runtime_meta (qo_indptr / paged_kv_*)",
            "ypk() launch",
        ],
        "meta_tensors": [
            "step",
            "tokens",
            "input_tokens",
            "output_tokens",
            "num_new_tokens",
            "prompt_lengths",
            "qo_indptr_buffer",
            "paged_kv_indptr_buffer",
            "paged_kv_indices_buffer",
            "paged_kv_last_page_len_buffer",
        ],
        "runtime_plan_ready": compile_plan["compile_plan_ready"],
        "requires_metax_gpu": True,
        "weight_source": "synthetic",
    }


def inspect_maca_pk_hf_runtime_plan(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    max_layers: int = 1,
) -> Dict[str, Any]:
    """Cloud-safe HF-weight PK runtime contract (weight injection implemented in R21)."""
    from demo.maca.qwen3_pk_hf_utils import inspect_maca_pk_hf_weight_plan

    scaffold = scaffold or Qwen3PKScaffold()
    runtime_plan = inspect_maca_pk_runtime_plan(scaffold, variant="stack", num_layers=max_layers)
    weight_plan = inspect_maca_pk_hf_weight_plan(scaffold, max_layers=max_layers)
    return {
        "cuda_reference": "demo/qwen3/demo.py --use-yirage (HF model weights)",
        "maca_pretrained_demo": "demo/maca/qwen_from_pretrained_demo.py",
        "maca_pk_runtime_entry": "maca_pk_hf_stack_runtime_smoke",
        "weight_injection_status": "implemented",
        "weight_injection_note": (
            "maca_pk_hf_stack_runtime_smoke loads HF Qwen3 via load_maca_pk_hf_weight_bundle "
            "and attach_input into PK stack graph (vocab_smoke=128 default; "
            "use_padded_lm_head=True for CUDA 153600 argmax)."
        ),
        "hf_runtime_ready": weight_plan["weight_plan_ready"] and runtime_plan["runtime_plan_ready"],
        "multi_layer_ready": weight_plan["weight_plan_ready"],
        "padded_lm_head_ready": True,
        "generation_loop_ready": True,
        "synthetic_runtime_ready": runtime_plan["runtime_plan_ready"],
        "weight_plan": weight_plan,
        "model": scaffold.model,
        "pk_compile_layers": scaffold.pk_compile_layers,
        "max_layers_default": max_layers,
        "requires": runtime_plan["requires"] + ["transformers", "HF hub access", "MetaX GPU for launch"],
        "next_steps": [
            "MetaX VM: --hf-runtime-stack --pk-compile-layers 2 (multi-layer)",
            "MetaX VM: --hf-runtime-stack --hf-padded-lm-head (153600 argmax)",
            "MetaX VM: --hf-generation-smoke --decode-steps 4 (multi-step ypk loop)",
            "tokenizer full-path generation e2e vs CUDA demo/qwen3/demo.py --use-yirage",
        ],
    }


def inspect_maca_pk_hf_generation_plan(
    scaffold: Optional[Qwen3PKScaffold] = None,
) -> Dict[str, Any]:
    """Cloud-safe HF PK generation loop contract (CUDA ``demo/qwen3/demo.py`` aligned)."""
    from demo.maca.qwen3_pk_generation_utils import (
        inspect_maca_pk_decode_step_contract,
        inspect_maca_pk_hf_tokenizer_generation_plan,
        inspect_maca_pk_multi_request_batch_plan,
    )

    scaffold = scaffold or Qwen3PKScaffold()
    runtime_plan = inspect_maca_pk_hf_runtime_plan(scaffold, max_layers=scaffold.pk_compile_layers)
    decode_contract = inspect_maca_pk_decode_step_contract(scaffold)
    tokenizer_plan = inspect_maca_pk_hf_tokenizer_generation_plan(scaffold)
    batch_plan = inspect_maca_pk_multi_request_batch_plan(scaffold)
    return {
        "cuda_reference": "demo/qwen3/demo.py --use-yirage (ypk() + tokenizer.decode)",
        "maca_runtime_entry": "maca_pk_hf_stack_runtime_smoke",
        "maca_generation_entry": "maca_pk_hf_generation_smoke",
        "maca_tokenizer_generation_entry": "maca_pk_hf_tokenizer_generation_smoke",
        "generation_ready": False,
        "multi_step_decode_ready": decode_contract["decode_step_contract_ready"],
        "tokenizer_generation_plan_ready": tokenizer_plan["tokenizer_generation_plan_ready"],
        "multi_request_batch_plan_ready": batch_plan["multi_request_batch_plan_ready"],
        "multi_request_decode_plan_ready": batch_plan["multi_request_batch_plan_ready"],
        "implemented_steps": [
            "HF load_maca_pk_hf_weight_bundle",
            "N-layer PK stack graph + mxcc compile",
            "prepare_maca_pk_runtime_meta (qo_indptr / paged_kv)",
            "prepare_maca_pk_prompt_meta (tokenizer prompt ids)",
            "prepare_maca_pk_batched_prompt_meta (multi-request)",
            "run_maca_pk_batched_decode_loop (multi-request ypk)",
            "encode_maca_pk_chat_prompt / decode_maca_pk_generated_tokens",
            "compute_maca_pk_generation_latency (per-token ms)",
            "maca_pk_hf_full_layer_tokenizer_generation_smoke",
        ],
        "backlog_steps": [
            "MetaX VM full-layer + batched generation e2e validation",
        ],
        "decode_step_contract": decode_contract,
        "tokenizer_generation_plan": tokenizer_plan,
        "multi_request_batch_plan": batch_plan,
        "hf_runtime_plan": runtime_plan,
        "model": scaffold.model,
        "pk_compile_layers": scaffold.pk_compile_layers,
        "generation_plan_ready": runtime_plan["hf_runtime_ready"],
        "requires_metax_gpu": True,
    }


def _maca_pk_hf_init_compiled_stack(
    scaffold: Qwen3PKScaffold,
    *,
    output_dir: Optional[str] = None,
    num_layers: int = 1,
    vocab_smoke: int = 128,
    use_padded_lm_head: bool = False,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    """Compile HF-weight N-layer PK stack; return ypk/meta/compile artifacts (caller finalizes)."""
    from demo.maca.qwen3_pk_hf_utils import load_maca_pk_hf_weight_bundle, resolve_maca_pk_lm_vocab_size

    lm_vocab = resolve_maca_pk_lm_vocab_size(
        vocab_smoke=vocab_smoke, use_padded_lm_head=use_padded_lm_head
    )
    ypk, meta, dims, device, num_workers, num_schedulers, yr = _init_maca_pk_yirage(scaffold)
    hf_bundle = load_maca_pk_hf_weight_bundle(
        scaffold.model,
        scaffold,
        device,
        max_layers=num_layers,
        vocab_smoke=vocab_smoke,
        local_files_only=local_files_only,
        use_padded_lm_head=use_padded_lm_head,
    )
    _build_maca_pk_stack_graph(
        ypk,
        meta,
        scaffold,
        dims,
        device=device,
        yr=yr,
        num_layers=num_layers,
        lm_head=True,
        vocab_smoke=lm_vocab,
        hf_bundle=hf_bundle,
    )
    compile_result = _maca_pk_compile_result(
        ypk,
        output_dir=output_dir,
        num_workers=num_workers,
        num_schedulers=num_schedulers,
        tasks=_maca_pk_stack_tasks(lm_head=True),
    )
    return {
        "ypk": ypk,
        "meta": meta,
        "compile_result": compile_result,
        "hf_bundle": hf_bundle,
        "pk_compile_layers": num_layers,
        "lm_vocab": lm_vocab,
    }


def maca_pk_hf_generation_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
    num_layers: Optional[int] = None,
    prompt_len: int = 4,
    decode_steps: int = 1,
    prompt_token_ids: Optional[Sequence[int]] = None,
    use_tokenizer: bool = False,
    chat_prompt: str = "Hello",
    use_padded_lm_head: bool = False,
    local_files_only: bool = False,
    eos_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """HF PK generation smoke: optional tokenizer prompt + multi-step ``ypk()`` decode loop."""
    from demo.maca.qwen3_pk_generation_utils import (
        DEFAULT_MACA_PK_CHAT_PROMPT,
        compute_maca_pk_generation_latency,
        decode_maca_pk_generated_tokens,
        encode_maca_pk_chat_prompt,
        prepare_maca_pk_prompt_meta,
        run_maca_pk_decode_loop,
    )

    scaffold = scaffold or Qwen3PKScaffold()
    layers = num_layers if num_layers is not None else 1
    if decode_steps < 1:
        raise ValueError("decode_steps must be >= 1")

    stack = _maca_pk_hf_init_compiled_stack(
        scaffold,
        output_dir=output_dir,
        num_layers=layers,
        vocab_smoke=128,
        use_padded_lm_head=use_padded_lm_head,
        local_files_only=local_files_only,
    )
    ypk = stack["ypk"]
    meta = stack["meta"]
    hf_bundle = stack["hf_bundle"]

    tokenizer = None
    resolved_prompt = chat_prompt or DEFAULT_MACA_PK_CHAT_PROMPT
    if use_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            scaffold.model, local_files_only=local_files_only
        )
        prompt_token_ids = encode_maca_pk_chat_prompt(tokenizer, resolved_prompt)

    if prompt_token_ids is not None:
        meta_summary = prepare_maca_pk_prompt_meta(
            meta, scaffold, prompt_token_ids, num_tokens=1
        )
    else:
        meta_summary = prepare_maca_pk_runtime_meta(
            meta, scaffold, prompt_len=prompt_len, num_tokens=1
        )

    decode_result = run_maca_pk_decode_loop(
        ypk,
        meta,
        scaffold,
        max_decode_steps=decode_steps,
        eos_token_id=eos_token_id,
    )
    ypk.finalize()

    output_token = decode_result["generated_tokens"][-1] if decode_result["generated_tokens"] else None
    result: Dict[str, Any] = {
        **stack["compile_result"],
        "launched": True,
        "launch_ms": decode_result["launch_ms"],
        "pk_compile_layers": layers,
        "meta_summary": meta_summary,
        "weight_source": "hf",
        "model_name": hf_bundle.model_name,
        "vocab_smoke": hf_bundle.vocab_smoke,
        "use_padded_lm_head": hf_bundle.use_padded_lm_head,
        "generation_steps": decode_result["decode_steps"],
        "generated_tokens": decode_result["generated_tokens"],
        "output_token": output_token,
        "final_step": decode_result["final_step"],
        "stopped_on_eos": decode_result["stopped_on_eos"],
        "use_tokenizer": use_tokenizer,
        "chat_prompt": resolved_prompt if use_tokenizer else None,
    }
    if tokenizer is not None:
        result["decoded_response"] = decode_maca_pk_generated_tokens(tokenizer, meta)
        prompt_len = int(meta_summary["prompt_len"])
        result["latency"] = compute_maca_pk_generation_latency(
            launch_ms=decode_result["launch_ms"],
            prompt_len=prompt_len,
            final_step=decode_result["final_step"],
        )
    return result


def maca_pk_hf_tokenizer_generation_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
    num_layers: Optional[int] = None,
    decode_steps: int = 4,
    chat_prompt: str = "Hello",
    use_padded_lm_head: bool = False,
    local_files_only: bool = False,
    ignore_eos: bool = False,
) -> Dict[str, Any]:
    """Tokenizer full-path HF PK generation smoke (CUDA ``demo/qwen3/demo.py --use-yirage``)."""
    from transformers import AutoTokenizer

    scaffold = scaffold or Qwen3PKScaffold()
    tokenizer = AutoTokenizer.from_pretrained(
        scaffold.model, local_files_only=local_files_only
    )
    eos_token_id = None if ignore_eos else tokenizer.eos_token_id

    result = maca_pk_hf_generation_smoke(
        scaffold,
        output_dir=output_dir,
        num_layers=num_layers,
        decode_steps=decode_steps,
        use_tokenizer=True,
        chat_prompt=chat_prompt,
        use_padded_lm_head=use_padded_lm_head,
        local_files_only=local_files_only,
        eos_token_id=eos_token_id,
    )
    return {
        **result,
        "tokenizer_generation_smoke": True,
        "eos_token_id": eos_token_id,
    }


def maca_pk_hf_batched_tokenizer_generation_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
    num_layers: Optional[int] = None,
    active_requests: int = 2,
    decode_steps: int = 4,
    chat_prompt: str = "Hello",
    use_padded_lm_head: bool = False,
    local_files_only: bool = False,
    ignore_eos: bool = False,
) -> Dict[str, Any]:
    """Multi-request tokenizer generation smoke (CUDA ``total_num_requests > 1``)."""
    from demo.maca.qwen3_pk_generation_utils import (
        compute_maca_pk_generation_latency,
        decode_maca_pk_batched_generated_tokens,
        encode_maca_pk_chat_prompt,
        prepare_maca_pk_batched_prompt_meta,
        run_maca_pk_batched_decode_loop,
    )
    from transformers import AutoTokenizer

    scaffold = scaffold or Qwen3PKScaffold()
    layers = num_layers if num_layers is not None else 1
    if active_requests < 2:
        raise ValueError("active_requests must be >= 2 for batched generation smoke")

    tokenizer = AutoTokenizer.from_pretrained(
        scaffold.model, local_files_only=local_files_only
    )
    eos_token_id = None if ignore_eos else tokenizer.eos_token_id
    prompt_token_ids = encode_maca_pk_chat_prompt(tokenizer, chat_prompt)

    stack = _maca_pk_hf_init_compiled_stack(
        scaffold,
        output_dir=output_dir,
        num_layers=layers,
        vocab_smoke=128,
        use_padded_lm_head=use_padded_lm_head,
        local_files_only=local_files_only,
    )
    ypk = stack["ypk"]
    meta = stack["meta"]
    hf_bundle = stack["hf_bundle"]

    meta_summary = prepare_maca_pk_batched_prompt_meta(
        meta, scaffold, prompt_token_ids, active_requests=active_requests
    )
    decode_result = run_maca_pk_batched_decode_loop(
        ypk,
        meta,
        scaffold,
        active_requests=active_requests,
        max_decode_steps=decode_steps,
        eos_token_id=eos_token_id,
    )
    ypk.finalize()

    decoded_responses = decode_maca_pk_batched_generated_tokens(
        tokenizer, meta, active_requests=active_requests
    )
    prompt_len = int(meta_summary["prompt_len"])
    return {
        **stack["compile_result"],
        "launched": True,
        "launch_ms": decode_result["launch_ms"],
        "pk_compile_layers": layers,
        "meta_summary": meta_summary,
        "weight_source": "hf",
        "model_name": hf_bundle.model_name,
        "vocab_smoke": hf_bundle.vocab_smoke,
        "use_padded_lm_head": hf_bundle.use_padded_lm_head,
        "batched_tokenizer_generation_smoke": True,
        "active_requests": active_requests,
        "generation_steps": decode_result["decode_steps"],
        "generated_tokens": decode_result["generated_tokens"],
        "generated_tokens_req0": decode_result["generated_tokens_req0"],
        "final_steps": decode_result["final_steps"],
        "steps_aligned": decode_result["steps_aligned"],
        "stopped_on_eos": decode_result["stopped_on_eos"],
        "decoded_responses": decoded_responses,
        "chat_prompt": chat_prompt,
        "eos_token_id": eos_token_id,
        "latency": compute_maca_pk_generation_latency(
            launch_ms=decode_result["launch_ms"],
            prompt_len=prompt_len,
            final_step=decode_result["final_step"],
        ),
    }


def maca_pk_hf_full_layer_tokenizer_generation_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
    decode_steps: int = 4,
    chat_prompt: str = "Hello",
    use_padded_lm_head: bool = False,
    local_files_only: bool = False,
    ignore_eos: bool = False,
) -> Dict[str, Any]:
    """Full-layer HF PK tokenizer generation e2e (``pk_compile_layers`` stack)."""
    scaffold = scaffold or Qwen3PKScaffold()
    result = maca_pk_hf_tokenizer_generation_smoke(
        scaffold,
        output_dir=output_dir,
        num_layers=scaffold.pk_compile_layers,
        decode_steps=decode_steps,
        chat_prompt=chat_prompt,
        use_padded_lm_head=use_padded_lm_head,
        local_files_only=local_files_only,
        ignore_eos=ignore_eos,
    )
    return {
        **result,
        "full_layer_tokenizer_generation_smoke": True,
        "pk_compile_layers": scaffold.pk_compile_layers,
    }


def maca_pk_hf_stack_runtime_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
    num_layers: Optional[int] = None,
    prompt_len: int = 4,
    num_tokens: int = 1,
    vocab_smoke: int = 128,
    use_padded_lm_head: bool = False,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    """Build HF-weight N-layer PK stack, compile via mxcc, fill meta, and ``ypk()`` launch."""
    import torch

    scaffold = scaffold or Qwen3PKScaffold()
    layers = num_layers if num_layers is not None else 1
    stack = _maca_pk_hf_init_compiled_stack(
        scaffold,
        output_dir=output_dir,
        num_layers=layers,
        vocab_smoke=vocab_smoke,
        use_padded_lm_head=use_padded_lm_head,
        local_files_only=local_files_only,
    )
    ypk = stack["ypk"]
    meta = stack["meta"]
    hf_bundle = stack["hf_bundle"]
    meta_summary = prepare_maca_pk_runtime_meta(
        meta, scaffold, prompt_len=prompt_len, num_tokens=num_tokens
    )

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    ypk()
    ender.record()
    torch.cuda.synchronize()
    launch_ms = starter.elapsed_time(ender)
    output_token = int(meta["output_tokens"][0, 0].item())
    ypk.finalize()

    return {
        **stack["compile_result"],
        "launched": True,
        "launch_ms": launch_ms,
        "pk_compile_layers": layers,
        "meta_summary": meta_summary,
        "weight_source": "hf",
        "model_name": hf_bundle.model_name,
        "vocab_smoke": hf_bundle.vocab_smoke,
        "use_padded_lm_head": hf_bundle.use_padded_lm_head,
        "output_token": output_token,
    }


def maca_pk_stack_runtime_smoke(
    scaffold: Optional[Qwen3PKScaffold] = None,
    *,
    output_dir: Optional[str] = None,
    num_layers: Optional[int] = None,
    prompt_len: int = 4,
    num_tokens: int = 1,
) -> Dict[str, Any]:
    """Build N-layer PK stack, compile via mxcc, fill meta tensors, and ``ypk()`` launch."""
    import torch

    scaffold = scaffold or Qwen3PKScaffold()
    layers = num_layers if num_layers is not None else 1
    ypk, meta, dims, device, num_workers, num_schedulers, yr = _init_maca_pk_yirage(scaffold)
    _build_maca_pk_stack_graph(
        ypk,
        meta,
        scaffold,
        dims,
        device=device,
        yr=yr,
        num_layers=layers,
        lm_head=True,
    )
    compile_result = _maca_pk_compile_result(
        ypk,
        output_dir=output_dir,
        num_workers=num_workers,
        num_schedulers=num_schedulers,
        tasks=_maca_pk_stack_tasks(lm_head=True),
    )
    meta_summary = prepare_maca_pk_runtime_meta(
        meta, scaffold, prompt_len=prompt_len, num_tokens=num_tokens
    )

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    ypk()
    ender.record()
    torch.cuda.synchronize()
    launch_ms = starter.elapsed_time(ender)
    ypk.finalize()

    return {
        **compile_result,
        "launched": True,
        "launch_ms": launch_ms,
        "pk_compile_layers": layers,
        "meta_summary": meta_summary,
        "weight_source": "synthetic",
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
        "pk_compile_layers": scaffold.pk_compile_layers,
        "hidden_size": dims.hidden_size,
        "intermediate_size": dims.intermediate_size,
        "fused_qkv_outdim": dims.fused_qkv_outdim,
        "compile_path": "mxcc",
        "compile_note": (
            "PersistentKernel.compile() selects mxcc when YIRAGE_BACKEND=maca "
            "(get_maca_pk_compile_command). Use --compile-plan/--compile-inspect "
            "for Cloud contract; MetaX --compile-only (embed), --compile-one-layer "
            "(decoder block), or --compile-stack (N layers + lm_head/argmax)."
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
    "inspect_maca_pk_hf_runtime_plan",
    "inspect_maca_pk_hf_generation_plan",
    "inspect_maca_pk_one_layer_compile_plan",
    "inspect_maca_pk_runtime_plan",
    "inspect_maca_pk_stack_compile_plan",
    "inspect_qwen3_pk_scaffold",
    "maca_pk_minimal_compile_smoke",
    "maca_pk_one_layer_compile_smoke",
    "maca_pk_stack_compile_smoke",
    "maca_pk_stack_runtime_smoke",
    "maca_pk_hf_generation_smoke",
    "maca_pk_hf_tokenizer_generation_smoke",
    "maca_pk_hf_batched_tokenizer_generation_smoke",
    "maca_pk_hf_full_layer_tokenizer_generation_smoke",
    "maca_pk_hf_stack_runtime_smoke",
    "maca_pk_runtime_smoke",
    "prepare_maca_pk_runtime_meta",
    "resolve_maca_pk_workers_schedulers",
]
