"""MACA Qwen3 PersistentKernel decode / generation helpers (CUDA ``demo/qwen3/demo.py`` aligned)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch

    from demo.maca.qwen3_pk_utils import Qwen3PKScaffold

DEFAULT_MACA_PK_CHAT_PROMPT = "Hello"


def encode_maca_pk_chat_prompt(tokenizer, prompt: str = DEFAULT_MACA_PK_CHAT_PROMPT) -> List[int]:
    """Encode a chat prompt like ``demo/qwen3/demo.py`` (apply_chat_template)."""
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    return model_inputs.input_ids[0].tolist()


def decode_maca_pk_generated_tokens(
    tokenizer,
    meta: Dict[str, "torch.Tensor"],
    *,
    req: int = 0,
    skip_special_tokens: bool = True,
) -> str:
    """Decode ``meta['tokens']`` up to ``meta['step'][req]`` (CUDA demo style)."""
    step = int(meta["step"][req].item())
    generated_ids = meta["tokens"][req, : step + 1]
    return tokenizer.decode(generated_ids, skip_special_tokens=skip_special_tokens)


def prepare_maca_pk_prompt_meta(
    meta: Dict[str, "torch.Tensor"],
    scaffold: "Qwen3PKScaffold",
    prompt_token_ids: Union[Sequence[int], "torch.Tensor"],
    *,
    active_requests: int = 1,
    num_tokens: int = 1,
) -> Dict[str, Any]:
    """Fill PK meta tensors from real tokenizer prompt ids (prefill contract)."""
    import torch

    if isinstance(prompt_token_ids, torch.Tensor):
        ids = prompt_token_ids.to(device=meta["tokens"].device, dtype=torch.long).flatten()
    else:
        ids = torch.tensor(list(prompt_token_ids), device=meta["tokens"].device, dtype=torch.long)
    prompt_len = int(ids.numel())
    if prompt_len < 1:
        raise ValueError("prompt_token_ids must be non-empty")
    if prompt_len > scaffold.max_seq_length:
        raise ValueError("prompt length exceeds max_seq_length")
    if active_requests < 1 or active_requests > scaffold.max_num_batched_requests:
        raise ValueError("active_requests out of range")
    if num_tokens < 1 or num_tokens > scaffold.max_num_batched_tokens:
        raise ValueError("num_tokens out of range")

    meta["tokens"].zero_()
    meta["input_tokens"].zero_()
    meta["output_tokens"].zero_()
    meta["step"].zero_()
    meta["num_new_tokens"].zero_()
    meta["prompt_lengths"].zero_()

    for req in range(active_requests):
        meta["tokens"][req, :prompt_len] = ids
        meta["prompt_lengths"][req] = prompt_len
        meta["step"][req] = prompt_len - 1
        meta["num_new_tokens"][req] = num_tokens

    meta["input_tokens"][:num_tokens, 0] = ids[-1]
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
        "prompt_token_ids_head": ids[: min(8, prompt_len)].tolist(),
        "num_tokens": num_tokens,
        "active_requests": active_requests,
        "qo_indptr": [0, num_tokens],
        "paged_kv_indptr": [0, 1],
        "paged_kv_indices_head": [0],
        "paged_kv_last_page_len": [prompt_len],
    }


def advance_maca_pk_decode_step(
    meta: Dict[str, "torch.Tensor"],
    scaffold: "Qwen3PKScaffold",
    *,
    req: int = 0,
) -> Dict[str, Any]:
    """Advance offline PK meta after ``ypk()`` for the next decode step."""
    output_token = int(meta["output_tokens"][0, 0].item())
    cur_step = int(meta["step"][req].item())
    new_step = cur_step + 1
    if new_step >= scaffold.max_seq_length:
        raise ValueError("decode step exceeds max_seq_length")

    meta["tokens"][req, new_step] = output_token
    meta["step"][req] = new_step
    meta["input_tokens"][0, 0] = output_token
    meta["num_new_tokens"][req] = 1
    meta["paged_kv_last_page_len_buffer"][req] = new_step + 1

    return {
        "output_token": output_token,
        "prev_step": cur_step,
        "new_step": new_step,
    }


def run_maca_pk_decode_loop(
    ypk,
    meta: Dict[str, "torch.Tensor"],
    scaffold: "Qwen3PKScaffold",
    *,
    max_decode_steps: int = 1,
    eos_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ``ypk()`` up to ``max_decode_steps`` times with step tensor advance."""
    import torch

    if max_decode_steps < 1:
        raise ValueError("max_decode_steps must be >= 1")

    generated: List[int] = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()

    steps_run = 0
    for step_idx in range(max_decode_steps):
        ypk()
        steps_run += 1
        output_token = int(meta["output_tokens"][0, 0].item())
        generated.append(output_token)
        if eos_token_id is not None and output_token == eos_token_id:
            break
        if step_idx + 1 < max_decode_steps:
            advance_maca_pk_decode_step(meta, scaffold, req=0)

    ender.record()
    torch.cuda.synchronize()
    launch_ms = starter.elapsed_time(ender)

    return {
        "decode_steps": steps_run,
        "generated_tokens": generated,
        "launch_ms": launch_ms,
        "final_step": int(meta["step"][0].item()),
        "stopped_on_eos": eos_token_id is not None and generated[-1] == eos_token_id,
    }


def compute_maca_pk_generation_latency(
    *,
    launch_ms: float,
    prompt_len: int,
    final_step: int,
) -> Dict[str, Any]:
    """Per-token latency metrics aligned with CUDA ``demo/qwen3/demo.py`` reporting."""
    generate_len = final_step + 1 - prompt_len
    if generate_len < 1:
        raise ValueError("generate_len must be >= 1")
    per_token_ms = launch_ms / generate_len
    return {
        "prompt_len": prompt_len,
        "generate_len": generate_len,
        "total_tokens": final_step + 1,
        "launch_ms": launch_ms,
        "per_token_latency_ms": per_token_ms,
    }


def prepare_maca_pk_batched_prompt_meta(
    meta: Dict[str, "torch.Tensor"],
    scaffold: "Qwen3PKScaffold",
    prompt_token_ids: Union[Sequence[int], "torch.Tensor"],
    *,
    active_requests: int = 2,
    num_tokens_per_request: int = 1,
) -> Dict[str, Any]:
    """Fill PK meta for replicated multi-request batch (CUDA ``total_num_requests`` style)."""
    import torch

    if active_requests < 2:
        raise ValueError("active_requests must be >= 2 for batched meta")
    if active_requests > scaffold.max_num_batched_requests:
        raise ValueError("active_requests exceeds max_num_batched_requests")

    num_tokens = active_requests * num_tokens_per_request
    if num_tokens > scaffold.max_num_batched_tokens:
        raise ValueError("active_requests * num_tokens_per_request exceeds max_num_batched_tokens")

    single = prepare_maca_pk_prompt_meta(
        meta,
        scaffold,
        prompt_token_ids,
        active_requests=1,
        num_tokens=num_tokens_per_request,
    )
    prompt_len = single["prompt_len"]

    if isinstance(prompt_token_ids, torch.Tensor):
        ids = prompt_token_ids.to(device=meta["tokens"].device, dtype=torch.long).flatten()
    else:
        ids = torch.tensor(list(prompt_token_ids), device=meta["tokens"].device, dtype=torch.long)

    for req in range(1, active_requests):
        meta["tokens"][req, :prompt_len] = ids
        meta["prompt_lengths"][req] = prompt_len
        meta["step"][req] = prompt_len - 1
        meta["num_new_tokens"][req] = num_tokens_per_request

    meta["qo_indptr_buffer"].zero_()
    meta["paged_kv_indptr_buffer"].zero_()
    meta["paged_kv_indices_buffer"].zero_()
    meta["paged_kv_last_page_len_buffer"].zero_()

    for req in range(active_requests):
        meta["qo_indptr_buffer"][req + 1] = (req + 1) * num_tokens_per_request
        meta["paged_kv_indptr_buffer"][req + 1] = req + 1
        meta["paged_kv_indices_buffer"][req] = req
        meta["paged_kv_last_page_len_buffer"][req] = prompt_len

    for req in range(active_requests):
        slot = req * num_tokens_per_request
        meta["input_tokens"][slot, 0] = ids[-1]

    return {
        **single,
        "active_requests": active_requests,
        "num_tokens": num_tokens,
        "num_tokens_per_request": num_tokens_per_request,
        "qo_indptr": meta["qo_indptr_buffer"][: active_requests + 1].tolist(),
        "paged_kv_indptr": meta["paged_kv_indptr_buffer"][: active_requests + 1].tolist(),
        "paged_kv_indices_head": meta["paged_kv_indices_buffer"][:active_requests].tolist(),
        "paged_kv_last_page_len": meta["paged_kv_last_page_len_buffer"][:active_requests].tolist(),
        "batched_prompt_meta_ready": True,
    }


def inspect_maca_pk_multi_request_batch_plan(
    scaffold: Optional["Qwen3PKScaffold"] = None,
) -> Dict[str, Any]:
    """Cloud-safe multi-request batch meta contract (CUDA ``total_num_requests > 1``)."""
    from demo.maca.qwen3_pk_utils import Qwen3PKScaffold

    scaffold = scaffold or Qwen3PKScaffold()
    return {
        "cuda_reference": "demo/qwen3/demo.py total_num_requests loop + step.max()==step.min()",
        "prepare_entry": "prepare_maca_pk_batched_prompt_meta",
        "max_num_batched_requests": scaffold.max_num_batched_requests,
        "max_num_batched_tokens": scaffold.max_num_batched_tokens,
        "default_active_requests": 2,
        "meta_tensors": [
            "qo_indptr_buffer",
            "paged_kv_indptr_buffer",
            "paged_kv_indices_buffer",
            "paged_kv_last_page_len_buffer",
            "step",
            "tokens",
        ],
        "multi_request_batch_plan_ready": scaffold.max_num_batched_requests >= 2,
    }


def inspect_maca_pk_hf_tokenizer_generation_plan(
    scaffold: Optional["Qwen3PKScaffold"] = None,
) -> Dict[str, Any]:
    """Cloud-safe tokenizer full-path generation contract vs CUDA demo."""
    from demo.maca.qwen3_pk_utils import Qwen3PKScaffold

    scaffold = scaffold or Qwen3PKScaffold()
    batch_plan = inspect_maca_pk_multi_request_batch_plan(scaffold)
    return {
        "cuda_reference": "demo/qwen3/demo.py --use-yirage (tokenizer + ypk + decode + latency)",
        "maca_entry": "maca_pk_hf_tokenizer_generation_smoke",
        "tokenizer_generation_ready": False,
        "tokenizer_generation_plan_ready": True,
        "pipeline_steps": [
            "AutoTokenizer.from_pretrained",
            "encode_maca_pk_chat_prompt",
            "maca_pk_hf_init_compiled_stack + prepare_maca_pk_prompt_meta",
            "run_maca_pk_decode_loop",
            "decode_maca_pk_generated_tokens",
            "compute_maca_pk_generation_latency",
        ],
        "latency_fields": [
            "prompt_len",
            "generate_len",
            "per_token_latency_ms",
        ],
        "multi_request_batch_plan": batch_plan,
        "model": scaffold.model,
        "requires_metax_gpu": True,
    }


def inspect_maca_pk_decode_step_contract(
    scaffold: Optional["Qwen3PKScaffold"] = None,
) -> Dict[str, Any]:
    """Cloud-safe decode step tensor semantics contract."""
    from demo.maca.qwen3_pk_utils import Qwen3PKScaffold

    scaffold = scaffold or Qwen3PKScaffold()
    return {
        "cuda_reference": "demo/qwen3/demo.py --use-yirage (step / input_tokens / tokens)",
        "prepare_prompt": "prepare_maca_pk_prompt_meta",
        "advance_step": "advance_maca_pk_decode_step",
        "decode_loop": "run_maca_pk_decode_loop",
        "tokenizer_encode": "encode_maca_pk_chat_prompt",
        "tokenizer_decode": "decode_maca_pk_generated_tokens",
        "initial_step": "prompt_len - 1",
        "meta_tensors": [
            "step",
            "tokens",
            "input_tokens",
            "output_tokens",
            "prompt_lengths",
            "paged_kv_last_page_len_buffer",
        ],
        "max_seq_length": scaffold.max_seq_length,
        "decode_step_contract_ready": True,
    }


__all__ = [
    "DEFAULT_MACA_PK_CHAT_PROMPT",
    "advance_maca_pk_decode_step",
    "compute_maca_pk_generation_latency",
    "decode_maca_pk_generated_tokens",
    "encode_maca_pk_chat_prompt",
    "inspect_maca_pk_decode_step_contract",
    "inspect_maca_pk_hf_tokenizer_generation_plan",
    "inspect_maca_pk_multi_request_batch_plan",
    "prepare_maca_pk_batched_prompt_meta",
    "prepare_maca_pk_prompt_meta",
    "run_maca_pk_decode_loop",
]
