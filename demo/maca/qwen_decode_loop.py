"""Qwen decode loops for MACA demos (eager + CUDA Graph, aligned to demo/qwen2.5/demo.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from demo._maca_utils import sync_device


@dataclass
class QwenDecodeResult:
    cur_pos: int
    run_time_ms: float
    tokens_generated: int
    used_cuda_graph: bool


def run_qwen_decode_loop(
    model: Any,
    *,
    tokens: torch.Tensor,
    prompt_len: int,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    max_tokens: int,
    warmup: int,
    device: torch.device,
    use_cuda_graph: bool = True,
) -> QwenDecodeResult:
    """Prefill (eager) + decode (CUDA Graph or eager), matching CUDA qwen2.5/demo.py."""
    prev_pos = 0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    step = torch.tensor([0], dtype=torch.int32, device=device)
    stream = torch.cuda.Stream(device=device)
    graph: Optional[torch.cuda.CUDAGraph] = None
    logits: Optional[torch.Tensor] = None
    static_input_ids: Optional[torch.Tensor] = None
    static_cos: Optional[torch.Tensor] = None
    static_sin: Optional[torch.Tensor] = None
    timed = False

    decode_end = prompt_len + max_tokens
    cur_pos = prompt_len

    for cur_pos in range(prompt_len, decode_end):
        step.fill_(cur_pos - 1)

        if not use_cuda_graph:
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            logits = model.forward(
                input_ids=input_ids,
                position_embeddings=(cos_embeddings, sin_embeddings),
                step=step,
                stream=stream,
            )
        elif cur_pos < prompt_len + 1:
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            logits = model.forward(
                input_ids=input_ids,
                position_embeddings=(cos_embeddings, sin_embeddings),
                step=step,
                stream=stream,
            )
        elif cur_pos == prompt_len + 1:
            static_input_ids = tokens[:, prev_pos:cur_pos]
            static_cos = position_embeddings[0][:, prev_pos:cur_pos]
            static_sin = position_embeddings[1][:, prev_pos:cur_pos]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                logits = model.forward(
                    input_ids=static_input_ids,
                    position_embeddings=(static_cos, static_sin),
                    step=step,
                    stream=stream,
                )
        else:
            assert graph is not None and logits is not None
            assert static_input_ids is not None and static_cos is not None and static_sin is not None
            static_input_ids.copy_(tokens[:, prev_pos:cur_pos])
            static_cos.copy_(position_embeddings[0][:, prev_pos:cur_pos])
            static_sin.copy_(position_embeddings[1][:, prev_pos:cur_pos])
            graph.replay()

        next_token = logits.argmax(dim=-1)[0, -1]
        tokens[0, cur_pos] = next_token
        prev_pos = cur_pos

        if next_token == model.config.eos_token_id:
            break

        if cur_pos == prompt_len + warmup:
            sync_device(device)
            starter.record()
            timed = True

    ender.record()
    sync_device(device)
    run_time_ms = starter.elapsed_time(ender) if timed else 0.0
    tokens_generated = max(cur_pos - prompt_len - warmup + 1, 0)

    return QwenDecodeResult(
        cur_pos=cur_pos,
        run_time_ms=run_time_ms,
        tokens_generated=tokens_generated,
        used_cuda_graph=use_cuda_graph and graph is not None,
    )
