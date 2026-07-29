# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Minimal single-process vLLM runtime for CPU Serving e2e (``vllm`` fork).

Requires the **CPU wheel** on headless CI::

    pip install https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl \\
        --extra-index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from typing import Any, Iterator, Optional

_VLLM_RUNTIME_READY = False
_VLLM_CONFIG: Optional[Any] = None
_VLLM_TEST_LAYER_SEQ = 0


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def allocate_vllm_test_layer_prefix() -> str:
    """Unique vLLM module prefix per build (attention registry is process-global)."""
    global _VLLM_TEST_LAYER_SEQ
    prefix = f"layers.{_VLLM_TEST_LAYER_SEQ}"
    _VLLM_TEST_LAYER_SEQ += 1
    return prefix


@contextmanager
def vllm_test_config_context() -> Iterator[Any]:
    """Re-enter ``set_current_vllm_config`` for layer build / forward."""
    from vllm.config import set_current_vllm_config

    ensure_vllm_single_process_runtime()
    assert _VLLM_CONFIG is not None
    with set_current_vllm_config(_VLLM_CONFIG):
        yield _VLLM_CONFIG


def ensure_vllm_single_process_runtime(*, master_port: Optional[int] = None) -> None:
    """Init gloo distributed + TP=1 vLLM config once per process."""
    global _VLLM_RUNTIME_READY, _VLLM_CONFIG
    if _VLLM_RUNTIME_READY:
        return

    from .vllm_plugin import require_vllm

    require_vllm()

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if master_port is None:
        master_port = _pick_free_tcp_port()
    os.environ["MASTER_PORT"] = str(int(master_port))

    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
        )

    _VLLM_CONFIG = EngineArgs(model="Qwen/Qwen2-0.5B").create_engine_config()
    with set_current_vllm_config(_VLLM_CONFIG):
        ensure_model_parallel_initialized(1, 1)

    _VLLM_RUNTIME_READY = True


def vllm_cpu_wheel_required_message() -> str:
    return (
        "CPU Serving verification requires the vLLM **CPU** wheel (not CUDA-only build). "
        "Run: bash scripts/setup_serving_vllm_cpu.sh"
    )
