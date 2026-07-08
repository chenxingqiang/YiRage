"""HF Qwen helpers for MACA demos (config load without flashinfer dependency)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"


@dataclass(frozen=True)
class QwenModelDims:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def fused_qkv_outdim(self) -> int:
        return (self.num_heads + 2 * self.num_kv_heads) * self.head_dim


def default_qwen_dims() -> QwenModelDims:
    """Built-in Qwen3-8B shapes (aligned with ``demo/qwen2.5/demo.py``)."""
    return QwenModelDims(
        hidden_size=4096,
        intermediate_size=12288,
        num_heads=32,
        num_kv_heads=8,
    )


def qwen_dims_from_hf_config(config: Any) -> QwenModelDims:
    """Extract MACA kernel shapes from a HuggingFace ``PretrainedConfig``."""
    return QwenModelDims(
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.intermediate_size),
        num_heads=int(config.num_attention_heads),
        num_kv_heads=int(config.num_key_value_heads),
    )


def load_qwen_dims_from_hf(
    model_name: str,
    *,
    local_files_only: bool = False,
) -> QwenModelDims:
    """Load ``config.json`` from HuggingFace Hub (no weight download)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    return qwen_dims_from_hf_config(config)


def resolve_qwen_dims(
    model_name: Optional[str],
    *,
    config_only: bool = True,
) -> QwenModelDims:
    """Resolve kernel shapes: built-in defaults or HF ``config.json``."""
    if not model_name:
        return default_qwen_dims()
    return load_qwen_dims_from_hf(model_name, local_files_only=not config_only)


def describe_from_pretrained_gap() -> str:
    """Document full-weight ``from_pretrained`` requirements on MACA."""
    return (
        "Full HF weights require MetaX VM + flashinfer (see demo/qwen2.5/models/modeling_qwen2.py). "
        "Use --model Qwen/Qwen3-8B --config-only for Hub config smoke without weight download."
    )
