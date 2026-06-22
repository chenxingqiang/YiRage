# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""LLaMA 3B MoE model configuration."""

from __future__ import annotations


class LLaMA3BMoEConfig:
    """
    Dataclass-style configuration for a LLaMA 3B Mixture-of-Experts model.

    Default dimensions mirror the "LLaMA-3B MoE" (DeepSeek-V2-Lite-style) variant:
      - hidden_size = 3072
      - num_hidden_layers = 28
      - num_attention_heads = 24 (GQA with 8 KV heads, head_dim = 128)
      - num_experts = 8, top_k = 2, intermediate_size = 8192
    """

    model_type: str = "llama3b_moe"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        num_experts: int = 8,
        top_k_experts: int = 2,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LLaMA3BMoEConfig("
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_hidden_layers}, "
            f"heads={self.num_attention_heads}/{self.num_key_value_heads}, "
            f"experts={self.num_experts}, top_k={self.top_k_experts}, "
            f"intermediate={self.intermediate_size})"
        )
