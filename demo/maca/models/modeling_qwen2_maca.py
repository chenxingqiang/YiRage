# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Modified for MetaX MACA GPU support by YiRage Project.
#
# CUDA reference: demo/qwen2.5/models/modeling_qwen2.py (flashinfer + default cuda transpiler)
# MACA path: backend=maca superoptimize via demo/maca/qwen_kernel_utils.py (no flashinfer)
"""PyTorch Qwen2 model for MetaX MACA GPU."""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo.maca.qwen_kernel_utils import (  # noqa: E402
    superoptimize_attn_qkv,
    superoptimize_mlp_down,
    superoptimize_mlp_gate_up,
)
from .configuration_qwen2 import Qwen2Config


def get_device_type() -> str:
    """mcPytorch exposes MetaX as ``torch.cuda``."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = get_device_type()
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=torch.bfloat16), sin.to(dtype=torch.bfloat16)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.enable_yirage = False

    def fuse_weights(self):
        self.fused_weight = torch.transpose(
            torch.cat((self.gate_proj.weight, self.up_proj.weight), 0), 0, 1
        )

    def superoptimize_kernels(self, *, quick: Optional[bool] = None, dtype_name: str = "bfloat16"):
        """Optimize MLP kernels with YiRage MACA backend (aligned to CUDA modeling_qwen2)."""
        try:
            self.kernel1 = superoptimize_mlp_gate_up(
                self.hidden_size,
                self.intermediate_size,
                dtype_name=dtype_name,
                quick=quick,
            )
            self.kernel2 = superoptimize_mlp_down(
                self.hidden_size,
                self.intermediate_size,
                dtype_name=dtype_name,
                quick=quick,
            )
            if self.kernel1 is not None and self.kernel2 is not None:
                self.enable_yirage = True
            else:
                print("  [MLP] No valid YiRage MACA kernels, using PyTorch")
        except Exception as exc:
            print(f"  [MLP] YiRage MACA optimization failed: {exc}, using PyTorch")

    def forward(
        self,
        input_layernorm,
        hidden_state,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        if hidden_state.shape[-2] == 1 and self.enable_yirage:
            kwargs = {"stream": stream} if stream is not None else {}
            output = self.kernel1(
                inputs=(hidden_state, input_layernorm.weight, self.fused_weight),
                **kwargs,
            )[0]
            gate_output, up_output = torch.chunk(output, 2, -1)
            output = self.kernel2(
                inputs=(gate_output, up_output, self.down_proj.weight),
                **kwargs,
            )[0]
        else:
            hidden_state = input_layernorm(hidden_state)
            output = torch.matmul(hidden_state, self.fused_weight)
            gate_output, up_output = torch.chunk(output, 2, -1)
            output = self.down_proj(self.act_fn(gate_output) * up_output)
        return output


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        assert self.layer_idx is not None
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.key_cache, self.value_cache = kv_cache
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        self.enable_yirage = False

    def fuse_weights(self):
        self.fused_weight = torch.transpose(
            torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), 0), 0, 1
        )
        self.fused_bias = torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias), 0)

    def superoptimize_kernels(self, *, quick: Optional[bool] = None, dtype_name: str = "bfloat16"):
        """Optimize attention QKV projection with YiRage MACA backend."""
        self.fused_outdim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        try:
            self.kernel = superoptimize_attn_qkv(
                self.hidden_size,
                self.fused_outdim,
                dtype_name=dtype_name,
                quick=quick,
            )
            if self.kernel is not None:
                self.enable_yirage = True
            else:
                print("  [Attn] No valid YiRage MACA kernel, using PyTorch")
        except Exception as exc:
            print(f"  [Attn] YiRage MACA optimization failed: {exc}, using PyTorch")

    def _scaled_dot_product_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = False
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(query.size(-1))
        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        if is_causal:
            seq_len = query.size(2)
            kv_len = key.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=query.device),
                diagonal=kv_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn_weights, value)

    def forward(
        self,
        input_layernorm,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        if q_len == 1 and self.enable_yirage:
            kwargs = {"stream": stream} if stream is not None else {}
            xqkv = self.kernel(
                inputs=(hidden_states, input_layernorm.weight, self.fused_weight),
                **kwargs,
            )[0]
            xqkv = xqkv.view(bsz, q_len, self.fused_outdim)
        else:
            hidden_states = input_layernorm(hidden_states)
            xqkv = torch.matmul(hidden_states, self.fused_weight)

        xqkv = xqkv + self.fused_bias
        query_states = xqkv[:, :, : (self.num_heads * self.head_dim)]
        xkv = xqkv[:, :, (self.num_heads * self.head_dim) :]
        key_states, value_states = xkv.chunk(2, -1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        step_val = step.item() if step is not None else 0
        if q_len > 1:
            self.key_cache[self.layer_idx, 0, :q_len] = key_states[0]
            self.value_cache[self.layer_idx, 0, :q_len] = value_states[0]
            kv_len = q_len
        else:
            self.key_cache[self.layer_idx, 0, step_val] = key_states[0]
            self.value_cache[self.layer_idx, 0, step_val] = value_states[0]
            kv_len = step_val + 1

        query_states = query_states.transpose(1, 2)
        key_cache = self.key_cache[self.layer_idx, :, :kv_len, :, :].transpose(1, 2)
        value_cache = self.value_cache[self.layer_idx, :, :kv_len, :, :].transpose(1, 2)

        attn_output = self._scaled_dot_product_attention(
            query_states, key_cache, value_cache, is_causal=(q_len > 1)
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self, config: Qwen2Config, kv_cache: Tuple[torch.Tensor, torch.Tensor], layer_idx: int
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config, kv_cache, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def fuse_weights(self):
        self.mlp.fuse_weights()
        self.self_attn.fuse_weights()

    def superoptimize_kernels(self, *, quick: Optional[bool] = None, dtype_name: str = "bfloat16"):
        self.mlp.superoptimize_kernels(quick=quick, dtype_name=dtype_name)
        self.self_attn.superoptimize_kernels(quick=quick, dtype_name=dtype_name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.self_attn(
            input_layernorm=self.input_layernorm,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            step=step,
            stream=stream,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(
            self.post_attention_layernorm, hidden_states, stream=stream
        )
        hidden_states = residual + hidden_states
        return (hidden_states,)


class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        head_dim = config.hidden_size // config.num_attention_heads
        key_cache = torch.empty(
            (
                config.num_hidden_layers,
                1,
                config.max_position_embeddings,
                config.num_key_value_heads,
                head_dim,
            )
        )
        value_cache = torch.empty(
            (
                config.num_hidden_layers,
                1,
                config.max_position_embeddings,
                config.num_key_value_heads,
                head_dim,
            )
        )
        self.kv_cache = (key_cache, value_cache)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, self.kv_cache, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.post_init()

    def fuse_weights(self):
        for decoder_layer in self.layers:
            decoder_layer.fuse_weights()

    def superoptimize_kernels(
        self,
        *,
        quick: Optional[bool] = None,
        dtype_name: str = "bfloat16",
        max_layers: Optional[int] = None,
    ):
        layers = self.layers[:max_layers] if max_layers is not None else self.layers
        for decoder_layer in layers:
            decoder_layer.superoptimize_kernels(quick=quick, dtype_name=dtype_name)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                step=step,
                stream=stream,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        return (hidden_states,)


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def fuse_weights(self):
        self.model.fuse_weights()

    def superoptimize_kernels(
        self,
        *,
        quick: Optional[bool] = None,
        dtype_name: str = "bfloat16",
        max_layers: Optional[int] = None,
    ):
        if quick is None:
            quick = os.environ.get("YIRAGE_MACA_SEARCH_QUICK", "1") == "1"
        self.model.superoptimize_kernels(
            quick=quick, dtype_name=dtype_name, max_layers=max_layers
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        num_logits_to_keep: int = 0,
        stream: Optional[torch.cuda.Stream] = None,
        **loss_kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            step=step,
            inputs_embeds=inputs_embeds,
            stream=stream,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        return logits
