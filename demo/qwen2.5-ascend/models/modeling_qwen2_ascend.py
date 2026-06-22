# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Modified for Huawei Ascend NPU support by YiRage Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2 model for Huawei Ascend NPU."""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from .configuration_qwen2 import Qwen2Config

import yirage as yr


def get_device_type():
    """Get the device type (npu, cuda, or cpu)"""
    try:
        import torch_npu

        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
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
        # Force float32 for RoPE computation
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=torch.bfloat16), sin.to(dtype=torch.bfloat16)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embedding to query and key tensors."""
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

    def superoptimize_kernels(self):
        """Optimize kernels using YiRage with Ascend backend"""
        # Note: If no valid kernels found, enable_yirage stays False
        # and forward() will use PyTorch fallback

        # Ascend search config - FIXED imaps to use valid dimension indices
        # imap values: -1 = no mapping, 0 = map to grid.x, 1 = map to grid.y
        # For 2D tensors (batch, dim), valid indices are 0 and 1
        # When grid_dim.x == 1, we MUST use imap.x = -1
        search_config = {
            # Valid imaps for 2D tensors with potential grid=(1,1,1)
            "imaps": [(-1, -1, -1), (0, -1, -1), (1, -1, -1)],
            "omaps": [(-1, -1, -1), (0, -1, -1)],
            # Grid/block dims - start with single block for stability
            "griddims": [(1, 1, 1), (2, 1, 1), (4, 1, 1), (8, 1, 1)],
            "blockdims": [(1, 1, 1), (2, 1, 1), (4, 1, 1)],
            # Forloop configs
            "fmaps": [-1, 0, 1],
            "franges": [4, 16, 64],
            "backend": "ascend",
        }

        try:
            # Kernel 1: RMSNorm -> Mul -> MatMul (gate + up projection)
            graph = yr.new_kernel_graph()
            X = graph.new_input(dims=(1, self.hidden_size), dtype=yr.bfloat16)
            G = graph.new_input(dims=(1, self.hidden_size), dtype=yr.bfloat16)
            W = graph.new_input(
                dims=(self.hidden_size, 2 * self.intermediate_size),
                strides=(1, self.hidden_size),
                dtype=yr.bfloat16,
            )
            D = graph.rms_norm(X, normalized_shape=(self.hidden_size,))
            D = graph.mul(D, G)
            O = graph.matmul(D, W)
            graph.mark_output(O)
            self.kernel1 = graph.superoptimize(**search_config)

            # Kernel 2: SiLU -> Mul -> MatMul (down projection)
            graph = yr.new_kernel_graph()
            X = graph.new_input(dims=(1, self.intermediate_size), dtype=yr.bfloat16)
            Y = graph.new_input(dims=(1, self.intermediate_size), dtype=yr.bfloat16)
            W = graph.new_input(
                dims=(self.intermediate_size, self.hidden_size),
                strides=(1, self.intermediate_size),
                dtype=yr.bfloat16,
            )
            D = graph.mul(graph.silu(X), Y)
            O = graph.matmul(D, W)
            graph.mark_output(O)
            self.kernel2 = graph.superoptimize(**search_config)

            # Only enable YiRage if both kernels are valid
            if self.kernel1 is not None and self.kernel2 is not None:
                self.enable_yirage = True
            else:
                print("  [MLP] No valid YiRage kernels found, using PyTorch")
        except Exception as e:
            print(f"  [MLP] YiRage optimization failed: {e}, using PyTorch")

    def forward(self, input_layernorm, hidden_state):
        if hidden_state.shape[-2] == 1 and self.enable_yirage:
            # Use YiRage kernels for decoding
            output = self.kernel1(inputs=(hidden_state, input_layernorm.weight, self.fused_weight))[
                0
            ]
            gate_output, up_output = torch.chunk(output, 2, -1)
            # Note: down_proj.weight is (hidden_size, intermediate_size), need to transpose
            # for matmul: (batch, seq, intermediate) @ (intermediate, hidden) = (batch, seq, hidden)
            output = self.kernel2(inputs=(gate_output, up_output, self.down_proj.weight.T))[0]
        else:
            # Use standard PyTorch for prefilling
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

    def superoptimize_kernels(self):
        """Optimize attention kernels using YiRage with Ascend backend"""
        self.fused_outdim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim

        # Ascend search config - FIXED imaps to use valid dimension indices
        # imap values: -1 = no mapping, 0 = map to grid.x, 1 = map to grid.y
        # For 2D tensors (batch, dim), valid indices are 0 and 1
        # When grid_dim.x == 1, we MUST use imap.x = -1
        search_config = {
            # Valid imaps for 2D tensors with potential grid=(1,1,1)
            "imaps": [(-1, -1, -1), (0, -1, -1), (1, -1, -1)],
            "omaps": [(-1, -1, -1), (0, -1, -1)],
            # Grid/block dims - start with single block for stability
            "griddims": [(1, 1, 1), (2, 1, 1), (4, 1, 1), (8, 1, 1)],
            "blockdims": [(1, 1, 1), (2, 1, 1), (4, 1, 1)],
            # Forloop configs
            "fmaps": [-1, 0, 1],
            "franges": [4, 16, 64],
            "backend": "ascend",
        }

        try:
            graph = yr.new_kernel_graph()
            X = graph.new_input(dims=(1, self.hidden_size), dtype=yr.bfloat16)
            G = graph.new_input(dims=(1, self.hidden_size), dtype=yr.bfloat16)
            W = graph.new_input(
                dims=(self.hidden_size, self.fused_outdim),
                strides=(1, self.hidden_size),
                dtype=yr.bfloat16,
            )
            D = graph.rms_norm(X, normalized_shape=(self.hidden_size,))
            D = graph.mul(D, G)
            O = graph.matmul(D, W)
            graph.mark_output(O)
            self.kernel = graph.superoptimize(**search_config)

            if self.kernel is not None:
                self.enable_yirage = True
            else:
                print("  [Attn] No valid YiRage kernel found, using PyTorch")
        except Exception as e:
            print(f"  [Attn] YiRage optimization failed: {e}, using PyTorch")

    def _scaled_dot_product_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = False
    ) -> torch.Tensor:
        """
        Scaled dot-product attention using native PyTorch operations.
        Compatible with Ascend NPU.
        """
        # query: (batch, num_heads, seq_len, head_dim)
        # key: (batch, num_kv_heads, kv_len, head_dim)
        # value: (batch, num_kv_heads, kv_len, head_dim)

        scale = 1.0 / math.sqrt(query.size(-1))

        # Repeat KV heads if needed (GQA)
        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Causal mask
        if is_causal:
            seq_len = query.size(2)
            kv_len = key.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=query.device),
                diagonal=kv_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(attn_weights, value)

        return output

    def forward(
        self,
        input_layernorm,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        if q_len == 1 and self.enable_yirage:
            # Use YiRage kernels for decoding
            xqkv = self.kernel(inputs=(hidden_states, input_layernorm.weight, self.fused_weight))[0]
            xqkv = xqkv.view(bsz, q_len, self.fused_outdim)
        else:
            # Use standard PyTorch for prefilling
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

        # Update KV cache
        step_val = step.item() if step is not None else 0
        if q_len > 1:
            self.key_cache[self.layer_idx, 0, :q_len] = key_states[0]
            self.value_cache[self.layer_idx, 0, :q_len] = value_states[0]
            kv_len = q_len
        else:
            self.key_cache[self.layer_idx, 0, step_val] = key_states[0]
            self.value_cache[self.layer_idx, 0, step_val] = value_states[0]
            kv_len = step_val + 1

        # Prepare for attention: (batch, num_heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_cache = self.key_cache[self.layer_idx, :, :kv_len, :, :].transpose(1, 2)
        value_cache = self.value_cache[self.layer_idx, :, :kv_len, :, :].transpose(1, 2)

        # Compute attention
        attn_output = self._scaled_dot_product_attention(
            query_states, key_cache, value_cache, is_causal=(q_len > 1)
        )

        # Reshape back: (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


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

    def superoptimize_kernels(self):
        self.mlp.superoptimize_kernels()
        self.self_attn.superoptimize_kernels()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        # Self Attention
        hidden_states = self.self_attn(
            input_layernorm=self.input_layernorm,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            step=step,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm, hidden_states)
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

        # KV cache: (num_layers, batch, max_seq_len, num_kv_heads, head_dim)
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

    def superoptimize_kernels(self):
        for decoder_layer in self.layers:
            decoder_layer.superoptimize_kernels()

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
    ):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                step=step,
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

    def superoptimize_kernels(self):
        self.model.superoptimize_kernels()

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            step=step,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        return logits
