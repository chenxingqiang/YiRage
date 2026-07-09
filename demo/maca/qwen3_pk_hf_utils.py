"""HF weight bundle for MACA Qwen3 PersistentKernel (CUDA ``demo/qwen3/demo.py`` aligned)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from demo.maca.qwen3_pk_utils import Qwen3PKScaffold


PK_HF_PAD_VOCAB_SIZE = 153600


@dataclass(frozen=True)
class MacaPkHfWeightBundle:
    """HF tensors wired into ``ypk.attach_input`` for MACA PK stack graphs."""

    model_name: str
    num_layers: int
    vocab_smoke: int
    cos_pos_embed: "torch.Tensor"
    sin_pos_embed: "torch.Tensor"
    embed_weight: "torch.Tensor"
    norm_weight: "torch.Tensor"
    lm_head_weight: "torch.Tensor"
    layers: tuple
    k_caches: tuple
    v_caches: tuple
    num_kv_heads: int
    use_padded_lm_head: bool = False

    def layer(self, layer_idx: int):
        return self.layers[layer_idx]

    def k_cache(self, layer_idx: int) -> "torch.Tensor":
        return self.k_caches[layer_idx]

    def v_cache(self, layer_idx: int) -> "torch.Tensor":
        return self.v_caches[layer_idx]


def maca_pk_hf_weight_attach_map(*, max_layers: int = 1) -> List[Dict[str, str]]:
    """Static PK attach_input name → HF model path mapping (Cloud contract)."""
    rows = [
        {"pk_name": "embed_tokens", "hf_path": "model.embed_tokens.weight"},
        {"pk_name": "cos_position_embedding", "hf_path": "model.rotary_emb (positions)"},
        {"pk_name": "sin_position_embedding", "hf_path": "model.rotary_emb (positions)"},
        {"pk_name": "model_norm_weight", "hf_path": "model.norm.weight"},
        {"pk_name": "lm_head", "hf_path": "lm_head.weight (padded or vocab_smoke slice)"},
    ]
    for layer_idx in range(max_layers):
        prefix = f"layer_{layer_idx}"
        rows.extend(
            [
                {"pk_name": f"{prefix}_input_layernorm", "hf_path": f"layers[{layer_idx}].input_layernorm.weight"},
                {"pk_name": f"{prefix}_q_proj", "hf_path": f"layers[{layer_idx}].self_attn.q_proj.weight"},
                {"pk_name": f"{prefix}_k_proj", "hf_path": f"layers[{layer_idx}].self_attn.k_proj.weight"},
                {"pk_name": f"{prefix}_v_proj", "hf_path": f"layers[{layer_idx}].self_attn.v_proj.weight"},
                {"pk_name": f"{prefix}_q_norm", "hf_path": f"layers[{layer_idx}].self_attn.q_norm.weight"},
                {"pk_name": f"{prefix}_k_norm", "hf_path": f"layers[{layer_idx}].self_attn.k_norm.weight"},
                {"pk_name": f"{prefix}_k_cache", "hf_path": f"model.kv_cache[0][{layer_idx}]"},
                {"pk_name": f"{prefix}_v_cache", "hf_path": f"model.kv_cache[1][{layer_idx}]"},
                {"pk_name": f"{prefix}_o_proj", "hf_path": f"layers[{layer_idx}].self_attn.o_proj.weight"},
                {"pk_name": f"{prefix}_post_attn_layernorm", "hf_path": f"layers[{layer_idx}].post_attention_layernorm.weight"},
                {"pk_name": f"{prefix}_gate_proj", "hf_path": f"layers[{layer_idx}].mlp.gate_proj.weight"},
                {"pk_name": f"{prefix}_up_proj", "hf_path": f"layers[{layer_idx}].mlp.up_proj.weight"},
                {"pk_name": f"{prefix}_down_proj", "hf_path": f"layers[{layer_idx}].mlp.down_proj.weight"},
            ]
        )
    return rows


def resolve_maca_pk_lm_vocab_size(
    *,
    vocab_smoke: int = 128,
    use_padded_lm_head: bool = False,
) -> int:
    """Return argmax/lm_head vocab dim (CUDA ``demo/qwen3/demo.py`` uses 153600 when padded)."""
    if use_padded_lm_head:
        return PK_HF_PAD_VOCAB_SIZE
    return vocab_smoke


def inspect_maca_pk_hf_padded_lm_head_plan(
    scaffold: Optional["Qwen3PKScaffold"] = None,
) -> Dict[str, Any]:
    """Cloud-safe padded lm_head (153600) argmax contract."""
    from demo.maca.qwen_hf_utils import default_qwen_dims
    from demo.maca.qwen3_pk_utils import Qwen3PKScaffold

    scaffold = scaffold or Qwen3PKScaffold()
    dims = default_qwen_dims()
    vocab = PK_HF_PAD_VOCAB_SIZE
    return {
        "cuda_reference": "demo/qwen3/demo.py (lm_head padded to 153600, vocab_size=153600)",
        "pad_helper": "pad_lm_head_weight",
        "pad_vocab_size": vocab,
        "hidden_size": dims.hidden_size,
        "lm_head_shape": [vocab, dims.hidden_size],
        "argmax_in_shape": [scaffold.max_num_batched_tokens, vocab],
        "loader_flag": "use_padded_lm_head=True",
        "runtime_entry": "maca_pk_hf_stack_runtime_smoke(use_padded_lm_head=True)",
        "padded_lm_head_plan_ready": vocab == 153600 and dims.hidden_size > 0,
        "requires_metax_gpu": True,
    }


def pad_lm_head_weight(
    lm_head_weight: "torch.Tensor",
    hidden_size: int,
    *,
    pad_vocab_size: int = PK_HF_PAD_VOCAB_SIZE,
) -> "torch.Tensor":
    """Pad lm_head to CUDA PK grid size (``demo/qwen3/demo.py`` uses 153600)."""
    import torch

    vocab_size = lm_head_weight.shape[0]
    if vocab_size >= pad_vocab_size:
        return lm_head_weight[:pad_vocab_size]
    pad_rows = pad_vocab_size - vocab_size
    padding = torch.zeros(pad_rows, hidden_size, device=lm_head_weight.device, dtype=lm_head_weight.dtype)
    return torch.cat((lm_head_weight, padding), dim=0)


def load_maca_pk_hf_weight_bundle(
    model_name: str,
    scaffold: "Qwen3PKScaffold",
    device: "torch.device",
    *,
    max_layers: int = 1,
    vocab_smoke: int = 128,
    local_files_only: bool = False,
    use_padded_lm_head: bool = False,
) -> MacaPkHfWeightBundle:
    """Load HF Qwen3 weights for MACA PK attach_input (MetaX VM)."""
    import torch

    from demo.qwen3.models.modeling_qwen3 import Qwen3ForCausalLM

    torch.set_default_dtype(torch.bfloat16)
    model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        1,
        max_num_pages=scaffold.max_num_pages,
        page_size=scaffold.page_size,
        local_files_only=local_files_only,
    ).to(device)

    if max_layers < 1 or max_layers > len(model.model.layers):
        raise ValueError(f"max_layers must be in [1, {len(model.model.layers)}], got {max_layers}")

    positions = torch.arange(scaffold.page_size, device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(positions)
    cos_pos = position_embeddings[0][0, : scaffold.page_size, :].contiguous()
    sin_pos = position_embeddings[1][0, : scaffold.page_size, :].contiguous()

    embed = model.model.embed_tokens.weight
    if vocab_smoke > 0 and not use_padded_lm_head:
        embed = embed[:vocab_smoke]

    lm_head = model.lm_head.weight
    if use_padded_lm_head:
        lm_head = pad_lm_head_weight(lm_head, model.config.hidden_size)
    elif vocab_smoke > 0:
        lm_head = lm_head[:vocab_smoke]

    key_cache, value_cache = model.model.kv_cache
    return MacaPkHfWeightBundle(
        model_name=model_name,
        num_layers=max_layers,
        vocab_smoke=vocab_smoke if not use_padded_lm_head else PK_HF_PAD_VOCAB_SIZE,
        cos_pos_embed=cos_pos,
        sin_pos_embed=sin_pos,
        embed_weight=embed,
        norm_weight=model.model.norm.weight,
        lm_head_weight=lm_head,
        layers=tuple(model.model.layers[:max_layers]),
        k_caches=tuple(key_cache[i] for i in range(max_layers)),
        v_caches=tuple(value_cache[i] for i in range(max_layers)),
        num_kv_heads=int(model.config.num_key_value_heads),
        use_padded_lm_head=use_padded_lm_head,
    )


def inspect_maca_pk_hf_weight_plan(
    scaffold: Optional["Qwen3PKScaffold"] = None,
    *,
    max_layers: int = 1,
) -> Dict[str, Any]:
    """Cloud-safe HF weight → PK attach_input mapping contract."""
    from demo.maca.qwen_hf_utils import default_qwen_dims
    from demo.maca.qwen3_pk_utils import Qwen3PKScaffold

    scaffold = scaffold or Qwen3PKScaffold()
    dims = default_qwen_dims()
    attach_map = maca_pk_hf_weight_attach_map(max_layers=max_layers)
    return {
        "cuda_reference": "demo/qwen3/demo.py --use-yirage (model.model.layers attach_input)",
        "loader": "load_maca_pk_hf_weight_bundle",
        "model": scaffold.model,
        "max_layers": max_layers,
        "vocab_smoke_default": 128,
        "pad_vocab_size": PK_HF_PAD_VOCAB_SIZE,
        "hidden_size": dims.hidden_size,
        "attach_map": attach_map,
        "attach_map_count": len(attach_map),
        "requires": ["transformers", "demo/qwen3/models/modeling_qwen3.py", "MetaX GPU for load"],
        "weight_plan_ready": len(attach_map) >= 5 + 13 * max_layers,
    }


__all__ = [
    "MacaPkHfWeightBundle",
    "PK_HF_PAD_VOCAB_SIZE",
    "inspect_maca_pk_hf_padded_lm_head_plan",
    "inspect_maca_pk_hf_weight_plan",
    "load_maca_pk_hf_weight_bundle",
    "maca_pk_hf_weight_attach_map",
    "pad_lm_head_weight",
    "resolve_maca_pk_lm_vocab_size",
]
