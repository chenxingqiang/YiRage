# Copyright 2025 YiRage Team
# TPU Embedding Kernels

"""
TPU embedding and position encoding kernels.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import math


# =============================================================================
# Embedding Lookup
# =============================================================================

def embedding_lookup(
    embedding_table: jax.Array,  # [vocab_size, hidden_dim]
    indices: jax.Array,          # [batch, seq_len]
) -> jax.Array:
    """Simple embedding lookup."""
    return embedding_table[indices]


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> tuple[jax.Array, jax.Array]:
    """Precompute RoPE sin/cos cache."""
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
    t = jnp.arange(max_seq_len)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(
    x: jax.Array,       # [batch, heads, seq, head_dim]
    cos: jax.Array,     # [seq, head_dim//2]
    sin: jax.Array,     # [seq, head_dim//2]
) -> jax.Array:
    """Apply Rotary Position Embedding."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    
    # Reshape for broadcasting
    cos = cos[None, None, :, :]  # [1, 1, seq, head_dim//2]
    sin = sin[None, None, :, :]
    
    return jnp.concatenate([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], axis=-1)


# =============================================================================
# Sinusoidal Position Embedding
# =============================================================================

def sinusoidal_position_embedding(
    seq_len: int,
    hidden_dim: int,
) -> jax.Array:
    """Generate sinusoidal position embeddings."""
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
    
    pe = jnp.zeros((seq_len, hidden_dim))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    
    return pe


# =============================================================================
# LM Head (Output Projection)
# =============================================================================

def lm_head(
    hidden_states: jax.Array,   # [batch, seq, hidden]
    weight: jax.Array,          # [vocab_size, hidden]
) -> jax.Array:
    """Language model output head."""
    return jnp.dot(hidden_states, weight.T)


def lm_head_with_bias(
    hidden_states: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
) -> jax.Array:
    """LM head with bias."""
    return jnp.dot(hidden_states, weight.T) + bias


# =============================================================================
# XLA HLO Generation
# =============================================================================

def generate_embedding_hlo(batch: int, seq: int, vocab: int, hidden: int) -> str:
    """Generate XLA HLO for embedding lookup."""
    return f"""
HloModule embedding_tpu

ENTRY main {{
  table = bf16[{vocab},{hidden}] parameter(0)
  indices = s32[{batch},{seq}] parameter(1)
  ROOT output = bf16[{batch},{seq},{hidden}] gather(table, indices),
       offset_dims={{2}}, collapsed_slice_dims={{0}},
       start_index_map={{0}}, index_vector_dim=2,
       slice_sizes={{{1},{hidden}}}
}}
"""
