# Copyright 2025 YiRage Team
# TPU Softmax Kernels

"""
TPU softmax kernels using Pallas.
Numerically stable implementations.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# =============================================================================
# Pallas Softmax Kernel
# =============================================================================

def softmax_kernel(
    x_ref,      # [TILE_M, TILE_N]
    o_ref,      # [TILE_M, TILE_N]
):
    """Row-wise softmax with numerical stability."""
    x = x_ref[...]
    x_max = jnp.max(x, axis=-1, keepdims=True)
    x_exp = jnp.exp(x - x_max)
    o_ref[...] = x_exp / jnp.sum(x_exp, axis=-1, keepdims=True)


def softmax_bf16(x: jax.Array, axis: int = -1) -> jax.Array:
    """Numerically stable softmax for TPU."""
    x_max = jnp.max(x, axis=axis, keepdims=True)
    x_exp = jnp.exp(x - x_max)
    return x_exp / jnp.sum(x_exp, axis=axis, keepdims=True)


# =============================================================================
# Online Softmax (for Flash Attention)
# =============================================================================

def online_softmax_update(
    m_prev: jax.Array,
    l_prev: jax.Array,
    x_new: jax.Array,
):
    """
    Online softmax update for streaming computation.
    Used in Flash Attention.
    """
    m_new = jnp.maximum(m_prev, jnp.max(x_new, axis=-1))
    
    scale_prev = jnp.exp(m_prev - m_new)
    p_new = jnp.exp(x_new - m_new[:, None])
    
    l_new = scale_prev * l_prev + jnp.sum(p_new, axis=-1)
    
    return m_new, l_new, p_new


# =============================================================================
# Causal Softmax
# =============================================================================

def causal_softmax_bf16(x: jax.Array) -> jax.Array:
    """Softmax with causal mask for autoregressive models."""
    seq_len = x.shape[-1]
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1) * -1e9
    x_masked = x + mask
    return softmax_bf16(x_masked)


# =============================================================================
# XLA HLO Generation
# =============================================================================

def generate_softmax_hlo(m: int, n: int) -> str:
    """Generate XLA HLO for softmax."""
    return f"""
HloModule softmax_tpu

ENTRY main {{
  x = bf16[{m},{n}] parameter(0)
  
  # Max reduction
  x_max = bf16[{m}] reduce(x, bf16[] constant(-inf)), dimensions={{1}}, to_apply=max_comp
  x_max_bc = bf16[{m},{n}] broadcast(x_max), dimensions={{0}}
  
  # Subtract max
  x_shifted = bf16[{m},{n}] subtract(x, x_max_bc)
  
  # Exp
  x_exp = bf16[{m},{n}] exponential(x_shifted)
  
  # Sum reduction
  x_sum = bf16[{m}] reduce(x_exp, bf16[] constant(0)), dimensions={{1}}, to_apply=add_comp
  x_sum_bc = bf16[{m},{n}] broadcast(x_sum), dimensions={{0}}
  
  # Divide
  ROOT output = bf16[{m},{n}] divide(x_exp, x_sum_bc)
}}

max_comp {{
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT max = bf16[] maximum(a, b)
}}

add_comp {{
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT add = bf16[] add(a, b)
}}
"""
