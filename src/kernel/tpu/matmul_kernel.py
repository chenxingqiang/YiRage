# Copyright 2025 YiRage Team
# TPU Matrix Multiplication Kernels via JAX/Pallas

"""
TPU GEMM kernels using Pallas and XLA.
Optimized for MXU (128x128 systolic array).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial


# =============================================================================
# Pallas-based GEMM kernel
# =============================================================================

def matmul_pallas_kernel(
    a_ref,      # [TILE_M, TILE_K]
    b_ref,      # [TILE_K, TILE_N]
    o_ref,      # [TILE_M, TILE_N]
):
    """MXU-optimized matrix multiplication tile."""
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...])


def matmul_bf16(
    a: jax.Array,  # [M, K]
    b: jax.Array,  # [K, N]
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
) -> jax.Array:
    """BF16 matrix multiplication optimized for TPU MXU."""
    m, k = a.shape
    _, n = b.shape
    
    return pl.pallas_call(
        matmul_pallas_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid=(m // tile_m, n // tile_n),
        in_specs=[
            pl.BlockSpec((tile_m, tile_k), lambda i, j: (i, 0)),
            pl.BlockSpec((tile_k, tile_n), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((tile_m, tile_n), lambda i, j: (i, j)),
    )(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16))


# =============================================================================
# Batched GEMM
# =============================================================================

def batched_matmul_kernel(
    a_ref,      # [TILE_M, TILE_K]
    b_ref,      # [TILE_K, TILE_N]
    o_ref,      # [TILE_M, TILE_N]
):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...])


@partial(jax.vmap, in_axes=(0, 0))
def batched_matmul_bf16(a: jax.Array, b: jax.Array) -> jax.Array:
    """Batched BF16 GEMM for transformer attention."""
    return matmul_bf16(a, b)


# =============================================================================
# XLA HLO Generation
# =============================================================================

def generate_matmul_hlo(m: int, n: int, k: int, dtype: str = "bf16") -> str:
    """Generate XLA HLO for TPU matmul."""
    return f"""
HloModule matmul_tpu

ENTRY main {{
  a = {dtype}[{m},{k}] parameter(0)
  b = {dtype}[{k},{n}] parameter(1)
  ROOT dot = {dtype}[{m},{n}] dot(a, b), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}
}}
"""


def generate_fused_matmul_bias_relu_hlo(m: int, n: int, k: int) -> str:
    """Generate fused GEMM + bias + ReLU for TPU."""
    return f"""
HloModule fused_gemm_bias_relu

ENTRY main {{
  a = bf16[{m},{k}] parameter(0)
  b = bf16[{k},{n}] parameter(1)
  bias = bf16[{n}] parameter(2)
  
  dot = bf16[{m},{n}] dot(a, b), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}
  bias_broadcast = bf16[{m},{n}] broadcast(bias), dimensions={{1}}
  add = bf16[{m},{n}] add(dot, bias_broadcast)
  zero = bf16[] constant(0)
  zero_broadcast = bf16[{m},{n}] broadcast(zero)
  ROOT relu = bf16[{m},{n}] maximum(add, zero_broadcast)
}}
"""
