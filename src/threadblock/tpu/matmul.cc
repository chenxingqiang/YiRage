/* Copyright 2025 YiRage Team */

#include "threadblock/graph.h"
#include "threadblock/matmul.h"
#include "threadblock/tpu/matmul.h"

#ifdef YIRAGE_USE_TPU

namespace yirage {
namespace threadblock {
namespace tpu {

// TPU matmul via XLA/Pallas - generates HLO or Pallas kernel
std::string generate_matmul_xla(int m, int n, int k) {
    std::ostringstream hlo;
    hlo << "HloModule matmul_fingerprint\n\n";
    hlo << "ENTRY main {\n";
    hlo << "  a = bf16[" << m << "," << k << "] parameter(0)\n";
    hlo << "  b = bf16[" << k << "," << n << "] parameter(1)\n";
    hlo << "  ROOT dot = bf16[" << m << "," << n << "] dot(a, b), ";
    hlo << "lhs_contracting_dims={1}, rhs_contracting_dims={0}\n";
    hlo << "}\n";
    return hlo.str();
}

std::string generate_matmul_pallas(int m, int n, int k, int tile_m, int tile_n) {
    std::ostringstream pallas;
    pallas << "import jax\n";
    pallas << "from jax.experimental import pallas as pl\n\n";
    pallas << "@pl.pallas_call(\n";
    pallas << "    out_shape=jax.ShapeDtypeStruct((" << m << ", " << n << "), jnp.bfloat16),\n";
    pallas << "    grid=(" << (m / tile_m) << ", " << (n / tile_n) << "),\n";
    pallas << ")\n";
    pallas << "def matmul_kernel(a_ref, b_ref, o_ref):\n";
    pallas << "    o_ref[...] = jnp.dot(a_ref[...], b_ref[...])\n";
    return pallas.str();
}

}  // namespace tpu
}  // namespace threadblock
}  // namespace yirage

#endif
