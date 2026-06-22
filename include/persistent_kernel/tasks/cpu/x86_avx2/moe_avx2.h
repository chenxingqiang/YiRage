/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file moe_avx2.h
 * @brief AVX2-specific MoE kernel micro-kernel constants and inline helpers.
 *
 * These constants guide the tile sizes used in cpu_moe_linear and
 * cpu_moe_silu_linear when the host CPU supports AVX2 (256-bit SIMD).
 *
 * Register allocation for expert GEMM micro-kernel (FP32):
 *   - SIMD register width: 256 bits = 8 × FP32
 *   - A micro-tile (token rows):     MR_MOE = 4  (4 token rows in registers)
 *   - B micro-tile (output cols):    NR_MOE = 16 (2 × 256-bit = 16 FP32)
 *
 * L1-resident tile for single expert:
 *   - A tile: MR_MOE × tile_k  floats  → kept in registers / L1
 *   - B tile: tile_k × NR_MOE  floats  → loaded into L1 once per tile_k strip
 *   - C tile: MR_MOE × NR_MOE  floats  → stays in AVX registers across k-loop
 *
 * Token-batching strategy:
 *   Group tokens assigned to the same expert in batches of EXPERT_BATCH_AVX2
 *   before running the blocked GEMM.  This maximises B-tile reuse (each row
 *   of the expert weight matrix is loaded once and dotted with multiple
 *   token rows), reducing DRAM bandwidth by up to EXPERT_BATCH_AVX2×.
 */

#include "task_header.h"  // pulls in cpu_common.h and AVX2 constants

namespace yirage {
namespace persistent_kernel {
namespace cpu {
namespace x86_avx2 {

// ---------------------------------------------------------------------------
// MoE-specific micro-kernel dimensions for AVX2
// ---------------------------------------------------------------------------

/// Number of token rows to process simultaneously (A micro-tile height).
/// Must fit in the floating-point and general-purpose registers while
/// leaving room for B/C tiles.
constexpr int MR_MOE = 4;

/// Output-column width of a single B micro-tile.
/// 2 × 256-bit AVX2 registers = 16 FP32 values.
constexpr int NR_MOE = 16;

/// K-dimension tile size.  Chosen so that one A-strip (MR_MOE × TILE_K_MOE)
/// and one B-strip (TILE_K_MOE × NR_MOE) together fit in the 32 KB L1 cache:
///   bytes = (MR_MOE + NR_MOE) × TILE_K_MOE × 4 ≤ L1_SIZE/2
///         = (4 + 16) × 256 × 4 = 20 480 bytes  (fits comfortably)
constexpr int TILE_K_MOE = 256;

/// Preferred number of tokens to group per expert before running the GEMM.
/// Larger batches improve weight-matrix reuse; smaller batches reduce the
/// overhead of the token-reordering step for sparsely activated experts.
constexpr int EXPERT_BATCH_AVX2 = 8;

/// Unroll factor for the K-loop inside the micro-kernel.
/// 4× unroll hides FMA latency (≈4 cycles on Skylake and later Intel
/// micro-architectures) by allowing the out-of-order engine to issue
/// independent FMA chains back-to-back.
constexpr int K_UNROLL_AVX2 = 4;

// ---------------------------------------------------------------------------
// SiLU activation helper (scalar, used inside the fused kernel)
// ---------------------------------------------------------------------------

/// Scalar SiLU: x * sigmoid(x) = x / (1 + exp(-x)).
/// The AVX2 path vectorises this loop automatically via -O3 + -mavx2.
inline float silu_f32(float x) noexcept {
    return x * (1.0f / (1.0f + __builtin_expf(-x)));
}

}  // namespace x86_avx2
}  // namespace cpu
}  // namespace persistent_kernel
}  // namespace yirage
