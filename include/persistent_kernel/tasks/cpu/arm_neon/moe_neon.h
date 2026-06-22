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
 * @file moe_neon.h
 * @brief ARM NEON-specific MoE kernel micro-kernel constants and inline helpers.
 *
 * These constants guide the tile sizes used in cpu_moe_linear and
 * cpu_moe_silu_linear when the host is an ARM core with NEON support
 * (Cortex-A series, Apple M-series, etc.).
 *
 * Register allocation for expert GEMM micro-kernel (FP32):
 *   - SIMD register width: 128 bits = 4 × FP32
 *   - A micro-tile (token rows):     MR_MOE = 4  (4 token rows)
 *   - B micro-tile (output cols):    NR_MOE = 8  (2 × 128-bit = 8 FP32)
 *
 * Token-batching strategy:
 *   Group tokens for the same expert in batches of EXPERT_BATCH_NEON to
 *   maximise NEON vfmaq_f32 throughput and weight-tile reuse.
 *
 * Apple M-series note:
 *   The M-series has 256-bit NEON execution units (two 128-bit pipes issued
 *   together).  Doubling NR_MOE to 16 on those cores can yield additional
 *   throughput; however, the conservative value of 8 is safe everywhere.
 */

#include "../task_header.h"  // pulls in cpu_common.h and NEON constants

namespace yirage {
namespace persistent_kernel {
namespace cpu {
namespace arm_neon {

// ---------------------------------------------------------------------------
// MoE-specific micro-kernel dimensions for NEON
// ---------------------------------------------------------------------------

/// Number of token rows processed simultaneously (A micro-tile height).
constexpr int MR_MOE = 4;

/// Output-column width of one B micro-tile (2 × 128-bit NEON vectors).
constexpr int NR_MOE = 8;

/// K-dimension tile size.
/// (MR_MOE + NR_MOE) × TILE_K_MOE × 4 = 12 × 256 × 4 = 12 288 bytes,
/// well within the 64 KB L1 cache on Cortex-A and Apple M cores.
constexpr int TILE_K_MOE = 256;

/// Preferred number of tokens to group per expert before running the GEMM.
/// Smaller than the AVX2 value because NEON register pressure is tighter.
constexpr int EXPERT_BATCH_NEON = 4;

/// Unroll factor for the K-loop — matches vfmaq_f32 throughput pipeline depth.
constexpr int K_UNROLL_NEON = 4;

// ---------------------------------------------------------------------------
// SiLU activation helper (scalar)
// ---------------------------------------------------------------------------

/// Scalar SiLU: x * sigmoid(x).
/// The NEON back-end auto-vectorises the SiLU loop via -O3 -mfpu=neon.
inline float silu_f32(float x) noexcept {
    return x * (1.0f / (1.0f + __builtin_expf(-x)));
}

}  // namespace arm_neon
}  // namespace cpu
}  // namespace persistent_kernel
}  // namespace yirage
