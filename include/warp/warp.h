/* Copyright 2023-2025 CMU, YiRage Project
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
 *
 * Unified Warp-level Operations Header
 * 
 * Provides a unified interface for warp-level (or wavefront-level) operations
 * across different GPU backends:
 *   - CUDA: 32-thread warps with CUTLASS
 *   - ROCm: 64-thread wavefronts with MFMA
 *   - MACA: 64-thread warps with MMU
 */

#pragma once

// CUDA warp-level operations (32 threads)
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include "warp/cuda/matmul.h"
#endif

// ROCm wavefront-level operations (64 threads)
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include "warp/rocm/matmul.h"
#endif

// MACA warp-level operations (64 threads)
#ifdef YIRAGE_BACKEND_MACA_ENABLED
#include "warp/maca/matmul.h"
#endif

namespace yirage {
namespace warp {

/**
 * @brief Get the warp/wavefront size for the current backend
 */
constexpr int get_warp_size() {
#if defined(YIRAGE_BACKEND_CUDA_ENABLED) || defined(YIRAGE_BACKEND_CUDNN_ENABLED)
  return 32;  // NVIDIA warp size
#elif defined(YIRAGE_BACKEND_ROCM_ENABLED)
  return 64;  // AMD wavefront size
#elif defined(YIRAGE_BACKEND_MACA_ENABLED)
  return 64;  // MetaX warp size
#else
  return 32;  // Default
#endif
}

/**
 * @brief Backend type for warp operations
 */
enum class WarpBackend {
  CUDA,    // NVIDIA CUDA (32-thread warps)
  ROCm,    // AMD ROCm/HIP (64-thread wavefronts)
  MACA,    // MetaX MACA (64-thread warps)
  Unknown
};

/**
 * @brief Get the current warp backend
 */
constexpr WarpBackend get_warp_backend() {
#if defined(YIRAGE_BACKEND_CUDA_ENABLED)
  return WarpBackend::CUDA;
#elif defined(YIRAGE_BACKEND_ROCM_ENABLED)
  return WarpBackend::ROCm;
#elif defined(YIRAGE_BACKEND_MACA_ENABLED)
  return WarpBackend::MACA;
#else
  return WarpBackend::Unknown;
#endif
}

/**
 * @brief Check if the current backend supports warp-level tensor operations
 */
constexpr bool has_tensor_cores() {
#if defined(YIRAGE_BACKEND_CUDA_ENABLED)
  return true;  // NVIDIA Tensor Cores (Volta+)
#elif defined(YIRAGE_BACKEND_ROCM_ENABLED)
  return true;  // AMD Matrix Cores (CDNA)
#elif defined(YIRAGE_BACKEND_MACA_ENABLED)
  return true;  // MetaX MMU
#else
  return false;
#endif
}

/**
 * @brief Common shape definition template
 */
template <int M, int N, int K>
struct Shape {
  static constexpr int kM = M;
  static constexpr int kN = N;
  static constexpr int kK = K;
};

// Common warp tile shapes that work across backends
namespace common_shapes {

// Small tiles for high occupancy
using WarpTile_16x16x16 = Shape<16, 16, 16>;
using WarpTile_32x32x16 = Shape<32, 32, 16>;

// Medium tiles for balanced performance
using WarpTile_64x64x16 = Shape<64, 64, 16>;
using WarpTile_64x64x32 = Shape<64, 64, 32>;

// Large tiles for high throughput
using WarpTile_128x128x16 = Shape<128, 128, 16>;
using WarpTile_128x128x32 = Shape<128, 128, 32>;

} // namespace common_shapes

} // namespace warp
} // namespace yirage
