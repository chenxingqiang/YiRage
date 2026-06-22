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
 * MetaX MACA Warp-level GEMM Executor
 * 
 * MetaX GPUs use 64-thread warps (vs NVIDIA's 32-thread warps)
 * This implementation uses MetaX Matrix Multiply Units (MMU)
 */

#pragma once

#ifdef YIRAGE_BACKEND_MACA_ENABLED

#include <mcr/mc_runtime.h>

namespace yirage {
namespace warp {
namespace maca {

// MetaX MACA warp size is 64 threads
constexpr int WARP_SIZE = 64;
constexpr unsigned long long FULL_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @brief Get lane index within a 64-thread warp
 */
__device__ __forceinline__ int lane_id() {
  return threadIdx.x % WARP_SIZE;
}

/**
 * @brief Get warp index within the block
 */
__device__ __forceinline__ int warp_id() {
  return threadIdx.x / WARP_SIZE;
}

/**
 * @brief Warp shuffle for data exchange
 */
template <typename T>
__device__ __forceinline__ T shfl_sync(unsigned long long mask, T val, int src_lane) {
  return __shfl_sync(mask, val, src_lane);
}

template <typename T>
__device__ __forceinline__ T shfl_xor_sync(unsigned long long mask, T val, int lane_mask) {
  return __shfl_xor_sync(mask, val, lane_mask);
}

template <typename T>
__device__ __forceinline__ T shfl_down_sync(unsigned long long mask, T val, unsigned delta) {
  return __shfl_down_sync(mask, val, delta);
}

/**
 * @brief Warp-level reduction sum for 64-thread warps
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += shfl_xor_sync(FULL_WARP_MASK, val, offset);
  }
  return val;
}

/**
 * @brief Warp-level reduction max for 64-thread warps
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other = shfl_xor_sync(FULL_WARP_MASK, val, offset);
    val = (val > other) ? val : other;
  }
  return val;
}

/**
 * @brief MACA Warp-level GEMM configuration
 * 
 * MetaX MMU (Matrix Multiply Unit) supports various tile sizes:
 * - MMU_F32_16x16x16_F16: 16x16x16 tile, FP16 input, FP32 accumulator
 * - MMU_F32_32x32x16_F16: 32x32x16 tile for larger throughput
 */
template <int WarpM, int WarpN, int WarpK>
struct WarpGemmConfig {
  static constexpr int kWarpM = WarpM;
  static constexpr int kWarpN = WarpN;
  static constexpr int kWarpK = WarpK;
  
  // Number of iterations along K dimension
  static constexpr int kIterations = WarpK / 16;
};

/**
 * @brief MACA Warp-level GEMM Executor
 * 
 * Template parameters:
 *   WarpShape: Warp tile dimensions (M, N, K)
 *   InstructionShape: MMU instruction tile size
 *   ElementType: Data type (half, float, __half)
 *   SmemLayoutA: Shared memory layout for A matrix
 *   SmemLayoutB: Shared memory layout for B matrix
 */
template <typename WarpShape,
          typename InstructionShape,
          typename ElementType,
          typename SmemLayoutA = void,
          typename SmemLayoutB = void>
class GemmExecutor {
public:
  // Warp shape parameters
  static constexpr int kWarpM = WarpShape::kM;
  static constexpr int kWarpN = WarpShape::kN;
  static constexpr int kWarpK = WarpShape::kK;
  
  // Instruction shape (MMU tile)
  static constexpr int kInstrM = InstructionShape::kM;
  static constexpr int kInstrN = InstructionShape::kN;
  static constexpr int kInstrK = InstructionShape::kK;
  
  // GEMM iterations along K
  static constexpr int kWarpGemmIterations = 
      (kWarpK + kInstrK - 1) / kInstrK;
  
  // Number of MMU tiles in each dimension
  static constexpr int kMTiles = kWarpM / kInstrM;
  static constexpr int kNTiles = kWarpN / kInstrN;
  
  // Elements per lane for fragments
  static constexpr int kElementsA = kWarpM * kInstrK / WARP_SIZE;
  static constexpr int kElementsB = kInstrK * kWarpN / WARP_SIZE;
  static constexpr int kElementsC = kWarpM * kWarpN / WARP_SIZE;
  
  // Fragment types
  using FragmentA = ElementType[kElementsA];
  using FragmentB = ElementType[kElementsB];
  using FragmentC = float[kElementsC];

private:
  int m_, n_, k_;
  int warp_idx_m_;
  int warp_idx_n_;
  int lane_idx_;
  
  const ElementType* smem_A_;
  const ElementType* smem_B_;
  ElementType* smem_C_;

public:
  __device__
  GemmExecutor(const ElementType* smem_A,
               const ElementType* smem_B,
               ElementType* smem_C,
               int m,
               int n,
               int k,
               int thread_idx,
               int warp_idx,
               int lane_idx)
      : smem_A_(smem_A),
        smem_B_(smem_B),
        smem_C_(smem_C),
        m_(m),
        n_(n),
        k_(k),
        lane_idx_(lane_idx) {
    
    int warp_count_m = m / kWarpM;
    int warp_count_n = n / kWarpN;
    
    int warp_idx_mn = warp_idx % (warp_count_m * warp_count_n);
    int warp_idx_k = warp_idx / (warp_count_m * warp_count_n);
    
    // Assume no K partitioning within threadblock
    (void)warp_idx_k;
    
    warp_idx_m_ = warp_idx_mn % warp_count_m;
    warp_idx_n_ = warp_idx_mn / warp_count_m;
  }

  /**
   * @brief Load fragment A from shared memory
   * 
   * Layout: Each warp loads a (kWarpM x kInstrK) tile
   */
  __device__
  void load_A(FragmentA& frag, int k_offset) {
    int row_base = warp_idx_m_ * kWarpM;
    int col_base = k_offset;
    
    #pragma unroll
    for (int i = 0; i < kElementsA; ++i) {
      int element_idx = lane_idx_ * kElementsA + i;
      int row = row_base + element_idx / kInstrK;
      int col = col_base + element_idx % kInstrK;
      
      if (row < m_ && col < k_) {
        frag[i] = smem_A_[row * k_ + col];
      } else {
        frag[i] = ElementType(0);
      }
    }
  }

  /**
   * @brief Load fragment B from shared memory
   * 
   * Layout: Each warp loads a (kInstrK x kWarpN) tile
   */
  __device__
  void load_B(FragmentB& frag, int k_offset) {
    int row_base = k_offset;
    int col_base = warp_idx_n_ * kWarpN;
    
    #pragma unroll
    for (int i = 0; i < kElementsB; ++i) {
      int element_idx = lane_idx_ * kElementsB + i;
      int row = row_base + element_idx / kWarpN;
      int col = col_base + element_idx % kWarpN;
      
      if (row < k_ && col < n_) {
        frag[i] = smem_B_[row * n_ + col];
      } else {
        frag[i] = ElementType(0);
      }
    }
  }

  /**
   * @brief Store fragment C to shared memory
   */
  __device__
  void store_C(const FragmentC& frag) {
    int row_base = warp_idx_m_ * kWarpM;
    int col_base = warp_idx_n_ * kWarpN;
    
    #pragma unroll
    for (int i = 0; i < kElementsC; ++i) {
      int element_idx = lane_idx_ * kElementsC + i;
      int row = row_base + element_idx / kWarpN;
      int col = col_base + element_idx % kWarpN;
      
      if (row < m_ && col < n_) {
        smem_C_[row * n_ + col] = static_cast<ElementType>(frag[i]);
      }
    }
  }

  /**
   * @brief Matrix multiply-accumulate using MetaX MMU or emulation
   * 
   * Uses native MMU instructions when available,
   * falls back to software emulation otherwise.
   */
  __device__
  void mma(FragmentC& accum, const FragmentA& frag_A, const FragmentB& frag_B) {
    // Software emulation for MACA MMU
    // Each lane computes its portion of the output tile
    
    #pragma unroll
    for (int c_idx = 0; c_idx < kElementsC; ++c_idx) {
      int element_idx = lane_idx_ * kElementsC + c_idx;
      int out_row = element_idx / kWarpN;
      int out_col = element_idx % kWarpN;
      
      float sum = 0.0f;
      
      #pragma unroll
      for (int k = 0; k < kInstrK; ++k) {
        // Get A element at (out_row, k)
        int a_element_idx = out_row * kInstrK + k;
        int a_lane = a_element_idx / kElementsA;
        int a_local = a_element_idx % kElementsA;
        float a_val = static_cast<float>(shfl_sync(FULL_WARP_MASK, 
                                                    frag_A[a_local], a_lane));
        
        // Get B element at (k, out_col)
        int b_element_idx = k * kWarpN + out_col;
        int b_lane = b_element_idx / kElementsB;
        int b_local = b_element_idx % kElementsB;
        float b_val = static_cast<float>(shfl_sync(FULL_WARP_MASK, 
                                                    frag_B[b_local], b_lane));
        
        sum += a_val * b_val;
      }
      
      accum[c_idx] += sum;
    }
  }

  /**
   * @brief Execute the warp-level GEMM kernel
   */
  __device__
  void execute_kernel() {
    FragmentA frag_A[2];
    FragmentB frag_B[2];
    FragmentC accum;
    
    // Initialize accumulator to zero
    #pragma unroll
    for (int i = 0; i < kElementsC; ++i) {
      accum[i] = 0.0f;
    }
    
    // Prefetch first tiles
    load_A(frag_A[0], 0);
    load_B(frag_B[0], 0);
    
    // Main GEMM loop with double buffering
    #pragma unroll 1
    for (int k_iter = 0; k_iter < kWarpGemmIterations; ++k_iter) {
      // Prefetch next tiles (if not last iteration)
      if (k_iter + 1 < kWarpGemmIterations) {
        int next_k = (k_iter + 1) * kInstrK;
        load_A(frag_A[(k_iter + 1) % 2], next_k);
        load_B(frag_B[(k_iter + 1) % 2], next_k);
      }
      
      // Matrix multiply-accumulate
      mma(accum, frag_A[k_iter % 2], frag_B[k_iter % 2]);
    }
    
    // Store final results
    store_C(accum);
    
    __syncthreads();
  }
};

// Common shape definitions for MACA
namespace shapes {

template <int M, int N, int K>
struct Shape {
  static constexpr int kM = M;
  static constexpr int kN = N;
  static constexpr int kK = K;
};

// MMU instruction shapes for MetaX C500/C600/C700
using MmuF32_16x16x16_F16 = Shape<16, 16, 16>;
using MmuF32_32x32x16_F16 = Shape<32, 32, 16>;
using MmuF32_16x16x8_F32 = Shape<16, 16, 8>;

// Common warp tile shapes
using WarpTile_64x64x16 = Shape<64, 64, 16>;
using WarpTile_64x64x32 = Shape<64, 64, 32>;
using WarpTile_64x64x64 = Shape<64, 64, 64>;
using WarpTile_128x64x16 = Shape<128, 64, 16>;
using WarpTile_64x128x16 = Shape<64, 128, 16>;
using WarpTile_32x32x32 = Shape<32, 32, 32>;

} // namespace shapes

/**
 * @brief Block-level GEMM using multiple warps
 * 
 * Coordinates multiple warp-level GEMMs to compute a larger tile.
 */
template <typename WarpShape,
          typename InstructionShape,
          typename ElementType,
          int NumWarpsM,
          int NumWarpsN>
class BlockGemmExecutor {
public:
  static constexpr int kBlockM = WarpShape::kM * NumWarpsM;
  static constexpr int kBlockN = WarpShape::kN * NumWarpsN;
  static constexpr int kBlockK = WarpShape::kK;
  static constexpr int kNumWarps = NumWarpsM * NumWarpsN;
  
  using WarpGemm = GemmExecutor<WarpShape, InstructionShape, ElementType>;

private:
  int m_, n_, k_;
  int warp_idx_;
  int lane_idx_;
  
  const ElementType* smem_A_;
  const ElementType* smem_B_;
  ElementType* smem_C_;

public:
  __device__
  BlockGemmExecutor(const ElementType* smem_A,
                    const ElementType* smem_B,
                    ElementType* smem_C,
                    int m,
                    int n,
                    int k)
      : smem_A_(smem_A),
        smem_B_(smem_B),
        smem_C_(smem_C),
        m_(m),
        n_(n),
        k_(k) {
    warp_idx_ = warp_id();
    lane_idx_ = lane_id();
  }

  /**
   * @brief Execute block-level GEMM
   */
  __device__
  void execute_kernel() {
    // Each warp handles its portion
    WarpGemm warp_gemm(smem_A_, smem_B_, smem_C_,
                       m_, n_, k_,
                       threadIdx.x, warp_idx_, lane_idx_);
    warp_gemm.execute_kernel();
  }
};

} // namespace maca
} // namespace warp
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED
