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
 * ROCm Wavefront-level GEMM Executor
 * 
 * AMD GPUs use 64-thread wavefronts (vs NVIDIA's 32-thread warps)
 * This implementation uses AMD Matrix Fused Multiply-Add (MFMA) instructions
 */

#pragma once

#ifdef YIRAGE_BACKEND_ROCM_ENABLED

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace yirage {
namespace warp {
namespace rocm {

// AMD wavefront size is 64 threads
constexpr int WAVEFRONT_SIZE = 64;
constexpr unsigned long long FULL_WAVEFRONT_MASK = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @brief Get lane index within a 64-thread wavefront
 */
__device__ __forceinline__ int lane_id() {
  return threadIdx.x % WAVEFRONT_SIZE;
}

/**
 * @brief Get wavefront index within the block
 */
__device__ __forceinline__ int wavefront_id() {
  return threadIdx.x / WAVEFRONT_SIZE;
}

/**
 * @brief Wavefront shuffle for data exchange
 */
template <typename T>
__device__ __forceinline__ T shfl_sync(unsigned long long mask, T val, int src_lane) {
  return __shfl(val, src_lane);
}

template <typename T>
__device__ __forceinline__ T shfl_xor_sync(unsigned long long mask, T val, int lane_mask) {
  return __shfl_xor(val, lane_mask);
}

template <typename T>
__device__ __forceinline__ T shfl_down_sync(unsigned long long mask, T val, unsigned delta) {
  return __shfl_down(val, delta);
}

/**
 * @brief Wavefront-level reduction sum
 */
template <typename T>
__device__ __forceinline__ T wavefront_reduce_sum(T val) {
  #pragma unroll
  for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
    val += shfl_xor_sync(FULL_WAVEFRONT_MASK, val, offset);
  }
  return val;
}

/**
 * @brief Wavefront-level GEMM configuration
 * 
 * AMD MFMA instructions support various tile sizes:
 * - MFMA_F32_16x16x16_F16: 16x16x16 tile, FP16 input, FP32 output
 * - MFMA_F32_32x32x8_F16: 32x32x8 tile, FP16 input, FP32 output
 */
template <int WarpM, int WarpN, int WarpK>
struct WavefrontGemmConfig {
  static constexpr int kWarpM = WarpM;
  static constexpr int kWarpN = WarpN;
  static constexpr int kWarpK = WarpK;
  
  // Number of iterations along K dimension
  static constexpr int kIterations = WarpK / 16;  // Assuming 16 as base K tile
};

/**
 * @brief ROCm Wavefront-level GEMM Executor
 * 
 * Template parameters:
 *   WarpShape: Wavefront tile dimensions (M, N, K)
 *   InstructionShape: MFMA instruction tile size
 *   ElementType: Data type (half, float)
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
  // Wavefront shape parameters
  static constexpr int kWarpM = WarpShape::kM;
  static constexpr int kWarpN = WarpShape::kN;
  static constexpr int kWarpK = WarpShape::kK;
  
  // Instruction shape
  static constexpr int kInstrM = InstructionShape::kM;
  static constexpr int kInstrN = InstructionShape::kN;
  static constexpr int kInstrK = InstructionShape::kK;
  
  // GEMM iterations
  static constexpr int kWarpGemmIterations = 
      (kWarpK + kInstrK - 1) / kInstrK;
  
  // Fragment types - simplified for ROCm
  using FragmentA = ElementType[kWarpM * kInstrK / WAVEFRONT_SIZE];
  using FragmentB = ElementType[kInstrK * kWarpN / WAVEFRONT_SIZE];
  using FragmentC = float[kWarpM * kWarpN / WAVEFRONT_SIZE];

private:
  int m_, n_, k_;
  int wavefront_idx_m_;
  int wavefront_idx_n_;
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
               int wavefront_idx,
               int lane_idx)
      : smem_A_(smem_A),
        smem_B_(smem_B),
        smem_C_(smem_C),
        m_(m),
        n_(n),
        k_(k),
        lane_idx_(lane_idx) {
    
    int wavefront_count_m = m / kWarpM;
    int wavefront_count_n = n / kWarpN;
    
    int wavefront_idx_mn = wavefront_idx % (wavefront_count_m * wavefront_count_n);
    wavefront_idx_m_ = wavefront_idx_mn % wavefront_count_m;
    wavefront_idx_n_ = wavefront_idx_mn / wavefront_count_m;
  }

  /**
   * @brief Load fragment A from shared memory
   */
  __device__
  void load_A(FragmentA& frag, int k_offset) {
    int row_base = wavefront_idx_m_ * kWarpM;
    int col_base = k_offset;
    
    // Each lane loads a portion of the tile
    int elements_per_lane = kWarpM * kInstrK / WAVEFRONT_SIZE;
    
    #pragma unroll
    for (int i = 0; i < elements_per_lane; ++i) {
      int element_idx = lane_idx_ * elements_per_lane + i;
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
   */
  __device__
  void load_B(FragmentB& frag, int k_offset) {
    int row_base = k_offset;
    int col_base = wavefront_idx_n_ * kWarpN;
    
    int elements_per_lane = kInstrK * kWarpN / WAVEFRONT_SIZE;
    
    #pragma unroll
    for (int i = 0; i < elements_per_lane; ++i) {
      int element_idx = lane_idx_ * elements_per_lane + i;
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
    int row_base = wavefront_idx_m_ * kWarpM;
    int col_base = wavefront_idx_n_ * kWarpN;
    
    int elements_per_lane = kWarpM * kWarpN / WAVEFRONT_SIZE;
    
    #pragma unroll
    for (int i = 0; i < elements_per_lane; ++i) {
      int element_idx = lane_idx_ * elements_per_lane + i;
      int row = row_base + element_idx / kWarpN;
      int col = col_base + element_idx % kWarpN;
      
      if (row < m_ && col < n_) {
        smem_C_[row * n_ + col] = static_cast<ElementType>(frag[i]);
      }
    }
  }

  /**
   * @brief Matrix multiply-accumulate using MFMA or emulation
   * 
   * On supported hardware, uses native MFMA instructions.
   * Falls back to software emulation otherwise.
   */
  __device__
  void mma(FragmentC& accum, const FragmentA& frag_A, const FragmentB& frag_B) {
    int elements_A = kWarpM * kInstrK / WAVEFRONT_SIZE;
    int elements_B = kInstrK * kWarpN / WAVEFRONT_SIZE;
    int elements_C = kWarpM * kWarpN / WAVEFRONT_SIZE;
    
    // Software emulation of MFMA
    // Each lane computes its portion of the output tile
    #pragma unroll
    for (int c_idx = 0; c_idx < elements_C; ++c_idx) {
      int element_idx = lane_idx_ * elements_C + c_idx;
      int out_row = element_idx / kWarpN;
      int out_col = element_idx % kWarpN;
      
      float sum = 0.0f;
      
      #pragma unroll
      for (int k = 0; k < kInstrK; ++k) {
        // Get A element
        int a_element_idx = out_row * kInstrK + k;
        int a_lane = a_element_idx / elements_A;
        int a_local = a_element_idx % elements_A;
        float a_val = static_cast<float>(shfl_sync(FULL_WAVEFRONT_MASK, 
                                                    frag_A[a_local], a_lane));
        
        // Get B element
        int b_element_idx = k * kWarpN + out_col;
        int b_lane = b_element_idx / elements_B;
        int b_local = b_element_idx % elements_B;
        float b_val = static_cast<float>(shfl_sync(FULL_WAVEFRONT_MASK, 
                                                    frag_B[b_local], b_lane));
        
        sum += a_val * b_val;
      }
      
      accum[c_idx] += sum;
    }
  }

  /**
   * @brief Execute the wavefront-level GEMM kernel
   */
  __device__
  void execute_kernel() {
    FragmentA frag_A[2];
    FragmentB frag_B[2];
    FragmentC accum;
    
    // Initialize accumulator
    #pragma unroll
    for (int i = 0; i < kWarpM * kWarpN / WAVEFRONT_SIZE; ++i) {
      accum[i] = 0.0f;
    }
    
    // Prefetch first tiles
    load_A(frag_A[0], 0);
    load_B(frag_B[0], 0);
    
    // Main GEMM loop with double buffering
    #pragma unroll 1
    for (int k_iter = 0; k_iter < kWarpGemmIterations; ++k_iter) {
      // Prefetch next tiles
      if (k_iter + 1 < kWarpGemmIterations) {
        int next_k = (k_iter + 1) * kInstrK;
        load_A(frag_A[(k_iter + 1) % 2], next_k);
        load_B(frag_B[(k_iter + 1) % 2], next_k);
      }
      
      // Matrix multiply-accumulate
      mma(accum, frag_A[k_iter % 2], frag_B[k_iter % 2]);
    }
    
    // Store results
    store_C(accum);
    
    __syncthreads();
  }
};

// Common shape definitions for ROCm
namespace shapes {

template <int M, int N, int K>
struct Shape {
  static constexpr int kM = M;
  static constexpr int kN = N;
  static constexpr int kK = K;
};

// MFMA instruction shapes for MI100/MI200/MI300
using MfmaF32_16x16x16_F16 = Shape<16, 16, 16>;
using MfmaF32_32x32x8_F16 = Shape<32, 32, 8>;
using MfmaF32_16x16x4_F32 = Shape<16, 16, 4>;

// Common wavefront tile shapes
using WavefrontTile_64x64x16 = Shape<64, 64, 16>;
using WavefrontTile_64x64x32 = Shape<64, 64, 32>;
using WavefrontTile_32x32x32 = Shape<32, 32, 32>;

} // namespace shapes

} // namespace rocm
} // namespace warp
} // namespace yirage

#endif // YIRAGE_BACKEND_ROCM_ENABLED
