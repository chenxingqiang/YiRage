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
 *
 * This file is part of YiRage (Yi Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */


#pragma once

#include "yirage/kernel/cpu/cpu_kernel_config.h"

#ifdef YIRAGE_BACKEND_MKL_ENABLED

namespace yirage {
namespace kernel {
namespace mkl {

/**
 * @brief MKL (Intel Math Kernel Library) configuration
 * 
 * Extends CPU configuration with MKL-specific options
 */
struct MKLKernelConfig : public cpu::CPUKernelConfig {
  // MKL BLAS configuration
  bool use_mkl_blas = true;
  bool use_mkl_sparse = false;
  
  // Threading mode
  enum class ThreadingMode {
    SEQUENTIAL,
    INTEL_THREADING,  // MKL's own threading
    GNU_THREADING,    // OpenMP
    TBB_THREADING     // Intel TBB
  };
  ThreadingMode threading_mode = ThreadingMode::INTEL_THREADING;
  
  // BLAS interface
  enum class BLASInterface {
    CBLAS,      // C BLAS interface
    LAPACKE,    // LAPACK interface
    SCALAPACK   // Distributed LAPACK
  };
  BLASInterface blas_interface = BLASInterface::CBLAS;
  
  // MKL optimization hints
  bool use_fast_mm = true;  // Fast matrix multiply
  bool use_packed_format = false;  // Packed matrix format
  
  // Memory allocation
  bool use_mkl_malloc = true;  // Use MKL's aligned allocator
  int alignment = 64;  // 64-byte alignment for AVX-512

  MKLKernelConfig() { 
    backend_type = type::BT_MKL;
    // MKL works best with its own threading
    use_openmp = false;
  }
};

/**
 * @brief MKL kernel optimizer
 */
class MKLOptimizer {
public:
  /**
   * @brief Check if MKL is available
   * @return true if MKL library is available
   */
  static bool is_mkl_available();

  /**
   * @brief Get MKL version
   * @return MKL version string
   */
  static std::string get_mkl_version();

  /**
   * @brief Select optimal threading mode
   * @param num_cores Number of CPU cores
   * @param problem_size Problem size
   * @return Optimal threading mode
   */
  static MKLKernelConfig::ThreadingMode
  select_threading_mode(int num_cores, size_t problem_size);

  /**
   * @brief Optimize for Intel CPU with MKL
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param config Configuration to optimize
   */
  static void optimize_for_intel(int m, int n, int k,
                                 MKLKernelConfig &config);

  /**
   * @brief Set MKL environment variables
   * @param config Configuration
   */
  static void set_mkl_env(MKLKernelConfig const &config);
};

} // namespace mkl
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MKL_ENABLED





