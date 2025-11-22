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


#include "yirage/kernel/mkl/mkl_kernel_config.h"

#ifdef YIRAGE_BACKEND_MKL_ENABLED

#include <cstdlib>
#include <iostream>

// Try to include MKL headers if available
#ifdef __has_include
#if __has_include(<mkl.h>)
#include <mkl.h>
#define HAS_MKL_HEADER 1
#endif
#endif

namespace yirage {
namespace kernel {
namespace mkl {

bool MKLOptimizer::is_mkl_available() {
#ifdef HAS_MKL_HEADER
  return true;
#else
  // Check if MKL library can be loaded dynamically
  // This is a simplified check
  return false;
#endif
}

std::string MKLOptimizer::get_mkl_version() {
#ifdef HAS_MKL_HEADER
  MKLVersion version;
  mkl_get_version(&version);
  return std::string(version.ProductVersion);
#else
  return "MKL not available";
#endif
}

MKLKernelConfig::ThreadingMode
MKLOptimizer::select_threading_mode(int num_cores, size_t problem_size) {
  // For large problems, use MKL's internal threading
  if (problem_size >= 1024 * 1024) {
    return MKLKernelConfig::ThreadingMode::INTEL_THREADING;
  }
  
  // For medium problems, consider TBB
  if (problem_size >= 256 * 256) {
    return MKLKernelConfig::ThreadingMode::TBB_THREADING;
  }
  
  // For small problems, sequential may be faster
  return MKLKernelConfig::ThreadingMode::SEQUENTIAL;
}

void MKLOptimizer::optimize_for_intel(int m, int n, int k,
                                     MKLKernelConfig &config) {
  // Use CPU optimizer as base
  cpu::CPUOptimizer::optimize_for_cpu(m, n, k, config);
  
  // MKL-specific optimizations
  config.use_mkl_blas = true;
  config.use_mkl_malloc = true;
  
  // Select threading mode
  size_t problem_size = static_cast<size_t>(m) * n * k;
  config.threading_mode = select_threading_mode(config.num_threads,
                                                 problem_size);
  
  // MKL prefers 64-byte alignment for AVX-512
  config.alignment = 64;
  
  // Enable fast matrix multiply
  config.use_fast_mm = true;
  
  // For large matrices, consider packed format
  if (m >= 1024 && n >= 1024) {
    config.use_packed_format = true;
  }
}

void MKLOptimizer::set_mkl_env(MKLKernelConfig const &config) {
#ifdef HAS_MKL_HEADER
  // Set MKL threading layer
  switch (config.threading_mode) {
  case MKLKernelConfig::ThreadingMode::INTEL_THREADING:
    mkl_set_threading_layer(MKL_THREADING_INTEL);
    break;
  case MKLKernelConfig::ThreadingMode::GNU_THREADING:
    mkl_set_threading_layer(MKL_THREADING_GNU);
    break;
  case MKLKernelConfig::ThreadingMode::TBB_THREADING:
    mkl_set_threading_layer(MKL_THREADING_TBB);
    break;
  case MKLKernelConfig::ThreadingMode::SEQUENTIAL:
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    break;
  }
  
  // Set number of threads
  if (config.num_threads > 0) {
    mkl_set_num_threads(config.num_threads);
  }
  
  // Enable fast memory allocator
  if (config.use_mkl_malloc) {
    // MKL's memory allocator is used automatically when linking with MKL
  }
#else
  std::cerr << "MKL library not available" << std::endl;
#endif
}

} // namespace mkl
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MKL_ENABLED





