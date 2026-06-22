/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_XPU

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

// XPU fingerprinting via SYCL-compatible reference (XMX simulation)
class TBMatmulFingerprinter {
public:
  void compute(FPType *A_ptr, FPType *B_ptr, FPType *C_ptr,
               int a_m_size, int c_n_size, int a_k_size,
               int thread_id = 0, int num_threads = 1) {
    // XMX 8x16 tile-based reference
    int num_elements = a_m_size * c_n_size;
    for (int i = thread_id; i < num_elements; i += num_threads) {
      FPType result = 0;
      int m = i / c_n_size;
      int n = i % c_n_size;
      for (int k = 0; k < a_k_size; k++) {
        FPType a = A_ptr[m * a_k_size + k];
        FPType b = B_ptr[k * c_n_size + n];
        result = compute_add_fingerprint(result, compute_mul_fingerprint(a, b));
      }
      C_ptr[i] = result;
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
