/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ROCM

#include "threadblock/smem_tensor.h"
#include "utils/rocm_helper.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

class TBMatmulFingerprinter {
public:
  __device__
  TBMatmulFingerprinter(FPType *A_ptr, FPType *B_ptr, FPType *C_ptr,
                        int a_m_size, int c_n_size, int a_k_size,
                        int thread_id, int num_threads) {
    int num_elements = a_m_size * c_n_size;
    int b_n_size = c_n_size;
    for (int i = thread_id; i < num_elements; i += num_threads) {
      FPType result = 0;
      int m = i / c_n_size;
      int n = i % c_n_size;
      for (int k = 0; k < a_k_size; k++) {
        FPType a = A_ptr[m * a_k_size + k];
        FPType b = B_ptr[k * b_n_size + n];
        FPType ab = compute_mul_fingerprint(a, b);
        result = compute_add_fingerprint(result, ab);
      }
      C_ptr[i] = result;
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_FINGERPRINT_USE_ROCM
