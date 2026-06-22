/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_FPGA

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

// FPGA fingerprinting via HLS-compatible reference
class TBMatmulFingerprinter {
public:
  void compute(FPType *A_ptr, FPType *B_ptr, FPType *C_ptr,
               int a_m_size, int c_n_size, int a_k_size) {
    // BRAM-tiled computation reference
    for (int m = 0; m < a_m_size; m++) {
      for (int n = 0; n < c_n_size; n++) {
        FPType result = 0;
        for (int k = 0; k < a_k_size; k++) {
          FPType a = A_ptr[m * a_k_size + k];
          FPType b = B_ptr[k * c_n_size + n];
          result = compute_add_fingerprint(result, compute_mul_fingerprint(a, b));
        }
        C_ptr[m * c_n_size + n] = result;
      }
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
