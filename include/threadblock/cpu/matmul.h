/* Copyright 2025 YiRage Team */
#pragma once
#include "type.h"

#ifdef YIRAGE_FINGERPRINT_USE_CPU

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

// CPU fingerprinting with optional OpenMP parallelization
class TBMatmulFingerprinter {
public:
  void compute(type::FPType *A_ptr, type::FPType *B_ptr, type::FPType *C_ptr,
               int a_m_size, int c_n_size, int a_k_size) {
    #pragma omp parallel for collapse(2) if(a_m_size * c_n_size > 1024)
    for (int m = 0; m < a_m_size; m++) {
      for (int n = 0; n < c_n_size; n++) {
        type::FPType result = 0;
        for (int k = 0; k < a_k_size; k++) {
          type::FPType a = A_ptr[m * a_k_size + k];
          type::FPType b = B_ptr[k * c_n_size + n];
          result = utils::compute_add_fingerprint(result, utils::compute_mul_fingerprint(a, b));
        }
        C_ptr[m * c_n_size + n] = result;
      }
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
