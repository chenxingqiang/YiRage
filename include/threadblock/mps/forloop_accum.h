/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_MPS

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

class TBForloopAccumFingerprinter {
public:
  void compute(FPType *accum_ptr, FPType *input_ptr, int num_elements) {
    for (int i = 0; i < num_elements; i++) {
      accum_ptr[i] = compute_add_fingerprint(accum_ptr[i], input_ptr[i]);
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
