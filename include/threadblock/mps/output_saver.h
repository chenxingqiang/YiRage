/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_MPS

#include "threadblock/smem_tensor.h"

namespace yirage {
namespace threadblock {

class TBOutputSaverFingerprinter {
public:
  void compute(FPType *smem_ptr, FPType *global_ptr, int num_elements) {
    for (int i = 0; i < num_elements; i++) {
      global_ptr[i] = smem_ptr[i];
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
