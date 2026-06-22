/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_TPU

#include "threadblock/smem_tensor.h"

namespace yirage {
namespace threadblock {

class TBOutputSaverFingerprinter {
public:
  void compute(FPType *vmem_ptr, FPType *global_ptr, int num_elements) {
    for (int i = 0; i < num_elements; i++) {
      global_ptr[i] = vmem_ptr[i];
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
