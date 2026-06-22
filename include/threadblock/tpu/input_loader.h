/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_TPU

#include "threadblock/smem_tensor.h"

namespace yirage {
namespace threadblock {

class TBInputLoaderFingerprinter {
public:
  void compute(FPType *global_ptr, FPType *vmem_ptr, int num_elements) {
    // TPU uses VMEM instead of SMEM
    for (int i = 0; i < num_elements; i++) {
      vmem_ptr[i] = global_ptr[i];
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
