/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ROCM

#include "threadblock/smem_tensor.h"

namespace yirage {
namespace threadblock {

class TBOutputSaverFingerprinter {
public:
  __device__
  TBOutputSaverFingerprinter(FPType *smem_ptr, FPType *global_ptr,
                              int num_elements, int thread_id, int num_threads) {
    for (int i = thread_id; i < num_elements; i += num_threads) {
      global_ptr[i] = smem_ptr[i];
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_FINGERPRINT_USE_ROCM
