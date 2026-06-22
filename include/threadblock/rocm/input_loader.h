/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ROCM

#include "threadblock/smem_tensor.h"

namespace yirage {
namespace threadblock {

class TBInputLoaderFingerprinter {
public:
  __device__
  TBInputLoaderFingerprinter(FPType *global_ptr, FPType *smem_ptr,
                              int num_elements, int thread_id, int num_threads) {
    for (int i = thread_id; i < num_elements; i += num_threads) {
      smem_ptr[i] = global_ptr[i];
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_FINGERPRINT_USE_ROCM
