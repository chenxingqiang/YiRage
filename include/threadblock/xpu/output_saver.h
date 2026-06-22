/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_XPU
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
class TBOutputSaverFingerprinter {
public:
  void compute(FPType *slm_ptr, FPType *global_ptr, int num_elements,
               int thread_id = 0, int num_threads = 1) {
    for (int i = thread_id; i < num_elements; i += num_threads) global_ptr[i] = slm_ptr[i];
  }
};
}}
#endif
