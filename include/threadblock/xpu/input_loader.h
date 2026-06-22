/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_XPU
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
class TBInputLoaderFingerprinter {
public:
  void compute(FPType *global_ptr, FPType *slm_ptr, int num_elements,
               int thread_id = 0, int num_threads = 1) {
    // XPU uses SLM (Shared Local Memory)
    for (int i = thread_id; i < num_elements; i += num_threads) slm_ptr[i] = global_ptr[i];
  }
};
}}
#endif
