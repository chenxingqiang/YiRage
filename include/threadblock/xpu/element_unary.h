/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_XPU
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
template <typename OpType>
class TBElementUnaryFingerprinter {
public:
  void compute(FPType *input_ptr, FPType *output_ptr, int num_elements,
               int thread_id = 0, int num_threads = 1) {
    for (int i = thread_id; i < num_elements; i += num_threads)
      output_ptr[i] = OpType::apply(input_ptr[i]);
  }
};
}}
#endif
