/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_XPU
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
template <typename OpType>
class TBElementBinaryFingerprinter {
public:
  void compute(FPType *A_ptr, FPType *B_ptr, FPType *C_ptr, int num_elements,
               int thread_id = 0, int num_threads = 1) {
    for (int i = thread_id; i < num_elements; i += num_threads)
      C_ptr[i] = OpType::apply(A_ptr[i], B_ptr[i]);
  }
};
}}
#endif
