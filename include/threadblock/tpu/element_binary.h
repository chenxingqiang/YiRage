/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_TPU

#include "threadblock/smem_tensor.h"

namespace yirage {
namespace threadblock {

template <typename OpType>
class TBElementBinaryFingerprinter {
public:
  void compute(FPType *A_ptr, FPType *B_ptr, FPType *C_ptr, int num_elements) {
    for (int i = 0; i < num_elements; i++) {
      C_ptr[i] = OpType::apply(A_ptr[i], B_ptr[i]);
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
