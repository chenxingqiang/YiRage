/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ROCM

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

template <typename OpType>
class TBElementBinaryFingerprinter {
public:
  __device__
  TBElementBinaryFingerprinter(FPType *A_ptr, FPType *B_ptr, FPType *C_ptr,
                                int num_elements, int thread_id, int num_threads) {
    for (int i = thread_id; i < num_elements; i += num_threads) {
      C_ptr[i] = OpType::apply(A_ptr[i], B_ptr[i]);
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_FINGERPRINT_USE_ROCM
