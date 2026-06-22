/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ROCM

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

template <typename OpType>
class TBElementUnaryFingerprinter {
public:
  __device__
  TBElementUnaryFingerprinter(FPType *input_ptr, FPType *output_ptr,
                               int num_elements, int thread_id, int num_threads) {
    for (int i = thread_id; i < num_elements; i += num_threads) {
      output_ptr[i] = OpType::apply(input_ptr[i]);
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_FINGERPRINT_USE_ROCM
