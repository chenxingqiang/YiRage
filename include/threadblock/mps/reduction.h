/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_MPS

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

class TBReductionFingerprinter {
public:
  void compute(FPType *input_ptr, FPType *output_ptr,
               int outer_size, int reduce_size) {
    for (int i = 0; i < outer_size; i++) {
      FPType result = 0;
      for (int j = 0; j < reduce_size; j++) {
        result = compute_add_fingerprint(result, input_ptr[i * reduce_size + j]);
      }
      output_ptr[i] = result;
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
