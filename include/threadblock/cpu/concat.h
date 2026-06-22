/* Copyright 2025 YiRage Team */
#pragma once
#include "type.h"
#ifdef YIRAGE_FINGERPRINT_USE_CPU
#include "threadblock/smem_tensor.h"
#include <cstring>
namespace yirage { namespace threadblock {
class TBConcatFingerprinter {
public:
  void compute(type::FPType **input_ptrs, int *input_sizes, int num_inputs, type::FPType *output_ptr) {
    int offset = 0;
    for (int n = 0; n < num_inputs; n++) {
      std::memcpy(output_ptr + offset, input_ptrs[n], input_sizes[n] * sizeof(type::FPType));
      offset += input_sizes[n];
    }
  }
};
}}
#endif
