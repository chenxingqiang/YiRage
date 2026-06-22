/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_FPGA
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
class TBConcatFingerprinter {
public:
  void compute(FPType **input_ptrs, int *input_sizes, int num_inputs, FPType *output_ptr) {
    int offset = 0;
    for (int n = 0; n < num_inputs; n++) {
      for (int i = 0; i < input_sizes[n]; i++) output_ptr[offset + i] = input_ptrs[n][i];
      offset += input_sizes[n];
    }
  }
};
}}
#endif
