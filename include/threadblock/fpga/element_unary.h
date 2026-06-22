/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_FPGA
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
template <typename OpType>
class TBElementUnaryFingerprinter {
public:
  void compute(FPType *input_ptr, FPType *output_ptr, int num_elements) {
    for (int i = 0; i < num_elements; i++) output_ptr[i] = OpType::apply(input_ptr[i]);
  }
};
}}
#endif
