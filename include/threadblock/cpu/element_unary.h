/* Copyright 2025 YiRage Team */
#pragma once
#include "type.h"
#ifdef YIRAGE_FINGERPRINT_USE_CPU
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
template <typename OpType>
class TBElementUnaryFingerprinter {
public:
  void compute(type::FPType *input_ptr, type::FPType *output_ptr, int num_elements) {
    #pragma omp parallel for if(num_elements > 1024)
    for (int i = 0; i < num_elements; i++) output_ptr[i] = OpType::apply(input_ptr[i]);
  }
};
}}
#endif
