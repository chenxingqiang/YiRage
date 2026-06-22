/* Copyright 2025 YiRage Team */
#pragma once
#include "type.h"
#ifdef YIRAGE_FINGERPRINT_USE_CPU
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
template <typename OpType>
class TBElementBinaryFingerprinter {
public:
  void compute(type::FPType *A_ptr, type::FPType *B_ptr, type::FPType *C_ptr, int num_elements) {
    #pragma omp parallel for if(num_elements > 1024)
    for (int i = 0; i < num_elements; i++) C_ptr[i] = OpType::apply(A_ptr[i], B_ptr[i]);
  }
};
}}
#endif
