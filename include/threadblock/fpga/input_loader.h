/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_FPGA
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
class TBInputLoaderFingerprinter {
public:
  void compute(FPType *global_ptr, FPType *bram_ptr, int num_elements) {
    for (int i = 0; i < num_elements; i++) bram_ptr[i] = global_ptr[i];
  }
};
}}
#endif
