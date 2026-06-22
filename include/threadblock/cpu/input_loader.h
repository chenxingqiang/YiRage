/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_CPU
#include "threadblock/smem_tensor.h"
#include "type.h"
#include <cstring>
namespace yirage { namespace threadblock {
class TBInputLoaderFingerprinter {
public:
  void compute(type::FPType *src_ptr, type::FPType *dst_ptr, int num_elements) {
    std::memcpy(dst_ptr, src_ptr, num_elements * sizeof(type::FPType));
  }
};
}}
#endif
