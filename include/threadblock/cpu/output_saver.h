/* Copyright 2025 YiRage Team */
#pragma once
#include "type.h"
#ifdef YIRAGE_FINGERPRINT_USE_CPU
#include "threadblock/smem_tensor.h"
#include <cstring>
namespace yirage { namespace threadblock {
class TBOutputSaverFingerprinter {
public:
  void compute(type::FPType *src_ptr, type::FPType *dst_ptr, int num_elements) {
    std::memcpy(dst_ptr, src_ptr, num_elements * sizeof(type::FPType));
  }
};
}}
#endif
