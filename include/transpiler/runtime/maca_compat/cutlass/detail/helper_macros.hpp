/* Force CUTLASS_HOST_DEVICE on mxcc (-x maca is not __NVCC__). */
#pragma once

#if !defined(__NVCC__) && !defined(__CUDACC_RTC__)
#if defined(__clang__) || defined(YIRAGE_BACKEND_MACA_ENABLED)
#define __NVCC__ 1
#endif
#endif

#include_next <cutlass/detail/helper_macros.hpp>
