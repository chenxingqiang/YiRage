/* CUDA fp16 shim for MetaX mxcc (CUTLASS half types). */
#pragma once

#include "maca_fp16.h"
#include "maca_fp16.hpp"

typedef __half __nv_half;
typedef __half2 __nv_half2;
typedef __half half;
typedef __half2 half2;
