/* CUDA bfloat16 shim for MetaX mxcc (CUTLASS bfloat16 types). */
#pragma once

#include "maca_bfloat16.h"
#include "maca_bfloat16.hpp"

typedef __maca_bfloat16 __nv_bfloat16;
typedef __maca_bfloat162 __nv_bfloat162;
typedef __maca_bfloat16_raw __nv_bfloat16_raw;
typedef __nv_bfloat16 nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;
