/* Shadow cute/config.hpp: mxcc -x maca does not predefine __CUDACC__. */
#pragma once

#if !defined(__CUDACC__) && !defined(_NVHPC_CUDA)
#ifndef YIRAGE_MACA_FORCED_CUDACC
#define YIRAGE_MACA_FORCED_CUDACC 1
#define __CUDACC__ 1
#endif
#endif

#include_next <cute/config.hpp>
