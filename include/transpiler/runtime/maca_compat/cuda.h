/* Minimal CUDA driver API shim for CUTLASS cute headers on MetaX mxcc. */
#pragma once

#include <cstdint>

#ifndef CUresult
typedef int CUresult;
#endif
#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS 0
#endif

typedef enum CUtensorMapDataType_enum {
  CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
  CU_TENSOR_MAP_DATA_TYPE_UINT16,
  CU_TENSOR_MAP_DATA_TYPE_UINT32,
  CU_TENSOR_MAP_DATA_TYPE_INT32,
  CU_TENSOR_MAP_DATA_TYPE_UINT64,
  CU_TENSOR_MAP_DATA_TYPE_INT64,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
  CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
  CU_TENSOR_MAP_DATA_TYPE_INT8,
} CUtensorMapDataType;

typedef enum CUtensorMapInterleave_enum {
  CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
} CUtensorMapInterleave;

typedef enum CUtensorMapSwizzle_enum {
  CU_TENSOR_MAP_SWIZZLE_NONE = 0,
} CUtensorMapSwizzle;

typedef enum CUtensorMapL2promotion_enum {
  CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
} CUtensorMapL2promotion;

typedef enum CUtensorMapFloatOOBfill_enum {
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
} CUtensorMapFloatOOBfill;

struct CUtensorMap_st {
  alignas(64) char opaque[128];
};
typedef CUtensorMap_st CUtensorMap;

#ifndef CUdeviceptr
typedef std::uintptr_t CUdeviceptr;
#endif
