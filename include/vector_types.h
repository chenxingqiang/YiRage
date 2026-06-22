/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// ============================================================================
// Vector Types Compatibility Layer
// ============================================================================
// Provides cross-platform vector types (dim3, int2, float4, etc.) for all
// backends. Each GPU platform provides these natively, while CPU/other
// backends use our manual definitions.
// ============================================================================

#if defined(YIRAGE_BACKEND_USE_CUDA) || defined(YIRAGE_BACKEND_CUDA_ENABLED)
// NVIDIA CUDA - native vector types
#include <vector_types.h>

#elif defined(YIRAGE_BACKEND_ROCM_ENABLED)
// AMD ROCm/HIP - provides CUDA-compatible vector types
#include <hip/hip_runtime.h>
// HIP defines dim3, int2, int3, int4, float2, float3, float4, etc.

#elif defined(YIRAGE_BACKEND_MACA_ENABLED)
// MetaX MACA - provides CUDA-compatible vector types
#include <mcr/mc_runtime_api.h>

#elif defined(YIRAGE_BACKEND_XPU_ENABLED)
// Intel/Baidu XPU - use SYCL or manual definitions
#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/sycl.hpp>
// Map SYCL types to CUDA-style names
using int2 = sycl::int2;
using int3 = sycl::int3;
using int4 = sycl::int4;
using float2 = sycl::float2;
using float3 = sycl::float3;
using float4 = sycl::float4;
struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1)
      : x(_x), y(_y), z(_z) {}
};
#else
// XPU host code - use manual definitions (see below)
#define YIRAGE_DEFINE_VECTOR_TYPES_MANUALLY
#endif

#elif defined(YIRAGE_BACKEND_ASCEND_ENABLED)
// Huawei Ascend NPU - no native vector types, use manual definitions
#define YIRAGE_DEFINE_VECTOR_TYPES_MANUALLY

#elif defined(YIRAGE_BACKEND_TPU_ENABLED)
// Google TPU - XLA-based, use manual definitions
#define YIRAGE_DEFINE_VECTOR_TYPES_MANUALLY

#elif defined(YIRAGE_BACKEND_FPGA_ENABLED)
// FPGA backends (Intel/Xilinx) - use manual definitions
#define YIRAGE_DEFINE_VECTOR_TYPES_MANUALLY

#elif defined(YIRAGE_BACKEND_MPS_ENABLED)
// Apple Metal Performance Shaders - use manual definitions
// Metal has its own simd types but we use CPU-compatible types for host code
#define YIRAGE_DEFINE_VECTOR_TYPES_MANUALLY

#else
// CPU backend or unknown - use manual definitions
#define YIRAGE_DEFINE_VECTOR_TYPES_MANUALLY
#endif

// ============================================================================
// Manual Vector Type Definitions
// ============================================================================
// Used for backends that don't provide native CUDA-style vector types
// ============================================================================
#ifdef YIRAGE_DEFINE_VECTOR_TYPES_MANUALLY
struct dim3 {
  unsigned int x, y, z;

  constexpr dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1)
      : x(_x), y(_y), z(_z) {}
};

struct int2 {
  int x, y;

  constexpr int2(int _x = 0, int _y = 0) : x(_x), y(_y) {}
};

struct int3 {
  int x, y, z;

  constexpr int3(int _x = 0, int _y = 0, int _z = 0) : x(_x), y(_y), z(_z) {}
};

struct int4 {
  int x, y, z, w;

  constexpr int4(int _x = 0, int _y = 0, int _z = 0, int _w = 0)
      : x(_x), y(_y), z(_z), w(_w) {}
};

struct float2 {
  float x, y;

  constexpr float2(float _x = 0, float _y = 0) : x(_x), y(_y) {}
};

struct float3 {
  float x, y, z;

  constexpr float3(float _x = 0, float _y = 0, float _z = 0)
      : x(_x), y(_y), z(_z) {}
};

struct float4 {
  float x, y, z, w;

  constexpr float4(float _x = 0, float _y = 0, float _z = 0, float _w = 0)
      : x(_x), y(_y), z(_z), w(_w) {}
};

// Helper functions to create vector types
inline int2 make_int2(int x, int y) {
  return int2(x, y);
}

inline int3 make_int3(int x, int y, int z) {
  return int3(x, y, z);
}

inline float2 make_float2(float x, float y) {
  return float2(x, y);
}

inline float3 make_float3(float x, float y, float z) {
  return float3(x, y, z);
}

inline int4 make_int4(int x, int y, int z, int w) {
  return int4(x, y, z, w);
}

inline float4 make_float4(float x, float y, float z, float w) {
  return float4(x, y, z, w);
}

// Additional unsigned vector types
struct uint2 {
  unsigned int x, y;
  constexpr uint2(unsigned int _x = 0, unsigned int _y = 0) : x(_x), y(_y) {}
};

struct uint3 {
  unsigned int x, y, z;
  constexpr uint3(unsigned int _x = 0, unsigned int _y = 0, unsigned int _z = 0)
      : x(_x), y(_y), z(_z) {}
};

struct uint4 {
  unsigned int x, y, z, w;
  constexpr uint4(unsigned int _x = 0, unsigned int _y = 0, 
                  unsigned int _z = 0, unsigned int _w = 0)
      : x(_x), y(_y), z(_z), w(_w) {}
};

inline uint2 make_uint2(unsigned int x, unsigned int y) {
  return uint2(x, y);
}

inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) {
  return uint3(x, y, z);
}

inline uint4 make_uint4(unsigned int x, unsigned int y, 
                        unsigned int z, unsigned int w) {
  return uint4(x, y, z, w);
}

// Double precision vector types
struct double2 {
  double x, y;
  constexpr double2(double _x = 0, double _y = 0) : x(_x), y(_y) {}
};

struct double3 {
  double x, y, z;
  constexpr double3(double _x = 0, double _y = 0, double _z = 0)
      : x(_x), y(_y), z(_z) {}
};

struct double4 {
  double x, y, z, w;
  constexpr double4(double _x = 0, double _y = 0, double _z = 0, double _w = 0)
      : x(_x), y(_y), z(_z), w(_w) {}
};

inline double2 make_double2(double x, double y) {
  return double2(x, y);
}

inline double3 make_double3(double x, double y, double z) {
  return double3(x, y, z);
}

inline double4 make_double4(double x, double y, double z, double w) {
  return double4(x, y, z, w);
}

#endif // YIRAGE_DEFINE_VECTOR_TYPES_MANUALLY

#undef YIRAGE_DEFINE_VECTOR_TYPES_MANUALLY
