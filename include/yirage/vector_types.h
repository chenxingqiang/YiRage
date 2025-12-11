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

#ifdef YIRAGE_BACKEND_USE_CUDA
#include <vector_types.h>
#elif defined(YIRAGE_BACKEND_MACA_ENABLED)
// MACA provides its own vector types in mc_runtime.h
// Don't define them here to avoid conflicts
// They will be included via maca_backend.h when needed
#include <mcr/mc_runtime_api.h>
#else
// CPU/Ascend backend - define vector types manually
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
#endif
