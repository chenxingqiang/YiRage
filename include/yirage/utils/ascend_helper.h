/* Copyright 2025 Chen Xingqiang (YiRage Project)
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

#include <cstdio>
#include <cstdlib>
#include "yirage/config.h"
#include "yirage/type.h"

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED

// ACL (Ascend Computing Language) headers
#ifdef __ASCEND__
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

namespace yirage {
namespace utils {

// Error checking macro for ACL runtime
#ifdef __ASCEND__
#define checkACL(status)                                                       \
  do {                                                                         \
    aclError err = (status);                                                   \
    if (err != ACL_SUCCESS) {                                                  \
      fprintf(stderr, "ACL Error %d at %s:%d\n", (int)err, __FILE__,           \
              __LINE__);                                                       \
      fprintf(stderr, "Error: %s\n", aclGetRecentErrMsg());                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define checkACLNoFail(status)                                                 \
  do {                                                                         \
    aclError err = (status);                                                   \
    if (err != ACL_SUCCESS) {                                                  \
      fprintf(stderr, "ACL Warning %d at %s:%d\n", (int)err, __FILE__,         \
              __LINE__);                                                       \
    }                                                                          \
  } while (0)
#else
// Stub macros when ACL not available
#define checkACL(status)       (void)(status)
#define checkACLNoFail(status) (void)(status)
#endif

// Ascend device constants
// Ascend 910/910B: 32 AI Cores
// Ascend 310P: 8 AI Cores
constexpr int ASCEND_910_AI_CORES = 32;
constexpr int ASCEND_310P_AI_CORES = 8;

// L1 Buffer sizes (per AI Core)
constexpr size_t ASCEND_910_L1_SIZE = 256 * 1024;   // 256 KB
constexpr size_t ASCEND_910B_L1_SIZE = 512 * 1024;  // 512 KB
constexpr size_t ASCEND_310P_L1_SIZE = 128 * 1024;  // 128 KB

// Cube unit tile sizes (native operations)
constexpr int ASCEND_CUBE_TILE_M = 16;
constexpr int ASCEND_CUBE_TILE_N = 16;
constexpr int ASCEND_CUBE_TILE_K = 16;

// Vector unit constants
constexpr int ASCEND_VECTOR_WIDTH = 256;  // bits

/**
 * @brief Ascend device type enumeration
 */
enum class AscendDeviceType {
  ASCEND_910 = 0,    // Original Ascend 910
  ASCEND_910B = 1,   // Enhanced Ascend 910B (Atlas 800)
  ASCEND_310P = 2,   // Inference card
  UNKNOWN = -1
};

/**
 * @brief Get AI Core count for device type
 * @param device_type Ascend device type
 * @return Number of AI Cores
 */
inline int get_ai_core_count(AscendDeviceType device_type) {
  switch (device_type) {
    case AscendDeviceType::ASCEND_910:
    case AscendDeviceType::ASCEND_910B:
      return ASCEND_910_AI_CORES;
    case AscendDeviceType::ASCEND_310P:
      return ASCEND_310P_AI_CORES;
    default:
      return ASCEND_910_AI_CORES;  // Default to 910
  }
}

/**
 * @brief Get L1 buffer size for device type
 * @param device_type Ascend device type
 * @return L1 buffer size in bytes
 */
inline size_t get_l1_buffer_size(AscendDeviceType device_type) {
  switch (device_type) {
    case AscendDeviceType::ASCEND_910:
      return ASCEND_910_L1_SIZE;
    case AscendDeviceType::ASCEND_910B:
      return ASCEND_910B_L1_SIZE;
    case AscendDeviceType::ASCEND_310P:
      return ASCEND_310P_L1_SIZE;
    default:
      return ASCEND_910_L1_SIZE;
  }
}

/**
 * @brief Check if tile size is optimal for Cube operations
 * @param tile_m M dimension
 * @param tile_n N dimension
 * @param tile_k K dimension
 * @return true if aligned for Cube
 */
inline bool is_cube_aligned(int tile_m, int tile_n, int tile_k) {
  return (tile_m % ASCEND_CUBE_TILE_M == 0) &&
         (tile_n % ASCEND_CUBE_TILE_N == 0) &&
         (tile_k % ASCEND_CUBE_TILE_K == 0);
}

/**
 * @brief FP pointer list for multi-device support
 */
struct FpPointerList {
  yirage::type::FPType *ptrs[yirage::config::MAX_NUM_DEVICES];
};

#ifdef __ASCEND__
/**
 * @brief Initialize ACL runtime
 * @param device_id Device to initialize
 * @return true on success
 */
inline bool acl_init(int device_id = 0) {
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
    fprintf(stderr, "aclInit failed with error: %d\n", (int)ret);
    return false;
  }
  
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    fprintf(stderr, "aclrtSetDevice failed with error: %d\n", (int)ret);
    return false;
  }
  
  return true;
}

/**
 * @brief Finalize ACL runtime
 */
inline void acl_finalize() {
  aclFinalize();
}

/**
 * @brief Allocate device memory
 * @param size Size in bytes
 * @return Device pointer or nullptr
 */
inline void* acl_malloc(size_t size) {
  void* ptr = nullptr;
  aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    fprintf(stderr, "aclrtMalloc failed with error: %d\n", (int)ret);
    return nullptr;
  }
  return ptr;
}

/**
 * @brief Free device memory
 * @param ptr Device pointer
 */
inline void acl_free(void* ptr) {
  if (ptr) {
    aclrtFree(ptr);
  }
}

/**
 * @brief Copy data to device
 * @param dst Device pointer
 * @param src Host pointer
 * @param size Size in bytes
 * @return true on success
 */
inline bool acl_memcpy_h2d(void* dst, const void* src, size_t size) {
  aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  return ret == ACL_SUCCESS;
}

/**
 * @brief Copy data from device
 * @param dst Host pointer
 * @param src Device pointer
 * @param size Size in bytes
 * @return true on success
 */
inline bool acl_memcpy_d2h(void* dst, const void* src, size_t size) {
  aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
  return ret == ACL_SUCCESS;
}

/**
 * @brief Synchronize device
 */
inline void acl_synchronize() {
  aclrtSynchronizeDevice();
}
#endif // __ASCEND__

} // namespace utils
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

