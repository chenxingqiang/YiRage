/* Copyright 2025 YiRage Project
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
 *
 * Ascend Device Memory Manager Implementation
 * Manages device memory allocation on Huawei Ascend NPU using ACL runtime
 * 
 * IMPORTANT: Like CUDA, fingerprint memory is allocated on NPU (not CPU)
 * This ensures consistency with the actual kernel execution environment.
 */

#ifdef YIRAGE_FINGERPRINT_USE_ASCEND

#include "kernel/device_memory_manager.h"
#include "utils/math_utils.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

#ifdef __ASCEND__
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

namespace yirage {
namespace kernel {

using namespace yirage::type;
using namespace yirage::config;

// Static singleton instance
DeviceMemoryManager *DeviceMemoryManager::singleton = nullptr;

// Flag to track if we initialized ACL ourselves
static bool acl_initialized_by_us = false;

DeviceMemoryManager::DeviceMemoryManager(int num_gpus, int device_id) {
  this->num_gpus = num_gpus;
  this->gpu_id = device_id;
  
  dmem_fp_size = config::MAX_DMEM_FP_SIZE;
  smem_fp_size = config::MAX_SMEM_FP_SIZE;
  
#ifdef __ASCEND__
  // Initialize ACL if not already done (e.g., by torch_npu)
  aclError ret = aclInit(nullptr);
  if (ret == ACL_SUCCESS) {
    acl_initialized_by_us = true;
    fprintf(stderr, "[Ascend] ACL initialized by DeviceMemoryManager\n");
  } else if (ret == ACL_ERROR_REPEAT_INITIALIZE) {
    // ACL already initialized (by torch_npu), that's fine
    acl_initialized_by_us = false;
  } else {
    fprintf(stderr, "[Ascend] Warning: aclInit failed with error %d\n", (int)ret);
  }
  
  // Set device
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    fprintf(stderr, "[Ascend] Warning: aclrtSetDevice(%d) failed with error %d\n", 
            device_id, (int)ret);
  }
  
  // Create stream for fingerprint operations
  ret = aclrtCreateStream(reinterpret_cast<aclrtStream*>(&stream));
  if (ret != ACL_SUCCESS) {
    fprintf(stderr, "[Ascend] Warning: aclrtCreateStream failed with error %d\n", (int)ret);
    stream = nullptr;
  }
  
  // Allocate fingerprint memory on NPU (like CUDA does on GPU)
  ret = aclrtMalloc(&dmem_fp_ptr, dmem_fp_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS || dmem_fp_ptr == nullptr) {
    fprintf(stderr, "[Ascend] Error: Failed to allocate NPU memory for fingerprint (%zu bytes), error=%d\n",
            dmem_fp_size, (int)ret);
    fprintf(stderr, "[Ascend] Falling back to CPU memory for fingerprint\n");
    dmem_fp_ptr = malloc(dmem_fp_size);
    use_npu_memory = false;
  } else {
    use_npu_memory = true;
    fprintf(stderr, "[Ascend] Allocated %zu bytes on NPU for fingerprint\n", dmem_fp_size);
  }
  
  // Allocate shared memory (L1 buffer simulation) on NPU
  ret = aclrtMalloc(&smem_fp_ptr, smem_fp_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS || smem_fp_ptr == nullptr) {
    fprintf(stderr, "[Ascend] Warning: Failed to allocate NPU smem (%zu bytes), using CPU\n", smem_fp_size);
    smem_fp_ptr = malloc(smem_fp_size);
  }
  
  // Allocate lookup tables on NPU
  void *exp_table_ptr = nullptr, *div_p_ptr = nullptr, *div_q_ptr = nullptr;
  void *sqrt_p_ptr = nullptr, *sqrt_q_ptr = nullptr;
  
  aclrtMalloc(&exp_table_ptr, FP_Q * sizeof(FPType), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&div_p_ptr, FP_P * sizeof(FPType), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&div_q_ptr, FP_Q * sizeof(FPType), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&sqrt_p_ptr, FP_P * sizeof(FPType), ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMalloc(&sqrt_q_ptr, FP_Q * sizeof(FPType), ACL_MEM_MALLOC_HUGE_FIRST);
  
  exp_lookup_table = reinterpret_cast<FPType*>(exp_table_ptr);
  div_p_lookup_table = reinterpret_cast<FPType*>(div_p_ptr);
  div_q_lookup_table = reinterpret_cast<FPType*>(div_q_ptr);
  sqrt_p_lookup_table = reinterpret_cast<FPType*>(sqrt_p_ptr);
  sqrt_q_lookup_table = reinterpret_cast<FPType*>(sqrt_q_ptr);
  
  // Initialize lookup tables on host then copy to device
  FPType *host_exp = new FPType[FP_Q];
  FPType *host_div_p = new FPType[FP_P];
  FPType *host_div_q = new FPType[FP_Q];
  FPType *host_sqrt_p = new FPType[FP_P];
  FPType *host_sqrt_q = new FPType[FP_Q];
  
  // Initialize exp lookup table
  host_exp[0] = 1;
  for (int i = 1; i < FP_Q; ++i) {
    host_exp[i] = (host_exp[i - 1] * FP_EXP_BASE) % FP_P;
  }
  
  // Initialize div lookup tables
  host_div_p[0] = 1;
  for (uint16_t i = 1; i < FP_P; ++i) {
    host_div_p[i] = yirage::mod_inverse(i, FP_P);
  }
  host_div_q[0] = 1;
  for (uint16_t i = 1; i < FP_Q; ++i) {
    host_div_q[i] = yirage::mod_inverse(i, FP_Q);
  }
  
  // Initialize sqrt lookup tables
  for (uint16_t i = 0; i < FP_P; ++i) {
    host_sqrt_p[i] = yirage::mod_power(i, static_cast<uint16_t>((FP_P + 1) / 4), FP_P);
  }
  for (uint16_t i = 0; i < FP_Q; ++i) {
    host_sqrt_q[i] = yirage::mod_power(i, static_cast<uint16_t>((FP_Q + 1) / 4), FP_Q);
  }
  
  // Copy to NPU
  aclrtMemcpy(exp_lookup_table, FP_Q * sizeof(FPType), 
              host_exp, FP_Q * sizeof(FPType), ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(div_p_lookup_table, FP_P * sizeof(FPType), 
              host_div_p, FP_P * sizeof(FPType), ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(div_q_lookup_table, FP_Q * sizeof(FPType), 
              host_div_q, FP_Q * sizeof(FPType), ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(sqrt_p_lookup_table, FP_P * sizeof(FPType), 
              host_sqrt_p, FP_P * sizeof(FPType), ACL_MEMCPY_HOST_TO_DEVICE);
  aclrtMemcpy(sqrt_q_lookup_table, FP_Q * sizeof(FPType), 
              host_sqrt_q, FP_Q * sizeof(FPType), ACL_MEMCPY_HOST_TO_DEVICE);
  
  delete[] host_exp;
  delete[] host_div_p;
  delete[] host_div_q;
  delete[] host_sqrt_p;
  delete[] host_sqrt_q;
  
#else
  // Non-Ascend build: use CPU memory as fallback
  stream = nullptr;
  use_npu_memory = false;
  dmem_fp_ptr = malloc(dmem_fp_size);
  smem_fp_ptr = malloc(smem_fp_size);
  
  if (!dmem_fp_ptr) {
    fprintf(stderr, "[Ascend] Error: Failed to allocate CPU memory for fingerprint (%zu bytes)\n",
            dmem_fp_size);
  }
  if (!smem_fp_ptr) {
    fprintf(stderr, "[Ascend] Error: Failed to allocate CPU memory for smem fingerprint (%zu bytes)\n",
            smem_fp_size);
  }
  
  // Initialize lookup tables on CPU
  auto initialize_exp_lookup_table =
      [](FPType *table, int size, int base, int modulus) {
        table[0] = 1;
        for (int i = 1; i < size; ++i) {
          table[i] = (table[i - 1] * base) % modulus;
        }
      };

  auto initialize_div_lookup_table = [](FPType *table, int size, int modulus) {
    table[0] = 1;
    for (int i = 1; i < size; ++i) {
      table[i] = yirage::mod_inverse(i, modulus);
    }
  };

  auto initialize_sqrt_lookup_table = [](FPType *table, int size, int modulus) {
    assert(modulus % 4 == 3 &&
           "Modulus must be of the form 4k + 3 for square roots to exist");
    for (int i = 0; i < size; ++i) {
      table[i] = yirage::mod_power(i, (modulus + 1) / 4, modulus);
    }
  };

  exp_lookup_table = new FPType[FP_Q];
  div_p_lookup_table = new FPType[FP_P];
  div_q_lookup_table = new FPType[FP_Q];
  sqrt_p_lookup_table = new FPType[FP_P];
  sqrt_q_lookup_table = new FPType[FP_Q];

  initialize_exp_lookup_table(exp_lookup_table, FP_Q, FP_EXP_BASE, FP_P);
  initialize_div_lookup_table(div_p_lookup_table, FP_P, FP_P);
  initialize_div_lookup_table(div_q_lookup_table, FP_Q, FP_Q);
  initialize_sqrt_lookup_table(sqrt_p_lookup_table, FP_P, FP_P);
  initialize_sqrt_lookup_table(sqrt_q_lookup_table, FP_Q, FP_Q);
#endif
  
  dmem_fp_offset = 0;
  smem_fp_offset = 0;
  
  // Initialize fp_base_ptr array (used by fingerprint functions)
  for (int i = 0; i < yirage::config::MAX_NUM_DEVICES; ++i) {
    fp_base_ptr[i] = reinterpret_cast<char *>(dmem_fp_ptr);
  }
  stensor_fp_base_ptr = reinterpret_cast<char *>(smem_fp_ptr);
}

DeviceMemoryManager::~DeviceMemoryManager() {
#ifdef __ASCEND__
  // Free NPU memory
  if (dmem_fp_ptr) {
    if (use_npu_memory) {
      aclrtFree(dmem_fp_ptr);
    } else {
      free(dmem_fp_ptr);
    }
    dmem_fp_ptr = nullptr;
  }
  if (smem_fp_ptr) {
    aclrtFree(smem_fp_ptr);
    smem_fp_ptr = nullptr;
  }
  
  // Free lookup tables
  if (exp_lookup_table) aclrtFree(exp_lookup_table);
  if (div_p_lookup_table) aclrtFree(div_p_lookup_table);
  if (div_q_lookup_table) aclrtFree(div_q_lookup_table);
  if (sqrt_p_lookup_table) aclrtFree(sqrt_p_lookup_table);
  if (sqrt_q_lookup_table) aclrtFree(sqrt_q_lookup_table);
  
  // Destroy stream
  if (stream) {
    aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream));
    stream = nullptr;
  }
  
  // Only finalize ACL if we initialized it
  if (acl_initialized_by_us) {
    aclFinalize();
  }
#else
  // CPU fallback cleanup
  if (dmem_fp_ptr) {
    free(dmem_fp_ptr);
    dmem_fp_ptr = nullptr;
  }
  if (smem_fp_ptr) {
    free(smem_fp_ptr);
    smem_fp_ptr = nullptr;
  }
  
  delete[] exp_lookup_table;
  delete[] div_p_lookup_table;
  delete[] div_q_lookup_table;
  delete[] sqrt_p_lookup_table;
  delete[] sqrt_q_lookup_table;
#endif
}

DeviceMemoryManager *DeviceMemoryManager::get_instance() {
  if (singleton == nullptr) {
    singleton = new DeviceMemoryManager(1, 0);
  }
  return singleton;
}

void DeviceMemoryManager::set_gpu_device_id(int device_id) {
  gpu_id = device_id;
#ifdef __ASCEND__
  aclrtSetDevice(device_id);
#endif
}

type::FPType *DeviceMemoryManager::allocate_dmem_fingerprint(size_t size, bool reset) {
  if (reset) {
    dmem_fp_offset = 0;
  }
  
  if (dmem_fp_offset + size > dmem_fp_size) {
    fprintf(stderr, "[Ascend] Error: Device memory overflow (requested %zu, available %zu)\n",
            size, dmem_fp_size - dmem_fp_offset);
    return nullptr;
  }
  
  type::FPType *ptr = reinterpret_cast<type::FPType*>(
      static_cast<char*>(dmem_fp_ptr) + dmem_fp_offset);
  dmem_fp_offset += size;
  return ptr;
}

type::FPType *DeviceMemoryManager::allocate_smem_fingerprint(size_t size, bool reset) {
  if (reset) {
    smem_fp_offset = 0;
  }
  
  if (smem_fp_offset + size > smem_fp_size) {
    fprintf(stderr, "[Ascend] Error: L1 buffer overflow (requested %zu, available %zu)\n",
            size, smem_fp_size - smem_fp_offset);
    return nullptr;
  }
  
  type::FPType *ptr = reinterpret_cast<type::FPType*>(
      static_cast<char*>(smem_fp_ptr) + smem_fp_offset);
  smem_fp_offset += size;
  return ptr;
}

void DeviceMemoryManager::copy_to_device(void *dst, void const *src, size_t size) {
#ifdef __ASCEND__
  if (use_npu_memory) {
    aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  } else {
    memcpy(dst, src, size);
  }
#else
  memcpy(dst, src, size);
#endif
}

void DeviceMemoryManager::copy_to_host(void *dst, void const *src, size_t size) {
#ifdef __ASCEND__
  if (use_npu_memory) {
    aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
  } else {
    memcpy(dst, src, size);
  }
#else
  memcpy(dst, src, size);
#endif
}

void DeviceMemoryManager::synchronize() {
#ifdef __ASCEND__
  if (stream) {
    aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
  } else {
    aclrtSynchronizeDevice();
  }
#endif
}

void *DeviceMemoryManager::get_stream() {
  return stream;
}

} // namespace kernel
} // namespace yirage

// C interface for Cython
extern "C" {

void cython_set_gpu_device_id(int device_id) {
  yirage::kernel::DeviceMemoryManager::get_instance()->set_gpu_device_id(device_id);
}

}

#endif // YIRAGE_FINGERPRINT_USE_ASCEND
