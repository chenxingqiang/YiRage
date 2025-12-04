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
 */

#ifdef YIRAGE_FINGERPRINT_USE_ASCEND

#include "yirage/kernel/device_memory_manager.h"
#include "yirage/utils/ascend_helper.h"

#ifdef __ASCEND__
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

namespace yirage {
namespace kernel {

// Static singleton instance
DeviceMemoryManager *DeviceMemoryManager::singleton = nullptr;

DeviceMemoryManager::DeviceMemoryManager(int num_gpus, int device_id) {
  this->num_gpus = num_gpus;
  this->gpu_id = device_id;
  
#ifdef __ASCEND__
  // Initialize ACL
  aclError ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
    fprintf(stderr, "aclInit failed: %d\n", (int)ret);
  }
  
  // Set device
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    fprintf(stderr, "aclrtSetDevice failed: %d\n", (int)ret);
  }
  
  // Create stream
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) {
    fprintf(stderr, "aclrtCreateStream failed: %d\n", (int)ret);
    stream = nullptr;
  }
#else
  stream = nullptr;
#endif
  
  // Allocate device memory for fingerprint
  dmem_fp_size = config::MAX_DMEM_FP_SIZE;
  smem_fp_size = config::MAX_SMEM_FP_SIZE;
  
#ifdef __ASCEND__
  aclError err = aclrtMalloc(&dmem_fp_ptr, dmem_fp_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (err != ACL_SUCCESS) {
    fprintf(stderr, "Ascend Error: Failed to allocate device memory (%zu bytes): %d\n",
            dmem_fp_size, (int)err);
    dmem_fp_ptr = nullptr;
  }
  
  err = aclrtMalloc(&smem_fp_ptr, smem_fp_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (err != ACL_SUCCESS) {
    fprintf(stderr, "Ascend Error: Failed to allocate L1 buffer (%zu bytes): %d\n",
            smem_fp_size, (int)err);
    smem_fp_ptr = nullptr;
  }
#else
  // CPU fallback
  dmem_fp_ptr = malloc(dmem_fp_size);
  smem_fp_ptr = malloc(smem_fp_size);
#endif
  
  dmem_fp_offset = 0;
  smem_fp_offset = 0;
}

DeviceMemoryManager::~DeviceMemoryManager() {
#ifdef __ASCEND__
  if (dmem_fp_ptr) {
    aclrtFree(dmem_fp_ptr);
    dmem_fp_ptr = nullptr;
  }
  if (smem_fp_ptr) {
    aclrtFree(smem_fp_ptr);
    smem_fp_ptr = nullptr;
  }
  if (stream) {
    aclrtDestroyStream(stream);
    stream = nullptr;
  }
  aclFinalize();
#else
  if (dmem_fp_ptr) {
    free(dmem_fp_ptr);
    dmem_fp_ptr = nullptr;
  }
  if (smem_fp_ptr) {
    free(smem_fp_ptr);
    smem_fp_ptr = nullptr;
  }
#endif
}

DeviceMemoryManager *DeviceMemoryManager::get_instance() {
  if (singleton == nullptr) {
    singleton = new DeviceMemoryManager(1, 0);
  }
  return singleton;
}

void DeviceMemoryManager::set_gpu_device_id(int device_id) {
  this->gpu_id = device_id;
#ifdef __ASCEND__
  aclrtSetDevice(device_id);
#endif
}

type::FPType *DeviceMemoryManager::allocate_dmem_fingerprint(size_t size, bool reset) {
  if (reset) {
    dmem_fp_offset = 0;
  }
  
  if (dmem_fp_offset + size > dmem_fp_size) {
    fprintf(stderr, "Ascend Error: Device memory overflow (requested %zu, available %zu)\n",
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
    fprintf(stderr, "Ascend Error: L1 buffer overflow (requested %zu, available %zu)\n",
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
  aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
#else
  memcpy(dst, src, size);
#endif
}

void DeviceMemoryManager::copy_to_host(void *dst, void const *src, size_t size) {
#ifdef __ASCEND__
  aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
#else
  memcpy(dst, src, size);
#endif
}

void DeviceMemoryManager::synchronize() {
#ifdef __ASCEND__
  if (stream) {
    aclrtSynchronizeStream(stream);
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

