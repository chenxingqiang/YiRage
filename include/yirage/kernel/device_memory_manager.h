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

#include "yirage/kernel/customized.h"
#include "yirage/kernel/matmul.h"
#include "yirage/kernel/operator.h"
#include <unordered_map>
#include <vector>

#ifdef YIRAGE_FINGERPRINT_USE_CUDA
#include <cublas_v2.h>
#endif
#ifdef YIRAGE_FINGERPRINT_USE_MACA
#include <mcr/mc_runtime.h>
#endif
#ifdef YIRAGE_FINGERPRINT_USE_ASCEND
// Ascend ACL headers included in source files when __ASCEND__ is defined
#endif
namespace yirage {
namespace kernel {

class DeviceMemoryManager {
public:
  int num_gpus;
#if defined(YIRAGE_FINGERPRINT_USE_CUDA)
  DeviceMemoryManager(int device_id, int num_gpus);
  static void set_gpu_device_id(int gpu_id);
  int gpu_id;
  cudaStream_t stream[yirage::config::MAX_NUM_DEVICES];
  cublasHandle_t blas[yirage::config::MAX_NUM_DEVICES];
#elif defined(YIRAGE_FINGERPRINT_USE_MACA)
  DeviceMemoryManager(int num_gpus, int device_id);
  static void set_gpu_device_id(int gpu_id);
  int gpu_id;
  // Note: MACA stream and blas handles can be added when needed
#elif defined(YIRAGE_FINGERPRINT_USE_ASCEND)
  DeviceMemoryManager(int num_gpus, int device_id);
  void set_gpu_device_id(int device_id);  // Non-static for Ascend (called via get_instance())
  int gpu_id;
  void *stream;  // aclrtStream
#else
  DeviceMemoryManager();
#endif
public:
  static DeviceMemoryManager *singleton;
  ~DeviceMemoryManager(void);

public:
  static DeviceMemoryManager *get_instance();

public:
  // fingerprint related fields
  yirage::type::FPType *exp_lookup_table;
  yirage::type::FPType *div_p_lookup_table;
  yirage::type::FPType *div_q_lookup_table;
  yirage::type::FPType *sqrt_p_lookup_table;
  yirage::type::FPType *sqrt_q_lookup_table;
  // fields for managing the preallocated cuda buffer
  // Note that all fp_base_ptrs refer
  // to buffers on the 0-th GPU since we compute
  // fingerprint on a single device to avoid inter-GPU
  // communication
  char *fp_base_ptr[yirage::config::MAX_NUM_DEVICES];
  char *stensor_fp_base_ptr;
  
#if defined(YIRAGE_FINGERPRINT_USE_ASCEND)
  // Ascend-specific memory management
  void *dmem_fp_ptr;  // Device memory for fingerprint
  void *smem_fp_ptr;  // L1 buffer (shared memory equivalent)
  size_t dmem_fp_size;
  size_t smem_fp_size;
  size_t dmem_fp_offset;
  size_t smem_fp_offset;
  
  yirage::type::FPType *allocate_dmem_fingerprint(size_t size, bool reset = false);
  yirage::type::FPType *allocate_smem_fingerprint(size_t size, bool reset = false);
  void copy_to_device(void *dst, void const *src, size_t size);
  void copy_to_host(void *dst, void const *src, size_t size);
  void synchronize();
  void *get_stream();
#endif
};

void cython_set_gpu_device_id(int gpu_id);

} // namespace kernel
} // namespace yirage
