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
 *
 * Ascend RMS Norm Fingerprint Implementation
 * 
 * Fingerprint verification for RMSNorm on Huawei Ascend NPU.
 * 
 * When use_npu_memory is true:
 *   - Copy data from NPU to CPU
 *   - Compute fingerprint on CPU
 *   - Copy result back to NPU
 * 
 * This matches the CUDA approach where fingerprint data lives on device,
 * but the actual computation may happen on host for simplicity.
 * 
 * TODO: Implement native Ascend C kernel for fingerprint computation
 *       to avoid D2H/H2D transfers.
 */

#ifdef YIRAGE_FINGERPRINT_USE_ASCEND

#include "kernel/device_memory_manager.h"
#include "kernel/rms_norm.h"
#include "utils/fingerprint_functions.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

#ifdef __ASCEND__
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

namespace yirage {
namespace kernel {

using namespace yirage::type;
using namespace yirage::config;
using namespace yirage::utils;

bool KNRMSNormOp::fingerprint(void) {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  
  // Safety checks
  if (input_tensors[0].fp_offset < 0 || output_tensors[0].fp_offset < 0) {
    std::cerr << "[ERROR] KNRMSNormOp::fingerprint: Invalid fp_offset" << std::endl;
    return false;
  }
  
  if (dmm->div_p_lookup_table == nullptr || dmm->sqrt_p_lookup_table == nullptr ||
      dmm->div_q_lookup_table == nullptr || dmm->sqrt_q_lookup_table == nullptr) {
    std::cerr << "[ERROR] KNRMSNormOp::fingerprint: Lookup tables not initialized" << std::endl;
    return false;
  }

  int num_samples = output_tensors[0].num_elements() / normalized_size;
  int total_elements = output_tensors[0].num_elements();
  size_t data_size = total_elements * sizeof(FPType);
  
  for (int device_id = 0; device_id < dmm->num_gpus; ++device_id) {
    if (dmm->fp_base_ptr[device_id] == nullptr) {
      std::cerr << "[ERROR] KNRMSNormOp::fingerprint: fp_base_ptr is null" << std::endl;
      return false;
    }
    
    FPType *input_ptr = reinterpret_cast<FPType *>(
        dmm->fp_base_ptr[device_id] + input_tensors[0].fp_offset);
    FPType *output_ptr = reinterpret_cast<FPType *>(
        dmm->fp_base_ptr[device_id] + output_tensors[0].fp_offset);
    
#ifdef __ASCEND__
    if (dmm->use_npu_memory) {
      // Data is on NPU - need to copy to CPU, compute, copy back
      
      // Allocate CPU buffers
      FPType *host_input = reinterpret_cast<FPType*>(malloc(data_size));
      FPType *host_output = reinterpret_cast<FPType*>(malloc(data_size));
      FPType *host_div_p = reinterpret_cast<FPType*>(malloc(FP_P * sizeof(FPType)));
      FPType *host_div_q = reinterpret_cast<FPType*>(malloc(FP_Q * sizeof(FPType)));
      FPType *host_sqrt_p = reinterpret_cast<FPType*>(malloc(FP_P * sizeof(FPType)));
      FPType *host_sqrt_q = reinterpret_cast<FPType*>(malloc(FP_Q * sizeof(FPType)));
      
      if (!host_input || !host_output || !host_div_p || !host_div_q || 
          !host_sqrt_p || !host_sqrt_q) {
        std::cerr << "[ERROR] KNRMSNormOp::fingerprint: Failed to allocate host memory" << std::endl;
        free(host_input);
        free(host_output);
        free(host_div_p);
        free(host_div_q);
        free(host_sqrt_p);
        free(host_sqrt_q);
        return false;
      }
      
      // Copy data from NPU to CPU
      aclrtMemcpy(host_input, data_size, input_ptr, data_size, ACL_MEMCPY_DEVICE_TO_HOST);
      aclrtMemcpy(host_div_p, FP_P * sizeof(FPType), 
                  dmm->div_p_lookup_table, FP_P * sizeof(FPType), ACL_MEMCPY_DEVICE_TO_HOST);
      aclrtMemcpy(host_div_q, FP_Q * sizeof(FPType), 
                  dmm->div_q_lookup_table, FP_Q * sizeof(FPType), ACL_MEMCPY_DEVICE_TO_HOST);
      aclrtMemcpy(host_sqrt_p, FP_P * sizeof(FPType), 
                  dmm->sqrt_p_lookup_table, FP_P * sizeof(FPType), ACL_MEMCPY_DEVICE_TO_HOST);
      aclrtMemcpy(host_sqrt_q, FP_Q * sizeof(FPType), 
                  dmm->sqrt_q_lookup_table, FP_Q * sizeof(FPType), ACL_MEMCPY_DEVICE_TO_HOST);
      
      // Compute fingerprint on CPU
      compute_rms_norm_fingerprint(host_input,
                                   host_output,
                                   host_div_p,
                                   host_div_q,
                                   host_sqrt_p,
                                   host_sqrt_q,
                                   num_samples,
                                   normalized_size);
      
      // Copy result back to NPU
      aclrtMemcpy(output_ptr, data_size, host_output, data_size, ACL_MEMCPY_HOST_TO_DEVICE);
      
      // Cleanup
      free(host_input);
      free(host_output);
      free(host_div_p);
      free(host_div_q);
      free(host_sqrt_p);
      free(host_sqrt_q);
    } else {
      // Data is on CPU (fallback mode) - compute directly
      compute_rms_norm_fingerprint(input_ptr,
                                   output_ptr,
                                   dmm->div_p_lookup_table,
                                   dmm->div_q_lookup_table,
                                   dmm->sqrt_p_lookup_table,
                                   dmm->sqrt_q_lookup_table,
                                   num_samples,
                                   normalized_size);
    }
#else
    // Non-Ascend build: compute on CPU
    compute_rms_norm_fingerprint(input_ptr,
                                 output_ptr,
                                 dmm->div_p_lookup_table,
                                 dmm->div_q_lookup_table,
                                 dmm->sqrt_p_lookup_table,
                                 dmm->sqrt_q_lookup_table,
                                 num_samples,
                                 normalized_size);
#endif
  }
  return true;
}

} // namespace kernel
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND
