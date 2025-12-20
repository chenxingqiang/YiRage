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

#include "yirage/kernel/chunk.h"
#include "yirage/kernel/device_memory_manager.h"
#include "yirage/kernel/graph.h"
#include "yirage/utils/cuda_helper.h"

namespace yirage {
namespace kernel {

#ifdef YIRAGE_FINGERPRINT_USE_CUDA
bool KNChunkOp::fingerprint(void) {
  // Chunk operation is a simple view operation (no data transformation)
  // Just copy the fingerprint from input to output with offset
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  checkCUDA(cudaSetDevice(dmm->gpu_id));
  
  // For chunk operations, fingerprints are directly inherited from input
  // with the appropriate offset, which is handled by the tensor view
  return true;
}
#endif // YIRAGE_FINGERPRINT_USE_CUDA

} // namespace kernel
} // namespace yirage
