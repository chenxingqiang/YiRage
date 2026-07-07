/* Stubs for Graph / device APIs that CUDA-only sources provide when MACA uses
 * the host transpiler (mxcc JIT) without YIRAGE_BACKEND_USE_CUDA. */

#include "kernel/device_memory_manager.h"
#include "kernel/graph.h"

namespace yirage {
namespace kernel {

#if defined(YIRAGE_BACKEND_MACA_ENABLED) && defined(YIRAGE_HOST_TRANSPILER_ENABLED)

void cython_set_gpu_device_id(int gpu_id) {
  (void)gpu_id;
  // mcPytorch selects the MACA device via torch.cuda; CPU fingerprint search
  // does not require DeviceMemoryManager GPU buffers at import time.
}

void Graph::generate_triton_program(char const *filepath) {
  (void)filepath;
}

#endif

} // namespace kernel
} // namespace yirage
