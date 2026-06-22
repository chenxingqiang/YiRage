/* Copyright 2025 YiRage Team */

#include "threadblock/element_unary.h"
#include "threadblock/rocm/element_unary.h"

#ifdef YIRAGE_USE_ROCM

#include <hip/hip_runtime.h>

namespace yirage {
namespace threadblock {
namespace rocm {

void launch_element_unary_fingerprint(
    void* input_ptr, void* output_ptr,
    int num_elements, int op_type,
    hipStream_t stream) {
    dim3 block(256);
    dim3 grid((num_elements + 255) / 256);
    // hipLaunchKernelGGL(...)
}

}  // namespace rocm
}  // namespace threadblock
}  // namespace yirage

#endif
