/* Copyright 2025 YiRage Team */

#include "threadblock/rocm/input_loader.h"

#ifdef YIRAGE_USE_ROCM

#include <hip/hip_runtime.h>

namespace yirage {
namespace threadblock {
namespace rocm {

void launch_input_loader(
    void* global_ptr, void* lds_ptr,
    int num_elements,
    hipStream_t stream) {
    dim3 block(256);
    dim3 grid((num_elements + 255) / 256);
    // LDS (Local Data Share) based loader
}

}  // namespace rocm
}  // namespace threadblock
}  // namespace yirage

#endif
