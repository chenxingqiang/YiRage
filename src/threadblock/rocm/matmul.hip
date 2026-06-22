/* Copyright 2025 YiRage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "threadblock/graph.h"
#include "threadblock/matmul.h"
#include "threadblock/rocm/matmul.h"

#ifdef YIRAGE_USE_ROCM

#include <hip/hip_runtime.h>

namespace yirage {
namespace threadblock {
namespace rocm {

// ROCm/HIP matmul kernel launch wrapper
void launch_matmul_fingerprint(
    void* A_ptr, void* B_ptr, void* C_ptr,
    int m, int n, int k,
    hipStream_t stream) {
    // 64-thread wavefront based execution
    dim3 block(64);
    dim3 grid((m * n + 63) / 64);
    
    // Kernel launch would go here
    // hipLaunchKernelGGL(...)
}

}  // namespace rocm
}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_USE_ROCM
