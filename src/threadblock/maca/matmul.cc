/* Copyright 2025 YiRage Team
 * Licensed under the Apache License, Version 2.0
 *
 * MetaX MACA GPU MatMul threadblock implementation
 * Host-side wrapper for MACA kernels (.maca files)
 */

#include "threadblock/maca/matmul.h"
#include "threadblock/graph.h"
#include "utils/hash_utils.h"

#include <cstring>
#include <cassert>

namespace yirage {
namespace threadblock {
namespace maca {

// =============================================================================
// MACA Architecture Constants
// =============================================================================

constexpr int MACA_WARP_SIZE = 64;  // Critical: NOT 32 like NVIDIA!
constexpr int MAX_BLOCK_DIM = 1024;
constexpr int PREFERRED_BLOCK_DIM = 256;  // 4 warps

// =============================================================================
// Configuration
// =============================================================================

MacaMatMulConfig::MacaMatMulConfig() {
    warp_size = MACA_WARP_SIZE;
    tile_m = 64;
    tile_n = 64;
    tile_k = 32;
    use_tensor_cores = true;
    shared_mem_bytes = 0;
    block_dim = PREFERRED_BLOCK_DIM;
}

MacaMatMulConfig get_optimal_matmul_config(int M, int N, int K) {
    MacaMatMulConfig config;
    
    // Large matrix: maximize tile sizes
    if (M >= 4096 && N >= 4096) {
        config.tile_m = 128;
        config.tile_n = 128;
        config.tile_k = 32;
        config.block_dim = 256;
    } 
    // Medium matrix
    else if (M >= 1024 && N >= 1024) {
        config.tile_m = 64;
        config.tile_n = 64;
        config.tile_k = 32;
        config.block_dim = 256;
    }
    // Small matrix: smaller tiles, fewer threads
    else {
        config.tile_m = 32;
        config.tile_n = 32;
        config.tile_k = 16;
        config.block_dim = 128;
    }
    
    // Calculate shared memory requirement
    config.shared_mem_bytes = 
        config.tile_m * config.tile_k * sizeof(float) +
        config.tile_k * config.tile_n * sizeof(float);
    
    return config;
}

// =============================================================================
// Device Query
// =============================================================================

MacaDeviceInfo get_maca_device_info(int device_id) {
    MacaDeviceInfo info;
    info.device_id = device_id;
    
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    // Query MACA device properties
    // Similar to cudaGetDeviceProperties
    info.name = "MetaX C500";  // Placeholder
    info.compute_units = 80;    // Example
    info.warp_size = 64;        // MACA uses 64!
    info.shared_mem_per_block = 49152;
    info.has_tensor_cores = true;
    info.max_threads_per_block = 1024;
#else
    info.name = "MACA (not available)";
    info.compute_units = 0;
    info.warp_size = 64;
    info.shared_mem_per_block = 0;
    info.has_tensor_cores = false;
    info.max_threads_per_block = 0;
#endif
    
    return info;
}

// =============================================================================
// External MACA Kernel Declarations
// =============================================================================

#ifdef YIRAGE_BACKEND_MACA_ENABLED
// Defined in matmul.maca
extern void maca_matmul_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K, void* stream
);

extern void maca_matmul_f16(
    const void* A, const void* B, void* C,
    int M, int N, int K, void* stream
);
#endif

// =============================================================================
// Host-side MatMul API
// =============================================================================

bool maca_matmul_execute(
    const void* A,
    const void* B,
    void* C,
    int M, int N, int K,
    const MacaMatMulConfig& config,
    void* stream
) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    if (config.dtype == DataType::FP32) {
        maca_matmul_f32(
            static_cast<const float*>(A),
            static_cast<const float*>(B),
            static_cast<float*>(C),
            M, N, K, stream
        );
    } else if (config.dtype == DataType::FP16) {
        maca_matmul_f16(A, B, C, M, N, K, stream);
    } else {
        return false;
    }
    return true;
#else
    return false;
#endif
}

// =============================================================================
// Fingerprint Computation (for verification)
// =============================================================================

void maca_matmul_fingerprint(
    const void* A,
    const void* B,
    void* C,
    int M, int N, int K,
    void* stream
) {
    // Fingerprint version uses integer arithmetic for verification
    // Implemented in the header file as TBMatmulFingerprinter class
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    // Launch fingerprint kernel
    // This is used during search to verify correctness
#endif
}

}  // namespace maca
}  // namespace threadblock
}  // namespace yirage
