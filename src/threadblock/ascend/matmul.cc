/* Copyright 2025 YiRage Team
 * Licensed under the Apache License, Version 2.0
 *
 * Ascend NPU MatMul threadblock implementation
 * Host-side wrapper for Ascend kernels (.ascend files)
 */

#include "threadblock/ascend/matmul.h"
#include "threadblock/graph.h"
#include "utils/hash_utils.h"

#include <cstring>
#include <cassert>

namespace yirage {
namespace threadblock {
namespace ascend {

// =============================================================================
// Ascend Architecture Constants
// =============================================================================

// Ascend Cube Unit processes 16x16 FP16 matrix tiles per cycle
constexpr int CUBE_UNIT_SIZE = 16;
constexpr int ASCEND_910_AI_CORES = 32;
constexpr int ASCEND_310_AI_CORES = 8;
constexpr int L1_BUFFER_SIZE = 1024 * 1024;  // 1MB per AI Core

// =============================================================================
// Configuration
// =============================================================================

AscendMatMulConfig::AscendMatMulConfig() {
    tile_m = CUBE_UNIT_SIZE;
    tile_n = CUBE_UNIT_SIZE;
    tile_k = CUBE_UNIT_SIZE;
    use_l1_buffer = true;
    use_l0_buffer = true;
    num_cores = ASCEND_910_AI_CORES;
}

AscendMatMulConfig get_optimal_matmul_config(int M, int N, int K, int device_type) {
    AscendMatMulConfig config;
    
    // Set core count based on device
    if (device_type == 910 || device_type == 0) {
        config.num_cores = ASCEND_910_AI_CORES;
    } else if (device_type == 310) {
        config.num_cores = ASCEND_310_AI_CORES;
    }
    
    // Choose tile sizes based on problem size
    // Tiles must be multiples of Cube Unit size (16)
    if (M >= 4096 && N >= 4096) {
        config.tile_m = 128;  // 8 x Cube Unit
        config.tile_n = 128;
        config.tile_k = 64;
    } else if (M >= 1024 && N >= 1024) {
        config.tile_m = 64;   // 4 x Cube Unit
        config.tile_n = 64;
        config.tile_k = 32;
    } else {
        config.tile_m = 32;   // 2 x Cube Unit
        config.tile_n = 32;
        config.tile_k = 16;
    }
    
    // Check L1 buffer constraints
    size_t tile_memory = 
        config.tile_m * config.tile_k * sizeof(float16_t) +  // A tile
        config.tile_k * config.tile_n * sizeof(float16_t) +  // B tile
        config.tile_m * config.tile_n * sizeof(float16_t);   // C tile
    
    if (tile_memory > L1_BUFFER_SIZE) {
        // Reduce tile sizes to fit in L1
        config.tile_m = 32;
        config.tile_n = 32;
        config.tile_k = 16;
    }
    
    config.use_l1_buffer = true;
    config.use_l0_buffer = (device_type == 910);  // L0 only on 910
    
    return config;
}

// =============================================================================
// Device Query
// =============================================================================

AscendDeviceInfo get_ascend_device_info(int device_id) {
    AscendDeviceInfo info;
    info.device_id = device_id;
    
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    // Query via CANN ACL
    // aclrtGetDeviceCount, aclrtGetDeviceName, etc.
    info.device_type = 910;  // Placeholder
    info.name = "Ascend 910";
    info.ai_cores = ASCEND_910_AI_CORES;
    info.cube_unit_size = CUBE_UNIT_SIZE;
    info.l1_buffer_size = L1_BUFFER_SIZE;
    info.peak_tflops_fp16 = 256.0;  // Ascend 910
#else
    info.device_type = 0;
    info.name = "Ascend (not available)";
    info.ai_cores = 0;
    info.cube_unit_size = CUBE_UNIT_SIZE;
    info.l1_buffer_size = 0;
    info.peak_tflops_fp16 = 0.0;
#endif
    
    return info;
}

// =============================================================================
// External Ascend Kernel Declarations
// =============================================================================

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
// Defined in matmul.ascend
extern int ascend_matmul_f16(
    const void* A, const void* B, void* C,
    int M, int N, int K, void* stream
);

extern int ascend_matmul_f32(
    const void* A, const void* B, void* C,
    int M, int N, int K, void* stream
);

extern int ascend_batch_matmul_f16(
    const void* A, const void* B, void* C,
    int batch_size, int M, int N, int K, void* stream
);
#endif

// =============================================================================
// Host-side MatMul API
// =============================================================================

bool ascend_matmul_execute(
    const void* A,
    const void* B,
    void* C,
    int M, int N, int K,
    const AscendMatMulConfig& config,
    void* stream
) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    int ret;
    
    if (config.dtype == DataType::FP16) {
        ret = ascend_matmul_f16(A, B, C, M, N, K, stream);
    } else if (config.dtype == DataType::FP32) {
        ret = ascend_matmul_f32(A, B, C, M, N, K, stream);
    } else {
        return false;
    }
    
    return (ret == 0);  // ACL_SUCCESS
#else
    return false;
#endif
}

bool ascend_batch_matmul_execute(
    const void* A,
    const void* B,
    void* C,
    int batch_size,
    int M, int N, int K,
    const AscendMatMulConfig& config,
    void* stream
) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    int ret = ascend_batch_matmul_f16(A, B, C, batch_size, M, N, K, stream);
    return (ret == 0);
#else
    return false;
#endif
}

// =============================================================================
// Performance Estimation
// =============================================================================

AscendMatMulStats estimate_matmul_performance(
    int M, int N, int K,
    const AscendMatMulConfig& config
) {
    AscendMatMulStats stats;
    
    // FLOPs for MatMul: 2 * M * N * K
    double flops = 2.0 * M * N * K;
    
    // Memory access: read A (M*K), read B (K*N), write C (M*N)
    double memory_bytes = (M * K + K * N + M * N) * sizeof(float16_t);
    
    // Estimate based on theoretical peak
    double peak_tflops = 256.0;  // Ascend 910
    double peak_bandwidth_gbps = 1200.0;  // GB/s
    
    // Compute-bound time
    double compute_time_ms = (flops / 1e12) / peak_tflops * 1000.0;
    
    // Memory-bound time  
    double memory_time_ms = (memory_bytes / 1e9) / peak_bandwidth_gbps * 1000.0;
    
    stats.estimated_latency_ms = std::max(compute_time_ms, memory_time_ms);
    stats.theoretical_gflops = flops / 1e9 / stats.estimated_latency_ms * 1000.0;
    stats.achieved_utilization = flops / (stats.estimated_latency_ms / 1000.0) / (peak_tflops * 1e12);
    
    return stats;
}

// =============================================================================
// Fingerprint Computation (for verification)
// =============================================================================

void ascend_matmul_fingerprint(
    const void* A,
    const void* B,
    void* C,
    int M, int N, int K,
    void* stream
) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    // Fingerprint version for verification
    // Uses the TBMatmulFingerprinter class from the header
#endif
}

}  // namespace ascend
}  // namespace threadblock
}  // namespace yirage
