/* Copyright 2025 YiRage Team
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

/**
 * @file mps_common.h
 * @brief Common definitions for MPS kernels across all Apple Silicon generations
 */

namespace yirage {
namespace persistent_kernel {
namespace mps {

// =============================================================================
// Apple Silicon Architecture Detection
// =============================================================================

enum class AppleSiliconGen {
    M1 = 1,      // A14 Bionic GPU architecture (2020)
    M2 = 2,      // Second generation (2022)
    M3 = 3,      // Third generation with ray tracing (2023)
    M4 = 4,      // Fourth generation with Neural Engine boost (2024)
    M5 = 5,      // Fifth generation (2025+)
    UNKNOWN = 0
};

enum class AppleSiliconVariant {
    BASE,        // Base chip
    PRO,         // Pro variant (more GPU cores)
    MAX,         // Max variant (even more GPU cores)
    ULTRA        // Ultra variant (2x Max, dual-die)
};

// =============================================================================
// Hardware Specifications by Generation
// =============================================================================

struct AppleSiliconSpecs {
    int gpu_cores;              // Number of GPU cores
    int simd_width;             // SIMD group size (always 32)
    int max_threadgroup_size;   // Max threads per threadgroup
    int max_threads_per_core;   // Max concurrent threads per GPU core
    int shared_memory_kb;       // Threadgroup memory in KB
    int unified_memory_bw_gbps; // Memory bandwidth
    bool has_ray_tracing;       // Hardware ray tracing support
    bool has_mesh_shaders;      // Mesh shader support
    bool has_dynamic_caching;   // Dynamic caching for better occupancy
    int neural_engine_tops;     // Neural Engine performance in TOPS
};

// M1 (2020) - First Apple Silicon
constexpr AppleSiliconSpecs M1_BASE_SPECS = {
    .gpu_cores = 8,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 768,
    .shared_memory_kb = 32,
    .unified_memory_bw_gbps = 68,
    .has_ray_tracing = false,
    .has_mesh_shaders = false,
    .has_dynamic_caching = false,
    .neural_engine_tops = 11
};

constexpr AppleSiliconSpecs M1_PRO_SPECS = {
    .gpu_cores = 16,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 768,
    .shared_memory_kb = 32,
    .unified_memory_bw_gbps = 200,
    .has_ray_tracing = false,
    .has_mesh_shaders = false,
    .has_dynamic_caching = false,
    .neural_engine_tops = 11
};

constexpr AppleSiliconSpecs M1_MAX_SPECS = {
    .gpu_cores = 32,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 768,
    .shared_memory_kb = 32,
    .unified_memory_bw_gbps = 400,
    .has_ray_tracing = false,
    .has_mesh_shaders = false,
    .has_dynamic_caching = false,
    .neural_engine_tops = 11
};

// M2 (2022) - Second generation
constexpr AppleSiliconSpecs M2_BASE_SPECS = {
    .gpu_cores = 10,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 768,
    .shared_memory_kb = 32,
    .unified_memory_bw_gbps = 100,
    .has_ray_tracing = false,
    .has_mesh_shaders = false,
    .has_dynamic_caching = false,
    .neural_engine_tops = 15
};

constexpr AppleSiliconSpecs M2_PRO_SPECS = {
    .gpu_cores = 19,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 768,
    .shared_memory_kb = 32,
    .unified_memory_bw_gbps = 200,
    .has_ray_tracing = false,
    .has_mesh_shaders = false,
    .has_dynamic_caching = false,
    .neural_engine_tops = 15
};

constexpr AppleSiliconSpecs M2_MAX_SPECS = {
    .gpu_cores = 38,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 768,
    .shared_memory_kb = 32,
    .unified_memory_bw_gbps = 400,
    .has_ray_tracing = false,
    .has_mesh_shaders = false,
    .has_dynamic_caching = false,
    .neural_engine_tops = 15
};

// M3 (2023) - Third generation with ray tracing
constexpr AppleSiliconSpecs M3_BASE_SPECS = {
    .gpu_cores = 10,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 1024,  // Increased
    .shared_memory_kb = 32,
    .unified_memory_bw_gbps = 100,
    .has_ray_tracing = true,       // New!
    .has_mesh_shaders = true,      // New!
    .has_dynamic_caching = true,   // New!
    .neural_engine_tops = 18
};

constexpr AppleSiliconSpecs M3_PRO_SPECS = {
    .gpu_cores = 18,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 1024,
    .shared_memory_kb = 32,
    .unified_memory_bw_gbps = 150,
    .has_ray_tracing = true,
    .has_mesh_shaders = true,
    .has_dynamic_caching = true,
    .neural_engine_tops = 18
};

constexpr AppleSiliconSpecs M3_MAX_SPECS = {
    .gpu_cores = 40,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 1024,
    .shared_memory_kb = 32,
    .unified_memory_bw_gbps = 400,
    .has_ray_tracing = true,
    .has_mesh_shaders = true,
    .has_dynamic_caching = true,
    .neural_engine_tops = 18
};

// M4 (2024) - Fourth generation with enhanced Neural Engine
constexpr AppleSiliconSpecs M4_BASE_SPECS = {
    .gpu_cores = 10,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 1024,
    .shared_memory_kb = 48,        // Increased!
    .unified_memory_bw_gbps = 120,
    .has_ray_tracing = true,
    .has_mesh_shaders = true,
    .has_dynamic_caching = true,
    .neural_engine_tops = 38       // Major boost!
};

constexpr AppleSiliconSpecs M4_PRO_SPECS = {
    .gpu_cores = 20,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 1024,
    .shared_memory_kb = 48,
    .unified_memory_bw_gbps = 270,
    .has_ray_tracing = true,
    .has_mesh_shaders = true,
    .has_dynamic_caching = true,
    .neural_engine_tops = 38
};

constexpr AppleSiliconSpecs M4_MAX_SPECS = {
    .gpu_cores = 40,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 1024,
    .shared_memory_kb = 48,
    .unified_memory_bw_gbps = 540,
    .has_ray_tracing = true,
    .has_mesh_shaders = true,
    .has_dynamic_caching = true,
    .neural_engine_tops = 38
};

// M5 (2025+) - Fifth generation (projected specs)
constexpr AppleSiliconSpecs M5_BASE_SPECS = {
    .gpu_cores = 12,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 1280,  // Projected increase
    .shared_memory_kb = 64,        // Projected increase
    .unified_memory_bw_gbps = 150,
    .has_ray_tracing = true,
    .has_mesh_shaders = true,
    .has_dynamic_caching = true,
    .neural_engine_tops = 50       // Projected
};

constexpr AppleSiliconSpecs M5_PRO_SPECS = {
    .gpu_cores = 24,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 1280,
    .shared_memory_kb = 64,
    .unified_memory_bw_gbps = 300,
    .has_ray_tracing = true,
    .has_mesh_shaders = true,
    .has_dynamic_caching = true,
    .neural_engine_tops = 50
};

constexpr AppleSiliconSpecs M5_MAX_SPECS = {
    .gpu_cores = 48,
    .simd_width = 32,
    .max_threadgroup_size = 1024,
    .max_threads_per_core = 1280,
    .shared_memory_kb = 64,
    .unified_memory_bw_gbps = 600,
    .has_ray_tracing = true,
    .has_mesh_shaders = true,
    .has_dynamic_caching = true,
    .neural_engine_tops = 50
};

// =============================================================================
// Kernel Configuration by Generation
// =============================================================================

struct MpsKernelConfig {
    int threadgroup_size;       // Threads per threadgroup
    int simd_groups_per_tg;     // SIMD groups per threadgroup
    int tile_size_m;            // GEMM M tile size
    int tile_size_n;            // GEMM N tile size
    int tile_size_k;            // GEMM K tile size
    int attention_tile_size;    // Attention tile size
    bool use_simd_shuffle;      // Use simd_shuffle operations
    bool use_simdgroup_matrix;  // Use simdgroup_matrix (M3+)
};

// Default configurations per generation
constexpr MpsKernelConfig M1_KERNEL_CONFIG = {
    .threadgroup_size = 256,
    .simd_groups_per_tg = 8,
    .tile_size_m = 32,
    .tile_size_n = 32,
    .tile_size_k = 32,
    .attention_tile_size = 64,
    .use_simd_shuffle = true,
    .use_simdgroup_matrix = false
};

constexpr MpsKernelConfig M2_KERNEL_CONFIG = {
    .threadgroup_size = 256,
    .simd_groups_per_tg = 8,
    .tile_size_m = 32,
    .tile_size_n = 64,
    .tile_size_k = 32,
    .attention_tile_size = 64,
    .use_simd_shuffle = true,
    .use_simdgroup_matrix = false
};

constexpr MpsKernelConfig M3_KERNEL_CONFIG = {
    .threadgroup_size = 256,
    .simd_groups_per_tg = 8,
    .tile_size_m = 64,
    .tile_size_n = 64,
    .tile_size_k = 32,
    .attention_tile_size = 128,
    .use_simd_shuffle = true,
    .use_simdgroup_matrix = true   // M3+ feature
};

constexpr MpsKernelConfig M4_KERNEL_CONFIG = {
    .threadgroup_size = 384,       // Larger threadgroups
    .simd_groups_per_tg = 12,
    .tile_size_m = 64,
    .tile_size_n = 128,
    .tile_size_k = 64,
    .attention_tile_size = 128,
    .use_simd_shuffle = true,
    .use_simdgroup_matrix = true
};

constexpr MpsKernelConfig M5_KERNEL_CONFIG = {
    .threadgroup_size = 512,       // Larger threadgroups
    .simd_groups_per_tg = 16,
    .tile_size_m = 128,
    .tile_size_n = 128,
    .tile_size_k = 64,
    .attention_tile_size = 256,
    .use_simd_shuffle = true,
    .use_simdgroup_matrix = true
};

// =============================================================================
// Runtime Detection Functions
// =============================================================================

/**
 * @brief Detect Apple Silicon generation at runtime
 * @return AppleSiliconGen enum value
 */
AppleSiliconGen detect_apple_silicon_gen();

/**
 * @brief Detect Apple Silicon variant (Base/Pro/Max/Ultra)
 * @return AppleSiliconVariant enum value
 */
AppleSiliconVariant detect_apple_silicon_variant();

/**
 * @brief Get hardware specs for current device
 * @return AppleSiliconSpecs for the detected hardware
 */
const AppleSiliconSpecs& get_device_specs();

/**
 * @brief Get optimal kernel config for current device
 * @return MpsKernelConfig for the detected hardware
 */
const MpsKernelConfig& get_kernel_config();

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
