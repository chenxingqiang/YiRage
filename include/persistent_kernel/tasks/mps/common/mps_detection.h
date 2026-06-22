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
 * @file mps_detection.h
 * @brief Runtime detection of Apple Silicon generation
 */

#include "mps_common.h"
#include <string>
#include <cstdlib>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace yirage {
namespace persistent_kernel {
namespace mps {

/**
 * @brief Detect Apple Silicon generation at runtime
 * 
 * Uses machdep.cpu.brand_string on macOS to identify the chip.
 * Falls back to environment variable YIRAGE_MPS_CHIP_GEN if available.
 */
inline AppleSiliconGen detect_apple_silicon_gen() {
    // Check environment override first
    const char* env_gen = std::getenv("YIRAGE_MPS_CHIP_GEN");
    if (env_gen) {
        std::string gen_str(env_gen);
        if (gen_str == "M1" || gen_str == "m1") return AppleSiliconGen::M1;
        if (gen_str == "M2" || gen_str == "m2") return AppleSiliconGen::M2;
        if (gen_str == "M3" || gen_str == "m3") return AppleSiliconGen::M3;
        if (gen_str == "M4" || gen_str == "m4") return AppleSiliconGen::M4;
        if (gen_str == "M5" || gen_str == "m5") return AppleSiliconGen::M5;
    }
    
#ifdef __APPLE__
    // Query system for chip info
    char brand[256] = {0};
    size_t brand_len = sizeof(brand);
    
    if (sysctlbyname("machdep.cpu.brand_string", brand, &brand_len, nullptr, 0) == 0) {
        std::string brand_str(brand);
        
        // Check for Apple Silicon chips
        if (brand_str.find("M5") != std::string::npos) {
            return AppleSiliconGen::M5;
        }
        if (brand_str.find("M4") != std::string::npos) {
            return AppleSiliconGen::M4;
        }
        if (brand_str.find("M3") != std::string::npos) {
            return AppleSiliconGen::M3;
        }
        if (brand_str.find("M2") != std::string::npos) {
            return AppleSiliconGen::M2;
        }
        if (brand_str.find("M1") != std::string::npos) {
            return AppleSiliconGen::M1;
        }
    }
    
    // Alternative: check hw.optional flags for Apple Silicon features
    int64_t val = 0;
    size_t val_len = sizeof(val);
    
    // Check for features that indicate generation
    // M3+ has hardware ray tracing
    if (sysctlbyname("hw.optional.arm.FEAT_RNG", &val, &val_len, nullptr, 0) == 0 && val) {
        // Check GPU core count as a rough indicator
        int gpu_cores = 0;
        size_t gpu_len = sizeof(gpu_cores);
        if (sysctlbyname("hw.perflevel0.logicalcpu", &gpu_cores, &gpu_len, nullptr, 0) == 0) {
            // M4 typically has 10+ performance cores in base model
            // M3 has similar but different power characteristics
            // This is a rough heuristic
            if (gpu_cores >= 12) return AppleSiliconGen::M4;
            return AppleSiliconGen::M3;
        }
    }
    
    // Default to M1 for Apple Silicon if we can't determine
    return AppleSiliconGen::M1;
#else
    // Non-Apple platforms
    return AppleSiliconGen::UNKNOWN;
#endif
}

/**
 * @brief Detect Apple Silicon variant (Base/Pro/Max/Ultra)
 */
inline AppleSiliconVariant detect_apple_silicon_variant() {
    // Check environment override
    const char* env_var = std::getenv("YIRAGE_MPS_CHIP_VARIANT");
    if (env_var) {
        std::string var_str(env_var);
        if (var_str == "PRO" || var_str == "pro") return AppleSiliconVariant::PRO;
        if (var_str == "MAX" || var_str == "max") return AppleSiliconVariant::MAX;
        if (var_str == "ULTRA" || var_str == "ultra") return AppleSiliconVariant::ULTRA;
        return AppleSiliconVariant::BASE;
    }
    
#ifdef __APPLE__
    char brand[256] = {0};
    size_t brand_len = sizeof(brand);
    
    if (sysctlbyname("machdep.cpu.brand_string", brand, &brand_len, nullptr, 0) == 0) {
        std::string brand_str(brand);
        
        if (brand_str.find("Ultra") != std::string::npos) {
            return AppleSiliconVariant::ULTRA;
        }
        if (brand_str.find("Max") != std::string::npos) {
            return AppleSiliconVariant::MAX;
        }
        if (brand_str.find("Pro") != std::string::npos) {
            return AppleSiliconVariant::PRO;
        }
    }
    
    return AppleSiliconVariant::BASE;
#else
    return AppleSiliconVariant::BASE;
#endif
}

/**
 * @brief Get GPU core count
 */
inline int get_gpu_core_count() {
#ifdef __APPLE__
    int cores = 8;  // Default for M1 base
    size_t len = sizeof(cores);
    
    // Try to get GPU core count (this may vary by macOS version)
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &len, nullptr, 0) != 0) {
        // Fallback: estimate from variant
        auto variant = detect_apple_silicon_variant();
        auto gen = detect_apple_silicon_gen();
        auto specs = get_specs_for_gen(gen, variant);
        return specs.gpu_cores;
    }
    
    return cores;
#else
    return 0;
#endif
}

/**
 * @brief Get unified memory size in GB
 */
inline int get_unified_memory_gb() {
#ifdef __APPLE__
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) {
        return static_cast<int>(memsize / (1024 * 1024 * 1024));
    }
    
    return 8;  // Default minimum
#else
    return 0;
#endif
}

/**
 * @brief Get string representation of generation
 */
inline const char* gen_to_string(AppleSiliconGen gen) {
    switch (gen) {
        case AppleSiliconGen::M1: return "M1";
        case AppleSiliconGen::M2: return "M2";
        case AppleSiliconGen::M3: return "M3";
        case AppleSiliconGen::M4: return "M4";
        case AppleSiliconGen::M5: return "M5";
        default: return "Unknown";
    }
}

/**
 * @brief Get string representation of variant
 */
inline const char* variant_to_string(AppleSiliconVariant variant) {
    switch (variant) {
        case AppleSiliconVariant::BASE:  return "Base";
        case AppleSiliconVariant::PRO:   return "Pro";
        case AppleSiliconVariant::MAX:   return "Max";
        case AppleSiliconVariant::ULTRA: return "Ultra";
        default: return "Unknown";
    }
}

/**
 * @brief Print device info for debugging
 */
inline void print_device_info() {
    auto gen = detect_apple_silicon_gen();
    auto variant = detect_apple_silicon_variant();
    auto specs = get_specs_for_gen(gen, variant);
    
    printf("Apple Silicon Device Info:\n");
    printf("  Generation: %s %s\n", gen_to_string(gen), variant_to_string(variant));
    printf("  GPU Cores: %d\n", specs.gpu_cores);
    printf("  SIMD Width: %d\n", specs.simd_width);
    printf("  Max Threadgroup Size: %d\n", specs.max_threadgroup_size);
    printf("  Shared Memory: %d KB\n", specs.shared_memory_kb);
    printf("  Memory Bandwidth: %d GB/s\n", specs.unified_memory_bw_gbps);
    printf("  Ray Tracing: %s\n", specs.has_ray_tracing ? "Yes" : "No");
    printf("  simdgroup_matrix: %s\n", has_simdgroup_matrix(gen) ? "Yes" : "No");
    printf("  Neural Engine: %d TOPS\n", specs.neural_engine_tops);
    printf("  Unified Memory: %d GB\n", get_unified_memory_gb());
}

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
