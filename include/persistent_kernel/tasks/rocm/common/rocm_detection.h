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
 * @file rocm_detection.h
 * @brief Runtime detection for AMD ROCm GPUs
 *
 * Detection methods:
 * 1. Environment variables (YIRAGE_ROCM_ARCH)
 * 2. rocm-smi command line tool
 * 3. HIP runtime API (hipGetDeviceProperties)
 */

#include "rocm_common.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

namespace yirage {
namespace persistent_kernel {
namespace rocm {

// =============================================================================
// Runtime Detection Implementation
// =============================================================================

/**
 * @brief Parse GFX architecture string to AMDArch enum
 */
inline AMDArch gfx_to_arch(const std::string& gfx) {
    if (gfx.find("gfx908") != std::string::npos) {
        return AMDArch::MI100;
    }
    if (gfx.find("gfx90a") != std::string::npos) {
        return AMDArch::MI250;  // Could be MI200/MI210/MI250
    }
    if (gfx.find("gfx942") != std::string::npos) {
        return AMDArch::MI300X;
    }
    if (gfx.find("gfx950") != std::string::npos) {
        return AMDArch::MI350;
    }
    if (gfx.find("gfx1030") != std::string::npos) {
        return AMDArch::RDNA2;
    }
    if (gfx.find("gfx110") != std::string::npos) {
        return AMDArch::RDNA3;
    }
    if (gfx.find("gfx120") != std::string::npos) {
        return AMDArch::RDNA4;
    }
    return AMDArch::UNKNOWN;
}

/**
 * @brief Detect AMD GPU architecture at runtime
 */
inline AMDArch detect_amd_arch() {
    // Check environment variable override
    const char* env_arch = std::getenv("YIRAGE_ROCM_ARCH");
    if (env_arch) {
        if (strcmp(env_arch, "MI100") == 0 || strcmp(env_arch, "gfx908") == 0) {
            return AMDArch::MI100;
        }
        if (strcmp(env_arch, "MI200") == 0 || strcmp(env_arch, "MI210") == 0) {
            return AMDArch::MI200;
        }
        if (strcmp(env_arch, "MI250") == 0 || strcmp(env_arch, "MI250X") == 0 ||
            strcmp(env_arch, "gfx90a") == 0) {
            return AMDArch::MI250;
        }
        if (strcmp(env_arch, "MI300X") == 0 || strcmp(env_arch, "gfx942") == 0) {
            return AMDArch::MI300X;
        }
        if (strcmp(env_arch, "MI300A") == 0) {
            return AMDArch::MI300A;
        }
        if (strcmp(env_arch, "MI325X") == 0) {
            return AMDArch::MI325X;
        }
        if (strcmp(env_arch, "MI350") == 0) {
            return AMDArch::MI350;
        }
    }

#ifdef __linux__
    // Try rocm-smi first
    FILE* pipe = popen("rocm-smi --showproductname 2>/dev/null | head -5", "r");
    if (pipe) {
        char buffer[256];
        std::string output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        pclose(pipe);
        
        // Parse product name
        if (output.find("MI350") != std::string::npos) {
            return AMDArch::MI350;
        }
        if (output.find("MI325X") != std::string::npos) {
            return AMDArch::MI325X;
        }
        if (output.find("MI300X") != std::string::npos) {
            return AMDArch::MI300X;
        }
        if (output.find("MI300A") != std::string::npos) {
            return AMDArch::MI300A;
        }
        if (output.find("MI250X") != std::string::npos) {
            return AMDArch::MI250;
        }
        if (output.find("MI250") != std::string::npos) {
            return AMDArch::MI250;
        }
        if (output.find("MI210") != std::string::npos ||
            output.find("MI200") != std::string::npos) {
            return AMDArch::MI200;
        }
        if (output.find("MI100") != std::string::npos) {
            return AMDArch::MI100;
        }
    }
    
    // Try hipInfo or rocminfo
    pipe = popen("rocminfo 2>/dev/null | grep 'gfx' | head -1", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            return gfx_to_arch(buffer);
        }
        pclose(pipe);
    }
#endif

    // Check HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES
    const char* hip_devices = std::getenv("HIP_VISIBLE_DEVICES");
    const char* rocr_devices = std::getenv("ROCR_VISIBLE_DEVICES");
    
    if (hip_devices || rocr_devices) {
        // Assume MI300X as default for modern setups
        return AMDArch::MI300X;
    }

    // Default to MI250 as common data center GPU
    return AMDArch::MI250;
}

/**
 * @brief Detect AMD GPU variant
 */
inline AMDVariant detect_amd_variant() {
    const char* env_variant = std::getenv("YIRAGE_ROCM_VARIANT");
    if (env_variant) {
        if (strcmp(env_variant, "X") == 0) return AMDVariant::X;
        if (strcmp(env_variant, "A") == 0) return AMDVariant::A;
        if (strcmp(env_variant, "DUAL") == 0) return AMDVariant::DUAL_DIE;
    }

#ifdef __linux__
    FILE* pipe = popen("rocm-smi --showproductname 2>/dev/null | head -5", "r");
    if (pipe) {
        char buffer[256];
        std::string output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        pclose(pipe);
        
        if (output.find("MI250X") != std::string::npos ||
            output.find("MI300X") != std::string::npos ||
            output.find("MI325X") != std::string::npos) {
            return AMDVariant::X;
        }
        if (output.find("MI300A") != std::string::npos) {
            return AMDVariant::A;
        }
    }
#endif

    return AMDVariant::STANDARD;
}

/**
 * @brief Get Compute Unit count
 */
inline int get_compute_unit_count() {
#ifdef __linux__
    FILE* pipe = popen("rocm-smi --showcomputeunits 2>/dev/null | grep -oE '[0-9]+' | head -1", "r");
    if (pipe) {
        char buffer[64];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            int cus = atoi(buffer);
            if (cus > 0) return cus;
        } else {
            pclose(pipe);
        }
    }
#endif

    AMDArch arch = detect_amd_arch();
    switch (arch) {
        case AMDArch::MI350: return 400;
        case AMDArch::MI325X: return 304;
        case AMDArch::MI300X: return 304;
        case AMDArch::MI300A: return 228;
        case AMDArch::MI250: return 220;
        case AMDArch::MI200: return 104;
        case AMDArch::MI100: return 120;
        default: return 120;
    }
}

/**
 * @brief Get HBM memory in GB
 */
inline int get_hbm_memory_gb() {
#ifdef __linux__
    FILE* pipe = popen("rocm-smi --showmeminfo vram 2>/dev/null | grep -oE '[0-9]+' | head -1", "r");
    if (pipe) {
        char buffer[64];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            long long bytes = atoll(buffer);
            if (bytes > 0) return bytes / (1024LL * 1024LL * 1024LL);
        } else {
            pclose(pipe);
        }
    }
#endif

    AMDArch arch = detect_amd_arch();
    switch (arch) {
        case AMDArch::MI350: return 288;
        case AMDArch::MI325X: return 256;
        case AMDArch::MI300X: return 192;
        case AMDArch::MI300A: return 128;
        case AMDArch::MI250: return 128;
        case AMDArch::MI200: return 64;
        case AMDArch::MI100: return 32;
        default: return 32;
    }
}

/**
 * @brief Convert AMDArch to string
 */
inline const char* arch_to_string(AMDArch arch) {
    switch (arch) {
        case AMDArch::MI100: return "AMD MI100 (gfx908)";
        case AMDArch::MI200: return "AMD MI210 (gfx90a)";
        case AMDArch::MI250: return "AMD MI250/MI250X (gfx90a)";
        case AMDArch::MI300X: return "AMD MI300X (gfx942)";
        case AMDArch::MI300A: return "AMD MI300A (gfx942)";
        case AMDArch::MI325X: return "AMD MI325X (gfx942)";
        case AMDArch::MI350: return "AMD MI350 (gfx950)";
        case AMDArch::RDNA2: return "AMD RDNA2 (gfx1030)";
        case AMDArch::RDNA3: return "AMD RDNA3 (gfx1100)";
        case AMDArch::RDNA4: return "AMD RDNA4 (gfx1200)";
        default: return "Unknown AMD GPU";
    }
}

/**
 * @brief Get GFX architecture string for compilation
 */
inline const char* arch_to_gfx(AMDArch arch) {
    switch (arch) {
        case AMDArch::MI100: return "gfx908";
        case AMDArch::MI200:
        case AMDArch::MI250: return "gfx90a";
        case AMDArch::MI300X:
        case AMDArch::MI300A:
        case AMDArch::MI325X: return "gfx942";
        case AMDArch::MI350: return "gfx950";
        case AMDArch::RDNA2: return "gfx1030";
        case AMDArch::RDNA3: return "gfx1100";
        case AMDArch::RDNA4: return "gfx1200";
        default: return "gfx942";
    }
}

/**
 * @brief Print device information for debugging
 */
inline void print_device_info() {
    AMDArch arch = detect_amd_arch();
    AMDVariant variant = detect_amd_variant();
    
    printf("=== YiRage AMD ROCm Device Info ===\n");
    printf("GPU: %s\n", arch_to_string(arch));
    printf("GFX: %s\n", arch_to_gfx(arch));
    printf("Compute Units: %d\n", get_compute_unit_count());
    printf("HBM: %d GB\n", get_hbm_memory_gb());
    printf("Wavefront Size: 64\n");
    
    // Print architecture features
    switch (arch) {
        case AMDArch::MI350:
            printf("Features: CDNA4, 64x64 MFMA, FP8, Sparsity, 128KB LDS\n");
            break;
        case AMDArch::MI325X:
        case AMDArch::MI300X:
        case AMDArch::MI300A:
            printf("Features: CDNA3, 32x32x16 MFMA, FP8, Sparsity\n");
            break;
        case AMDArch::MI250:
        case AMDArch::MI200:
            printf("Features: CDNA2, 32x32x8 MFMA, Async Copy\n");
            break;
        case AMDArch::MI100:
        default:
            printf("Features: CDNA1, 32x32x8 MFMA\n");
            break;
    }
    printf("===================================\n");
}

/**
 * @brief Check if MFMA (Matrix Core) is supported
 */
inline bool has_matrix_core(AMDArch arch) {
    return arch == AMDArch::MI100 || arch == AMDArch::MI200 ||
           arch == AMDArch::MI250 || arch == AMDArch::MI300X ||
           arch == AMDArch::MI300A || arch == AMDArch::MI325X ||
           arch == AMDArch::MI350;
}

/**
 * @brief Check if FP8 is supported
 */
inline bool has_fp8(AMDArch arch) {
    return arch == AMDArch::MI300X || arch == AMDArch::MI300A ||
           arch == AMDArch::MI325X || arch == AMDArch::MI350;
}

/**
 * @brief Check if sparse MFMA is supported
 */
inline bool has_sparsity(AMDArch arch) {
    return arch == AMDArch::MI300X || arch == AMDArch::MI300A ||
           arch == AMDArch::MI325X || arch == AMDArch::MI350;
}

/**
 * @brief Check if this is a CDNA architecture (data center)
 */
inline bool is_cdna(AMDArch arch) {
    return arch == AMDArch::MI100 || arch == AMDArch::MI200 ||
           arch == AMDArch::MI250 || arch == AMDArch::MI300X ||
           arch == AMDArch::MI300A || arch == AMDArch::MI325X ||
           arch == AMDArch::MI350;
}

}  // namespace rocm
}  // namespace persistent_kernel
}  // namespace yirage
