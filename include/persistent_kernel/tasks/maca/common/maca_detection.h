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
 * @file maca_detection.h
 * @brief Runtime detection for MetaX MACA GPUs
 *
 * Detection methods:
 * 1. Environment variables (YIRAGE_MACA_GEN, YIRAGE_MACA_VARIANT)
 * 2. mx-smi command line tool
 * 3. MACA runtime API (if available)
 */

#include "maca_common.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

namespace yirage {
namespace persistent_kernel {
namespace maca {

// =============================================================================
// Runtime Detection Implementation
// =============================================================================

/**
 * @brief Detect MetaX GPU generation at runtime
 */
inline MetaXGen detect_metax_gen() {
    // Check environment variable override first
    const char* env_gen = std::getenv("YIRAGE_MACA_GEN");
    if (env_gen) {
        if (strcmp(env_gen, "C500") == 0 || strcmp(env_gen, "500") == 0) {
            return MetaXGen::C500;
        }
        if (strcmp(env_gen, "C550") == 0 || strcmp(env_gen, "550") == 0) {
            return MetaXGen::C550;
        }
        if (strcmp(env_gen, "C600") == 0 || strcmp(env_gen, "600") == 0) {
            return MetaXGen::C600;
        }
        if (strcmp(env_gen, "C650") == 0 || strcmp(env_gen, "650") == 0) {
            return MetaXGen::C650;
        }
        if (strcmp(env_gen, "C700") == 0 || strcmp(env_gen, "700") == 0) {
            return MetaXGen::C700;
        }
    }

#ifdef __linux__
    // Try to detect via mx-smi (MetaX System Management Interface)
    FILE* pipe = popen("mx-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            
            std::string gpu_name(buffer);
            
            // Parse GPU name to detect generation
            if (gpu_name.find("C700") != std::string::npos ||
                gpu_name.find("c700") != std::string::npos) {
                return MetaXGen::C700;
            }
            if (gpu_name.find("C650") != std::string::npos ||
                gpu_name.find("c650") != std::string::npos) {
                return MetaXGen::C650;
            }
            if (gpu_name.find("C600") != std::string::npos ||
                gpu_name.find("c600") != std::string::npos) {
                return MetaXGen::C600;
            }
            if (gpu_name.find("C550") != std::string::npos ||
                gpu_name.find("c550") != std::string::npos) {
                return MetaXGen::C550;
            }
            if (gpu_name.find("C500") != std::string::npos ||
                gpu_name.find("c500") != std::string::npos) {
                return MetaXGen::C500;
            }
        } else {
            pclose(pipe);
        }
    }
#endif

    // Check MACA_HOME environment for SDK version hints
    const char* maca_home = std::getenv("MACA_HOME");
    if (maca_home) {
        std::string path(maca_home);
        // Version in path might indicate generation
        if (path.find("3.") != std::string::npos) {
            return MetaXGen::C700;  // SDK 3.x for C700
        }
        if (path.find("2.") != std::string::npos) {
            return MetaXGen::C600;  // SDK 2.x for C600
        }
    }

    // Default to C500 if unable to detect
    return MetaXGen::C500;
}

/**
 * @brief Detect MetaX GPU variant
 */
inline MetaXVariant detect_metax_variant() {
    const char* env_variant = std::getenv("YIRAGE_MACA_VARIANT");
    if (env_variant) {
        if (strcmp(env_variant, "STANDARD") == 0 || strcmp(env_variant, "standard") == 0) {
            return MetaXVariant::STANDARD;
        }
        if (strcmp(env_variant, "PRO") == 0 || strcmp(env_variant, "pro") == 0) {
            return MetaXVariant::PRO;
        }
        if (strcmp(env_variant, "MAX") == 0 || strcmp(env_variant, "max") == 0) {
            return MetaXVariant::MAX;
        }
        if (strcmp(env_variant, "ULTRA") == 0 || strcmp(env_variant, "ultra") == 0) {
            return MetaXVariant::ULTRA;
        }
    }

#ifdef __linux__
    // Try to detect via mx-smi
    FILE* pipe = popen("mx-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            
            std::string gpu_name(buffer);
            
            if (gpu_name.find("Ultra") != std::string::npos ||
                gpu_name.find("ULTRA") != std::string::npos) {
                return MetaXVariant::ULTRA;
            }
            if (gpu_name.find("Max") != std::string::npos ||
                gpu_name.find("MAX") != std::string::npos) {
                return MetaXVariant::MAX;
            }
            if (gpu_name.find("Pro") != std::string::npos ||
                gpu_name.find("PRO") != std::string::npos) {
                return MetaXVariant::PRO;
            }
        } else {
            pclose(pipe);
        }
    }
#endif

    return MetaXVariant::STANDARD;
}

/**
 * @brief Get SM count for detected GPU
 */
inline int get_sm_count() {
#ifdef __linux__
    FILE* pipe = popen("mx-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buffer[64];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            
            // Estimate SM count from memory size
            int memory_mb = atoi(buffer);
            if (memory_mb >= 250000) return 200;  // C700 Pro
            if (memory_mb >= 180000) return 160;  // C700 Standard
            if (memory_mb >= 180000) return 180;  // C650 Pro
            if (memory_mb >= 120000) return 144;  // C650 Standard
            if (memory_mb >= 120000) return 160;  // C600 Pro
            if (memory_mb >= 90000) return 128;   // C600 Standard
            if (memory_mb >= 90000) return 140;   // C550 Pro
            if (memory_mb >= 75000) return 112;   // C550 Standard
            if (memory_mb >= 75000) return 128;   // C500 Pro
            return 104;                            // C500 Standard
        } else {
            pclose(pipe);
        }
    }
#endif

    MetaXGen gen = detect_metax_gen();
    MetaXVariant variant = detect_metax_variant();
    
    switch (gen) {
        case MetaXGen::C700:
            return (variant == MetaXVariant::PRO) ? 200 : 160;
        case MetaXGen::C650:
            return (variant == MetaXVariant::PRO) ? 180 : 144;
        case MetaXGen::C600:
            return (variant == MetaXVariant::PRO) ? 160 : 128;
        case MetaXGen::C550:
            return (variant == MetaXVariant::PRO) ? 140 : 112;
        case MetaXGen::C500:
        default:
            return (variant == MetaXVariant::PRO) ? 128 : 104;
    }
}

/**
 * @brief Get HBM memory size in GB
 */
inline int get_hbm_memory_gb() {
#ifdef __linux__
    FILE* pipe = popen("mx-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buffer[64];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            int memory_mb = atoi(buffer);
            return memory_mb / 1024;
        } else {
            pclose(pipe);
        }
    }
#endif

    MetaXGen gen = detect_metax_gen();
    MetaXVariant variant = detect_metax_variant();
    
    switch (gen) {
        case MetaXGen::C700:
            return (variant == MetaXVariant::PRO) ? 256 : 192;
        case MetaXGen::C650:
            return (variant == MetaXVariant::PRO) ? 192 : 128;
        case MetaXGen::C600:
            return (variant == MetaXVariant::PRO) ? 128 : 96;
        case MetaXGen::C550:
            return (variant == MetaXVariant::PRO) ? 96 : 80;
        case MetaXGen::C500:
        default:
            return (variant == MetaXVariant::PRO) ? 80 : 64;
    }
}

/**
 * @brief Convert MetaXGen to string
 */
inline const char* gen_to_string(MetaXGen gen) {
    switch (gen) {
        case MetaXGen::C500: return "MetaX C500";
        case MetaXGen::C550: return "MetaX C550";
        case MetaXGen::C600: return "MetaX C600";
        case MetaXGen::C650: return "MetaX C650";
        case MetaXGen::C700: return "MetaX C700";
        default: return "Unknown MetaX GPU";
    }
}

/**
 * @brief Convert MetaXVariant to string
 */
inline const char* variant_to_string(MetaXVariant variant) {
    switch (variant) {
        case MetaXVariant::STANDARD: return "Standard";
        case MetaXVariant::PRO: return "Pro";
        case MetaXVariant::MAX: return "Max";
        case MetaXVariant::ULTRA: return "Ultra";
        default: return "Unknown";
    }
}

/**
 * @brief Print device information for debugging
 */
inline void print_device_info() {
    MetaXGen gen = detect_metax_gen();
    MetaXVariant variant = detect_metax_variant();
    
    printf("=== YiRage MACA Device Info ===\n");
    printf("GPU: %s %s\n", gen_to_string(gen), variant_to_string(variant));
    printf("SMs: %d\n", get_sm_count());
    printf("HBM: %d GB\n", get_hbm_memory_gb());
    printf("Warp Size: 64 (CRITICAL: NOT 32!)\n");
    
    // Print generation-specific features
    switch (gen) {
        case MetaXGen::C700:
            printf("Features: 2048 threads/block, 192KB shared mem, advanced sparsity\n");
            break;
        case MetaXGen::C650:
        case MetaXGen::C600:
            printf("Features: 128KB shared mem, 2:4 sparsity acceleration\n");
            break;
        case MetaXGen::C550:
            printf("Features: 96KB shared mem, improved tensor cores\n");
            break;
        case MetaXGen::C500:
        default:
            printf("Features: 64KB shared mem, tensor cores\n");
            break;
    }
    printf("===============================\n");
}

}  // namespace maca
}  // namespace persistent_kernel
}  // namespace yirage
