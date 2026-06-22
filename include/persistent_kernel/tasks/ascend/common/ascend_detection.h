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
 * @file ascend_detection.h
 * @brief Runtime detection for Huawei Ascend NPUs
 *
 * Detection methods:
 * 1. Environment variables (YIRAGE_ASCEND_MODEL)
 * 2. npu-smi command line tool
 * 3. CANN/AscendCL runtime API (if available)
 */

#include "ascend_common.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

namespace yirage {
namespace persistent_kernel {
namespace ascend {

// =============================================================================
// Runtime Detection Implementation
// =============================================================================

/**
 * @brief Detect Ascend NPU model at runtime
 */
inline AscendModel detect_ascend_model() {
    // Check environment variable override first
    const char* env_model = std::getenv("YIRAGE_ASCEND_MODEL");
    if (env_model) {
        if (strcmp(env_model, "910C") == 0 || strcmp(env_model, "Ascend910C") == 0) {
            return AscendModel::ASCEND_910C;
        }
        if (strcmp(env_model, "910B") == 0 || strcmp(env_model, "Ascend910B") == 0) {
            return AscendModel::ASCEND_910B;
        }
        if (strcmp(env_model, "910A") == 0 || strcmp(env_model, "Ascend910A") == 0) {
            return AscendModel::ASCEND_910A;
        }
        if (strcmp(env_model, "910") == 0 || strcmp(env_model, "Ascend910") == 0) {
            return AscendModel::ASCEND_910;
        }
        if (strcmp(env_model, "310P") == 0 || strcmp(env_model, "Ascend310P") == 0) {
            return AscendModel::ASCEND_310P;
        }
        if (strcmp(env_model, "310") == 0 || strcmp(env_model, "Ascend310") == 0) {
            return AscendModel::ASCEND_310;
        }
    }

#ifdef __linux__
    // Try to detect via npu-smi (Ascend System Management Interface)
    FILE* pipe = popen("npu-smi info 2>/dev/null | grep 'Name'", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            
            std::string npu_info(buffer);
            
            // Parse NPU name to detect model
            if (npu_info.find("910C") != std::string::npos ||
                npu_info.find("Ascend 910C") != std::string::npos) {
                return AscendModel::ASCEND_910C;
            }
            if (npu_info.find("910B") != std::string::npos ||
                npu_info.find("Ascend 910B") != std::string::npos) {
                return AscendModel::ASCEND_910B;
            }
            if (npu_info.find("910A") != std::string::npos ||
                npu_info.find("Ascend 910A") != std::string::npos) {
                return AscendModel::ASCEND_910A;
            }
            if (npu_info.find("910") != std::string::npos ||
                npu_info.find("Ascend 910") != std::string::npos) {
                return AscendModel::ASCEND_910;
            }
            if (npu_info.find("310P") != std::string::npos ||
                npu_info.find("Ascend 310P") != std::string::npos) {
                return AscendModel::ASCEND_310P;
            }
            if (npu_info.find("310") != std::string::npos ||
                npu_info.find("Ascend 310") != std::string::npos) {
                return AscendModel::ASCEND_310;
            }
        } else {
            pclose(pipe);
        }
    }
    
    // Alternative: Check via /dev/davinci* devices
    FILE* check_910 = fopen("/dev/davinci0", "r");
    if (check_910) {
        fclose(check_910);
        
        // Check memory size to distinguish models
        FILE* mem_pipe = popen("npu-smi info -t memory 2>/dev/null | grep -i 'total' | head -1", "r");
        if (mem_pipe) {
            char mem_buffer[128];
            if (fgets(mem_buffer, sizeof(mem_buffer), mem_pipe) != nullptr) {
                pclose(mem_pipe);
                
                std::string mem_info(mem_buffer);
                // Parse memory to distinguish 910 variants
                if (mem_info.find("96") != std::string::npos) {
                    return AscendModel::ASCEND_910C;
                }
                if (mem_info.find("64") != std::string::npos) {
                    return AscendModel::ASCEND_910B;
                }
                if (mem_info.find("32") != std::string::npos) {
                    return AscendModel::ASCEND_910;
                }
            } else {
                pclose(mem_pipe);
            }
        }
        
        // Default to 910B if davinci device exists but can't determine model
        return AscendModel::ASCEND_910B;
    }
#endif

    // Check ASCEND_HOME for CANN SDK hints
    const char* ascend_home = std::getenv("ASCEND_HOME");
    if (!ascend_home) {
        ascend_home = std::getenv("ASCEND_TOOLKIT_HOME");
    }
    
    if (ascend_home) {
        std::string path(ascend_home);
        // CANN version might indicate supported hardware
        if (path.find("8.") != std::string::npos) {
            return AscendModel::ASCEND_910C;  // CANN 8.x for latest
        }
        if (path.find("7.") != std::string::npos) {
            return AscendModel::ASCEND_910B;  // CANN 7.x for 910B
        }
    }

    // Default to 910B if unable to detect
    return AscendModel::ASCEND_910B;
}

/**
 * @brief Detect Ascend series
 */
inline AscendSeries detect_ascend_series() {
    AscendModel model = detect_ascend_model();
    
    switch (model) {
        case AscendModel::ASCEND_310:
        case AscendModel::ASCEND_310P:
            return AscendSeries::SERIES_300;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
        case AscendModel::ASCEND_910B:
        case AscendModel::ASCEND_910C:
            return AscendSeries::SERIES_900;
        default:
            return AscendSeries::UNKNOWN;
    }
}

/**
 * @brief Get AI Core count for detected NPU
 */
inline int get_ai_core_count() {
#ifdef __linux__
    FILE* pipe = popen("npu-smi info -t usages 2>/dev/null | grep -i 'aicore' | wc -l", "r");
    if (pipe) {
        char buffer[64];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            int count = atoi(buffer);
            if (count > 0) return count;
        } else {
            pclose(pipe);
        }
    }
#endif

    AscendModel model = detect_ascend_model();
    
    switch (model) {
        case AscendModel::ASCEND_910C:
            return 48;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
        case AscendModel::ASCEND_910B:
            return 32;
        case AscendModel::ASCEND_310P:
            return 8;
        case AscendModel::ASCEND_310:
        default:
            return 2;
    }
}

/**
 * @brief Get HBM memory size in GB
 */
inline int get_hbm_memory_gb() {
#ifdef __linux__
    FILE* pipe = popen("npu-smi info -t memory 2>/dev/null | grep -i 'total' | awk '{print $NF}'", "r");
    if (pipe) {
        char buffer[64];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            int memory_mb = atoi(buffer);
            if (memory_mb > 0) return memory_mb / 1024;
        } else {
            pclose(pipe);
        }
    }
#endif

    AscendModel model = detect_ascend_model();
    
    switch (model) {
        case AscendModel::ASCEND_910C:
            return 96;
        case AscendModel::ASCEND_910B:
            return 64;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
            return 32;
        case AscendModel::ASCEND_310P:
            return 24;
        case AscendModel::ASCEND_310:
        default:
            return 8;
    }
}

/**
 * @brief Get L1 buffer size per AI Core in KB
 */
inline int get_l1_buffer_kb() {
    AscendModel model = detect_ascend_model();
    
    switch (model) {
        case AscendModel::ASCEND_910C:
            return 1024;
        case AscendModel::ASCEND_910B:
            return 512;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
        case AscendModel::ASCEND_310P:
            return 256;
        case AscendModel::ASCEND_310:
        default:
            return 128;
    }
}

/**
 * @brief Convert AscendModel to string
 */
inline const char* model_to_string(AscendModel model) {
    switch (model) {
        case AscendModel::ASCEND_310: return "Ascend 310";
        case AscendModel::ASCEND_310P: return "Ascend 310P";
        case AscendModel::ASCEND_910: return "Ascend 910";
        case AscendModel::ASCEND_910A: return "Ascend 910A";
        case AscendModel::ASCEND_910B: return "Ascend 910B";
        case AscendModel::ASCEND_910C: return "Ascend 910C";
        default: return "Unknown Ascend NPU";
    }
}

/**
 * @brief Convert AscendSeries to string
 */
inline const char* series_to_string(AscendSeries series) {
    switch (series) {
        case AscendSeries::SERIES_300: return "Ascend 300 Series (Inference)";
        case AscendSeries::SERIES_900: return "Ascend 900 Series (Training)";
        default: return "Unknown";
    }
}

/**
 * @brief Print device information for debugging
 */
inline void print_device_info() {
    AscendModel model = detect_ascend_model();
    AscendSeries series = detect_ascend_series();
    
    printf("=== YiRage Ascend NPU Device Info ===\n");
    printf("NPU: %s\n", model_to_string(model));
    printf("Series: %s\n", series_to_string(series));
    printf("AI Cores: %d\n", get_ai_core_count());
    printf("L1 Buffer: %d KB per AI Core\n", get_l1_buffer_kb());
    printf("HBM: %d GB\n", get_hbm_memory_gb());
    
    // Print model-specific features
    switch (model) {
        case AscendModel::ASCEND_910C:
            printf("Features: 32x32 Cube, BF16, INT4, 1MB L1\n");
            break;
        case AscendModel::ASCEND_910B:
            printf("Features: 16x16 Cube, BF16, INT4, 512KB L1\n");
            break;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
            printf("Features: 16x16 Cube, FP32 Cube, 256KB L1\n");
            break;
        case AscendModel::ASCEND_310P:
            printf("Features: 16x16 Cube, BF16, Inference-optimized\n");
            break;
        case AscendModel::ASCEND_310:
        default:
            printf("Features: 16x16 Cube, Edge inference\n");
            break;
    }
    printf("=====================================\n");
}

/**
 * @brief Check if BF16 is supported
 */
inline bool has_bf16_support() {
    AscendModel model = detect_ascend_model();
    return (model == AscendModel::ASCEND_310P ||
            model == AscendModel::ASCEND_910B ||
            model == AscendModel::ASCEND_910C);
}

/**
 * @brief Check if INT4 quantization is supported
 */
inline bool has_int4_support() {
    AscendModel model = detect_ascend_model();
    return (model == AscendModel::ASCEND_910B ||
            model == AscendModel::ASCEND_910C);
}

/**
 * @brief Check if 32x32 Cube is available
 */
inline bool has_large_cube() {
    AscendModel model = detect_ascend_model();
    return (model == AscendModel::ASCEND_910C);
}

}  // namespace ascend
}  // namespace persistent_kernel
}  // namespace yirage
