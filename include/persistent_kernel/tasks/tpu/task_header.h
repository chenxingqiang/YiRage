/* Copyright 2025 YiRage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include "common/tpu_common.h"
#include "common/tpu_detection.h"

/**
 * @file task_header.h
 * @brief Main include file for TPU persistent kernel tasks
 * 
 * TPU kernels are written in:
 * - XLA HLO (high-level operations)
 * - Pallas (JAX-based kernel language)
 * - Mosaic (low-level TPU programming)
 */

namespace yirage {
namespace persistent_kernel {
namespace tpu {

// Task types for TPU
enum class TPUTaskType {
    GEMM,
    ATTENTION,
    SOFTMAX,
    RMS_NORM,
    EMBEDDING,
    ARGMAX,
    TOPK,
    FLASH_ATTENTION,
    MOE
};

// Task descriptor for TPU
struct TPUTaskDesc {
    TPUTaskType type;
    TPUVersion target_version;
    int m, n, k;              // Problem dimensions
    bool use_sparsity;
    bool use_int4;
    int num_attention_heads;
    int head_dim;
};

/**
 * @brief Get optimal config for task
 */
inline TPUKernelConfig get_optimal_config(TPUTaskDesc const& task) {
    switch (task.target_version) {
        case TPUVersion::V2: return TPU_V2_KERNEL_CONFIG;
        case TPUVersion::V3: return TPU_V3_KERNEL_CONFIG;
        case TPUVersion::V4: return TPU_V4_KERNEL_CONFIG;
        case TPUVersion::V5E: return TPU_V5E_KERNEL_CONFIG;
        case TPUVersion::V5P: return TPU_V5P_KERNEL_CONFIG;
        default: return TPU_V4_KERNEL_CONFIG;
    }
}

}  // namespace tpu
}  // namespace persistent_kernel
}  // namespace yirage
