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
 * @file silu_mul.metal.h
 * @brief MPS SiLU activation and multiply kernel
 *
 * Computes: output = SiLU(gate) * up
 * where SiLU(x) = x * sigmoid(x)
 *
 * Used in LLaMA/Qwen FFN layers (SwiGLU activation).
 */

namespace yirage {
namespace persistent_kernel {
namespace mps {

constexpr const char* SILU_MUL_KERNEL_SOURCE = R"(
// SiLU (Swish) activation with element-wise multiply
// output = SiLU(gate) * up = gate * sigmoid(gate) * up
kernel void silu_mul_kernel(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)size) return;
    
    float g = gate[tid];
    float sigmoid_g = 1.0f / (1.0f + exp(-g));
    output[tid] = (g * sigmoid_g) * up[tid];
}

// Fused SiLU + multiply for half precision
kernel void silu_mul_kernel_half(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant int& size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)size) return;
    
    float g = float(gate[tid]);
    float sigmoid_g = 1.0f / (1.0f + exp(-g));
    output[tid] = half((g * sigmoid_g) * float(up[tid]));
}

// GELU activation (for BERT-style models)
// output = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)size) return;
    
    float x = input[tid];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[tid] = x * cdf;
}
)";

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
