/* Copyright 2025 Chen Xingqiang (YiRage Project)
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
 *
 * Metal Element-wise Operations for Apple Silicon
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Binary Operations
// =============================================================================

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
}

kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
}

// Vectorized add (4x throughput)
kernel void add_f16_vec4(
    device const half4* a [[buffer(0)]],
    device const half4* b [[buffer(1)]],
    device half4* c [[buffer(2)]],
    constant uint& size_vec4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size_vec4) {
        c[gid] = a[gid] + b[gid];
    }
}

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        c[gid] = a[gid] * b[gid];
    }
}

kernel void mul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        c[gid] = a[gid] * b[gid];
    }
}

// =============================================================================
// SiLU + Mul (Gated activation)
// =============================================================================

kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        float g = gate[gid];
        float u = up[gid];
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        float silu = g / (1.0f + exp(-g));
        output[gid] = silu * u;
    }
}

kernel void silu_mul_f16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        float g = float(gate[gid]);
        float u = float(up[gid]);
        float silu = g / (1.0f + exp(-g));
        output[gid] = half(silu * u);
    }
}

// Vectorized SiLU+Mul
kernel void silu_mul_f16_vec4(
    device const half4* gate [[buffer(0)]],
    device const half4* up [[buffer(1)]],
    device half4* output [[buffer(2)]],
    constant uint& size_vec4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size_vec4) {
        float4 g = float4(gate[gid]);
        float4 u = float4(up[gid]);
        float4 silu = g / (1.0f + exp(-g));
        output[gid] = half4(silu * u);
    }
}

// =============================================================================
// Activation Functions
// =============================================================================

kernel void relu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        output[gid] = max(0.0f, input[gid]);
    }
}

kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        float x = input[gid];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr float sqrt_2_over_pi = 0.7978845608f;
        constexpr float coeff = 0.044715f;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        output[gid] = 0.5f * x * (1.0f + tanh(inner));
    }
}

kernel void gelu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        float x = float(input[gid]);
        constexpr float sqrt_2_over_pi = 0.7978845608f;
        constexpr float coeff = 0.044715f;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        output[gid] = half(0.5f * x * (1.0f + tanh(inner)));
    }
}

kernel void silu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        float x = input[gid];
        output[gid] = x / (1.0f + exp(-x));
    }
}

kernel void sigmoid_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        output[gid] = 1.0f / (1.0f + exp(-input[gid]));
    }
}

kernel void tanh_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        output[gid] = tanh(input[gid]);
    }
}

// =============================================================================
// Fused Operations
// =============================================================================

// Fused Add + ReLU
kernel void add_relu_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        c[gid] = max(0.0f, a[gid] + b[gid]);
    }
}

// Fused Mul + Add (FMA)
kernel void fma_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* d [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        d[gid] = fma(a[gid], b[gid], c[gid]);
    }
}

// Scale (multiply by constant)
kernel void scale_f16(
    device half* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        data[gid] = half(float(data[gid]) * scale);
    }
}

// =============================================================================
// Type Conversion
// =============================================================================

kernel void cast_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        output[gid] = half(input[gid]);
    }
}

kernel void cast_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        output[gid] = float(input[gid]);
    }
}
