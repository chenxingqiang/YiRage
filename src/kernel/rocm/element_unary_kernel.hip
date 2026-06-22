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
 * ROCm/HIP Element-wise Unary Kernels
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// =============================================================================
// Activation Functions
// =============================================================================

/**
 * @brief ReLU FP32
 */
__global__ void relu_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float val = input[i];
        output[i] = val > 0.0f ? val : 0.0f;
    }
}

/**
 * @brief GELU FP32 (approximation)
 */
__global__ void gelu_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    
    for (int i = idx; i < size; i += stride) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/**
 * @brief GELU FP16 (approximation)
 */
__global__ void gelu_f16_hip(
    const half* __restrict__ input,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    
    for (int i = idx; i < size; i += stride) {
        float x = __half2float(input[i]);
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        output[i] = __float2half(0.5f * x * (1.0f + tanhf(inner)));
    }
}

/**
 * @brief SiLU (Swish) FP32
 */
__global__ void silu_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

/**
 * @brief SiLU FP16 (vectorized)
 */
__global__ void silu_f16_vec_hip(
    const half2* __restrict__ input,
    half2* __restrict__ output,
    int size_half2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size_half2; i += stride) {
        float2 x = __half22float2(input[i]);
        x.x = x.x / (1.0f + expf(-x.x));
        x.y = x.y / (1.0f + expf(-x.y));
        output[i] = __float22half2_rn(x);
    }
}

/**
 * @brief Sigmoid FP32
 */
__global__ void sigmoid_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

/**
 * @brief Tanh FP32
 */
__global__ void tanh_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

/**
 * @brief Exp FP32
 */
__global__ void exp_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = expf(input[i]);
    }
}

/**
 * @brief Log FP32
 */
__global__ void log_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = logf(input[i]);
    }
}

/**
 * @brief Rsqrt FP32
 */
__global__ void rsqrt_f32_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = rsqrtf(input[i]);
    }
}

/**
 * @brief Cast FP32 to FP16
 */
__global__ void cast_f32_to_f16_hip(
    const float* __restrict__ input,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = __float2half(input[i]);
    }
}

/**
 * @brief Cast FP16 to FP32
 */
__global__ void cast_f16_to_f32_hip(
    const half* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = __half2float(input[i]);
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_relu_f32_hip(
    const float* input, float* output, int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(relu_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, input, output, size);
}

void launch_gelu_f32_hip(
    const float* input, float* output, int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(gelu_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, input, output, size);
}

void launch_gelu_f16_hip(
    const half* input, half* output, int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(gelu_f16_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, input, output, size);
}

void launch_silu_f32_hip(
    const float* input, float* output, int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(silu_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, input, output, size);
}

void launch_silu_f16_hip(
    const half* input, half* output, int size, hipStream_t stream
) {
    int block_size = 256;
    int size_half2 = size / 2;
    int num_blocks = min((size_half2 + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(silu_f16_vec_hip, dim3(num_blocks), dim3(block_size),
                       0, stream,
                       reinterpret_cast<const half2*>(input),
                       reinterpret_cast<half2*>(output),
                       size_half2);
}

void launch_sigmoid_f32_hip(
    const float* input, float* output, int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(sigmoid_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, input, output, size);
}

void launch_tanh_f32_hip(
    const float* input, float* output, int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    hipLaunchKernelGGL(tanh_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, input, output, size);
}

}  // extern "C"
