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
 * ROCm/HIP Element-wise Binary Kernels
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// =============================================================================
// Binary Operation Kernels
// =============================================================================

template<typename T, typename Op>
__global__ void binary_op_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int size,
    Op op
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = op(a[idx], b[idx]);
    }
}

/**
 * @brief Element-wise Add FP32
 */
__global__ void add_f32_hip(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief Element-wise Add FP16 (vectorized)
 */
__global__ void add_f16_hip(
    const half2* __restrict__ a,
    const half2* __restrict__ b,
    half2* __restrict__ c,
    int size_half2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size_half2; i += stride) {
        c[i] = __hadd2(a[i], b[i]);
    }
}

/**
 * @brief Element-wise Mul FP32
 */
__global__ void mul_f32_hip(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        c[i] = a[i] * b[i];
    }
}

/**
 * @brief Element-wise Mul FP16 (vectorized)
 */
__global__ void mul_f16_hip(
    const half2* __restrict__ a,
    const half2* __restrict__ b,
    half2* __restrict__ c,
    int size_half2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size_half2; i += stride) {
        c[i] = __hmul2(a[i], b[i]);
    }
}

/**
 * @brief SiLU + Mul (fused gate activation)
 * 
 * output = silu(gate) * up
 * where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
__global__ void silu_mul_f16_hip(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 4;  // Process 4 elements per thread
    
    if (base + 3 < size) {
        // Load 4 elements at once
        half2 g0 = *reinterpret_cast<const half2*>(gate + base);
        half2 g1 = *reinterpret_cast<const half2*>(gate + base + 2);
        half2 u0 = *reinterpret_cast<const half2*>(up + base);
        half2 u1 = *reinterpret_cast<const half2*>(up + base + 2);
        
        // Convert to float for computation
        float2 gf0 = __half22float2(g0);
        float2 gf1 = __half22float2(g1);
        float2 uf0 = __half22float2(u0);
        float2 uf1 = __half22float2(u1);
        
        // SiLU + Mul
        gf0.x = gf0.x / (1.0f + expf(-gf0.x)) * uf0.x;
        gf0.y = gf0.y / (1.0f + expf(-gf0.y)) * uf0.y;
        gf1.x = gf1.x / (1.0f + expf(-gf1.x)) * uf1.x;
        gf1.y = gf1.y / (1.0f + expf(-gf1.y)) * uf1.y;
        
        // Store
        *reinterpret_cast<half2*>(output + base) = __float22half2_rn(gf0);
        *reinterpret_cast<half2*>(output + base + 2) = __float22half2_rn(gf1);
    } else {
        // Handle tail elements
        for (int i = base; i < size && i < base + 4; i++) {
            float g = __half2float(gate[i]);
            float u = __half2float(up[i]);
            output[i] = __float2half(g / (1.0f + expf(-g)) * u);
        }
    }
}

/**
 * @brief Fused Add + ReLU
 */
__global__ void add_relu_f32_hip(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        float val = a[i] + b[i];
        c[i] = val > 0.0f ? val : 0.0f;
    }
}

/**
 * @brief Fused Mul + Add (a * b + c) = FMA
 */
__global__ void fma_f32_hip(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ d,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        d[i] = fmaf(a[i], b[i], c[i]);
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_add_f32_hip(
    const float* a, const float* b, float* c,
    int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    num_blocks = min(num_blocks, 65535);
    
    hipLaunchKernelGGL(add_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, a, b, c, size);
}

void launch_add_f16_hip(
    const half* a, const half* b, half* c,
    int size, hipStream_t stream
) {
    int size_half2 = size / 2;
    int block_size = 256;
    int num_blocks = (size_half2 + block_size - 1) / block_size;
    num_blocks = min(num_blocks, 65535);
    
    hipLaunchKernelGGL(add_f16_hip, dim3(num_blocks), dim3(block_size),
                       0, stream,
                       reinterpret_cast<const half2*>(a),
                       reinterpret_cast<const half2*>(b),
                       reinterpret_cast<half2*>(c),
                       size_half2);
}

void launch_mul_f32_hip(
    const float* a, const float* b, float* c,
    int size, hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    num_blocks = min(num_blocks, 65535);
    
    hipLaunchKernelGGL(mul_f32_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, a, b, c, size);
}

void launch_silu_mul_f16_hip(
    const half* gate, const half* up, half* output,
    int size, hipStream_t stream
) {
    int block_size = 256;
    int elements_per_thread = 4;
    int num_threads = (size + elements_per_thread - 1) / elements_per_thread;
    int num_blocks = (num_threads + block_size - 1) / block_size;
    num_blocks = min(num_blocks, 65535);
    
    hipLaunchKernelGGL(silu_mul_f16_hip, dim3(num_blocks), dim3(block_size),
                       0, stream, gate, up, output, size);
}

}  // extern "C"
