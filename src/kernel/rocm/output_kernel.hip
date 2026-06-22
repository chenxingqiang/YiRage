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
 * ROCm/HIP Output Kernels
 * 
 * Kernels for LM head, sampling, and output processing.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVEFRONT_SIZE 64

// =============================================================================
// LM Head (Linear projection to vocabulary)
// =============================================================================

/**
 * @brief LM Head projection: hidden -> vocab logits
 * 
 * Essentially a GEMM: logits = hidden @ weight.T
 */
__global__ void lm_head_f16_hip(
    const half* __restrict__ hidden,
    const half* __restrict__ weight,
    float* __restrict__ logits,
    int batch_size,
    int hidden_dim,
    int vocab_size
) {
    // Each block handles one batch element
    int batch = blockIdx.x;
    int vocab_start = blockIdx.y * blockDim.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size) return;
    
    const half* h = hidden + batch * hidden_dim;
    float* out = logits + batch * vocab_size;
    
    int vocab_idx = vocab_start + tid;
    if (vocab_idx >= vocab_size) return;
    
    const half* w = weight + vocab_idx * hidden_dim;
    
    // Dot product
    float sum = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        sum += __half2float(h[i]) * __half2float(w[i]);
    }
    
    out[vocab_idx] = sum;
}

// =============================================================================
// Sampling Operations
// =============================================================================

/**
 * @brief Top-K selection using wavefront operations
 */
__global__ void topk_f32_hip(
    const float* __restrict__ logits,
    float* __restrict__ top_values,
    int* __restrict__ top_indices,
    int batch_size,
    int vocab_size,
    int k
) {
    __shared__ float shared_vals[64];
    __shared__ int shared_idxs[64];
    
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    const float* in = logits + batch * vocab_size;
    float* out_vals = top_values + batch * k;
    int* out_idxs = top_indices + batch * k;
    
    int tid = threadIdx.x;
    
    // Find top-k using insertion sort in registers
    float my_vals[16];  // Assume k <= 16
    int my_idxs[16];
    
    for (int i = 0; i < k; i++) {
        my_vals[i] = -INFINITY;
        my_idxs[i] = -1;
    }
    
    // Each thread scans a portion of vocabulary
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = in[v];
        
        // Insert into sorted list if larger than smallest
        if (val > my_vals[k - 1]) {
            // Find insertion position
            int pos = k - 1;
            while (pos > 0 && val > my_vals[pos - 1]) {
                my_vals[pos] = my_vals[pos - 1];
                my_idxs[pos] = my_idxs[pos - 1];
                pos--;
            }
            my_vals[pos] = val;
            my_idxs[pos] = v;
        }
    }
    
    // Merge across threads (simplified - use parallel reduction)
    // For now, use atomic compare-and-swap
    __syncthreads();
    
    // First thread collects results (simplified)
    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            out_vals[i] = my_vals[i];
            out_idxs[i] = my_idxs[i];
        }
    }
}

/**
 * @brief Apply temperature to logits
 */
__global__ void apply_temperature_f32_hip(
    float* __restrict__ logits,
    int size,
    float temperature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float inv_temp = 1.0f / temperature;
    
    for (int i = idx; i < size; i += stride) {
        logits[i] *= inv_temp;
    }
}

/**
 * @brief Apply top-p (nucleus) filtering
 */
__global__ void apply_top_p_f32_hip(
    float* __restrict__ probs,
    int batch_size,
    int vocab_size,
    float top_p
) {
    // Each block handles one batch
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    float* p = probs + batch * vocab_size;
    
    __shared__ float cumsum;
    
    int tid = threadIdx.x;
    
    // Cumulative sum (simplified - needs sorting first)
    // This is a placeholder - real implementation needs parallel sort + scan
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            sum += p[i];
            if (sum >= top_p) {
                // Zero out remaining
                for (int j = i + 1; j < vocab_size; j++) {
                    p[j] = 0.0f;
                }
                break;
            }
        }
    }
}

/**
 * @brief Multinomial sampling with Gumbel-max trick
 */
__global__ void sample_multinomial_hip(
    const float* __restrict__ probs,
    const float* __restrict__ uniform_samples,
    int* __restrict__ output_tokens,
    int batch_size,
    int vocab_size
) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    const float* p = probs + batch * vocab_size;
    float u = uniform_samples[batch];
    
    // Inverse CDF sampling
    float cumsum = 0.0f;
    int selected = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        cumsum += p[i];
        if (cumsum >= u) {
            selected = i;
            break;
        }
    }
    
    output_tokens[batch] = selected;
}

/**
 * @brief Argmax (greedy decoding)
 */
__global__ void argmax_f32_hip(
    const float* __restrict__ logits,
    int* __restrict__ output_tokens,
    int batch_size,
    int vocab_size
) {
    __shared__ float shared_max[4];
    __shared__ int shared_idx[4];
    
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    const float* in = logits + batch * vocab_size;
    
    int tid = threadIdx.x;
    int wave_id = tid / WAVEFRONT_SIZE;
    int lane_id = tid % WAVEFRONT_SIZE;
    
    float local_max = -INFINITY;
    int local_idx = 0;
    
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = in[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }
    
    // Wavefront max reduction
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_xor(local_max, offset);
        int other_idx = __shfl_xor(local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }
    
    if (lane_id == 0) {
        shared_max[wave_id] = local_max;
        shared_idx[wave_id] = local_idx;
    }
    __syncthreads();
    
    if (wave_id == 0 && lane_id < 4) {
        local_max = shared_max[lane_id];
        local_idx = shared_idx[lane_id];
        
        for (int i = 1; i < 4; i++) {
            float other_max = __shfl_xor(local_max, i);
            int other_idx = __shfl_xor(local_idx, i);
            if (other_max > local_max) {
                local_max = other_max;
                local_idx = other_idx;
            }
        }
        
        if (lane_id == 0) {
            output_tokens[batch] = local_idx;
        }
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_lm_head_f16_hip(
    const half* hidden, const half* weight, float* logits,
    int batch_size, int hidden_dim, int vocab_size,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size, (vocab_size + 255) / 256);
    
    hipLaunchKernelGGL(lm_head_f16_hip, grid, block, 0, stream,
                       hidden, weight, logits,
                       batch_size, hidden_dim, vocab_size);
}

void launch_apply_temperature_f32_hip(
    float* logits, int size, float temperature,
    hipStream_t stream
) {
    int block_size = 256;
    int num_blocks = min((size + block_size - 1) / block_size, 65535);
    
    hipLaunchKernelGGL(apply_temperature_f32_hip,
                       dim3(num_blocks), dim3(block_size), 0, stream,
                       logits, size, temperature);
}

void launch_argmax_f32_hip(
    const float* logits, int* output_tokens,
    int batch_size, int vocab_size,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size);
    
    hipLaunchKernelGGL(argmax_f32_hip, grid, block, 0, stream,
                       logits, output_tokens, batch_size, vocab_size);
}

void launch_sample_multinomial_hip(
    const float* probs, const float* uniform_samples, int* output_tokens,
    int batch_size, int vocab_size,
    hipStream_t stream
) {
    dim3 block(1);
    dim3 grid(batch_size);
    
    hipLaunchKernelGGL(sample_multinomial_hip, grid, block, 0, stream,
                       probs, uniform_samples, output_tokens,
                       batch_size, vocab_size);
}

}  // extern "C"
