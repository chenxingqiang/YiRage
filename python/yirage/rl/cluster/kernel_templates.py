"""
Hardware-Specific Kernel Templates

Provides optimized kernel implementations for each hardware backend.
These templates are used by the KernelGenerator to produce production-ready code.
"""

from typing import Dict, List, Optional
from .kernel_coverage import KernelOpType
from .topology import DeviceType


# ============================================================================
# CUDA Templates (NVIDIA)
# ============================================================================

CUDA_TEMPLATES: Dict[KernelOpType, str] = {
    KernelOpType.MATMUL: """
// CUDA MATMUL with Tensor Cores (FP16)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Tile dimensions for Tensor Core WMMA
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void matmul_tensor_core(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K) {
    
    // Warp-level matrix multiply
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    
    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_n = (blockIdx.y * blockDim.y + threadIdx.y);
    
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    
    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = warp_m * WMMA_M;
        int a_col = k;
        int b_row = k;
        int b_col = warp_n * WMMA_N;
        
        if (a_row < M && a_col < K && b_row < K && b_col < N) {
            wmma::load_matrix_sync(a_frag, A + a_row * K + a_col, K);
            wmma::load_matrix_sync(b_frag, B + b_row * N + b_col, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    int c_row = warp_m * WMMA_M;
    int c_col = warp_n * WMMA_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
    }
}
""",
    KernelOpType.RMS_NORM: """
// CUDA RMSNorm with vectorized loads
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void rms_norm_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    int hidden_size,
    float eps) {
    
    int row = blockIdx.x;
    const half* row_input = input + row * hidden_size;
    half* row_output = output + row * hidden_size;
    
    // Compute variance using warp reduction
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }
    
    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    // Block reduce
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    if (lane == 0) shared_sum[wid] = sum_sq;
    __syncthreads();
    
    if (wid == 0) {
        sum_sq = (lane < blockDim.x / 32) ? shared_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
    
    __shared__ float rrms;
    if (threadIdx.x == 0) {
        rrms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    
    // Apply normalization
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]) * rrms;
        row_output[i] = __float2half(val * __half2float(weight[i]));
    }
}
""",
    KernelOpType.SILU: """
// CUDA SiLU (Swish) activation
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void silu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int N) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = __half2float(input[idx]);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = __float2half(x * sigmoid);
    }
}
""",
    KernelOpType.SOFTMAX: """
// CUDA Softmax with online algorithm
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void softmax_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch_size,
    int seq_len) {
    
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const half* row_input = input + row * seq_len;
    half* row_output = output + row * seq_len;
    
    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, __half2float(row_input[i]));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        sum += expf(__half2float(row_input[i]) - max_val);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    sum = shared_sum;
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = expf(__half2float(row_input[i]) - max_val) * inv_sum;
        row_output[i] = __float2half(val);
    }
}
""",
    KernelOpType.ATTENTION: """
// CUDA Multi-Head Attention (simplified)
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// For production, use FlashAttention from flash-attn library
__global__ void attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {
    
    // This is a simplified reference implementation
    // For production use FlashAttention-2
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    int qkv_offset = (b * num_heads + h) * seq_len * head_dim;
    
    // Compute attention scores
    extern __shared__ float scores[];
    
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = __half2float(Q[qkv_offset + q_idx * head_dim + d]);
            float k_val = __half2float(K[qkv_offset + k_idx * head_dim + d]);
            score += q_val * k_val;
        }
        scores[k_idx] = score * scale;
    }
    __syncthreads();
    
    // Softmax
    float max_score = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_score = fmaxf(max_score, scores[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }
    
    for (int i = 0; i < seq_len; i++) {
        scores[i] /= sum;
    }
    
    // Weighted sum of values
    for (int d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (int v_idx = 0; v_idx < seq_len; v_idx++) {
            out_val += scores[v_idx] * __half2float(V[qkv_offset + v_idx * head_dim + d]);
        }
        output[qkv_offset + q_idx * head_dim + d] = __float2half(out_val);
    }
}
""",
}


# ============================================================================
# ROCm Templates (AMD)
# ============================================================================

ROCM_TEMPLATES: Dict[KernelOpType, str] = {
    KernelOpType.MATMUL: """
// ROCm MATMUL with Matrix FMA (MFMA)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void matmul_mfma(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K) {
    
    // MFMA-based matrix multiplication
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
    
    int wavefront_m = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    int wavefront_n = blockIdx.y;
    
    fill_fragment(c_frag, __float2half(0.0f));
    
    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = wavefront_m * WMMA_M;
        int b_col = wavefront_n * WMMA_N;
        
        if (a_row < M && k < K && b_col < N) {
            load_matrix_sync(a_frag, A + a_row * K + k, K);
            load_matrix_sync(b_frag, B + k * N + b_col, N);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    int c_row = wavefront_m * WMMA_M;
    int c_col = wavefront_n * WMMA_N;
    if (c_row < M && c_col < N) {
        store_matrix_sync(C + c_row * N + c_col, c_frag, N, mem_row_major);
    }
}
""",
    KernelOpType.RMS_NORM: """
// ROCm RMSNorm using LDS (Local Data Share)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void rms_norm_hip(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const __half* __restrict__ weight,
    int hidden_size,
    float eps) {
    
    int row = blockIdx.x;
    const __half* row_input = input + row * hidden_size;
    __half* row_output = output + row * hidden_size;
    
    // Use LDS for reduction
    __shared__ float lds_sum[256];
    
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }
    
    lds_sum[threadIdx.x] = sum_sq;
    __syncthreads();
    
    // Reduction in LDS
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            lds_sum[threadIdx.x] += lds_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float rrms = rsqrtf(lds_sum[0] / hidden_size + eps);
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]) * rrms;
        row_output[i] = __float2half(val * __half2float(weight[i]));
    }
}
""",
}


# ============================================================================
# Triton Templates (Cross-Platform)
# ============================================================================

TRITON_TEMPLATES: Dict[KernelOpType, str] = {
    KernelOpType.MATMUL: """
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
""",
    KernelOpType.RMS_NORM: """
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    
    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    y = x * rrms * w
    
    tl.store(Y_ptr + row * stride_x + cols, y.to(tl.float16), mask=mask)
""",
    KernelOpType.FLASH_ATTENTION: """
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N, D,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Load Q block
    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.0)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Iterate over K, V blocks
    for start_n in range(0, N, BLOCK_N):
        offs_n_block = start_n + offs_n
        
        # Load K, V
        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vk
        
        k = tl.load(k_ptrs, mask=(offs_n_block[:, None] < N) & (offs_d[None, :] < D), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n_block[:, None] < N) & (offs_d[None, :] < D), other=0.0)
        
        # Compute attention scores
        s = tl.dot(q, tl.trans(k)) * scale
        
        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * tl.sum(tl.exp(s - m_ij[:, None]), axis=1)
        
        # Update accumulator
        p = tl.exp(s - m_new[:, None])
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v)
        
        m_i = m_new
        l_i = l_new
    
    # Finalize
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = O_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))
""",
    KernelOpType.GELU: """
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    X_ptr, Y_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    x = tl.load(X_ptr + offs, mask=mask).to(tl.float32)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    y = 0.5 * x * (1.0 + tl.libdevice.tanh(c * (x + 0.044715 * x * x * x)))
    
    tl.store(Y_ptr + offs, y.to(tl.float16), mask=mask)
""",
    KernelOpType.SILU: """
import triton
import triton.language as tl

@triton.jit
def silu_kernel(
    X_ptr, Y_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    x = tl.load(X_ptr + offs, mask=mask).to(tl.float32)
    y = x * tl.sigmoid(x)
    
    tl.store(Y_ptr + offs, y.to(tl.float16), mask=mask)
""",
}


# ============================================================================
# Ascend Templates (Huawei)
# ============================================================================

ASCEND_TEMPLATES: Dict[KernelOpType, str] = {
    KernelOpType.MATMUL: """
// Ascend C MATMUL using Cube Unit
#include "kernel_operator.h"
using namespace AscendC;

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int BLOCK_K = 16;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}
    
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                int M, int N, int K) {
        this->M = M;
        this->N = N;
        this->K = K;
        
        aGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a), M * K);
        bGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b), K * N);
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(c), M * N);
        
        pipe.InitBuffer(aQueue, 2, BLOCK_M * BLOCK_K * sizeof(half));
        pipe.InitBuffer(bQueue, 2, BLOCK_K * BLOCK_N * sizeof(half));
        pipe.InitBuffer(cQueue, 2, BLOCK_M * BLOCK_N * sizeof(half));
    }
    
    __aicore__ inline void Process() {
        int blockIdx = GetBlockIdx();
        int numBlocks = GetBlockNum();
        
        int tilesM = (M + BLOCK_M - 1) / BLOCK_M;
        int tilesN = (N + BLOCK_N - 1) / BLOCK_N;
        int totalTiles = tilesM * tilesN;
        
        for (int tile = blockIdx; tile < totalTiles; tile += numBlocks) {
            int tileM = tile / tilesN;
            int tileN = tile % tilesN;
            
            LocalTensor<half> aLocal = aQueue.AllocTensor<half>();
            LocalTensor<half> bLocal = bQueue.AllocTensor<half>();
            LocalTensor<half> cLocal = cQueue.AllocTensor<half>();
            
            // Load tiles
            DataCopy(aLocal, aGM[tileM * BLOCK_M * K], BLOCK_M * BLOCK_K);
            DataCopy(bLocal, bGM[tileN * BLOCK_N], BLOCK_K * BLOCK_N);
            
            // Cube Unit compute
            Matmul(cLocal, aLocal, bLocal, BLOCK_M, BLOCK_N, BLOCK_K);
            
            // Store result
            DataCopy(cGM[tileM * BLOCK_M * N + tileN * BLOCK_N], cLocal, BLOCK_M * BLOCK_N);
            
            aQueue.FreeTensor(aLocal);
            bQueue.FreeTensor(bLocal);
            cQueue.FreeTensor(cLocal);
        }
    }

private:
    int M, N, K;
    GlobalTensor<half> aGM, bGM, cGM;
    TPipe pipe;
    TQue<QuePosition::A1, 2> aQueue;
    TQue<QuePosition::B1, 2> bQueue;
    TQue<QuePosition::C1, 2> cQueue;
};

extern "C" __global__ __aicore__ void matmul_ascend(
    GM_ADDR a, GM_ADDR b, GM_ADDR c,
    int M, int N, int K) {
    KernelMatmul op;
    op.Init(a, b, c, M, N, K);
    op.Process();
}
""",
    KernelOpType.RMS_NORM: """
// Ascend C RMSNorm using Vector Unit
#include "kernel_operator.h"
using namespace AscendC;

class KernelRmsNorm {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                int batch_size, int hidden_size, float eps) {
        this->batch_size = batch_size;
        this->hidden_size = hidden_size;
        this->eps = eps;
        
        inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        weightGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight));
        outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    }
    
    __aicore__ inline void Process() {
        int blockIdx = GetBlockIdx();
        
        LocalTensor<half> x = inputQueue.AllocTensor<half>();
        LocalTensor<half> w = weightQueue.AllocTensor<half>();
        LocalTensor<half> y = outputQueue.AllocTensor<half>();
        
        // Load row
        DataCopy(x, inputGM[blockIdx * hidden_size], hidden_size);
        DataCopy(w, weightGM, hidden_size);
        
        // Compute variance
        LocalTensor<float> sum_sq = tmpQueue.AllocTensor<float>();
        Mul(sum_sq, x, x, hidden_size);
        float var = ReduceSum(sum_sq, hidden_size) / hidden_size;
        
        // Compute rrms
        float rrms = Rsqrt(var + eps);
        
        // Normalize and scale
        Muls(y, x, rrms, hidden_size);
        Mul(y, y, w, hidden_size);
        
        // Store
        DataCopy(outputGM[blockIdx * hidden_size], y, hidden_size);
    }

private:
    int batch_size, hidden_size;
    float eps;
    GlobalTensor<half> inputGM, weightGM, outputGM;
    TQue<QuePosition::VECIN, 2> inputQueue, weightQueue;
    TQue<QuePosition::VECOUT, 2> outputQueue;
    TQue<QuePosition::VECCALC, 1> tmpQueue;
};

extern "C" __global__ __aicore__ void rms_norm_ascend(
    GM_ADDR input, GM_ADDR weight, GM_ADDR output,
    int batch_size, int hidden_size, float eps) {
    KernelRmsNorm op;
    op.Init(input, weight, output, batch_size, hidden_size, eps);
    op.Process();
}
""",
}


# ============================================================================
# TPU Templates (Google)
# ============================================================================

TPU_TEMPLATES: Dict[KernelOpType, str] = {
    KernelOpType.MATMUL: """
# TPU MATMUL using Pallas
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(a_ref, b_ref, c_ref, *, acc_ref, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    
    i_m = pl.program_id(0)
    i_n = pl.program_id(1)
    i_k = pl.program_id(2)
    
    a_tile = a_ref[i_m, i_k]
    b_tile = b_ref[i_k, i_n]
    
    acc_ref[...] += jnp.dot(a_tile, b_tile)
    
    @pl.when(pl.program_id(2) == K // BLOCK_K - 1)
    def _():
        c_ref[i_m, i_n] = acc_ref[...].astype(c_ref.dtype)

@jax.jit
def matmul_tpu(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    M, K = A.shape
    K_, N = B.shape
    assert K == K_
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    grid = (M // BLOCK_M, N // BLOCK_N, K // BLOCK_K)
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), A.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_M, BLOCK_K), lambda i, j, k: (i, k)),
            pl.BlockSpec((BLOCK_K, BLOCK_N), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j, k: (i, j)),
        scratch_shapes=[pltpu.VMEM((BLOCK_M, BLOCK_N), jnp.float32)],
        compiler_params=dict(
            mosaic=dict(dimension_semantics=("parallel", "parallel", "arbitrary"))
        ),
    )(A, B, M=M, N=N, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
""",
    KernelOpType.FLASH_ATTENTION: """
# TPU FlashAttention using Pallas
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def flash_attention_kernel(
    q_ref, k_ref, v_ref, o_ref, *,
    m_ref, l_ref,  # Running max and sum
    scale,
    BLOCK_M, BLOCK_N, D,
):
    m_i = m_ref[...]
    l_i = l_ref[...]
    o_i = o_ref[...]
    
    q = q_ref[...]
    
    for j in range(0, k_ref.shape[0], BLOCK_N):
        k = pl.load(k_ref, (pl.ds(j, BLOCK_N), slice(None)))
        v = pl.load(v_ref, (pl.ds(j, BLOCK_N), slice(None)))
        
        s = jnp.dot(q, k.T) * scale
        
        m_ij = jnp.max(s, axis=-1, keepdims=True)
        m_new = jnp.maximum(m_i, m_ij)
        
        alpha = jnp.exp(m_i - m_new)
        beta = jnp.exp(s - m_new)
        
        l_new = alpha * l_i + jnp.sum(beta, axis=-1, keepdims=True)
        
        o_i = alpha * o_i + jnp.dot(beta, v)
        
        m_i = m_new
        l_i = l_new
    
    o_ref[...] = o_i / l_i
    m_ref[...] = m_i
    l_ref[...] = l_i

@jax.jit
def flash_attention_tpu(Q, K, V, scale=None):
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    
    if scale is None:
        scale = D ** -0.5
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    return pl.pallas_call(
        flash_attention_kernel,
        out_shape=jax.ShapeDtypeStruct((B, H, M, D), Q.dtype),
        grid=(B, H, M // BLOCK_M),
        in_specs=[
            pl.BlockSpec((1, 1, BLOCK_M, D), lambda b, h, i: (b, h, i, 0)),
            pl.BlockSpec((1, 1, N, D), lambda b, h, i: (b, h, 0, 0)),
            pl.BlockSpec((1, 1, N, D), lambda b, h, i: (b, h, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, 1, BLOCK_M, D), lambda b, h, i: (b, h, i, 0)),
    )(Q, K, V, scale=scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D)
""",
}


# ============================================================================
# CPU Templates (x86/ARM)
# ============================================================================

CPU_TEMPLATES: Dict[KernelOpType, str] = {
    KernelOpType.MATMUL: """
// CPU MATMUL with AVX-512 and OpenMP
#include <immintrin.h>
#include <omp.h>
#include <cstring>

void matmul_avx512(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    constexpr int BLOCK = 64;
    
    // Initialize C to zero
    memset(C, 0, M * N * sizeof(float));
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += BLOCK) {
        for (int jj = 0; jj < N; jj += BLOCK) {
            for (int kk = 0; kk < K; kk += BLOCK) {
                int i_end = (ii + BLOCK < M) ? ii + BLOCK : M;
                int j_end = (jj + BLOCK < N) ? jj + BLOCK : N;
                int k_end = (kk + BLOCK < K) ? kk + BLOCK : K;
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        __m512 a_val = _mm512_set1_ps(A[i * K + k]);
                        
                        for (int j = jj; j < j_end; j += 16) {
                            __m512 b_val = _mm512_loadu_ps(&B[k * N + j]);
                            __m512 c_val = _mm512_loadu_ps(&C[i * N + j]);
                            c_val = _mm512_fmadd_ps(a_val, b_val, c_val);
                            _mm512_storeu_ps(&C[i * N + j], c_val);
                        }
                    }
                }
            }
        }
    }
}
""",
    KernelOpType.RMS_NORM: """
// CPU RMSNorm with AVX-512
#include <immintrin.h>
#include <omp.h>
#include <cmath>

void rms_norm_avx512(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    int batch_size,
    int hidden_size,
    float eps) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        const float* row = input + b * hidden_size;
        float* out_row = output + b * hidden_size;
        
        // Compute sum of squares
        __m512 sum_sq = _mm512_setzero_ps();
        for (int i = 0; i < hidden_size; i += 16) {
            __m512 x = _mm512_loadu_ps(&row[i]);
            sum_sq = _mm512_fmadd_ps(x, x, sum_sq);
        }
        
        float var = _mm512_reduce_add_ps(sum_sq) / hidden_size;
        float rrms = 1.0f / sqrtf(var + eps);
        __m512 rrms_vec = _mm512_set1_ps(rrms);
        
        // Normalize and scale
        for (int i = 0; i < hidden_size; i += 16) {
            __m512 x = _mm512_loadu_ps(&row[i]);
            __m512 w = _mm512_loadu_ps(&weight[i]);
            __m512 y = _mm512_mul_ps(_mm512_mul_ps(x, rrms_vec), w);
            _mm512_storeu_ps(&out_row[i], y);
        }
    }
}
""",
}


# ============================================================================
# Template Registry
# ============================================================================

ALL_TEMPLATES: Dict[DeviceType, Dict[KernelOpType, str]] = {
    DeviceType.CUDA: CUDA_TEMPLATES,
    DeviceType.ROCM: ROCM_TEMPLATES,
    DeviceType.ASCEND: ASCEND_TEMPLATES,
    DeviceType.TPU: TPU_TEMPLATES,
    DeviceType.CPU: CPU_TEMPLATES,
}

# Triton templates can target multiple backends
TRITON_TARGETS = [DeviceType.CUDA, DeviceType.ROCM, DeviceType.XPU]


def get_template(op: KernelOpType, target: DeviceType) -> Optional[str]:
    """Get kernel template for operation and target."""
    # Check native templates first
    if target in ALL_TEMPLATES:
        templates = ALL_TEMPLATES[target]
        if op in templates:
            return templates[op]

    # Fall back to Triton for supported targets
    if target in TRITON_TARGETS and op in TRITON_TEMPLATES:
        return TRITON_TEMPLATES[op]

    return None


def list_available_templates() -> List[tuple]:
    """List all available (op, target) template pairs."""
    result = []

    for target, templates in ALL_TEMPLATES.items():
        for op in templates:
            result.append((op, target))

    for op in TRITON_TEMPLATES:
        for target in TRITON_TARGETS:
            if (op, target) not in result:
                result.append((op, target))

    return result
