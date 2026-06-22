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
 * AWS Neuron Kernel Interface (NKI) Operations Implementation
 */

#include "threadblock/nki/nki_ops.h"

#include <sstream>
#include <map>

namespace yirage {
namespace threadblock {
namespace nki {

// =============================================================================
// NKI Code Generator
// =============================================================================

std::string NKICodeGenerator::generate_matmul(
    int M, int N, int K,
    type::DataType dtype,
    const NKITileConfig& config
) {
    std::stringstream ss;
    
    ss << R"(
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
    """
    Matrix multiplication using NKI Tensor Engine
    C = A @ B where A is [M, K], B is [K, N], C is [M, N]
    """
    # Get partition ID
    i_p = nl.program_id(0)  # Partition index
    
    # Tile sizes - must be power of 2, max 128 for partition dim
    BLOCK_M = )" << config.partition_dim << R"(
    BLOCK_N = )" << config.free_dim << R"(
    BLOCK_K = 128  # Match Tensor Engine size
    
    # Allocate SBUF tiles
    a_tile = nl.ndarray((BLOCK_M, BLOCK_K), dtype=nl.)" << (dtype == type::DT_FLOAT16 ? "float16" : "bfloat16") << R"()
    b_tile = nl.ndarray((BLOCK_K, BLOCK_N), dtype=nl.)" << (dtype == type::DT_FLOAT16 ? "float16" : "bfloat16") << R"()
    c_tile = nl.ndarray((BLOCK_M, BLOCK_N), dtype=nl.float32)
    
    # Initialize accumulator
    c_tile[...] = 0.0
    
    # Loop over K dimension
    for k in nl.affine_range(0, K, BLOCK_K):
        # Load A tile from HBM to SBUF
        a_hbm = nisa.load(a_ptr + i_p * BLOCK_M * K + k, (BLOCK_M, BLOCK_K))
        nl.store(a_tile, a_hbm)
        
        # Load B tile from HBM to SBUF
        b_hbm = nisa.load(b_ptr + k * N, (BLOCK_K, BLOCK_N))
        nl.store(b_tile, b_hbm)
        
        # Tensor Engine matrix multiply
        # Uses 128x128 systolic array
        c_partial = nisa.nc_matmul(a_tile, b_tile)
        
        # Accumulate partial results
        c_tile[...] += c_partial
    
    # Store result to HBM
    nisa.store(c_ptr + i_p * BLOCK_M * N, c_tile)
)";
    
    return ss.str();
}

std::string NKICodeGenerator::generate_attention(
    int batch, int heads, int seq_len, int head_dim,
    bool causal,
    const NKITileConfig& config
) {
    std::stringstream ss;
    
    ss << R"(
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr, 
                           batch, heads, seq_len, head_dim):
    """
    Flash Attention implementation for NKI
    Uses Tensor Engine for Q@K^T and attention@V
    """
    # Partition over batch and heads
    b = nl.program_id(0)
    h = nl.program_id(1)
    
    # Tile configuration
    BLOCK_Q = )" << config.partition_dim << R"(
    BLOCK_KV = )" << config.free_dim << R"(
    
    # SBUF allocations
    q_tile = nl.ndarray((BLOCK_Q, head_dim), dtype=nl.bfloat16)
    k_tile = nl.ndarray((BLOCK_KV, head_dim), dtype=nl.bfloat16)
    v_tile = nl.ndarray((BLOCK_KV, head_dim), dtype=nl.bfloat16)
    
    # Running statistics for online softmax
    m_i = nl.ndarray((BLOCK_Q,), dtype=nl.float32)
    l_i = nl.ndarray((BLOCK_Q,), dtype=nl.float32)
    acc = nl.ndarray((BLOCK_Q, head_dim), dtype=nl.float32)
    
    m_i[...] = -float('inf')
    l_i[...] = 0.0
    acc[...] = 0.0
    
    # Iterate over sequence blocks
    for q_start in nl.affine_range(0, seq_len, BLOCK_Q):
        # Load Q block
        q_offset = b * heads * seq_len * head_dim + h * seq_len * head_dim + q_start * head_dim
        q_tile[...] = nisa.load(q_ptr + q_offset, (BLOCK_Q, head_dim))
        
        # Iterate over K/V blocks
        kv_end = q_start + BLOCK_Q if )" << (causal ? "True" : "False") << R"( else seq_len
        
        for kv_start in nl.affine_range(0, kv_end, BLOCK_KV):
            # Load K and V
            kv_offset = b * heads * seq_len * head_dim + h * seq_len * head_dim + kv_start * head_dim
            k_tile[...] = nisa.load(k_ptr + kv_offset, (BLOCK_KV, head_dim))
            v_tile[...] = nisa.load(v_ptr + kv_offset, (BLOCK_KV, head_dim))
            
            # Q @ K^T using Tensor Engine
            scores = nisa.nc_matmul(q_tile, nl.transpose(k_tile))
            
            # Scale by 1/sqrt(d)
            scale = 1.0 / nl.sqrt(float(head_dim))
            scores = scores * scale
            
            # Online softmax update
            m_ij = nl.max(scores, axis=1)
            m_new = nl.maximum(m_i, m_ij)
            alpha = nl.exp(m_i - m_new)
            beta = nl.exp(m_ij - m_new)
            
            p = nl.exp(scores - m_ij[:, None])
            l_new = alpha * l_i + beta * nl.sum(p, axis=1)
            
            # Attention @ V
            attn_v = nisa.nc_matmul(p.astype(nl.bfloat16), v_tile)
            acc = acc * (alpha * l_i / l_new)[:, None] + attn_v * (beta / l_new)[:, None]
            
            m_i = m_new
            l_i = l_new
        
        # Store output
        out_offset = b * heads * seq_len * head_dim + h * seq_len * head_dim + q_start * head_dim
        nisa.store(out_ptr + out_offset, acc.astype(nl.bfloat16))
)";
    
    return ss.str();
}

std::string NKICodeGenerator::generate_rms_norm(
    int hidden_dim,
    float epsilon
) {
    std::stringstream ss;
    
    ss << R"(
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit  
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, hidden_dim):
    """
    RMS Normalization using NKI Vector Engine
    """
    # Each partition handles one row
    row_idx = nl.program_id(0)
    
    # Load row to SBUF
    x = nl.ndarray((hidden_dim,), dtype=nl.bfloat16)
    x[...] = nisa.load(x_ptr + row_idx * hidden_dim, (hidden_dim,))
    
    # Compute variance using Vector Engine
    x_fp32 = x.astype(nl.float32)
    variance = nl.mean(x_fp32 * x_fp32)
    
    # RMS normalization
    rstd = nl.rsqrt(variance + )" << epsilon << R"()
    
    # Load weights and apply
    w = nl.ndarray((hidden_dim,), dtype=nl.bfloat16)
    w[...] = nisa.load(weight_ptr, (hidden_dim,))
    
    # Normalize
    out = (x_fp32 * rstd * w.astype(nl.float32)).astype(nl.bfloat16)
    
    # Store result
    nisa.store(out_ptr + row_idx * hidden_dim, out)
)";
    
    return ss.str();
}

std::string NKICodeGenerator::generate_elementwise(
    const std::string& op_name,
    int size,
    type::DataType dtype
) {
    std::stringstream ss;
    
    ss << R"(
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def )" << op_name << R"(_kernel(x_ptr, out_ptr, n):
    """
    Element-wise )" << op_name << R"( using NKI Vector Engine
    """
    # Partition over elements
    i = nl.program_id(0)
    BLOCK_SIZE = 128  # Vector Engine width
    
    # Load block
    offset = i * BLOCK_SIZE
    x = nl.ndarray((BLOCK_SIZE,), dtype=nl.bfloat16)
    x[...] = nisa.load(x_ptr + offset, (BLOCK_SIZE,))
    
    # Apply operation using Vector Engine
)";
    
    if (op_name == "silu") {
        ss << "    out = x * nl.sigmoid(x)\n";
    } else if (op_name == "gelu") {
        ss << "    out = 0.5 * x * (1.0 + nl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))\n";
    } else if (op_name == "relu") {
        ss << "    out = nl.maximum(x, 0.0)\n";
    } else if (op_name == "exp") {
        ss << "    out = nl.exp(x)\n";
    } else {
        ss << "    out = x  # Identity\n";
    }
    
    ss << R"(
    # Store result
    nisa.store(out_ptr + offset, out.astype(nl.bfloat16))
)";
    
    return ss.str();
}

// =============================================================================
// NKI Kernel Registry
// =============================================================================

NKIKernelRegistry& NKIKernelRegistry::instance() {
    static NKIKernelRegistry instance;
    return instance;
}

void NKIKernelRegistry::register_kernel(const std::string& name,
                                        const std::string& code) {
    kernels_[name] = code;
}

std::string NKIKernelRegistry::get_kernel(const std::string& name) const {
    auto it = kernels_.find(name);
    return (it != kernels_.end()) ? it->second : "";
}

bool NKIKernelRegistry::has_kernel(const std::string& name) const {
    return kernels_.find(name) != kernels_.end();
}

bool NKIKernelRegistry::compile_kernel(const std::string& name) {
    // This would invoke the Neuron compiler
    // neuronx-cc compile --target=trn1 kernel.py
    return true;
}

// =============================================================================
// Performance Estimator
// =============================================================================

NKIPerformanceEstimate estimate_nki_performance(
    const std::string& op_type,
    int M, int N, int K,
    type::DataType dtype
) {
    NKIPerformanceEstimate estimate;
    
    // Trainium v2 peak: ~380 TFLOPS BF16
    double peak_tflops = 380.0;
    
    if (op_type == "matmul") {
        // FLOPs for MatMul: 2 * M * N * K
        double flops = 2.0 * M * N * K;
        
        // Estimate utilization based on tile efficiency
        double utilization = 1.0;
        if (M < constants::TENSOR_ENGINE_SIZE || 
            N < constants::TENSOR_ENGINE_SIZE) {
            utilization = 0.5;
        }
        
        estimate.estimated_tflops = peak_tflops * utilization;
        estimate.tensor_engine_utilization = utilization;
        
        // SBUF usage
        size_t sbuf_used = (M + K + N) * constants::TENSOR_ENGINE_SIZE * 2;  // BF16
        estimate.sbuf_utilization = static_cast<double>(sbuf_used) / constants::SBUF_SIZE;
        
        // Latency estimate
        double time_s = flops / (estimate.estimated_tflops * 1e12);
        estimate.estimated_latency_us = static_cast<int>(time_s * 1e6);
    }
    
    return estimate;
}

}  // namespace nki
}  // namespace threadblock
}  // namespace yirage
