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
 * TPU Kernel Optimizer
 * 
 * Optimization logic for Google TPU kernel generation.
 */

#include "kernel/tpu/tpu_kernel_config.h"

#ifdef YIRAGE_BACKEND_TPU_ENABLED

#include <algorithm>
#include <cmath>
#include <sstream>

namespace yirage {
namespace kernel {
namespace tpu {

void TPUOptimizer::compute_optimal_tiling(int m, int n, int k,
                                          TPUVersion version,
                                          TPUKernelConfig &config) {
    int mxu_size = config.get_mxu_size();
    
    // Tiles should be multiples of MXU size (128)
    config.tile_m = std::min(m, ((m + mxu_size - 1) / mxu_size) * mxu_size);
    config.tile_n = std::min(n, ((n + mxu_size - 1) / mxu_size) * mxu_size);
    config.tile_k = std::min(k, mxu_size);
    
    // Limit by VMEM
    size_t vmem_size = config.get_vmem_size();
    size_t tile_bytes = (config.tile_m * config.tile_k + 
                         config.tile_k * config.tile_n +
                         config.tile_m * config.tile_n) * 2;  // BF16
    
    while (tile_bytes > vmem_size * 0.8 && config.tile_m > mxu_size) {
        config.tile_m /= 2;
        tile_bytes = (config.tile_m * config.tile_k + 
                     config.tile_k * config.tile_n +
                     config.tile_m * config.tile_n) * 2;
    }
    
    while (tile_bytes > vmem_size * 0.8 && config.tile_n > mxu_size) {
        config.tile_n /= 2;
        tile_bytes = (config.tile_m * config.tile_k + 
                     config.tile_k * config.tile_n +
                     config.tile_m * config.tile_n) * 2;
    }
}

size_t TPUOptimizer::estimate_memory_usage(TPUKernelConfig const &config,
                                           int m, int n, int k) {
    size_t tile_bytes = (config.tile_m * config.tile_k + 
                         config.tile_k * config.tile_n +
                         config.tile_m * config.tile_n) * 2;
    
    // Account for double buffering
    if (config.enable_double_buffering) {
        tile_bytes *= 2;
    }
    
    return tile_bytes;
}

bool TPUOptimizer::fits_in_vmem(TPUKernelConfig const &config,
                                size_t required_bytes) {
    return required_bytes <= config.get_vmem_size();
}

void TPUOptimizer::optimize_pipeline(TPUKernelConfig &config,
                                     int m, int n, int k) {
    // Larger problems benefit from deeper pipelines
    size_t problem_size = static_cast<size_t>(m) * n * k;
    
    if (problem_size > 1e9) {
        config.pipeline_depth = 4;
        config.enable_double_buffering = true;
    } else if (problem_size > 1e6) {
        config.pipeline_depth = 2;
        config.enable_double_buffering = true;
    } else {
        config.pipeline_depth = 1;
        config.enable_double_buffering = false;
    }
}

std::string TPUOptimizer::generate_matmul_xla(int m, int n, int k,
                                              TPUKernelConfig const &config) {
    std::ostringstream oss;
    
    oss << "# XLA HLO for TPU GEMM\n";
    oss << "# M=" << m << ", N=" << n << ", K=" << k << "\n";
    oss << "# Tile: " << config.tile_m << "x" << config.tile_n << "x" << config.tile_k << "\n";
    oss << "\n";
    oss << "HloModule gemm_module\n\n";
    oss << "ENTRY gemm {\n";
    oss << "  a = bf16[" << m << "," << k << "] parameter(0)\n";
    oss << "  b = bf16[" << k << "," << n << "] parameter(1)\n";
    oss << "  ROOT dot = bf16[" << m << "," << n << "] dot(a, b), ";
    oss << "lhs_contracting_dims={1}, rhs_contracting_dims={0}\n";
    oss << "}\n";
    
    return oss.str();
}

std::string TPUOptimizer::generate_pallas_kernel(std::string const &op_name,
                                                 TPUKernelConfig const &config) {
    std::ostringstream oss;
    
    oss << "# Pallas kernel for TPU: " << op_name << "\n";
    oss << "import jax\n";
    oss << "from jax.experimental import pallas as pl\n\n";
    
    if (op_name == "matmul") {
        oss << "@pl.pallas_call(\n";
        oss << "    out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),\n";
        oss << "    grid=(" << config.tile_m << ", " << config.tile_n << "),\n";
        oss << "    in_specs=[pl.BlockSpec((" << config.tile_m << ", " << config.tile_k << "), lambda i, j: (i, 0)),\n";
        oss << "              pl.BlockSpec((" << config.tile_k << ", " << config.tile_n << "), lambda i, j: (0, j))],\n";
        oss << "    out_specs=pl.BlockSpec((" << config.tile_m << ", " << config.tile_n << "), lambda i, j: (i, j)),\n";
        oss << ")\n";
        oss << "def matmul_kernel(a_ref, b_ref, o_ref):\n";
        oss << "    o_ref[...] = jnp.dot(a_ref[...], b_ref[...])\n";
    } else if (op_name == "rms_norm") {
        oss << "def rms_norm_kernel(x_ref, weight_ref, o_ref, eps=1e-5):\n";
        oss << "    x = x_ref[...]\n";
        oss << "    rms = jnp.sqrt(jnp.mean(x ** 2) + eps)\n";
        oss << "    o_ref[...] = x / rms * weight_ref[...]\n";
    }
    
    return oss.str();
}

void TPUOptimizer::get_optimal_mesh(int num_tpus, int &mesh_x, int &mesh_y) {
    // Find factors closest to square root
    mesh_x = 1;
    mesh_y = num_tpus;
    
    for (int i = static_cast<int>(std::sqrt(num_tpus)); i >= 1; i--) {
        if (num_tpus % i == 0) {
            mesh_x = i;
            mesh_y = num_tpus / i;
            break;
        }
    }
}

}  // namespace tpu
}  // namespace kernel
}  // namespace yirage

#endif  // YIRAGE_BACKEND_TPU_ENABLED
