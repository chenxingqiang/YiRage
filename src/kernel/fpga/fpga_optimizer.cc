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
 * FPGA Kernel Optimizer
 */

#include "kernel/fpga/fpga_kernel_config.h"

#ifdef YIRAGE_BACKEND_FPGA_ENABLED

#include <algorithm>
#include <cmath>
#include <sstream>

namespace yirage {
namespace kernel {
namespace fpga {

int FPGAOptimizer::compute_optimal_parallelism(size_t problem_size,
                                               FPGADevice device,
                                               int available_dsp) {
    // DSP usage: ~2 DSPs per FP16 MAC, ~1 DSP per INT8 MAC
    int max_parallel = available_dsp / 2;
    
    // Balance parallelism with routing complexity
    int optimal = static_cast<int>(std::sqrt(max_parallel));
    optimal = std::min(optimal, 64);  // Practical limit
    
    // Round to power of 2
    int power = 1;
    while (power * 2 <= optimal) power *= 2;
    
    return power;
}

void FPGAOptimizer::compute_optimal_tiling(int m, int n, int k,
                                           FPGAKernelConfig &config) {
    // BRAM-based tiling
    size_t bram_bytes = config.memory.bram_kb * 1024;
    
    // Start with reasonable tile sizes
    int tile_m = 32, tile_n = 32, tile_k = 32;
    
    // Increase while fits in BRAM
    while (tile_m * 2 <= m && 
           (tile_m * 2 * tile_k + tile_k * tile_n + tile_m * 2 * tile_n) * 2 <= bram_bytes) {
        tile_m *= 2;
    }
    
    while (tile_n * 2 <= n && 
           (tile_m * tile_k + tile_k * tile_n * 2 + tile_m * tile_n * 2) * 2 <= bram_bytes) {
        tile_n *= 2;
    }
}

void FPGAOptimizer::estimate_resources(FPGAKernelConfig const &config,
                                       float &lut_util, float &ff_util,
                                       float &bram_util, float &dsp_util) {
    // Rough estimates based on parallelism
    int parallel = config.num_parallel_ops;
    int cus = config.num_compute_units;
    
    // DSP usage
    int dsp_per_op = config.use_half ? 2 : (config.use_int8 ? 1 : 4);
    dsp_util = static_cast<float>(parallel * cus * dsp_per_op) / config.dsp_slices;
    
    // LUT usage (roughly proportional to DSP)
    lut_util = dsp_util * 0.8f;
    
    // FF usage
    ff_util = dsp_util * 0.6f;
    
    // BRAM usage
    bram_util = 0.3f + 0.1f * cus;
}

float FPGAOptimizer::estimate_latency_us(FPGAKernelConfig const &config,
                                         int m, int n, int k) {
    // Cycles = (m * n * k) / (parallel_ops * compute_units)
    float ops = static_cast<float>(m) * n * k * 2;
    float ops_per_cycle = config.num_parallel_ops * config.num_compute_units;
    float cycles = ops / ops_per_cycle;
    
    // Account for pipeline II
    cycles *= config.hls.pipeline_ii;
    
    // Convert to microseconds
    float freq_mhz = config.target_frequency_mhz;
    return cycles / freq_mhz;
}

float FPGAOptimizer::estimate_throughput_gops(FPGAKernelConfig const &config,
                                              int m, int n, int k) {
    float latency_us = estimate_latency_us(config, m, n, k);
    float ops = static_cast<float>(m) * n * k * 2;
    return ops / latency_us / 1000.0f;  // GOPS
}

std::string FPGAOptimizer::generate_hls_code(std::string const &op_name,
                                             FPGAKernelConfig const &config) {
    std::ostringstream oss;
    
    oss << "// HLS C++ kernel for FPGA: " << op_name << "\n";
    oss << "#include <hls_stream.h>\n";
    oss << "#include <ap_fixed.h>\n\n";
    
    if (op_name == "matmul") {
        oss << "void krnl_gemm(\n";
        oss << "    hls::stream<half>& a_stream,\n";
        oss << "    hls::stream<half>& b_stream,\n";
        oss << "    hls::stream<half>& c_stream,\n";
        oss << "    int M, int N, int K\n";
        oss << ") {\n";
        oss << "#pragma HLS INTERFACE axis port=a_stream\n";
        oss << "#pragma HLS INTERFACE axis port=b_stream\n";
        oss << "#pragma HLS INTERFACE axis port=c_stream\n";
        oss << "#pragma HLS DATAFLOW\n\n";
        oss << "    // Tiled matmul with parallel MACs\n";
        oss << "    half a_tile[TILE_M][TILE_K];\n";
        oss << "    half b_tile[TILE_K][TILE_N];\n";
        oss << "    half c_tile[TILE_M][TILE_N];\n";
        oss << "#pragma HLS ARRAY_PARTITION variable=a_tile cyclic factor=" << config.hls.array_partition_factor << " dim=2\n";
        oss << "#pragma HLS ARRAY_PARTITION variable=b_tile cyclic factor=" << config.hls.array_partition_factor << " dim=1\n\n";
        oss << "    // Compute loop with pipeline\n";
        oss << "    for (int i = 0; i < TILE_M; i++) {\n";
        oss << "        for (int j = 0; j < TILE_N; j++) {\n";
        oss << "#pragma HLS PIPELINE II=" << config.hls.pipeline_ii << "\n";
        oss << "            half sum = 0;\n";
        oss << "            for (int k = 0; k < TILE_K; k++) {\n";
        oss << "#pragma HLS UNROLL factor=" << config.hls.unroll_factor << "\n";
        oss << "                sum += a_tile[i][k] * b_tile[k][j];\n";
        oss << "            }\n";
        oss << "            c_tile[i][j] = sum;\n";
        oss << "        }\n";
        oss << "    }\n";
        oss << "}\n";
    }
    
    return oss.str();
}

std::string FPGAOptimizer::generate_opencl_kernel(std::string const &op_name,
                                                  FPGAKernelConfig const &config) {
    std::ostringstream oss;
    
    oss << "// OpenCL kernel for FPGA: " << op_name << "\n";
    oss << "__kernel __attribute__((reqd_work_group_size(1, 1, 1)))\n";
    oss << "void " << config.kernel_name << "(\n";
    oss << "    __global half* restrict a,\n";
    oss << "    __global half* restrict b,\n";
    oss << "    __global half* restrict c,\n";
    oss << "    int M, int N, int K\n";
    oss << ") {\n";
    oss << "    // Streaming compute with local memory\n";
    oss << "    __local half a_local[TILE_M][TILE_K];\n";
    oss << "    __local half b_local[TILE_K][TILE_N];\n";
    oss << "    // ... tiled matmul implementation ...\n";
    oss << "}\n";
    
    return oss.str();
}

std::string FPGAOptimizer::generate_hls_directives(FPGAKernelConfig const &config) {
    std::ostringstream oss;
    
    oss << "# Vitis HLS Directives\n";
    oss << "set_directive_pipeline -II " << config.hls.pipeline_ii << " \"compute_loop\"\n";
    oss << "set_directive_unroll -factor " << config.hls.unroll_factor << " \"inner_loop\"\n";
    oss << "set_directive_array_partition -type cyclic -factor " << config.hls.array_partition_factor << " -dim 2 \"a_tile\"\n";
    if (config.hls.enable_dataflow) {
        oss << "set_directive_dataflow \"top_function\"\n";
    }
    
    return oss.str();
}

void FPGAOptimizer::optimize_memory_access(FPGAKernelConfig &config,
                                           int data_width, int access_pattern) {
    // Optimize for coalesced memory access
    config.hls.array_partition_factor = std::max(4, data_width / 16);
    config.stream_depth = 32;
    config.enable_ping_pong = true;
}

std::string FPGAOptimizer::generate_connectivity_cfg(FPGAKernelConfig const &config) {
    std::ostringstream oss;
    
    oss << "[connectivity]\n";
    oss << "nk=" << config.kernel_name << ":" << config.num_compute_units << "\n";
    
    for (int i = 0; i < config.num_compute_units; i++) {
        oss << "sp=" << config.kernel_name << "_" << i + 1 << ".a:DDR[" << i % 4 << "]\n";
        oss << "sp=" << config.kernel_name << "_" << i + 1 << ".b:DDR[" << (i + 1) % 4 << "]\n";
        oss << "sp=" << config.kernel_name << "_" << i + 1 << ".c:DDR[" << (i + 2) % 4 << "]\n";
    }
    
    return oss.str();
}

}  // namespace fpga
}  // namespace kernel
}  // namespace yirage

#endif  // YIRAGE_BACKEND_FPGA_ENABLED
