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
 * Intel XPU Kernel Optimizer
 */

#include "kernel/xpu/xpu_kernel_config.h"

#ifdef YIRAGE_BACKEND_XPU_ENABLED

#include <algorithm>
#include <cmath>
#include <sstream>

namespace yirage {
namespace kernel {
namespace xpu {

int XPUOptimizer::compute_optimal_simd_width(size_t problem_size, XPUArch arch) {
    // PVC and Arc support SIMD8, SIMD16, SIMD32
    if (problem_size < 1024) {
        return 8;
    } else if (problem_size < 65536) {
        return 16;
    } else {
        return 32;
    }
}

size_t XPUOptimizer::compute_optimal_slm(size_t data_size, SLMLayout layout) {
    // Round up to 256-byte boundary for bank efficiency
    size_t slm = ((data_size + 255) / 256) * 256;
    
    // Add padding for blocked layout to avoid bank conflicts
    if (layout == SLMLayout::BLOCKED) {
        slm = slm * 33 / 32;  // ~3% overhead
    }
    
    return slm;
}

bool XPUOptimizer::has_bank_conflict(SLMLayout layout, int stride) {
    // SLM has 32 banks, 4 bytes per bank
    if (layout == SLMLayout::VNNI_PACKED) {
        return false;  // VNNI layout is optimized
    }
    
    return (stride % 32 == 0);  // Power-of-2 strides cause conflicts
}

float XPUOptimizer::estimate_occupancy(XPUKernelConfig const &config,
                                       int registers_used) {
    int threads_per_wg = config.simd_width * config.num_sub_groups;
    int threads_per_eu = config.threads_per_eu;
    
    // Limit by threads
    int wgs_by_threads = threads_per_eu * config.eu_per_subslice / threads_per_wg;
    
    // Limit by registers (256 per thread on PVC)
    int regs_per_eu = 256 * threads_per_eu;
    int wgs_by_regs = regs_per_eu / (registers_used * threads_per_wg);
    
    // Limit by SLM
    size_t slm_per_subslice = config.slm_size;
    int wgs_by_slm = slm_per_subslice / (config.slm_size / config.num_sub_groups);
    
    int max_wgs = std::min({wgs_by_threads, wgs_by_regs, wgs_by_slm, 8});
    
    return static_cast<float>(max_wgs) / 8.0f;
}

bool XPUOptimizer::select_xmx_config(int m, int n, int k,
                                     XPUArch arch,
                                     XPUKernelConfig &config) {
    if (!config.has_xmx()) {
        return false;
    }
    
    // XMX supports 8x16 systolic array
    config.use_xmx = true;
    config.xmx_m = 8;
    config.xmx_n = 16;
    config.xmx_k = 16;
    
    // Select precision based on problem size
    if (m * n * k > 1e9) {
        config.use_bf16 = true;  // Large problems: use BF16 for speed
    } else {
        config.use_bf16 = true;  // Default to BF16
    }
    
    return true;
}

std::string XPUOptimizer::generate_sycl_kernel(std::string const &op_name,
                                               XPUKernelConfig const &config) {
    std::ostringstream oss;
    
    oss << "// SYCL kernel for Intel XPU: " << op_name << "\n";
    oss << "#include <sycl/sycl.hpp>\n";
    oss << "#include <sycl/ext/intel/experimental/esimd.hpp>\n\n";
    
    if (op_name == "matmul") {
        oss << "template<int M, int N, int K>\n";
        oss << "void gemm_xmx(sycl::queue& q,\n";
        oss << "              sycl::buffer<sycl::ext::oneapi::bfloat16>& a,\n";
        oss << "              sycl::buffer<sycl::ext::oneapi::bfloat16>& b,\n";
        oss << "              sycl::buffer<float>& c) {\n";
        oss << "    constexpr int TILE_M = " << config.xmx_m << ";\n";
        oss << "    constexpr int TILE_N = " << config.xmx_n << ";\n";
        oss << "    constexpr int TILE_K = " << config.xmx_k << ";\n";
        oss << "    constexpr int SIMD = " << config.simd_width << ";\n\n";
        oss << "    q.submit([&](sycl::handler& h) {\n";
        oss << "        auto a_acc = a.get_access<sycl::access::mode::read>(h);\n";
        oss << "        auto b_acc = b.get_access<sycl::access::mode::read>(h);\n";
        oss << "        auto c_acc = c.get_access<sycl::access::mode::write>(h);\n\n";
        oss << "        h.parallel_for<class gemm_kernel>(\n";
        oss << "            sycl::nd_range<2>({M/TILE_M, N/TILE_N}, {1, SIMD}),\n";
        oss << "            [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SIMD)]] {\n";
        oss << "                // XMX-based matrix multiplication\n";
        oss << "                // Uses DPAS (Dot Product Accumulate Systolic)\n";
        oss << "                auto sg = item.get_sub_group();\n";
        oss << "                // ... XMX compute ...\n";
        oss << "            });\n";
        oss << "    });\n";
        oss << "}\n";
    } else if (op_name == "rms_norm") {
        oss << "void rms_norm_xpu(sycl::queue& q,\n";
        oss << "                  sycl::buffer<float>& input,\n";
        oss << "                  sycl::buffer<float>& weight,\n";
        oss << "                  sycl::buffer<float>& output,\n";
        oss << "                  int hidden_dim, float eps) {\n";
        oss << "    constexpr int SIMD = " << config.simd_width << ";\n\n";
        oss << "    q.submit([&](sycl::handler& h) {\n";
        oss << "        auto in = input.get_access<sycl::access::mode::read>(h);\n";
        oss << "        auto w = weight.get_access<sycl::access::mode::read>(h);\n";
        oss << "        auto out = output.get_access<sycl::access::mode::write>(h);\n";
        oss << "        sycl::local_accessor<float, 1> slm(hidden_dim, h);\n\n";
        oss << "        h.parallel_for<class rms_kernel>(\n";
        oss << "            sycl::nd_range<1>({batch_size}, {SIMD}),\n";
        oss << "            [=](sycl::nd_item<1> item) {\n";
        oss << "                // Sub-group reduction for sum of squares\n";
        oss << "                // ...\n";
        oss << "            });\n";
        oss << "    });\n";
        oss << "}\n";
    }
    
    return oss.str();
}

std::string XPUOptimizer::generate_onednn_config(std::string const &op_name,
                                                 XPUKernelConfig const &config) {
    std::ostringstream oss;
    
    oss << "// oneDNN primitive configuration for: " << op_name << "\n";
    oss << "dnnl::engine engine(dnnl::engine::kind::gpu, 0);\n";
    oss << "dnnl::stream stream(engine);\n\n";
    
    if (op_name == "matmul") {
        oss << "auto src_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::ab);\n";
        oss << "auto weights_md = dnnl::memory::desc({K, N}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::ba);\n";
        oss << "auto dst_md = dnnl::memory::desc({M, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);\n\n";
        oss << "auto matmul_pd = dnnl::matmul::primitive_desc(engine, src_md, weights_md, dst_md);\n";
        oss << "auto matmul_prim = dnnl::matmul(matmul_pd);\n";
    }
    
    return oss.str();
}

void XPUOptimizer::optimize_work_groups(int problem_m, int problem_n,
                                        int problem_k, XPUArch arch,
                                        XPUKernelConfig &config) {
    // Determine optimal SIMD width
    size_t problem_size = static_cast<size_t>(problem_m) * problem_n;
    config.simd_width = compute_optimal_simd_width(problem_size, arch);
    
    // Determine number of sub-groups
    if (arch == XPUArch::PONTE_VECCHIO) {
        config.num_sub_groups = 8;  // High occupancy for PVC
    } else {
        config.num_sub_groups = 4;  // Arc/Flex
    }
    
    // Enable multi-tile for large problems
    if (problem_size > 1e6 && arch == XPUArch::PONTE_VECCHIO) {
        config.enable_multi_tile = true;
    }
}

}  // namespace xpu
}  // namespace kernel
}  // namespace yirage

#endif  // YIRAGE_BACKEND_XPU_ENABLED
