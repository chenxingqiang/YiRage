/* Copyright 2025 YiRage Team */

#include "threadblock/graph.h"
#include "threadblock/matmul.h"
#include "threadblock/xpu/matmul.h"

#ifdef YIRAGE_USE_XPU

#include <sstream>

namespace yirage {
namespace threadblock {
namespace xpu {

// Generate SYCL kernel for Intel XPU
std::string generate_matmul_sycl(int m, int n, int k, int tile_m, int tile_n) {
    std::ostringstream sycl;
    sycl << "#include <sycl/sycl.hpp>\n";
    sycl << "#include <sycl/ext/intel/experimental/esimd.hpp>\n\n";
    sycl << "void matmul_xmx(sycl::queue& q,\n";
    sycl << "                sycl::buffer<sycl::ext::oneapi::bfloat16>& a,\n";
    sycl << "                sycl::buffer<sycl::ext::oneapi::bfloat16>& b,\n";
    sycl << "                sycl::buffer<float>& c,\n";
    sycl << "                int M, int N, int K) {\n";
    sycl << "    constexpr int TILE_M = " << tile_m << ";\n";
    sycl << "    constexpr int TILE_N = " << tile_n << ";\n";
    sycl << "    constexpr int SIMD = 16;\n\n";
    sycl << "    q.submit([&](sycl::handler& h) {\n";
    sycl << "        auto a_acc = a.get_access<sycl::access::mode::read>(h);\n";
    sycl << "        auto b_acc = b.get_access<sycl::access::mode::read>(h);\n";
    sycl << "        auto c_acc = c.get_access<sycl::access::mode::write>(h);\n";
    sycl << "        sycl::local_accessor<float, 2> slm({TILE_M, TILE_N}, h);\n\n";
    sycl << "        h.parallel_for<class gemm_xmx>(\n";
    sycl << "            sycl::nd_range<2>({M/TILE_M, N/TILE_N}, {1, SIMD}),\n";
    sycl << "            [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SIMD)]] {\n";
    sycl << "                // XMX-based DPAS matmul\n";
    sycl << "                auto sg = item.get_sub_group();\n";
    sycl << "                // ... XMX compute ...\n";
    sycl << "            });\n";
    sycl << "    });\n";
    sycl << "}\n";
    return sycl.str();
}

}  // namespace xpu
}  // namespace threadblock
}  // namespace yirage

#endif
