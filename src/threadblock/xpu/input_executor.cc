/* Copyright 2025 YiRage Team */

#include "threadblock/xpu/input_loader.h"

#ifdef YIRAGE_USE_XPU

#include <sstream>

namespace yirage {
namespace threadblock {
namespace xpu {

std::string generate_slm_load_sycl(int num_elements) {
    std::ostringstream sycl;
    sycl << "#include <sycl/sycl.hpp>\n\n";
    sycl << "void load_to_slm(sycl::queue& q,\n";
    sycl << "                 sycl::buffer<float>& global_buf,\n";
    sycl << "                 int num_elements) {\n";
    sycl << "    q.submit([&](sycl::handler& h) {\n";
    sycl << "        auto global = global_buf.get_access<sycl::access::mode::read>(h);\n";
    sycl << "        sycl::local_accessor<float, 1> slm(num_elements, h);\n";
    sycl << "        h.parallel_for<class slm_load>(\n";
    sycl << "            sycl::nd_range<1>({num_elements}, {256}),\n";
    sycl << "            [=](sycl::nd_item<1> item) {\n";
    sycl << "                int i = item.get_global_id(0);\n";
    sycl << "                slm[item.get_local_id(0)] = global[i];\n";
    sycl << "            });\n";
    sycl << "    });\n";
    sycl << "}\n";
    return sycl.str();
}

}  // namespace xpu
}  // namespace threadblock
}  // namespace yirage

#endif
