/* Copyright 2025 YiRage Team */

#include "threadblock/element_unary.h"
#include "threadblock/xpu/element_unary.h"

#ifdef YIRAGE_USE_XPU

#include <sstream>

namespace yirage {
namespace threadblock {
namespace xpu {

std::string generate_element_unary_sycl(const std::string& op_name, int num_elements) {
    std::ostringstream sycl;
    sycl << "#include <sycl/sycl.hpp>\n\n";
    sycl << "void " << op_name << "_kernel(sycl::queue& q,\n";
    sycl << "                             sycl::buffer<float>& input,\n";
    sycl << "                             sycl::buffer<float>& output,\n";
    sycl << "                             int num_elements) {\n";
    sycl << "    q.submit([&](sycl::handler& h) {\n";
    sycl << "        auto in = input.get_access<sycl::access::mode::read>(h);\n";
    sycl << "        auto out = output.get_access<sycl::access::mode::write>(h);\n";
    sycl << "        h.parallel_for<class " << op_name << "_kernel>(\n";
    sycl << "            sycl::range<1>(num_elements),\n";
    sycl << "            [=](sycl::id<1> i) {\n";
    sycl << "                out[i] = " << op_name << "_op(in[i]);\n";
    sycl << "            });\n";
    sycl << "    });\n";
    sycl << "}\n";
    return sycl.str();
}

}  // namespace xpu
}  // namespace threadblock
}  // namespace yirage

#endif
