/* Copyright 2025 YiRage Team */

#include "threadblock/element_unary.h"
#include "threadblock/fpga/element_unary.h"

#ifdef YIRAGE_USE_FPGA

#include <sstream>

namespace yirage {
namespace threadblock {
namespace fpga {

std::string generate_element_unary_hls(const std::string& op_name, int num_elements, int unroll) {
    std::ostringstream hls;
    hls << "void krnl_" << op_name << "(\n";
    hls << "    hls::stream<half>& input,\n";
    hls << "    hls::stream<half>& output,\n";
    hls << "    int num_elements\n";
    hls << ") {\n";
    hls << "#pragma HLS INTERFACE axis port=input\n";
    hls << "#pragma HLS INTERFACE axis port=output\n";
    hls << "    for (int i = 0; i < num_elements; i++) {\n";
    hls << "#pragma HLS PIPELINE II=1\n";
    hls << "#pragma HLS UNROLL factor=" << unroll << "\n";
    hls << "        half val = input.read();\n";
    hls << "        output.write(" << op_name << "_op(val));\n";
    hls << "    }\n";
    hls << "}\n";
    return hls.str();
}

}  // namespace fpga
}  // namespace threadblock
}  // namespace yirage

#endif
