/* Copyright 2025 YiRage Team */

#include "threadblock/fpga/input_loader.h"

#ifdef YIRAGE_USE_FPGA

#include <sstream>

namespace yirage {
namespace threadblock {
namespace fpga {

std::string generate_ddr_to_bram_hls(int num_elements, int burst_len) {
    std::ostringstream hls;
    hls << "void load_ddr_to_bram(\n";
    hls << "    half* ddr_ptr,\n";
    hls << "    half bram[BRAM_SIZE],\n";
    hls << "    int num_elements\n";
    hls << ") {\n";
    hls << "#pragma HLS INTERFACE m_axi port=ddr_ptr bundle=gmem\n";
    hls << "#pragma HLS INTERFACE bram port=bram\n";
    hls << "    for (int i = 0; i < num_elements; i += " << burst_len << ") {\n";
    hls << "#pragma HLS PIPELINE II=1\n";
    hls << "        memcpy(&bram[i], &ddr_ptr[i], " << burst_len << " * sizeof(half));\n";
    hls << "    }\n";
    hls << "}\n";
    return hls.str();
}

}  // namespace fpga
}  // namespace threadblock
}  // namespace yirage

#endif
