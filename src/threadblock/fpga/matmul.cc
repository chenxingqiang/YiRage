/* Copyright 2025 YiRage Team */

#include "threadblock/graph.h"
#include "threadblock/matmul.h"
#include "threadblock/fpga/matmul.h"

#ifdef YIRAGE_USE_FPGA

#include <sstream>

namespace yirage {
namespace threadblock {
namespace fpga {

// Generate Vitis HLS C++ for matmul
std::string generate_matmul_hls(int m, int n, int k, int tile_size) {
    std::ostringstream hls;
    hls << "#include <hls_stream.h>\n";
    hls << "#include <ap_fixed.h>\n\n";
    hls << "void krnl_matmul_fingerprint(\n";
    hls << "    hls::stream<half>& a_stream,\n";
    hls << "    hls::stream<half>& b_stream,\n";
    hls << "    hls::stream<half>& c_stream,\n";
    hls << "    int M, int N, int K\n";
    hls << ") {\n";
    hls << "#pragma HLS INTERFACE axis port=a_stream\n";
    hls << "#pragma HLS INTERFACE axis port=b_stream\n";
    hls << "#pragma HLS INTERFACE axis port=c_stream\n";
    hls << "#pragma HLS DATAFLOW\n\n";
    hls << "    half a_tile[" << tile_size << "][" << tile_size << "];\n";
    hls << "    half b_tile[" << tile_size << "][" << tile_size << "];\n";
    hls << "#pragma HLS ARRAY_PARTITION variable=a_tile cyclic factor=8 dim=2\n";
    hls << "#pragma HLS ARRAY_PARTITION variable=b_tile cyclic factor=8 dim=1\n\n";
    hls << "    // Tiled matmul with pipelining\n";
    hls << "    for (int i = 0; i < " << tile_size << "; i++) {\n";
    hls << "        for (int j = 0; j < " << tile_size << "; j++) {\n";
    hls << "#pragma HLS PIPELINE II=1\n";
    hls << "            half sum = 0;\n";
    hls << "            for (int kk = 0; kk < " << tile_size << "; kk++) {\n";
    hls << "#pragma HLS UNROLL factor=4\n";
    hls << "                sum += a_tile[i][kk] * b_tile[kk][j];\n";
    hls << "            }\n";
    hls << "            c_stream.write(sum);\n";
    hls << "        }\n";
    hls << "    }\n";
    hls << "}\n";
    return hls.str();
}

// Generate OpenCL kernel for FPGA
std::string generate_matmul_opencl(int m, int n, int k) {
    std::ostringstream ocl;
    ocl << "__kernel __attribute__((reqd_work_group_size(1, 1, 1)))\n";
    ocl << "void krnl_matmul(\n";
    ocl << "    __global half* restrict a,\n";
    ocl << "    __global half* restrict b,\n";
    ocl << "    __global half* restrict c,\n";
    ocl << "    int M, int N, int K\n";
    ocl << ") {\n";
    ocl << "    // FPGA streaming matmul\n";
    ocl << "}\n";
    return ocl.str();
}

}  // namespace fpga
}  // namespace threadblock
}  // namespace yirage

#endif
