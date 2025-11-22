/* Transpiler stub for non-CUDA builds */

#ifndef YIRAGE_BACKEND_CUDA_ENABLED

#include "yirage/kernel/graph.h"
#include <vector>
#include <string>

namespace yirage {
namespace transpiler {

struct TranspilerConfig {
    int target_cc;
    bool profiling;
    bool enable_online_softmax;
    int num_producer_wgs;
    int num_consumer_wgs;
    int pipeline_stages;
};

struct TranspileResult {
    std::string code;
    std::string exec_nvcc_command;
    std::vector<size_t> output_shapes;
};

// Stub implementation for when CUDA is disabled
TranspileResult transpile(kernel::Graph const *graph,
                         TranspilerConfig const &config,
                         std::vector<std::vector<size_t>> const &input_strides) {
    TranspileResult result;
    result.code = "// CUDA transpiler not available (CUDA disabled)";
    result.exec_nvcc_command = "";
    return result;
}

} // namespace transpiler

namespace nki_transpiler {

struct NKITranspilerConfig {
    int target_cc;
};

struct NKITranspileResult {
    std::string code;
    struct {
        std::vector<std::string> errors;
    } error_state;
};

NKITranspileResult transpile(kernel::Graph const *graph,
                             NKITranspilerConfig const &config) {
    NKITranspileResult result;
    result.code = "// NKI transpiler not available";
    return result;
}

} // namespace nki_transpiler

namespace triton_transpiler {

struct TritonTranspilerConfig {
    int target_cc;
};

struct TritonTranspileResult {
    std::string code;
};

TritonTranspileResult transpile(kernel::Graph const *graph,
                                TritonTranspilerConfig const &config) {
    TritonTranspileResult result;
    result.code = "// Triton transpiler not available";
    return result;
}

} // namespace triton_transpiler

// Additional stubs for Python bindings
namespace kernel {

void cython_set_gpu_device_id(int device_id) {
    // Stub for non-CUDA builds - does nothing
}

void Graph::generate_triton_program(char const *filepath) {
    // Stub for non-CUDA builds - does nothing
}

} // namespace kernel

} // namespace yirage

#endif // !YIRAGE_BACKEND_CUDA_ENABLED

