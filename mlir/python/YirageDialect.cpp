//===- YirageDialect.cpp - Yirage MLIR Python Bindings ----------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file implements Python bindings for the Yirage MLIR dialect.
// Uses pybind11 to expose C++ MLIR functionality to Python.
//
// Features:
//   - Dialect registration
//   - Operation construction
//   - Pass execution
//   - GPU code generation
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.h"
#include "yirage-mlir/Dialect/Yirage/IR/YirageOps.h"
#include "yirage-mlir/Dialect/Yirage/Transforms/Passes.h"
#include "yirage-mlir/Execution/CPUJITKernel.h"
#include "yirage-mlir/Execution/GPUCodeGen.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace mlir;
using namespace yirage;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Parse MLIR text to a module
OwningOpRef<ModuleOp> parseModule(MLIRContext &context, const std::string &mlirText) {
  return parseSourceString<ModuleOp>(mlirText, &context);
}

/// Run a pass pipeline on a module
bool runPipeline(MLIRContext &context, ModuleOp module, const std::string &pipeline) {
  PassManager pm(&context);
  
  if (failed(parsePassPipeline(pipeline, pm))) {
    return false;
  }
  
  return succeeded(pm.run(module));
}

} // namespace

//===----------------------------------------------------------------------===//
// Python Module Definition
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_yirage_mlir, m) {
  m.doc() = "YiRage MLIR Python bindings";

  //===--------------------------------------------------------------------===//
  // GPU Backend Enum
  //===--------------------------------------------------------------------===//
  
  py::enum_<GPUBackend>(m, "GPUBackend")
      .value("CUDA", GPUBackend::CUDA)
      .value("ROCm", GPUBackend::ROCm)
      .value("SPIRV", GPUBackend::SPIRV)
      .value("Metal", GPUBackend::Metal)
      .value("MACA", GPUBackend::MACA)
      .value("TPU", GPUBackend::TPU)
      .value("Ascend", GPUBackend::Ascend)
      .value("FPGA_OpenCL", GPUBackend::FPGA_OpenCL)
      .value("FPGA_HLS", GPUBackend::FPGA_HLS)
      .value("CPU_LLVM", GPUBackend::CPU_LLVM)
      .value("CPU_OpenMP", GPUBackend::CPU_OpenMP);

#if defined(YIRAGE_MLIR_GPU_PYBIND)
  //===--------------------------------------------------------------------===//
  // GPUTargetConfig
  //===--------------------------------------------------------------------===//
  
  py::class_<GPUTargetConfig>(m, "GPUTargetConfig")
      .def(py::init<>())
      .def_readwrite("backend", &GPUTargetConfig::backend)
      .def_readwrite("arch", &GPUTargetConfig::arch)
      .def_readwrite("triple", &GPUTargetConfig::triple)
      .def_readwrite("features", &GPUTargetConfig::features)
      .def_readwrite("optLevel", &GPUTargetConfig::optLevel)
      .def_readwrite("useFastMath", &GPUTargetConfig::useFastMath)
      .def_readwrite("useFMA", &GPUTargetConfig::useFMA)
      .def_readwrite("debug", &GPUTargetConfig::debug)
      .def_static("forCUDA", &GPUTargetConfig::forCUDA,
                  py::arg("computeCapability") = 80)
      .def_static("forROCm", &GPUTargetConfig::forROCm,
                  py::arg("gpuArch") = "gfx942")
      .def_static("forSPIRV", &GPUTargetConfig::forSPIRV)
      .def_static("forMetal", &GPUTargetConfig::forMetal,
                  py::arg("arch") = "m3");

  //===--------------------------------------------------------------------===//
  // CompilationResult
  //===--------------------------------------------------------------------===//
  
  py::class_<CompilationResult>(m, "CompilationResult")
      .def(py::init<>())
      .def_readwrite("success", &CompilationResult::success)
      .def_readwrite("errorMessage", &CompilationResult::errorMessage)
      .def_readwrite("textCode", &CompilationResult::textCode)
      .def_property_readonly("binaryCode", [](const CompilationResult &r) {
        return py::bytes(reinterpret_cast<const char*>(r.binaryCode.data()),
                        r.binaryCode.size());
      })
      .def_readwrite("targetArch", &CompilationResult::targetArch)
      .def_readwrite("kernelNames", &CompilationResult::kernelNames)
      .def_readwrite("registerUsage", &CompilationResult::registerUsage)
      .def_readwrite("sharedMemUsage", &CompilationResult::sharedMemUsage)
      .def("__bool__", [](const CompilationResult &r) { return r.success; });
#endif

  //===--------------------------------------------------------------------===//
  // MLIRContext Wrapper
  //===--------------------------------------------------------------------===//
  
  py::class_<MLIRContext>(m, "MLIRContext")
      .def(py::init<>())
      .def("loadDialect", [](MLIRContext &ctx, const std::string &name) {
        if (name == "yirage") {
          ctx.loadDialect<ir::YirageDialect>();
        }
      })
      .def("loadAllDialects", [](MLIRContext &ctx) {
        ctx.loadDialect<ir::YirageDialect>();
        ctx.loadDialect<func::FuncDialect>();
        ctx.loadDialect<arith::ArithDialect>();
        ctx.loadDialect<linalg::LinalgDialect>();
        ctx.loadDialect<tensor::TensorDialect>();
        ctx.loadDialect<memref::MemRefDialect>();
      });

  //===--------------------------------------------------------------------===//
  // Module Operations
  //===--------------------------------------------------------------------===//
  
  m.def("parseMLIR", [](const std::string &mlirText) -> py::object {
    auto context = std::make_shared<MLIRContext>();
    context->loadDialect<ir::YirageDialect>();
    context->loadDialect<func::FuncDialect>();
    context->loadDialect<arith::ArithDialect>();
    context->loadDialect<linalg::LinalgDialect>();
    context->loadDialect<tensor::TensorDialect>();
    
    auto module = parseModule(*context, mlirText);
    if (!module) {
      return py::none();
    }
    
    // Store context and module together
    return py::make_tuple(context, py::cast(module.release()));
  }, "Parse MLIR text and return (context, module) tuple");

  m.def("printMLIR", [](ModuleOp module) -> std::string {
    std::string output;
    llvm::raw_string_ostream os(output);
    module.print(os);
    return output;
  }, "Print module to MLIR text");

  //===--------------------------------------------------------------------===//
  // Pass Execution
  //===--------------------------------------------------------------------===//
  
  m.def("runYirageToLinalg", [](MLIRContext &ctx, ModuleOp module) -> bool {
    return runPipeline(ctx, module, "yirage-to-linalg");
  }, "Lower Yirage dialect to Linalg");

  m.def("runGPUPipeline", [](MLIRContext &ctx, ModuleOp module) -> bool {
    return runPipeline(ctx, module, "yirage-gpu-pipeline");
  }, "Run complete GPU lowering pipeline");

  m.def("runCUDAPipeline", [](MLIRContext &ctx, ModuleOp module) -> bool {
    return runPipeline(ctx, module, "yirage-cuda-pipeline");
  }, "Run CUDA-specific pipeline");

  m.def("runROCmPipeline", [](MLIRContext &ctx, ModuleOp module) -> bool {
    return runPipeline(ctx, module, "yirage-rocm-pipeline");
  }, "Run ROCm-specific pipeline");

  m.def("runCPUPipeline", [](MLIRContext &ctx, ModuleOp module) -> bool {
    return runPipeline(ctx, module, "yirage-cpu-pipeline");
  }, "Run CPU lowering pipeline");

  m.def("runCPUJITPipeline", [](MLIRContext &ctx, ModuleOp module) -> bool {
    return runPipeline(ctx, module, "yirage-cpu-jit-pipeline");
  }, "Run CPU JIT prep pipeline (Linalg + bufferize + loops)");

  m.def("runCustomPipeline", [](MLIRContext &ctx, ModuleOp module,
                                const std::string &pipeline) -> bool {
    return runPipeline(ctx, module, pipeline);
  }, "Run custom pass pipeline");

  //===--------------------------------------------------------------------===//
  // GPU Code Generation (optional; full GPU link needs YIRAGE_MLIR_GPU_PYBIND)
  //===--------------------------------------------------------------------===//
#if defined(YIRAGE_MLIR_GPU_PYBIND)

  m.def("generatePTX", [](MLIRContext &ctx, ModuleOp module,
                          int computeCapability) -> std::string {
    auto codegen = createCUDACodeGen(&ctx, computeCapability);
    return codegen->exportCode(module);
  }, "Generate PTX code for NVIDIA GPU",
     py::arg("ctx"), py::arg("module"), py::arg("computeCapability") = 80);

  m.def("generateROCm", [](MLIRContext &ctx, ModuleOp module,
                           const std::string &arch) -> std::string {
    auto codegen = createROCmCodeGen(&ctx, arch);
    return codegen->exportCode(module);
  }, "Generate GCN assembly for AMD GPU",
     py::arg("ctx"), py::arg("module"), py::arg("arch") = "gfx908");

  m.def("generateSPIRV", [](MLIRContext &ctx, ModuleOp module) -> std::string {
    auto codegen = createSPIRVCodeGen(&ctx);
    return codegen->exportCode(module);
  }, "Generate SPIR-V code");

  m.def("generateMetal", [](MLIRContext &ctx, ModuleOp module) -> std::string {
    auto codegen = createMetalCodeGen(&ctx);
    return codegen->exportCode(module);
  }, "Generate Metal Shading Language code");

  m.def("generateCubin", [](MLIRContext &ctx, ModuleOp module,
                            int computeCapability) -> py::bytes {
    auto codegen = createCUDACodeGen(&ctx, computeCapability);
    auto binary = codegen->getBinary(module);
    return py::bytes(reinterpret_cast<const char*>(binary.data()), binary.size());
  }, "Generate cubin binary for NVIDIA GPU",
     py::arg("ctx"), py::arg("module"), py::arg("computeCapability") = 80);

  m.def("generateHSACO", [](MLIRContext &ctx, ModuleOp module,
                            const std::string &arch) -> py::bytes {
    auto codegen = createROCmCodeGen(&ctx, arch);
    auto binary = codegen->getBinary(module);
    return py::bytes(reinterpret_cast<const char*>(binary.data()), binary.size());
  }, "Generate HSACO binary for AMD GPU",
     py::arg("ctx"), py::arg("module"), py::arg("arch") = "gfx908");

  m.def("generateSPIRVBinary", [](MLIRContext &ctx, ModuleOp module) -> py::bytes {
    auto codegen = createSPIRVCodeGen(&ctx);
    auto binary = codegen->getBinary(module);
    return py::bytes(reinterpret_cast<const char*>(binary.data()), binary.size());
  }, "Generate SPIR-V binary");

#else

  m.def("isBackendAvailable", [](GPUBackend backend) -> bool {
    return backend == GPUBackend::CPU_LLVM || backend == GPUBackend::CPU_OpenMP;
  }, "Check if a GPU backend is available on this system");

  m.def("getAvailableBackends", []() -> std::vector<GPUBackend> {
    return {GPUBackend::CPU_LLVM};
  }, "Get list of available GPU backends");

#endif

  //===--------------------------------------------------------------------===//
  // Pass Registration
  //===--------------------------------------------------------------------===//
  
  m.def("registerPasses", []() {
    registerYiragePasses();
    registerYiragePassPipelines();
  }, "Register all Yirage MLIR passes");

  //===--------------------------------------------------------------------===//
  // Utility Functions
  //===--------------------------------------------------------------------===//
  
#if defined(YIRAGE_MLIR_GPU_PYBIND)
  m.def("backendToString", [](GPUBackend backend) -> std::string {
    return backendToString(backend).str();
  }, "Convert backend enum to string");

  m.def("stringToBackend", [](const std::string &name) -> GPUBackend {
    return stringToBackend(name);
  }, "Convert string to backend enum");
#else
  m.def("backendToString", [](GPUBackend backend) -> std::string {
    switch (backend) {
    case GPUBackend::CUDA: return "cuda";
    case GPUBackend::ROCm: return "rocm";
    case GPUBackend::CPU_LLVM: return "cpu_llvm";
    case GPUBackend::CPU_OpenMP: return "cpu_openmp";
    default: return "unknown";
    }
  }, "Convert backend enum to string");

  m.def("stringToBackend", [](const std::string &name) -> GPUBackend {
    if (name == "cpu_llvm" || name == "cpu")
      return GPUBackend::CPU_LLVM;
    return GPUBackend::CPU_LLVM;
  }, "Convert string to backend enum");
#endif

  //===--------------------------------------------------------------------===//
  // CPU JIT (LLVM ExecutionEngine)
  //===--------------------------------------------------------------------===//

  py::class_<CPUJITKernel>(m, "CPUJITKernel")
      .def(py::init<>())
      .def("compile_mlir", [](CPUJITKernel &kernel, const std::string &mlirText,
                              const std::string &entry) -> bool {
        return succeeded(kernel.compileFromText(mlirText, entry));
      }, py::arg("mlir_text"), py::arg("entry") = "mugraph")
      .def("is_ready", &CPUJITKernel::isReady)
      .def("last_error", &CPUJITKernel::lastError)
      .def(
          "invoke_rms_matmul_f16",
          [](CPUJITKernel &kernel, uintptr_t xPtr, uintptr_t wPtr,
             uintptr_t outPtr, int64_t m, int64_t k, int64_t n) -> bool {
            return succeeded(kernel.invokeRmsMatmulF16(
                reinterpret_cast<void *>(xPtr), reinterpret_cast<void *>(wPtr),
                reinterpret_cast<void *>(outPtr), m, k, n));
          },
          py::arg("x_ptr"), py::arg("w_ptr"), py::arg("out_ptr"), py::arg("m"),
          py::arg("k"), py::arg("n"));
}
