//===- CPUJITKernel.h - CPU JIT kernel session -----------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_MLIR_EXECUTION_CPUJITKERNEL_H
#define YIRAGE_MLIR_EXECUTION_CPUJITKERNEL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <memory>
#include <string>

namespace yirage {

class JITRunner;

/// Session that lowers YiRage MLIR (tensor) to LLVM and JIT-executes on CPU.
class CPUJITKernel {
public:
  CPUJITKernel();
  ~CPUJITKernel();

  CPUJITKernel(const CPUJITKernel &) = delete;
  CPUJITKernel &operator=(const CPUJITKernel &) = delete;

  /// Parse MLIR text, run ``yirage-cpu-jit-pipeline``, and JIT-compile ``@mugraph``.
  mlir::LogicalResult compileFromText(const std::string &mlirText,
                                      const std::string &entry = "mugraph");

  bool isReady() const;

  /// Row-major contiguous f16 tensors: x[M,K], w[K,N] -> out[M,N].
  mlir::LogicalResult invokeRmsMatmulF16(void *x, void *w, void *out, int64_t m,
                                        int64_t k, int64_t n);

  std::string lastError() const { return lastError_; }

private:
  mlir::LogicalResult runCpuJitPipeline(mlir::ModuleOp module);

  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<JITRunner> runner_;
  std::string entry_;
  std::string lastError_;
};

} // namespace yirage

#endif // YIRAGE_MLIR_EXECUTION_CPUJITKERNEL_H
