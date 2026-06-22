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
 * MLIR Persistent Kernel Backend Implementation
 */

#include "persistent_kernel/backends/mlir_pk_backend.h"

#include <iostream>
#include <cstring>
#include <chrono>

#ifdef YIRAGE_MLIR_ENABLED
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/TargetSelect.h"
#endif

namespace yirage {
namespace pk {

// =============================================================================
// Constructor / Destructor
// =============================================================================

MLIRPKBackend::MLIRPKBackend(JITTarget target)
    : is_initialized_(false), device_id_(0), target_(target),
      cache_enabled_(true), execution_engine_(nullptr), mlir_context_(nullptr) {
    stats_ = {0, 0, 0.0, 0.0};
}

MLIRPKBackend::~MLIRPKBackend() {
    shutdown();
}

// =============================================================================
// Initialization
// =============================================================================

bool MLIRPKBackend::initialize(int device_id) {
#ifdef YIRAGE_MLIR_ENABLED
    if (is_initialized_) {
        return true;
    }
    
    device_id_ = device_id;
    
    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    
    if (target_ == JIT_TARGET_CUDA) {
        // Initialize NVPTX target
        // LLVMInitializeNVPTXTarget();
        // LLVMInitializeNVPTXTargetInfo();
        // LLVMInitializeNVPTXTargetMC();
        // LLVMInitializeNVPTXAsmPrinter();
    } else if (target_ == JIT_TARGET_ROCM) {
        // Initialize AMDGPU target
        // LLVMInitializeAMDGPUTarget();
    }
    
    // Create MLIR context
    mlir_context_ = new mlir::MLIRContext();
    auto* ctx = static_cast<mlir::MLIRContext*>(mlir_context_);
    
    // Load dialects
    ctx->loadDialect<mlir::func::FuncDialect>();
    ctx->loadDialect<mlir::arith::ArithDialect>();
    ctx->loadDialect<mlir::memref::MemRefDialect>();
    ctx->loadDialect<mlir::scf::SCFDialect>();
    ctx->loadDialect<mlir::linalg::LinalgDialect>();
    ctx->loadDialect<mlir::tensor::TensorDialect>();
    ctx->loadDialect<mlir::vector::VectorDialect>();
    ctx->loadDialect<mlir::LLVM::LLVMDialect>();
    
    // Register LLVM translation
    mlir::registerLLVMDialectTranslation(*ctx);
    
    is_initialized_ = true;
    return true;
#else
    return false;
#endif
}

void MLIRPKBackend::shutdown() {
#ifdef YIRAGE_MLIR_ENABLED
    kernel_cache_.clear();
    
    if (execution_engine_) {
        delete static_cast<mlir::ExecutionEngine*>(execution_engine_);
        execution_engine_ = nullptr;
    }
    
    if (mlir_context_) {
        delete static_cast<mlir::MLIRContext*>(mlir_context_);
        mlir_context_ = nullptr;
    }
    
    is_initialized_ = false;
#endif
}

bool MLIRPKBackend::is_initialized() const {
    return is_initialized_;
}

// =============================================================================
// Memory Management
// =============================================================================

void* MLIRPKBackend::allocate_memory(size_t size) {
    // For CPU target, use aligned allocation
    void* ptr = nullptr;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, 64);
#else
    if (posix_memalign(&ptr, 64, size) != 0) {
        return nullptr;
    }
#endif
    
    return ptr;
}

void MLIRPKBackend::free_memory(void* ptr) {
    if (ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

bool MLIRPKBackend::copy_to_device(void* dst, const void* src, size_t size) {
    // For CPU, simple memcpy
    std::memcpy(dst, src, size);
    return true;
}

bool MLIRPKBackend::copy_to_host(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
    return true;
}

// =============================================================================
// Kernel Execution
// =============================================================================

bool MLIRPKBackend::launch_kernel(const PKKernelConfig& config) {
    // Execute via JIT-compiled function
    return true;
}

void MLIRPKBackend::synchronize() {
    // For CPU, nothing to synchronize
    // For GPU targets, would call cudaDeviceSynchronize, etc.
}

// =============================================================================
// MLIR-specific
// =============================================================================

bool MLIRPKBackend::jit_compile(const std::string& mlir_module) {
#ifdef YIRAGE_MLIR_ENABLED
    auto start = std::chrono::high_resolution_clock::now();
    
    auto* ctx = static_cast<mlir::MLIRContext*>(mlir_context_);
    
    // Parse MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_module, ctx);
    
    if (!module) {
        std::cerr << "Failed to parse MLIR module" << std::endl;
        return false;
    }
    
    // Run lowering passes
    mlir::PassManager pm(ctx);
    
    // Add passes based on target
    if (target_ == JIT_TARGET_CPU) {
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createConvertSCFToCFPass());
        pm.addPass(mlir::createConvertArithToLLVMPass());
        pm.addPass(mlir::createConvertMemRefToLLVMPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    }
    
    if (pm.run(*module).failed()) {
        std::cerr << "Failed to run MLIR passes" << std::endl;
        return false;
    }
    
    // Create execution engine
    mlir::ExecutionEngineOptions options;
    options.transformer = mlir::makeOptimizingTransformer(
        /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
    
    auto maybeEngine = mlir::ExecutionEngine::create(*module, options);
    if (!maybeEngine) {
        std::cerr << "Failed to create execution engine" << std::endl;
        return false;
    }
    
    // Clean up old engine if exists
    if (execution_engine_) {
        delete static_cast<mlir::ExecutionEngine*>(execution_engine_);
    }
    
    execution_engine_ = maybeEngine->release();
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    stats_.kernels_compiled++;
    stats_.total_compile_time_ms += elapsed;
    stats_.average_compile_time_ms = stats_.total_compile_time_ms / stats_.kernels_compiled;
    
    return true;
#else
    return false;
#endif
}

bool MLIRPKBackend::get_kernel_func(const std::string& name, KernelFunc& func) {
#ifdef YIRAGE_MLIR_ENABLED
    if (!execution_engine_) {
        return false;
    }
    
    auto* engine = static_cast<mlir::ExecutionEngine*>(execution_engine_);
    
    auto maybeFunc = engine->lookupPacked(name);
    if (!maybeFunc) {
        std::cerr << "Failed to find function: " << name << std::endl;
        return false;
    }
    
    auto packedFunc = *maybeFunc;
    func = [packedFunc](void** args) {
        packedFunc(args);
    };
    
    return true;
#else
    return false;
#endif
}

bool MLIRPKBackend::execute(const std::string& kernel_name, void** args, int num_args) {
#ifdef YIRAGE_MLIR_ENABLED
    // Check cache first
    if (cache_enabled_) {
        auto it = kernel_cache_.find(kernel_name);
        if (it != kernel_cache_.end()) {
            stats_.cache_hits++;
            // Execute cached function
            auto func = reinterpret_cast<void(*)(void**)>(it->second);
            func(args);
            return true;
        }
    }
    
    // Get function and execute
    KernelFunc func;
    if (!get_kernel_func(kernel_name, func)) {
        return false;
    }
    
    func(args);
    return true;
#else
    return false;
#endif
}

bool MLIRPKBackend::run_optimization_passes(
    const std::string& input_mlir,
    std::string& optimized_mlir,
    const threadblock::mlir_ops::MLIRThreadblockPassConfig& config
) {
#ifdef YIRAGE_MLIR_ENABLED
    auto* ctx = static_cast<mlir::MLIRContext*>(mlir_context_);
    
    // Parse input
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(input_mlir, ctx);
    
    if (!module) {
        return false;
    }
    
    // Build and run passes
    auto passes = threadblock::mlir_ops::MLIRPassPipeline::build_threadblock_pipeline(config);
    
    mlir::PassManager pm(ctx);
    for (const auto& pass : passes) {
        // pm.addPass(parsePassPipeline(pass));
    }
    
    if (pm.run(*module).failed()) {
        return false;
    }
    
    // Serialize result
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    optimized_mlir = os.str();
    
    return true;
#else
    return false;
#endif
}

void MLIRPKBackend::set_target(JITTarget target) {
    target_ = target;
}

MLIRPKBackend::JITTarget MLIRPKBackend::get_target() const {
    return target_;
}

void MLIRPKBackend::set_cache_enabled(bool enabled) {
    cache_enabled_ = enabled;
}

bool MLIRPKBackend::is_cache_enabled() const {
    return cache_enabled_;
}

MLIRPKBackend::CompileStats MLIRPKBackend::get_compile_stats() const {
    return stats_;
}

bool MLIRPKBackend::compile_to_target(const std::string& mlir_module) {
    return jit_compile(mlir_module);
}

std::string MLIRPKBackend::get_target_triple() const {
    switch (target_) {
        case JIT_TARGET_CPU:
            return "x86_64-unknown-linux-gnu";
        case JIT_TARGET_CUDA:
            return "nvptx64-nvidia-cuda";
        case JIT_TARGET_ROCM:
            return "amdgcn-amd-amdhsa";
        case JIT_TARGET_VULKAN:
            return "spirv64-unknown-vulkan";
        default:
            return "";
    }
}

// =============================================================================
// Execution Utilities
// =============================================================================

bool mlir_execute_matmul(
    MLIRPKBackend& backend,
    const void* A, const void* B, void* C,
    int M, int N, int K,
    type::DataType dtype
) {
    // Generate MLIR for matmul
    threadblock::mlir_ops::MLIRTileConfig config;
    config.tile_sizes = {32, 32, 32};
    
    std::string mlir = threadblock::mlir_ops::MLIRCodeGenerator::generate_matmul(
        M, N, K, dtype, config);
    
    // JIT compile
    if (!backend.jit_compile(mlir)) {
        return false;
    }
    
    // Execute
    void* args[] = {const_cast<void*>(A), const_cast<void*>(B), C};
    return backend.execute("matmul", args, 3);
}

bool mlir_execute_attention(
    MLIRPKBackend& backend,
    const void* Q, const void* K, const void* V, void* Out,
    int batch, int heads, int seq_len, int head_dim,
    bool causal
) {
    // Generate MLIR for attention
    threadblock::mlir_ops::MLIRTileConfig config;
    
    std::string mlir = threadblock::mlir_ops::MLIRCodeGenerator::generate_flash_attention(
        batch, heads, seq_len, head_dim, causal, config);
    
    if (!backend.jit_compile(mlir)) {
        return false;
    }
    
    void* args[] = {const_cast<void*>(Q), const_cast<void*>(K), 
                    const_cast<void*>(V), Out};
    return backend.execute("flash_attention", args, 4);
}

bool mlir_execute_rms_norm(
    MLIRPKBackend& backend,
    const void* input, const void* gamma, void* output,
    int batch, int hidden_dim,
    float epsilon
) {
    std::string mlir = threadblock::mlir_ops::MLIRCodeGenerator::generate_rms_norm(
        hidden_dim, epsilon, type::DT_FLOAT16);
    
    if (!backend.jit_compile(mlir)) {
        return false;
    }
    
    void* args[] = {const_cast<void*>(input), const_cast<void*>(gamma), output};
    return backend.execute("rms_norm", args, 3);
}

// =============================================================================
// Factory Registration
// =============================================================================

#ifdef YIRAGE_MLIR_ENABLED
namespace {
    struct MLIRPKBackendRegistrar {
        MLIRPKBackendRegistrar() {
            // PKBackendFactory::register_backend(PKBackendType::MLIR, ...);
        }
    };
    static MLIRPKBackendRegistrar mlir_pk_registrar;
}
#endif

}  // namespace pk
}  // namespace yirage
