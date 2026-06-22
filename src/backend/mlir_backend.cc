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
 * MLIR Backend Implementation
 */

#include "backend/mlir_backend.h"
#include "backend/backend_registry.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>

#ifdef YIRAGE_MLIR_ENABLED
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#endif

namespace yirage {
namespace backend {

// =============================================================================
// Constructor / Destructor
// =============================================================================

MLIRBackend::MLIRBackend(MLIRTarget target)
    : target_(target), is_available_(false), device_id_(0),
      mlir_context_(nullptr), execution_engine_(nullptr) {
    is_available_ = check_mlir_availability();
}

MLIRBackend::~MLIRBackend() {
#ifdef YIRAGE_MLIR_ENABLED
    if (execution_engine_) {
        delete static_cast<mlir::ExecutionEngine*>(execution_engine_);
    }
    if (mlir_context_) {
        delete static_cast<mlir::MLIRContext*>(mlir_context_);
    }
#endif
}

// =============================================================================
// Availability Check
// =============================================================================

bool MLIRBackend::check_mlir_availability() {
#ifdef YIRAGE_MLIR_ENABLED
    try {
        mlir_context_ = new mlir::MLIRContext();
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

// =============================================================================
// Backend Information
// =============================================================================

type::BackendType MLIRBackend::get_type() const {
    return type::BT_MLIR;
}

std::string MLIRBackend::get_name() const {
    return "mlir";
}

std::string MLIRBackend::get_display_name() const {
    std::string target_name;
    switch (target_) {
        case MLIR_TARGET_LLVM: target_name = "LLVM"; break;
        case MLIR_TARGET_NVVM: target_name = "NVVM"; break;
        case MLIR_TARGET_ROCDL: target_name = "ROCDL"; break;
        case MLIR_TARGET_SPIRV: target_name = "SPIR-V"; break;
        case MLIR_TARGET_VULKAN: target_name = "Vulkan"; break;
        case MLIR_TARGET_WASM: target_name = "WebAssembly"; break;
    }
    return "MLIR (" + target_name + ")";
}

bool MLIRBackend::is_available() const {
    return is_available_;
}

type::BackendInfo MLIRBackend::get_info() const {
    type::BackendInfo info;
    info.type = type::BT_MLIR;
    info.name = "mlir";
    info.display_name = get_display_name();
    info.requires_gpu = (target_ == MLIR_TARGET_NVVM || 
                        target_ == MLIR_TARGET_ROCDL ||
                        target_ == MLIR_TARGET_VULKAN);
    info.required_libs = {"MLIR", "LLVM"};
    return info;
}

// =============================================================================
// Compilation
// =============================================================================

bool MLIRBackend::compile(CompileContext const& ctx) {
    std::string output;
    return compile_mlir(ctx.source_code, output);
}

std::string MLIRBackend::get_compile_flags() const {
    std::string flags = "-O3";
    
    switch (target_) {
        case MLIR_TARGET_LLVM:
            flags += " -march=native";
            break;
        case MLIR_TARGET_NVVM:
            flags += " -target nvptx64-nvidia-cuda";
            break;
        case MLIR_TARGET_ROCDL:
            flags += " -target amdgcn-amd-amdhsa";
            break;
        default:
            break;
    }
    
    return flags;
}

std::vector<std::string> MLIRBackend::get_include_dirs() const {
    std::vector<std::string> dirs;
    
    const char* llvm_path = getenv("LLVM_PATH");
    if (llvm_path) {
        dirs.push_back(std::string(llvm_path) + "/include");
    }
    
    return dirs;
}

std::vector<std::string> MLIRBackend::get_library_dirs() const {
    std::vector<std::string> dirs;
    
    const char* llvm_path = getenv("LLVM_PATH");
    if (llvm_path) {
        dirs.push_back(std::string(llvm_path) + "/lib");
    }
    
    return dirs;
}

std::vector<std::string> MLIRBackend::get_link_libraries() const {
    return {"MLIR", "LLVM", "LLVMCore", "LLVMSupport"};
}

// =============================================================================
// Memory Management (delegated to underlying target)
// =============================================================================

void* MLIRBackend::allocate_memory(size_t size) {
    // MLIR itself doesn't manage memory - delegate to target
    return std::malloc(size);
}

void MLIRBackend::free_memory(void* ptr) {
    std::free(ptr);
}

bool MLIRBackend::copy_to_device(void* dst, void const* src, size_t size) {
    std::memcpy(dst, src, size);
    return true;
}

bool MLIRBackend::copy_to_host(void* dst, void const* src, size_t size) {
    std::memcpy(dst, src, size);
    return true;
}

bool MLIRBackend::copy_device_to_device(void* dst, void const* src, size_t size) {
    std::memcpy(dst, src, size);
    return true;
}

// =============================================================================
// Synchronization
// =============================================================================

void MLIRBackend::synchronize() {
    // No-op for CPU targets
    // Would sync GPU for NVVM/ROCDL targets
}

// =============================================================================
// Capability Query
// =============================================================================

size_t MLIRBackend::get_max_memory() const {
    // Return system memory for CPU targets
    return 16ULL * 1024 * 1024 * 1024;  // 16GB default
}

size_t MLIRBackend::get_max_shared_memory() const {
    return 48 * 1024;  // 48KB typical
}

bool MLIRBackend::supports_data_type(type::DataType dt) const {
    // MLIR supports all data types via appropriate dialects
    return true;
}

int MLIRBackend::get_compute_capability() const {
    return 0;  // N/A for MLIR
}

int MLIRBackend::get_num_compute_units() const {
    return 1;  // MLIR is a compiler, not an executor
}

// =============================================================================
// Device Management
// =============================================================================

bool MLIRBackend::set_device(int device_id) {
    device_id_ = device_id;
    return true;
}

int MLIRBackend::get_device() const {
    return device_id_;
}

int MLIRBackend::get_device_count() const {
    return 1;
}

// =============================================================================
// MLIR-specific
// =============================================================================

void MLIRBackend::set_target(MLIRTarget target) {
    target_ = target;
}

MLIRBackend::MLIRTarget MLIRBackend::get_target() const {
    return target_;
}

bool MLIRBackend::compile_mlir(const std::string& mlir_module,
                               std::string& output) {
#ifdef YIRAGE_MLIR_ENABLED
    auto* ctx = static_cast<mlir::MLIRContext*>(mlir_context_);
    
    // Parse MLIR module
    // mlir::OwningOpRef<mlir::ModuleOp> module = 
    //     mlir::parseSourceString<mlir::ModuleOp>(mlir_module, ctx);
    
    // Run optimization passes
    // mlir::PassManager pm(ctx);
    // pm.addPass(mlir::createCanonicalizerPass());
    // pm.addPass(mlir::createCSEPass());
    // pm.run(*module);
    
    // Lower to target
    switch (target_) {
        case MLIR_TARGET_LLVM:
            // Lower to LLVM IR
            break;
        case MLIR_TARGET_NVVM:
            // Lower to NVVM/PTX
            break;
        case MLIR_TARGET_ROCDL:
            // Lower to ROCDL
            break;
        default:
            break;
    }
    
    return true;
#else
    return false;
#endif
}

bool MLIRBackend::run_passes(const std::string& input_mlir,
                             const std::vector<std::string>& passes,
                             std::string& output_mlir) {
#ifdef YIRAGE_MLIR_ENABLED
    auto* ctx = static_cast<mlir::MLIRContext*>(mlir_context_);
    
    // mlir::PassManager pm(ctx);
    // for (const auto& pass : passes) {
    //     if (pass == "canonicalize") pm.addPass(mlir::createCanonicalizerPass());
    //     else if (pass == "cse") pm.addPass(mlir::createCSEPass());
    //     // ... more passes
    // }
    
    return true;
#else
    return false;
#endif
}

bool MLIRBackend::lower_yirage_dialect(const std::string& yirage_mlir,
                                       std::string& lowered_mlir) {
#ifdef YIRAGE_MLIR_ENABLED
    // Use YiRage lowering passes
    std::vector<std::string> passes = {
        "convert-yirage-to-linalg",
        "linalg-tile-and-fuse",
        "convert-linalg-to-loops",
        "convert-scf-to-cf",
        "convert-func-to-llvm"
    };
    
    return run_passes(yirage_mlir, passes, lowered_mlir);
#else
    return false;
#endif
}

bool MLIRBackend::jit_compile(const std::string& mlir_module,
                              const std::string& entry_point,
                              JITFunction& func) {
#ifdef YIRAGE_MLIR_ENABLED
    // auto* ctx = static_cast<mlir::MLIRContext*>(mlir_context_);
    
    // Parse and compile
    // auto engine = mlir::ExecutionEngine::create(module);
    // auto jitFunc = engine->lookupPacked(entry_point);
    
    // func = [jitFunc](void** args) {
    //     jitFunc(args);
    // };
    
    return true;
#else
    return false;
#endif
}

// =============================================================================
// Pass Pipeline Builder
// =============================================================================

std::vector<std::string> build_pass_pipeline(const MLIRPassConfig& config) {
    std::vector<std::string> passes;
    
    if (config.canonicalize) {
        passes.push_back("canonicalize");
    }
    
    if (config.cse) {
        passes.push_back("cse");
    }
    
    if (config.loop_fusion) {
        passes.push_back("affine-loop-fusion");
    }
    
    if (config.tile_loops) {
        std::stringstream ss;
        ss << "linalg-tile{tile-sizes=";
        for (size_t i = 0; i < config.tile_sizes.size(); i++) {
            if (i > 0) ss << ",";
            ss << config.tile_sizes[i];
        }
        ss << "}";
        passes.push_back(ss.str());
    }
    
    if (config.vectorize) {
        passes.push_back("linalg-vectorization{vector-width=" + 
                        std::to_string(config.vector_width) + "}");
    }
    
    if (config.loop_unroll) {
        passes.push_back("affine-loop-unroll{unroll-factor=" +
                        std::to_string(config.unroll_factor) + "}");
    }
    
    if (config.bufferize) {
        passes.push_back("one-shot-bufferize");
    }
    
    if (config.convert_to_llvm) {
        passes.push_back("convert-func-to-llvm");
        passes.push_back("convert-arith-to-llvm");
        passes.push_back("convert-memref-to-llvm");
    }
    
    return passes;
}

// =============================================================================
// Dialect Registration
// =============================================================================

bool register_yirage_dialect(void* mlir_context) {
#ifdef YIRAGE_MLIR_ENABLED
    // auto* ctx = static_cast<mlir::MLIRContext*>(mlir_context);
    // ctx->loadDialect<yirage::mlir::YirageDialect>();
    return true;
#else
    return false;
#endif
}

std::vector<std::string> get_available_dialects() {
    return {
        "yirage",       // YiRage dialect
        "linalg",       // Linear algebra
        "tensor",       // Tensor operations
        "arith",        // Arithmetic
        "scf",          // Structured control flow
        "memref",       // Memory references
        "vector",       // Vector operations
        "affine",       // Affine loops
        "func",         // Functions
        "llvm",         // LLVM dialect
        "gpu",          // GPU operations
        "nvvm",         // NVIDIA VM
        "rocdl",        // ROCm DL
        "spirv",        // SPIR-V
    };
}

// =============================================================================
// Backend Registration
// =============================================================================

#ifdef YIRAGE_MLIR_ENABLED
namespace {
    struct MLIRBackendRegistrar {
        MLIRBackendRegistrar() {
            BackendRegistry::get_instance().register_backend(
                std::make_unique<MLIRBackend>());
        }
    };
    static MLIRBackendRegistrar mlir_registrar;
}
#endif

}  // namespace backend
}  // namespace yirage
