/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * Backend initialization for non-automatic registration scenarios
 */

#include "backend/backend_registry.h"
#include <memory>

#ifdef YIRAGE_BACKEND_CPU_ENABLED
#include "backend/cpu_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
#include "backend/mps_backend.h"
#endif

#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include "backend/cuda_backend.h"
#endif

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
#include "backend/triton_backend.h"
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
#include "backend/nki_backend.h"
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
#include "backend/ascend_backend.h"
#endif

#ifdef YIRAGE_BACKEND_CUDNN_ENABLED
#include "backend/cudnn_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MKL_ENABLED
#include "backend/mkl_backend.h"
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include "backend/rocm_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MACA_ENABLED
#include "backend/maca_backend.h"
#endif

#ifdef YIRAGE_BACKEND_XPU_ENABLED
#include "backend/xpu_backend.h"
#endif

#ifdef YIRAGE_BACKEND_TPU_ENABLED
#include "backend/tpu_backend.h"
#endif

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
#include "backend/fpga_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MLIR_ENABLED
#include "backend/mlir_backend.h"
#endif

namespace yirage {
namespace backend {

// Manual backend registration for static library builds
void register_all_backends() {
    auto& registry = BackendRegistry::get_instance();
    
#ifdef YIRAGE_BACKEND_CPU_ENABLED
    registry.register_backend(std::make_unique<CPUBackend>());
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
    registry.register_backend(std::make_unique<MPSBackend>());
#endif

#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    registry.register_backend(std::make_unique<CUDABackend>());
#endif

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
    registry.register_backend(std::make_unique<TritonBackend>());
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
    registry.register_backend(std::make_unique<NKIBackend>());
#endif

#ifdef YIRAGE_BACKEND_CUDNN_ENABLED
    registry.register_backend(std::make_unique<CUDNNBackend>());
#endif

#ifdef YIRAGE_BACKEND_MKL_ENABLED
    registry.register_backend(std::make_unique<MKLBackend>());
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    registry.register_backend(std::make_unique<AscendBackend>());
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    registry.register_backend(std::make_unique<ROCmBackend>());
#endif

#ifdef YIRAGE_BACKEND_MACA_ENABLED
    registry.register_backend(std::make_unique<MACABackend>());
#endif

#ifdef YIRAGE_BACKEND_XPU_ENABLED
    registry.register_backend(std::make_unique<XPUBackend>());
#endif

#ifdef YIRAGE_BACKEND_TPU_ENABLED
    registry.register_backend(std::make_unique<TPUBackend>());
#endif

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
    registry.register_backend(std::make_unique<FPGABackend>());
#endif

#ifdef YIRAGE_BACKEND_MLIR_ENABLED
    registry.register_backend(std::make_unique<MLIRBackend>());
#endif
}

} // namespace backend
} // namespace yirage

