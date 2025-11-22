/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * Backend initialization for non-automatic registration scenarios
 */

#include "yirage/backend/backend_registry.h"
#include <memory>

#ifdef YIRAGE_BACKEND_CPU_ENABLED
#include "yirage/backend/cpu_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
#include "yirage/backend/mps_backend.h"
#endif

#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include "yirage/backend/cuda_backend.h"
#endif

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
#include "yirage/backend/triton_backend.h"
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
#include "yirage/backend/nki_backend.h"
#endif

#ifdef YIRAGE_BACKEND_CUDNN_ENABLED
#include "yirage/backend/cudnn_backend.h"
#endif

#ifdef YIRAGE_BACKEND_MKL_ENABLED
#include "yirage/backend/mkl_backend.h"
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
}

} // namespace backend
} // namespace yirage

