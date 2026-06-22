# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage - Multi-Backend Kernel Optimization Framework

Provides kernel optimization for CUDA, MPS, MACA, Ascend, and CPU backends.

The Python package always depends on the native runtime (``libyirage_runtime``,
linked into ``yirage.core``). Install with a full build (e.g.
``YIRAGE_BACKEND=cpu pip install -e .``); ``import yirage`` fails if C++ bindings
are missing.
"""

import ctypes
import os
import platform
import sys

_this_dir = os.path.dirname(__file__)
_yirage_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
lib_ext = ".dylib" if platform.system() == "Darwin" else ".so"


def _preload_z3_shared_library() -> None:
    """
    On macOS, core.cpython-*.so links with a bare ``libz3.dylib`` install name;
    dyld often does not apply embedded LC_RPATH when resolving that dependency.
    Preload the wheel-shipped Z3 library (without importing the z3 Python package)
    so ``import yirage.core`` succeeds. Linux typically resolves via rpath already.
    """
    if platform.system() != "Darwin":
        return

    name = f"libz3{lib_ext}"

    # 1) Locate the z3 package directory via importlib (does not import z3).
    try:
        from importlib.util import find_spec
        spec = find_spec("z3")
        if spec is not None and spec.origin is not None:
            z3_init = spec.origin
            candidate = os.path.join(os.path.dirname(z3_init), "lib", name)
            if os.path.isfile(candidate):
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                return
    except (ImportError, ValueError, AttributeError):
        pass

    # 2) Try ctypes.util.find_library as a system-level fallback.
    try:
        from ctypes.util import find_library
        lib_path = find_library("z3")
        if lib_path and os.path.isfile(lib_path):
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            return
    except (ImportError, OSError):
        pass

    # 3) Walk sys.path for the wheel layout (z3/lib/libz3.dylib).
    for root in sys.path:
        if not root or not os.path.isdir(root):
            continue
        candidate = os.path.join(root, "z3", "lib", name)
        if os.path.isfile(candidate):
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return


# Core C++ bindings — preload Z3 dylib on Darwin, then load core before the z3
# Python package (see comment below) to reduce Z3 init / mutex issues.
CORE_AVAILABLE = False
Z3_AVAILABLE = False
NATIVE_LIBS_AVAILABLE = False

_preload_z3_shared_library()

try:
    from .core import *

    CORE_AVAILABLE = True
    Z3_AVAILABLE = True  # Core loaded successfully, Z3 is available through it
    NATIVE_LIBS_AVAILABLE = True
except (ImportError, OSError) as e:
    raise ImportError(
        "YiRage requires the native runtime (libyirage_runtime) and Cython "
        "module yirage.core. Build from the repository root with CMake + "
        "setuptools (e.g. YIRAGE_BACKEND=cpu pip install -e . --no-build-isolation). "
        f"Original error: {e}"
    ) from e

# Kernel graphs (new location) - depends on core and torch
KERNEL_AVAILABLE = False
try:
    from .kernel.graph import KNGraph, get_key_paths
    from .kernel.threadblock import TBGraph
    from .kernel.speculative import (
        SpecDecodeConfig,
        PromptLookupConfig,
        LookaheadConfig,
        spec_decode_class,
    )

    KERNEL_AVAILABLE = True
except (ImportError, OSError) as e:
    import warnings

    warnings.warn(f"Kernel module not available: {e}")

# op_registry has no torch/C++ dependency — always import.
try:
    from .kernel.op_registry import (
        CustomOpSpec,
        OpRegistry,
        global_registry,
        register_op,
        custom_op,
        list_ops,
        get_op,
    )
except (ImportError, OSError) as _e:
    import warnings
    warnings.warn(f"op_registry not available: {_e}")

# Backend API (new location)
from .backends.api import (
    get_available_backends,
    is_backend_available,
    get_default_backend,
    get_backend_info,
    set_default_backend,
    list_backends,
    available_backends,
    default_backend,
)

# Backend Compiler (new location)
from .backends.compiler import (
    CompilerBackend,
    CompileConfig,
    CompileResult,
    CompilerFactory,
    compile_kernel,
    get_target_arch_from_device,
)

# Backend-specific configurations (new locations) - all optional
try:
    from .backends.mps.config import (
        get_mps_search_config,
        get_mps_memory_config,
        apply_backend_config,
    )
except ImportError:
    pass

try:
    from .backends.ascend.config import (
        get_ascend_search_config,
        get_ascend_memory_config,
        get_ascend_device_info,
        check_torch_npu,
        require_torch_npu,
        is_ascend_available,
    )
except ImportError:
    pass

try:
    from .backends.maca.config import (
        get_maca_search_config,
        get_maca_memory_config,
        get_maca_device_info,
        is_maca_available,
        get_maca_sdk_path,
        apply_maca_config,
        MACA_WARP_SIZE,
    )
except ImportError:
    pass

try:
    from .backends.cuda.config import (
        get_cuda_search_config,
        get_cuda_memory_config,
        is_cuda_available,
    )
except ImportError:
    pass

try:
    from .backends.rocm.config import (
        get_rocm_search_config,
        get_rocm_memory_config,
        is_rocm_available,
    )
except ImportError:
    pass

try:
    from .backends.cpu.config import get_cpu_search_config, get_cpu_info
except ImportError:
    pass

try:
    from .backends.tpu.config import get_tpu_search_config, get_tpu_info, is_tpu_available
except ImportError:
    pass

try:
    from .backends.xpu.config import get_xpu_search_config, get_xpu_info, is_xpu_available
except ImportError:
    pass

try:
    from .backends.fpga.config import get_fpga_search_config, get_fpga_info, is_fpga_available
except ImportError:
    pass

# Ascend Transpiler (new location) - optional
try:
    from .backends.ascend.transpiler import (
        AscendTranspileConfig,
        AscendTranspileResult,
        CodeGenPath,
        AscendDeviceType,
        detect_ascend_environment,
        transpile_to_ascend,
    )
except ImportError:
    pass

# Multi-Backend Kernel (new location) - optional
try:
    from .kernel.multi_backend import (
        MultiBackendKernel,
        KernelBackend,
        KernelConfig,
        KernelExecutionContext,
        create_kernel,
    )
except (ImportError, OSError):
    pass

# Storage (new location) - optional
try:
    from .storage.graph_dataset import graph_dataset
    from .storage.mugraph_store import (
        MuGraphStore,
        MuGraphMetadata,
        MuGraphEntry,
        get_mugraph_store,
        save_mugraph,
        find_mugraph,
        find_best_mugraph,
    )
except ImportError:
    pass

# Profiler (new location) - optional
try:
    from .profiler.hardware import (
        HardwareProfiler,
        ProfileConfig,
        TimingResult,
        HardwareCounters,
        TrainingDataCollector,
        TrainingBenchmarkResult,
        check_google_benchmark_available,
    )
except ImportError:
    pass

# Unified Compiler - integrates muGraph search with MLIR compilation
try:
    from .compiler import (
        UnifiedCompiler,
        CompileMode,
        CompileOptions,
        CompileResult,
        compile_graph,
        hardware_aware_compile,
        jit_compile,
        CompilePipeline,
        get_compile_cache,
        clear_compile_cache,
    )
except ImportError:
    pass

# MLIR Integration - optional (requires USE_MLIR=ON build)
MLIR_AVAILABLE = False
try:
    # Try to import native MLIR bindings
    from ._yirage_mlir import (
        GPUBackend,
        GPUTargetConfig,
        CompilationResult,
        MLIRContext,
        parseMLIR,
        printMLIR,
        runYirageToLinalg,
        runGPUPipeline,
        runCUDAPipeline,
        runROCmPipeline,
        runCPUPipeline,
        runCustomPipeline,
        generatePTX,
        generateROCm,
        generateSPIRV,
        generateMetal,
        generateCubin,
        generateHSACO,
        generateSPIRVBinary,
        registerPasses,
        isBackendAvailable,
        getAvailableBackends,
        backendToString,
        stringToBackend,
    )
    MLIR_AVAILABLE = True
except ImportError:
    # Fall back to pure Python MLIR API
    try:
        import sys
        from pathlib import Path
        _mlir_python_path = Path(__file__).parent.parent.parent / "mlir" / "python"
        if _mlir_python_path.exists():
            sys.path.insert(0, str(_mlir_python_path))
        
        from yirage_mlir import (
            Target as MLIRTarget,
            YirageModule,
            CompiledKernel,
        )
        MLIR_AVAILABLE = True
    except ImportError:
        pass


def is_mlir_available() -> bool:
    """Check if MLIR compilation support is available."""
    return MLIR_AVAILABLE

# Utils (new location) - optional
try:
    from .utils.common import get_shared_memory_capacity, get_nvcc_compiler
except ImportError:
    pass

# Core Bridge - unified interface to C++ core
try:
    from .core_bridge import (
        CoreBridge,
        CoreCapabilities,
        get_core_bridge,
        get_capabilities,
        is_core_available,
        is_rl_available,
        is_mlir_available,
    )
except ImportError:
    pass

# Logging and Error Handling
try:
    from .logging_config import (
        get_logger,
        LogConfig,
        LogLevel,
        PerfLogger,
        StructuredLogger,
        YirageError,
        CoreError,
        SearchError,
        BackendError,
        CompilationError,
        RLError,
        ErrorCode,
    )
except ImportError:
    pass

# Hardware Device Management (registry for chip architectures)
try:
    from .hardware import (
        HardwareRegistry,
        ChipArchitecture,
        detect_current_chip,
    )
except ImportError:
    pass

# Global config and version
try:
    from .global_config import global_config
except ImportError:
    global_config = None
from .version import __version__

# Auto-initialize backends
try:
    if hasattr(core, "init_backends"):
        core.init_backends()
except Exception as e:
    import sys

    print(f"Warning: Backend initialization failed: {e}", file=sys.stderr)


class InputNotFoundError(Exception):
    """Raised when cannot find input tensors"""

    pass


def set_gpu_device_id(device_id: int):
    """Set GPU device ID for kernels."""
    if global_config is not None:
        global_config.gpu_device_id = device_id
    core.set_gpu_device_id(device_id)


def bypass_compile_errors(value: bool = True):
    """Bypass compile errors during kernel optimization."""
    if global_config is not None:
        global_config.bypass_compile_errors = value


def _get_initial_backend() -> str:
    """Pick the execution backend for newly created graphs."""
    backend = get_default_backend()
    return backend or "cpu"


def new_kernel_graph():
    """Create a new kernel graph."""
    kgraph = core.CyKNGraph()
    return KNGraph(kgraph, backend=_get_initial_backend())


def new_threadblock_graph(
    grid_dim: tuple, block_dim: tuple, forloop_range: int, reduction_dimx: int
):
    """Create a new threadblock graph."""
    bgraph = core.CyTBGraph(grid_dim, block_dim, forloop_range, reduction_dimx)
    return TBGraph(bgraph)


# Ray Distributed Search (optional)
try:
    from .ray import (
        DistributedSearchCoordinator,
        SearchWorker,
        SearchFeedback,
        SearchPartition,
        create_partitions,
    )

    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False


def is_ray_available() -> bool:
    return _RAY_AVAILABLE


# Backwards compatibility alias
is_distributed_available = is_ray_available


# RL-Guided Search (optional)
try:
    from .rl import (
        YiRageSearchEnv,
        EnvConfig,
        train_rl_search,
        TrainingConfig,
    )

    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False


def is_rl_available() -> bool:
    return _RL_AVAILABLE


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# Old module-level imports (deprecated, use new paths)
backend_api = backends = None
try:
    from . import backends

    backend_api = getattr(backends, "api", None)
except ImportError:
    pass

# Re-export for backwards compatibility
PersistentKernel = None
try:
    from .persistent_kernel.kernel import PersistentKernel
except ImportError:
    pass
