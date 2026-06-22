# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Backend Support Module.

Provides multi-backend support for CUDA, MPS, MACA, Ascend, ROCm, CPU, and TPU.
"""

from .api import (
    get_available_backends,
    get_default_backend,
    is_backend_available,
    get_backend_info,
)

from .compiler import (
    CompilerBackend,
    CompileConfig,
    CompileResult,
    CompilerFactory,
    compile_kernel,
    get_target_arch_from_device,
)


# Lazy imports for backend configs to avoid import errors if dependencies missing
def get_cuda_config():
    """Get CUDA backend configuration."""
    from .cuda.config import get_cuda_search_config

    return get_cuda_search_config()


def get_mps_config():
    """Get MPS backend configuration."""
    from .mps.config import get_mps_search_config

    return get_mps_search_config()


def get_rocm_config():
    """Get ROCm backend configuration."""
    from .rocm.config import get_rocm_search_config

    return get_rocm_search_config()


def get_cpu_config():
    """Get CPU backend configuration."""
    from .cpu.config import get_cpu_search_config

    return get_cpu_search_config()


def get_ascend_config():
    """Get Ascend backend configuration."""
    from .ascend.config import get_ascend_search_config

    return get_ascend_search_config()


def get_maca_config():
    """Get MACA backend configuration."""
    from .maca.config import get_maca_search_config

    return get_maca_search_config()


def get_tpu_config():
    """Get TPU backend configuration."""
    from .tpu.config import get_tpu_search_config

    return get_tpu_search_config()


def get_xpu_config():
    """Get Intel XPU backend configuration."""
    from .xpu.config import get_xpu_search_config

    return get_xpu_search_config()


def get_fpga_config():
    """Get FPGA backend configuration."""
    from .fpga.config import get_fpga_search_config

    return get_fpga_search_config()


def get_chip_registry():
    """Get the hardware device registry with all known chip architectures."""
    from ..hardware import HardwareRegistry

    return HardwareRegistry.instance()


__all__ = [
    # API
    "get_available_backends",
    "get_default_backend",
    "is_backend_available",
    "get_backend_info",
    # Compiler
    "CompilerBackend",
    "CompileConfig",
    "CompileResult",
    "CompilerFactory",
    "compile_kernel",
    "get_target_arch_from_device",
    # Backend configs
    "get_cuda_config",
    "get_mps_config",
    "get_rocm_config",
    "get_cpu_config",
    "get_ascend_config",
    "get_maca_config",
    "get_tpu_config",
    "get_xpu_config",
    "get_fpga_config",
    # Hardware registry
    "get_chip_registry",
]
