# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""Apple MPS backend support."""

from .config import (
    get_mps_search_config,
    get_mps_memory_config,
    apply_backend_config,
    AppleChipFamily,
    AppleSiliconSpecs,
    detect_apple_silicon,
    get_apple_gpu_info,
    get_chip_optimized_config,
    get_all_chip_specs,
    print_apple_silicon_info,
)

__all__ = [
    "get_mps_search_config",
    "get_mps_memory_config",
    "apply_backend_config",
    "AppleChipFamily",
    "AppleSiliconSpecs",
    "detect_apple_silicon",
    "get_apple_gpu_info",
    "get_chip_optimized_config",
    "get_all_chip_specs",
    "print_apple_silicon_info",
]
