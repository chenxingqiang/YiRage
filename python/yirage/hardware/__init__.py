# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Hardware Device Management Module.

Provides a unified registry for chip architectures so that new accelerator
chips can be registered at runtime without modifying any existing code.

Quick start::

    from yirage.hardware import HardwareRegistry, ChipArchitecture

    # Look up a built-in chip
    reg = HardwareRegistry.instance()
    h100 = reg.get("nvidia_h100")
    print(h100.summary())

    # Register a brand-new chip
    my_chip = ChipArchitecture(
        chip_id="myvendor_x1",
        chip_name="MyVendor X1 Accelerator",
        backend="cuda",
        ...
    )
    reg.register(my_chip)

    # Auto-detect the current machine's chip
    from yirage.hardware import detect_current_chip
    chip = detect_current_chip()
"""

# Auto-populate built-in chips on first import
from .builtin_chips import register_builtin_chips as _register_builtin_chips
from .chip_arch import (
    ChipArchitecture,
    ChipCategory,
    ChipVendor,
    ComputeSpec,
    FeatureFlags,
    MemorySpec,
    MemoryType,
)
from .detector import detect_current_chip
from .registry import HardwareRegistry

_register_builtin_chips()

__all__ = [
    # Core types
    "ChipArchitecture",
    "ChipCategory",
    "ChipVendor",
    "ComputeSpec",
    "FeatureFlags",
    "MemorySpec",
    "MemoryType",
    # Registry
    "HardwareRegistry",
    # Detection
    "detect_current_chip",
]
