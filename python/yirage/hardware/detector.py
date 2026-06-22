# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Runtime hardware detection.

Probes the running system to find which chip is present and returns the
matching :class:`~yirage.hardware.chip_arch.ChipArchitecture` from the
:class:`~yirage.hardware.registry.HardwareRegistry`.
"""

from __future__ import annotations

import subprocess

from .chip_arch import ChipArchitecture
from .registry import HardwareRegistry


def detect_current_chip(
    registry: HardwareRegistry | None = None,
) -> ChipArchitecture | None:
    """
    Auto-detect the chip present on this machine.

    Probing order:
      1. NVIDIA GPU (via ``nvidia-smi``)
      2. AMD GPU (via ``rocm-smi``)
      3. Huawei Ascend (via ``npu-smi``)
      4. MetaX MACA (via ``mx-smi``)
      5. Apple MPS (via PyTorch)
      6. Fallback → ``None``

    The detected chip_id is looked up in the registry.  If a chip is detected
    but not registered, ``None`` is returned (you can register it first).

    Returns:
        The matching :class:`ChipArchitecture` or *None*.
    """
    if registry is None:
        registry = HardwareRegistry.instance()

    chip_id = _detect_nvidia() or _detect_amd() or _detect_ascend() or _detect_metax() or _detect_mps()
    if chip_id:
        return registry.get(chip_id)
    return None


# ------------------------------------------------------------------ probes


def _run(cmd: list[str], timeout: int = 5) -> str | None:
    """Run a command, return stdout or None."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _detect_nvidia() -> str | None:
    out = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
    if out is None:
        return None
    try:
        cc = float(out.strip().split("\n")[0])
        if cc >= 10.0:
            return "nvidia_b200"
        if cc >= 9.0:
            return "nvidia_h100"
        if cc >= 8.9:
            return "nvidia_rtx4090"
        if cc >= 8.6:
            return "nvidia_rtx3090"
        if cc >= 8.0:
            return "nvidia_a100"
        if cc >= 7.5:
            return "nvidia_t4"
        if cc >= 7.0:
            return "nvidia_v100"
    except (ValueError, IndexError):
        pass
    return None


def _detect_amd() -> str | None:
    out = _run(["rocm-smi", "--showproductname"])
    if out is None:
        return None
    text = out.upper()
    if "MI300" in text:
        return "amd_mi300x"
    if "MI250" in text:
        return "amd_mi250x"
    return None


def _detect_ascend() -> str | None:
    out = _run(["npu-smi", "info"])
    if out is None:
        return None
    if "910B" in out:
        return "ascend_910b"
    if "910" in out:
        return "ascend_910"
    if "310P" in out:
        return "ascend_310p"
    return None


def _detect_metax() -> str | None:
    out = _run(["mx-smi"])
    if out is None:
        return None
    if "C500 Pro" in out:
        return "metax_c500_pro"
    if "C500" in out:
        return "metax_c500"
    return None


def _detect_mps() -> str | None:
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out:
                brand = out.strip()
                # brand looks like "Apple M1 Pro", "Apple M2 Max", "Apple M3", etc.
                if brand.startswith("Apple "):
                    parts = brand[6:].lower().split()
                    # parts[0] is the generation (e.g. "m1", "m2")
                    # parts[1:] is the variant (e.g. [], ["pro"], ["max"], ["ultra"])
                    chip_id = "apple_" + "_".join(parts)
                    return chip_id
            return None
    except ImportError:
        pass
    return None
