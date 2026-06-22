# Copyright 2025 YiRage Project
# SPDX-License-Identifier: Apache-2.0
#
# Map repo-root config.cmake (set(VAR ON|OFF)) to C/C++ preprocessor macros for
# Cython and other tools. Kept in sync with CMakeLists.txt target_compile_definitions
# for yirage_runtime.

import re
from typing import Dict, List, Optional, Set, Tuple

# Allow trailing CMake line comments: set(USE_MPS ON)  # note
_CONFIG_SET_RE = re.compile(
    r"^\s*set\s*\(\s*([A-Z_][A-Z0-9_]*)\s+(ON|OFF)\s*\)\s*(?:#.*)?\s*$",
    re.IGNORECASE,
)

Macro = Tuple[str, Optional[str]]


def parse_config_cmake(config_file: str) -> Dict[str, bool]:
    """Parse set(NAME ON|OFF) lines from config.cmake. Missing variables are not entries."""
    flags: Dict[str, bool] = {}
    with open(config_file, encoding="utf-8") as f:
        for line in f:
            m = _CONFIG_SET_RE.match(line)
            if m:
                flags[m.group(1)] = m.group(2).upper() == "ON"
    return flags


def _on(flags: Dict[str, bool], key: str) -> bool:
    return flags.get(key, False)


def _add(seen: set[str], out: List[Macro], name: str, value: Optional[str] = None) -> None:
    if name not in seen:
        seen.add(name)
        out.append((name, value))


def macros_from_config(config_file: str) -> List[Macro]:
    """
    Return define_macros list matching CMake's yirage_runtime definitions for enabled flags.
    Unknown / missing keys are treated as OFF.
    """
    flags = parse_config_cmake(config_file)
    if not flags:
        raise ValueError(f"No set(NAME ON|OFF) entries parsed from {config_file!r}")

    use_any = any(k.startswith("USE_") and v for k, v in flags.items())
    if not use_any:
        raise ValueError(
            f"At least one USE_* backend flag must be ON in {config_file!r}"
        )

    out: List[Macro] = []
    seen: Set[str] = set()

    if _on(flags, "USE_CUDA"):
        _add(seen, out, "YIRAGE_BACKEND_CUDA_ENABLED")
        _add(seen, out, "YIRAGE_BACKEND_USE_CUDA")
        _add(seen, out, "YIRAGE_FINGERPRINT_USE_CUDA")

    if _on(flags, "USE_CPU"):
        _add(seen, out, "YIRAGE_BACKEND_CPU_ENABLED")

    if _on(flags, "USE_MPS"):
        _add(seen, out, "YIRAGE_BACKEND_MPS_ENABLED")

    if _on(flags, "USE_ROCM"):
        _add(seen, out, "YIRAGE_BACKEND_ROCM_ENABLED")
        _add(seen, out, "YIRAGE_FINGERPRINT_USE_ROCM")

    if _on(flags, "USE_XPU"):
        _add(seen, out, "YIRAGE_BACKEND_XPU_ENABLED")

    if _on(flags, "USE_ASCEND"):
        _add(seen, out, "YIRAGE_BACKEND_ASCEND_ENABLED")
        _add(seen, out, "__ASCEND__")
        _add(seen, out, "YIRAGE_FINGERPRINT_USE_ASCEND")

    if _on(flags, "USE_MACA"):
        _add(seen, out, "YIRAGE_BACKEND_MACA_ENABLED")
        _add(seen, out, "YIRAGE_FINGERPRINT_USE_MACA")

    if _on(flags, "USE_TPU"):
        _add(seen, out, "YIRAGE_BACKEND_TPU_ENABLED")

    if _on(flags, "USE_FPGA"):
        _add(seen, out, "YIRAGE_BACKEND_FPGA_ENABLED")

    if _on(flags, "USE_CUDNN"):
        _add(seen, out, "YIRAGE_BACKEND_CUDNN_ENABLED")

    if _on(flags, "USE_MKL"):
        _add(seen, out, "YIRAGE_BACKEND_MKL_ENABLED")

    if _on(flags, "USE_MKLDNN"):
        _add(seen, out, "YIRAGE_BACKEND_MKLDNN_ENABLED")

    if _on(flags, "USE_OPENMP"):
        _add(seen, out, "YIRAGE_BACKEND_OPENMP_ENABLED")

    if _on(flags, "USE_XEON"):
        _add(seen, out, "YIRAGE_BACKEND_XEON_ENABLED")

    if _on(flags, "USE_NKI"):
        _add(seen, out, "YIRAGE_BACKEND_NKI_ENABLED")
        _add(seen, out, "YIRAGE_BACKEND_USE_NKI")
        _add(seen, out, "YIRAGE_FINGERPRINT_USE_CPU")

    if _on(flags, "USE_TRITON"):
        _add(seen, out, "YIRAGE_BACKEND_TRITON_ENABLED")

    if _on(flags, "USE_MHA"):
        _add(seen, out, "YIRAGE_BACKEND_MHA_ENABLED")

    if _on(flags, "USE_NNPACK"):
        _add(seen, out, "YIRAGE_BACKEND_NNPACK_ENABLED")

    if _on(flags, "USE_OPT_EINSUM"):
        _add(seen, out, "YIRAGE_BACKEND_OPT_EINSUM_ENABLED")

    if _on(flags, "USE_CUSPARSELT"):
        _add(seen, out, "YIRAGE_BACKEND_CUSPARSELT_ENABLED")

    if _on(flags, "USE_CUTLASS"):
        _add(seen, out, "YIRAGE_BACKEND_CUTLASS_ENABLED")

    if _on(flags, "USE_MLIR"):
        _add(seen, out, "YIRAGE_BACKEND_MLIR_ENABLED")
        _add(seen, out, "YIRAGE_MLIR_ENABLED")

    if _on(flags, "USE_STABLEHLO"):
        _add(seen, out, "YIRAGE_BACKEND_STABLEHLO_ENABLED")

    if _on(flags, "USE_TVM"):
        _add(seen, out, "YIRAGE_BACKEND_TVM_ENABLED")

    if _on(flags, "USE_IREE"):
        _add(seen, out, "YIRAGE_BACKEND_IREE_ENABLED")

    if _on(flags, "USE_FORMAL_VERIFIER"):
        _add(seen, out, "YIRAGE_USE_FORMAL_VERIFIER")

    # Fingerprint: mirrors CMakeLists first if/elseif chain (CUDA / MACA / Ascend vs else CPU).
    if _on(flags, "USE_CUDA"):
        pass
    elif _on(flags, "USE_MACA"):
        pass
    elif _on(flags, "USE_ASCEND"):
        pass
    else:
        _add(seen, out, "YIRAGE_FINGERPRINT_USE_CPU")

    # YIRAGE_ENABLE_PARALLEL_SEARCH: CUDA / default-CPU-branch / MPS when OpenMP is on;
    # not defined for MACA or Ascend in CMakeLists (they only link OpenMP).
    if _on(flags, "USE_OPENMP") and not _on(flags, "USE_MACA") and not _on(
        flags, "USE_ASCEND"
    ):
        _add(seen, out, "YIRAGE_ENABLE_PARALLEL_SEARCH")

    return out
