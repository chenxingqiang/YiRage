# Copyright 2026 YiRage team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Environment / backend helpers for repo-root setup.py. Standalone module so
# tests can import it without running CMake, Rust, or setuptools.

from __future__ import annotations

import os
import re
from typing import List, Mapping, MutableMapping, Optional, Union

_CONFIG_SET_RE = re.compile(
    r"^\s*set\s*\(\s*([A-Z_][A-Z0-9_]*)\s+(ON|OFF)\s*\)\s*(?:#.*)?\s*$",
    re.IGNORECASE,
)

Environ = Union[Mapping[str, str], MutableMapping[str, str]]


def env_to_cmake_onoff(val: Optional[str]) -> Optional[str]:
    """Normalize shell truthy values to ON/OFF for CMake set(USE_X ...) and -DUSE_X=."""
    if val is None:
        return None
    v = str(val).strip().upper()
    if v in ("ON", "1", "TRUE", "YES"):
        return "ON"
    if v in ("OFF", "0", "FALSE", "NO"):
        return "OFF"
    return v


def merge_extra_use_flags_from_env(
    backends: MutableMapping[str, str], environ: Optional[Environ] = None
) -> None:
    """Apply USE_* env vars that are not only from YIRAGE_BACKEND (e.g. USE_MLIR)."""
    env = os.environ if environ is None else environ
    extra = (
        "USE_MLIR",
        "USE_STABLEHLO",
        "USE_TVM",
        "USE_IREE",
        "USE_FORMAL_VERIFIER",
        "USE_CUSPARSELT",
        "USE_CUTLASS",
        "USE_OPENMP",
        "USE_MKLDNN",
        "USE_MKL",
        "USE_MHA",
        "USE_NNPACK",
        "USE_OPT_EINSUM",
        "USE_XEON",
    )
    for flag in extra:
        raw = env.get(flag)
        if raw is None or str(raw).strip() == "":
            continue
        coerced = env_to_cmake_onoff(str(raw))
        if coerced is not None:
            backends[flag] = coerced


def should_regenerate_config_cmake(environ: Optional[Environ] = None) -> bool:
    """Regenerate config.cmake when YIRAGE_BACKEND or any non-empty USE_* is set."""
    env = os.environ if environ is None else environ
    if env.get("YIRAGE_BACKEND"):
        return True
    for name, val in env.items():
        if name.startswith("USE_") and str(val).strip() != "":
            return True
    return False


def cmake_mlir_extra_definitions(
    yirage_path: str, environ: Optional[Environ] = None
) -> List[str]:
    """
    Extra cmake argv fragments for LLVM/MLIR when USE_MLIR is ON.

    setup_llvm_mlir() reads CMake cache variables, not the shell alone, so MLIR_DIR
    and YIRAGE_LLVM_* must be passed as -D when set in the environment.
    """
    env = os.environ if environ is None else environ
    out: List[str] = []
    if env_to_cmake_onoff(env.get("USE_MLIR")) != "ON":
        return out
    for var in (
        "YIRAGE_LLVM_SOURCE",
        "YIRAGE_LLVM_VERSION",
        "YIRAGE_LLVM_BUILD_TYPE",
        "MLIR_DIR",
        "LLVM_DIR",
    ):
        val = env.get(var)
        if val is not None and str(val).strip() != "":
            out.append(f"-D{var}={val}")
    src = env.get("YIRAGE_LLVM_SOURCE")
    mlir_d = env.get("MLIR_DIR")
    if (src is None or str(src).strip() == "") and (
        mlir_d is None or str(mlir_d).strip() == ""
    ):
        llvm_sub_cmake = os.path.join(
            yirage_path, "deps", "llvm-project", "llvm", "CMakeLists.txt"
        )
        if not os.path.isfile(llvm_sub_cmake):
            out.append("-DYIRAGE_LLVM_SOURCE=system")
    return out


def _parse_config_cmake_on_flags(config_file: str) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    if not os.path.isfile(config_file):
        return flags
    with open(config_file, encoding="utf-8") as f:
        for line in f:
            m = _CONFIG_SET_RE.match(line)
            if m:
                flags[m.group(1)] = m.group(2).upper() == "ON"
    return flags


def resolve_llvm_library_dir(environ: Optional[Environ] = None) -> Optional[str]:
    """Directory containing libLLVM-*.so for system / prebuilt MLIR installs."""
    env = os.environ if environ is None else environ

    for var in ("MLIR_DIR", "LLVM_DIR"):
        raw = env.get(var)
        if raw is None or str(raw).strip() == "":
            continue
        cmake_dir = os.path.abspath(str(raw).strip())
        # .../lib/cmake/mlir or .../lib/cmake/llvm -> .../lib
        lib_dir = os.path.dirname(os.path.dirname(cmake_dir))
        if os.path.isdir(lib_dir) and (
            os.path.isfile(os.path.join(lib_dir, "libLLVM.so"))
            or any(
                name.startswith("libLLVM-") and ".so" in name
                for name in os.listdir(lib_dir)
            )
        ):
            return lib_dir

    for ver in ("19", "18", "17", "16"):
        lib_dir = f"/usr/lib/llvm-{ver}/lib"
        if os.path.isdir(lib_dir) and (
            os.path.isfile(os.path.join(lib_dir, "libLLVM.so"))
            or os.path.isfile(os.path.join(lib_dir, f"libLLVM-{ver}.so"))
        ):
            return lib_dir

    homebrew = "/opt/homebrew/opt/llvm/lib"
    if os.path.isdir(homebrew) and os.path.isfile(
        os.path.join(homebrew, "libLLVM.dylib")
    ):
        return homebrew
    return None


def _llvm_dylib_link_names(lib_dir: str) -> tuple[str, str]:
    """Return (-lLLVM*, -lMLIR*) basenames for the dylib pair in lib_dir."""
    llvm_name = "LLVM-17"
    mlir_name = "MLIR"
    for entry in os.listdir(lib_dir):
        if (
            entry.startswith("libLLVM-")
            and entry.endswith(".so")
            and ".so." not in entry[8:]
        ):
            llvm_name = entry[3:-3]
        elif entry == "libMLIR.so":
            mlir_name = "MLIR"
        elif (
            entry.startswith("libMLIR-")
            and entry.endswith(".so")
            and ".so." not in entry[8:]
        ):
            mlir_name = entry[3:-3]
    return llvm_name, mlir_name


def cython_mlir_extra_link_args(
    config_file: str,
    environ: Optional[Environ] = None,
    *,
    platform: str = "linux",
) -> List[str]:
    """
    Link libLLVM/libMLIR when yirage.core loads libyirage_runtime.a built with USE_MLIR.

    Static whole-archive linking does not propagate CMake PUBLIC deps (MLIR/LLVM dylibs),
    which surfaces as undefined ``llvm::DisableABIBreakingChecks`` at import time.
    """
    flags = _parse_config_cmake_on_flags(config_file)
    if not flags.get("USE_MLIR"):
        return []

    lib_dir = resolve_llvm_library_dir(environ)
    if lib_dir is None:
        return []

    llvm_lib, mlir_lib = _llvm_dylib_link_names(lib_dir)
    if platform == "darwin":
        return [
            f"-L{lib_dir}",
            f"-l{llvm_lib}",
            f"-l{mlir_lib}",
            f"-Wl,-rpath,{lib_dir}",
        ]

    return [
        "-Wl,--no-as-needed",
        f"-L{lib_dir}",
        f"-l{llvm_lib}",
        f"-l{mlir_lib}",
        "-Wl,--as-needed",
        f"-Wl,-rpath,{lib_dir}",
    ]
