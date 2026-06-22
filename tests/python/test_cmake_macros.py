# Copyright 2025 YiRage Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for yirage.cmake_macros (Cython / setup.py alignment with config.cmake)."""

import importlib.util
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CMAKE_MACROS_PATH = _REPO_ROOT / "python" / "yirage" / "cmake_macros.py"


def _load_cmake_macros():
    spec = importlib.util.spec_from_file_location("cmake_macros", _CMAKE_MACROS_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_parse_config_ignores_trailing_hash_comments(tmp_path):
    """Repo config.cmake uses inline # comments after set(); parser must accept them."""
    cm = _load_cmake_macros()
    p = tmp_path / "config.cmake"
    p.write_text(
        "set(USE_MPS ON)   # Apple\nset(USE_CPU OFF)\n",
        encoding="utf-8",
    )
    flags = cm.parse_config_cmake(str(p))
    assert flags.get("USE_MPS") is True
    assert flags.get("USE_CPU") is False


def test_macros_match_repo_config_cmake():
    """Repo config.cmake (auto-generated or checked in) must map consistently to macros."""
    cm = _load_cmake_macros()
    cfg = _REPO_ROOT / "config.cmake"
    if not cfg.is_file():
        pytest.skip("repo config.cmake not present")
    flags = cm.parse_config_cmake(str(cfg))
    names = {m[0] for m in cm.macros_from_config(str(cfg))}
    if flags.get("USE_MPS"):
        assert "YIRAGE_BACKEND_MPS_ENABLED" in names
    if flags.get("USE_CPU"):
        assert "YIRAGE_BACKEND_CPU_ENABLED" in names
    if flags.get("USE_MLIR"):
        assert "YIRAGE_BACKEND_MLIR_ENABLED" in names
        assert "YIRAGE_MLIR_ENABLED" in names
    assert "YIRAGE_FINGERPRINT_USE_CPU" in names
    if flags.get("USE_OPENMP") and not flags.get("USE_MACA") and not flags.get(
        "USE_ASCEND"
    ):
        assert "YIRAGE_ENABLE_PARALLEL_SEARCH" in names


def test_macros_cuda_includes_compat_and_fingerprint(tmp_path):
    cm = _load_cmake_macros()
    p = tmp_path / "config.cmake"
    p.write_text(
        textwrap.dedent(
            """
            set(USE_CUDA ON)
            set(USE_CPU ON)
            set(USE_OPENMP ON)
            set(USE_CUDNN OFF)
            set(USE_ROCM OFF)
            set(USE_MPS OFF)
            set(USE_XPU OFF)
            set(USE_ASCEND OFF)
            set(USE_MACA OFF)
            set(USE_TPU OFF)
            set(USE_FPGA OFF)
            set(USE_MKL OFF)
            set(USE_MKLDNN OFF)
            set(USE_XEON OFF)
            set(USE_NKI OFF)
            set(USE_TRITON OFF)
            set(USE_MLIR OFF)
            set(USE_STABLEHLO OFF)
            set(USE_TVM OFF)
            set(USE_IREE OFF)
            set(USE_MHA OFF)
            set(USE_NNPACK OFF)
            set(USE_OPT_EINSUM OFF)
            set(USE_CUSPARSELT OFF)
            set(USE_CUTLASS OFF)
            set(USE_FORMAL_VERIFIER OFF)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    names = {m[0] for m in cm.macros_from_config(str(p))}
    assert "YIRAGE_BACKEND_CUDA_ENABLED" in names
    assert "YIRAGE_BACKEND_USE_CUDA" in names
    assert "YIRAGE_FINGERPRINT_USE_CUDA" in names
    assert "YIRAGE_FINGERPRINT_USE_CPU" not in names
    assert "YIRAGE_ENABLE_PARALLEL_SEARCH" in names


def test_macros_mlir_matches_cpu_openmp(tmp_path):
    """USE_MLIR ON must expose YIRAGE_BACKEND_MLIR_ENABLED for Cython/CMake parity."""
    cm = _load_cmake_macros()
    p = tmp_path / "config.cmake"
    p.write_text(
        textwrap.dedent(
            """
            set(USE_CUDA OFF)
            set(USE_CPU ON)
            set(USE_OPENMP ON)
            set(USE_MLIR ON)
            set(USE_CUDNN OFF)
            set(USE_ROCM OFF)
            set(USE_MPS OFF)
            set(USE_XPU OFF)
            set(USE_ASCEND OFF)
            set(USE_MACA OFF)
            set(USE_TPU OFF)
            set(USE_FPGA OFF)
            set(USE_MKL OFF)
            set(USE_MKLDNN OFF)
            set(USE_XEON OFF)
            set(USE_NKI OFF)
            set(USE_TRITON OFF)
            set(USE_STABLEHLO OFF)
            set(USE_TVM OFF)
            set(USE_IREE OFF)
            set(USE_MHA OFF)
            set(USE_NNPACK OFF)
            set(USE_OPT_EINSUM OFF)
            set(USE_CUSPARSELT OFF)
            set(USE_CUTLASS OFF)
            set(USE_FORMAL_VERIFIER OFF)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    names = {m[0] for m in cm.macros_from_config(str(p))}
    assert "YIRAGE_BACKEND_MLIR_ENABLED" in names
    assert "YIRAGE_MLIR_ENABLED" in names


def test_macros_ascend_no_cpu_fingerprint(tmp_path):
    cm = _load_cmake_macros()
    p = tmp_path / "config.cmake"
    p.write_text(
        textwrap.dedent(
            """
            set(USE_CUDA OFF)
            set(USE_CPU ON)
            set(USE_OPENMP ON)
            set(USE_ASCEND ON)
            set(USE_CUDNN OFF)
            set(USE_ROCM OFF)
            set(USE_MPS OFF)
            set(USE_XPU OFF)
            set(USE_MACA OFF)
            set(USE_TPU OFF)
            set(USE_FPGA OFF)
            set(USE_MKL OFF)
            set(USE_MKLDNN OFF)
            set(USE_XEON OFF)
            set(USE_NKI OFF)
            set(USE_TRITON OFF)
            set(USE_MLIR OFF)
            set(USE_STABLEHLO OFF)
            set(USE_TVM OFF)
            set(USE_IREE OFF)
            set(USE_MHA OFF)
            set(USE_NNPACK OFF)
            set(USE_OPT_EINSUM OFF)
            set(USE_CUSPARSELT OFF)
            set(USE_CUTLASS OFF)
            set(USE_FORMAL_VERIFIER OFF)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    names = {m[0] for m in cm.macros_from_config(str(p))}
    assert "YIRAGE_BACKEND_ASCEND_ENABLED" in names
    assert "YIRAGE_FINGERPRINT_USE_ASCEND" in names
    assert "__ASCEND__" in names
    assert "YIRAGE_FINGERPRINT_USE_CPU" not in names
    assert "YIRAGE_ENABLE_PARALLEL_SEARCH" not in names
