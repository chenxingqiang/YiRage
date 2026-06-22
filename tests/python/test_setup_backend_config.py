# Copyright 2025 YiRage Project
# SPDX-License-Identifier: Apache-2.0

"""Functional tests for tools/setup_backend_config (pip / CMake env alignment)."""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = str(_REPO_ROOT / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import setup_backend_config as ysc


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("1", "ON"),
        ("0", "OFF"),
        ("yes", "ON"),
        ("no", "OFF"),
        ("ON", "ON"),
        ("OFF", "OFF"),
        ("maybe", "MAYBE"),
    ],
)
def test_env_to_cmake_onoff(raw, expected):
    assert ysc.env_to_cmake_onoff(raw) == expected


def test_should_regenerate_yirage_backend():
    assert ysc.should_regenerate_config_cmake({"YIRAGE_BACKEND": "cpu"}) is True
    assert ysc.should_regenerate_config_cmake({}) is False


def test_should_regenerate_any_use_prefix():
    """Any non-empty USE_* env var triggers config.cmake regen (CMake-style flags)."""
    assert ysc.should_regenerate_config_cmake({"USE_MLIR": "1"}) is True
    assert ysc.should_regenerate_config_cmake({"USE_FOO": "1"}) is True
    assert ysc.should_regenerate_config_cmake({"USE_CUDA": ""}) is False


def test_merge_extra_use_merges_mlir():
    backends: dict = {"USE_CPU": "ON"}
    ysc.merge_extra_use_flags_from_env(
        backends, {"USE_MLIR": "1", "USE_STABLEHLO": "0"}
    )
    assert backends["USE_MLIR"] == "ON"
    assert backends["USE_STABLEHLO"] == "OFF"


def test_cmake_mlir_extra_off_when_use_mlir_unset():
    assert ysc.cmake_mlir_extra_definitions(str(_REPO_ROOT), {}) == []


def test_cmake_mlir_forwards_mlir_dir():
    env = {"USE_MLIR": "1", "MLIR_DIR": "/opt/llvm/lib/cmake/mlir"}
    out = ysc.cmake_mlir_extra_definitions(str(_REPO_ROOT), env)
    assert "-DMLIR_DIR=/opt/llvm/lib/cmake/mlir" in out
    assert "-DYIRAGE_LLVM_SOURCE=system" not in out


def test_cmake_mlir_respects_explicit_submodule_no_auto_system(tmp_path):
    env = {"USE_MLIR": "1", "YIRAGE_LLVM_SOURCE": "submodule"}
    out = ysc.cmake_mlir_extra_definitions(str(tmp_path), env)
    assert "-DYIRAGE_LLVM_SOURCE=submodule" in out
    assert "-DYIRAGE_LLVM_SOURCE=system" not in out


def test_cmake_mlir_auto_system_without_submodule(tmp_path):
    env = {"USE_MLIR": "1"}
    out = ysc.cmake_mlir_extra_definitions(str(tmp_path), env)
    assert "-DYIRAGE_LLVM_SOURCE=system" in out


def test_cmake_mlir_no_auto_system_when_submodule_present(tmp_path):
    llvm_cmake = tmp_path / "deps" / "llvm-project" / "llvm" / "CMakeLists.txt"
    llvm_cmake.parent.mkdir(parents=True)
    llvm_cmake.write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    env = {"USE_MLIR": "1"}
    out = ysc.cmake_mlir_extra_definitions(str(tmp_path), env)
    assert "-DYIRAGE_LLVM_SOURCE=system" not in out


def test_resolve_llvm_library_dir_from_mlir_dir():
    env = {"MLIR_DIR": "/usr/lib/llvm-17/lib/cmake/mlir"}
    assert ysc.resolve_llvm_library_dir(env) == "/usr/lib/llvm-17/lib"


def test_cython_mlir_link_args_off_without_use_mlir(tmp_path):
    cfg = tmp_path / "config.cmake"
    cfg.write_text("set(USE_CPU ON)\nset(USE_MLIR OFF)\n", encoding="utf-8")
    assert ysc.cython_mlir_extra_link_args(str(cfg), {}) == []


def test_cython_mlir_link_args_when_use_mlir_on(tmp_path):
    cfg = tmp_path / "config.cmake"
    cfg.write_text("set(USE_CPU ON)\nset(USE_MLIR ON)\n", encoding="utf-8")
    env = {"MLIR_DIR": "/usr/lib/llvm-17/lib/cmake/mlir"}
    out = ysc.cython_mlir_extra_link_args(str(cfg), env, platform="linux")
    assert "-L/usr/lib/llvm-17/lib" in out
    assert "-lLLVM-17" in out
    assert "-lMLIR" in out
    assert any("rpath" in arg for arg in out)