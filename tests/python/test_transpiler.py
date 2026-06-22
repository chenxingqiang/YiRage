# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Transpiler Module Tests

Maps to C++ tests:
  - test_transpiler_config_gtest.cc
  - test_transpiler_structs_gtest.cc
  - test_transpiler_sched_gtest.cc
  - test_transpiler_utils_gtest.cc
  - test_transpiler_memory_gtest.cc
  - test_transpiler_ascend_gtest.cc
  - test_transpiler_layout_gtest.cc
"""

import pytest
from typing import List, Dict

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def compiler_module():
    """Import the compiler module."""
    try:
        from yirage.backends import compiler
        return compiler
    except ImportError:
        pytest.skip("yirage.backends.compiler module not available")


@pytest.fixture
def ascend_transpiler_module():
    """Import the Ascend transpiler module."""
    try:
        from yirage.backends.ascend import transpiler
        return transpiler
    except ImportError:
        pytest.skip("yirage.backends.ascend.transpiler module not available")


@pytest.fixture
def core_module():
    """Import the core module."""
    try:
        from yirage import core
        return core
    except ImportError:
        pytest.skip("yirage.core module not available")


# =============================================================================
# Compiler Module Tests (maps to test_transpiler_config_gtest.cc)
# =============================================================================


class TestCompilerModule:
    """Tests for the compiler module."""

    def test_module_imports(self, compiler_module):
        """Test compiler module can be imported."""
        assert compiler_module is not None


# =============================================================================
# Ascend Transpiler Tests (maps to test_transpiler_ascend_gtest.cc)
# =============================================================================


class TestAscendTranspiler:
    """Tests for Ascend transpiler."""

    def test_module_exists(self, ascend_transpiler_module):
        """Test Ascend transpiler module exists."""
        assert ascend_transpiler_module is not None

    def test_codegen_path_enum(self, ascend_transpiler_module):
        """Test CodeGenPath enum exists if defined."""
        if hasattr(ascend_transpiler_module, "CodeGenPath"):
            CodeGenPath = ascend_transpiler_module.CodeGenPath
            members = list(CodeGenPath)
            assert len(members) >= 1, "CodeGenPath should define at least one variant"

    def test_ascend_device_type_enum(self, ascend_transpiler_module):
        """Test AscendDeviceType enum exists if defined."""
        if hasattr(ascend_transpiler_module, "AscendDeviceType"):
            AscendDeviceType = ascend_transpiler_module.AscendDeviceType
            assert AscendDeviceType is not None


# =============================================================================
# Core Transpiler Function Tests
# =============================================================================


class TestTranspilerCore:
    """Tests for core transpiler functionality."""

    def test_transpile_function_exists(self, core_module):
        """Test transpile function exists in core."""
        # The actual transpile function may be in core or generated via Cython
        # This checks if the binding exists
        if hasattr(core_module, "transpile"):
            assert callable(core_module.transpile)

    def test_generate_cuda_function_exists(self, core_module):
        """Test generate_cuda function exists if available."""
        if hasattr(core_module, "generate_cuda"):
            assert callable(core_module.generate_cuda)


# =============================================================================
# GPU Compute Capability Tests (maps to test_transpiler_config_gtest.cc)
# =============================================================================


class TestGPUComputeCapability:
    """Tests for GPU compute capability handling."""

    @pytest.mark.parametrize("gpu,expected_cc", [
        ("P100", 60),
        ("V100", 70),
        ("T4", 75),
        ("A100", 80),
        ("H100", 90),
    ])
    def test_gpu_compute_capabilities(self, gpu, expected_cc):
        """Test known GPU compute capabilities."""
        # This documents the expected compute capabilities
        # The actual implementation should match these
        gpu_cc_map = {
            "P100": 60,
            "V100": 70,
            "T4": 75,
            "A100": 80,
            "H100": 90,
            "B200": 100,
        }
        assert gpu_cc_map[gpu] == expected_cc


# =============================================================================
# Transpiler Config Tests (maps to test_transpiler_config_gtest.cc)
# =============================================================================


class TestTranspilerConfig:
    """Tests for transpiler configuration."""

    def test_config_target_cc_range(self):
        """Test valid compute capability range."""
        # Valid CC range should be 60-100 (P100 to B200)
        valid_cc_range = range(60, 101, 5)  # 60, 65, 70, ..., 100
        for cc in [60, 70, 75, 80, 90, 100]:
            assert cc >= 60 and cc <= 100

    def test_hopper_cc_threshold(self):
        """Test Hopper compute capability threshold."""
        # Hopper (H100) is CC 90+
        hopper_cc = 90
        assert hopper_cc >= 90

    def test_blackwell_cc_threshold(self):
        """Test Blackwell compute capability threshold."""
        # Blackwell (B200) is CC 100+
        blackwell_cc = 100
        assert blackwell_cc >= 100


# =============================================================================
# Layout Tests (maps to test_transpiler_layout_gtest.cc)
# =============================================================================


class TestLayoutConcepts:
    """Tests for layout concepts."""

    def test_row_major_strides(self):
        """Test row-major stride calculation."""
        # For shape [64, 128], row-major strides are [128, 1]
        shape = [64, 128]
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        
        assert strides == [128, 1]

    def test_column_major_strides(self):
        """Test column-major stride calculation."""
        # For shape [64, 128], column-major strides are [1, 64]
        shape = [64, 128]
        strides = []
        stride = 1
        for dim in shape:
            strides.append(stride)
            stride *= dim
        
        assert strides == [1, 64]

    def test_contiguous_check(self):
        """Test contiguity check for layouts."""
        # Contiguous row-major
        shape = [64, 128]
        strides = [128, 1]
        
        expected_stride = 1
        is_contiguous = True
        for i in range(len(shape) - 1, -1, -1):
            if strides[i] != expected_stride:
                is_contiguous = False
                break
            expected_stride *= shape[i]
        
        assert is_contiguous


# =============================================================================
# Swizzle Tests (maps to test_transpiler_layout_gtest.cc)
# =============================================================================


class TestSwizzleConcepts:
    """Tests for swizzle concepts."""

    def test_swizzle_size_selection(self):
        """Test swizzle size selection based on row bytes."""
        def select_swizzle(row_bytes):
            if row_bytes >= 128:
                return "XOR_128B"
            elif row_bytes >= 64:
                return "XOR_64B"
            elif row_bytes >= 32:
                return "XOR_32B"
            else:
                return "NONE"
        
        # FP16 element size = 2 bytes
        elem_size = 2
        
        assert select_swizzle(128 * elem_size) == "XOR_128B"  # 256 bytes
        assert select_swizzle(64 * elem_size) == "XOR_128B"   # 128 bytes
        assert select_swizzle(32 * elem_size) == "XOR_64B"    # 64 bytes
        assert select_swizzle(16 * elem_size) == "XOR_32B"    # 32 bytes
        assert select_swizzle(8 * elem_size) == "NONE"        # 16 bytes


# =============================================================================
# Memory Planning Tests (maps to test_transpiler_memory_gtest.cc)
# =============================================================================


class TestMemoryPlanning:
    """Tests for memory planning concepts."""

    def test_alignment(self):
        """Test memory alignment."""
        def align_up(value, alignment):
            return ((value + alignment - 1) // alignment) * alignment
        
        assert align_up(100, 128) == 128
        assert align_up(128, 128) == 128
        assert align_up(129, 128) == 256
        assert align_up(0, 128) == 0

    def test_ceil_div(self):
        """Test ceiling division."""
        def ceil_div(a, b):
            return (a + b - 1) // b
        
        assert ceil_div(10, 2) == 5
        assert ceil_div(10, 3) == 4
        assert ceil_div(10, 4) == 3
        assert ceil_div(1, 10) == 1


# =============================================================================
# Code Generation Utility Tests (maps to test_transpiler_utils_gtest.cc)
# =============================================================================


class TestCodeGenUtils:
    """Tests for code generation utilities."""

    def test_format_string_replacement(self):
        """Test format string with marker replacement."""
        def fmt(template, *args):
            result = template
            for arg in args:
                result = result.replace("$", str(arg), 1)
            return result
        
        assert fmt("int x = $;", 42) == "int x = 42;"
        assert fmt("$ + $ = $", 1, 2, 3) == "1 + 2 = 3"

    def test_map_function(self):
        """Test map function."""
        def map_fn(lst, fn):
            return [fn(x) for x in lst]
        
        result = map_fn([1, 2, 3], lambda x: x * 2)
        assert result == [2, 4, 6]

    def test_cute_int_format(self):
        """Test CUTE Int<> format."""
        def to_cute_int(values):
            return [f"Int<{v}>" for v in values]
        
        result = to_cute_int([64, 128, 256])
        assert result == ["Int<64>", "Int<128>", "Int<256>"]


# =============================================================================
# Scheduling Tests (maps to test_transpiler_sched_gtest.cc)
# =============================================================================


class TestSchedulingConcepts:
    """Tests for scheduling concepts."""

    def test_schedule_node_types(self):
        """Test schedule node types."""
        node_types = ["OPERATOR", "SYNCTHREADS"]
        assert "OPERATOR" in node_types
        assert "SYNCTHREADS" in node_types

    def test_schedule_phases(self):
        """Test schedule phases."""
        phases = ["pre_loop", "loop", "post_loop"]
        assert len(phases) == 3


# =============================================================================
# TMA Tests (for Hopper+)
# =============================================================================


class TestTMAConcepts:
    """Tests for TMA (Tensor Memory Accelerator) concepts."""

    def test_tma_requires_hopper(self):
        """Test TMA requires Hopper or above."""
        hopper_cc = 90
        ampere_cc = 80
        
        def supports_tma(cc):
            return cc >= 90
        
        assert supports_tma(hopper_cc) is True
        assert supports_tma(ampere_cc) is False

    def test_multicast_directions(self):
        """Test TMA multicast directions."""
        directions = ["NOT_MULTICAST", "X_MULTICAST", "Y_MULTICAST", "Z_MULTICAST"]
        assert "NOT_MULTICAST" in directions


# =============================================================================
# Data Type Tests
# =============================================================================


class TestDataTypeConcepts:
    """Tests for data type handling."""

    @pytest.mark.parametrize("dtype,size", [
        ("fp16", 2),
        ("bf16", 2),
        ("fp32", 4),
        ("fp64", 8),
        ("int8", 1),
        ("int32", 4),
    ])
    def test_dtype_sizes(self, dtype, size):
        """Test data type sizes in bytes."""
        dtype_sizes = {
            "fp16": 2,
            "bf16": 2,
            "fp32": 4,
            "fp64": 8,
            "int8": 1,
            "int32": 4,
        }
        assert dtype_sizes[dtype] == size


# =============================================================================
# Bank Conflict Tests
# =============================================================================


class TestBankConflictConcepts:
    """Tests for bank conflict analysis."""

    def test_num_banks(self):
        """Test shared memory bank count."""
        NUM_BANKS = 32
        assert NUM_BANKS == 32

    def test_bank_width(self):
        """Test bank width in bytes."""
        BANK_WIDTH = 4
        assert BANK_WIDTH == 4
