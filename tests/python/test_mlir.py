#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
MLIR Module Unit Tests

Tests for yirage MLIR dialect, converter, and compiler.
Run with: pytest tests/python/test_mlir.py -v
"""

import pytest
import tempfile
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
MLIR_ROOT = PROJECT_ROOT / "mlir" / "python"
sys.path.insert(0, str(PYTHON_ROOT))
sys.path.insert(0, str(MLIR_ROOT))

from conftest import load_module


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def mugraph_to_mlir_module():
    """Load mugraph_to_mlir module."""
    return load_module("mugraph_to_mlir", MLIR_ROOT / "mugraph_to_mlir.py")


@pytest.fixture(scope="module")
def yirage_compiler_module():
    """Load yirage_compiler module."""
    return load_module("yirage_compiler", MLIR_ROOT / "yirage_compiler.py")


@pytest.fixture
def sample_matmul_json():
    """Sample matmul graph JSON."""
    return {
        "operators": [
            {"type": "matmul", "inputs": [0, 1], "outputs": [{"id": 2, "dims": [32, 128], "dtype": "fp32"}]}
        ],
        "inputs": [
            {"id": 0, "dims": [32, 64], "dtype": "fp32"},
            {"id": 1, "dims": [64, 128], "dtype": "fp32"}
        ],
        "outputs": [
            {"id": 2, "dims": [32, 128], "dtype": "fp32"}
        ]
    }


@pytest.fixture
def sample_mlp_json():
    """Sample MLP (matmul + silu) graph JSON."""
    return {
        "operators": [
            {"type": "matmul", "inputs": [0, 1], "outputs": [{"id": 2, "dims": [8, 4096], "dtype": "fp16"}]},
            {"type": "silu", "inputs": [2], "outputs": [{"id": 3, "dims": [8, 4096], "dtype": "fp16"}]}
        ],
        "inputs": [
            {"id": 0, "dims": [8, 4096], "dtype": "fp16"},
            {"id": 1, "dims": [4096, 4096], "dtype": "fp16"}
        ],
        "outputs": [
            {"id": 3, "dims": [8, 4096], "dtype": "fp16"}
        ]
    }


# =============================================================================
# MuGraphToMLIR Tests
# =============================================================================

class TestMuGraphToMLIR:
    """Tests for MuGraphToMLIR converter."""

    def test_converter_class_exists(self, mugraph_to_mlir_module):
        """Test MuGraphToMLIR class exists."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        assert hasattr(mugraph_to_mlir_module, "MuGraphToMLIR")

    def test_converter_creation(self, mugraph_to_mlir_module):
        """Test MuGraphToMLIR can be created."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        converter = MuGraphToMLIR()
        assert converter is not None

    def test_op_mapping_exists(self, mugraph_to_mlir_module):
        """Test OP_MAPPING is defined."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        assert hasattr(MuGraphToMLIR, "OP_MAPPING")
        assert len(MuGraphToMLIR.OP_MAPPING) > 0

    def test_dtype_mapping_exists(self, mugraph_to_mlir_module):
        """Test DTYPE_MAPPING is defined."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        assert hasattr(MuGraphToMLIR, "DTYPE_MAPPING")
        # Check common dtypes
        assert "fp16" in MuGraphToMLIR.DTYPE_MAPPING
        assert "fp32" in MuGraphToMLIR.DTYPE_MAPPING
        assert "bf16" in MuGraphToMLIR.DTYPE_MAPPING

    def test_convert_matmul_json(self, mugraph_to_mlir_module, sample_matmul_json):
        """Test converting matmul from JSON."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        converter = MuGraphToMLIR()
        mlir_text = converter._convert_json_data(sample_matmul_json)

        assert mlir_text is not None
        assert isinstance(mlir_text, str)
        assert "module" in mlir_text
        assert "func.func" in mlir_text
        assert "yirage.matmul" in mlir_text

    def test_convert_mlp_json(self, mugraph_to_mlir_module, sample_mlp_json):
        """Test converting MLP from JSON."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        converter = MuGraphToMLIR()
        mlir_text = converter._convert_json_data(sample_mlp_json)

        assert "yirage.matmul" in mlir_text
        assert "yirage.silu" in mlir_text

    def test_convert_from_json_file(self, mugraph_to_mlir_module, sample_matmul_json):
        """Test converting from JSON file."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_matmul_json, f)
            json_path = f.name

        try:
            converter = MuGraphToMLIR()
            mlir_text = converter.convert_from_json(json_path)

            assert mlir_text is not None
            assert "yirage.matmul" in mlir_text
        finally:
            Path(json_path).unlink(missing_ok=True)


# =============================================================================
# MLIR Type Generation Tests
# =============================================================================

class TestMLIRTypeGeneration:
    """Tests for MLIR type generation."""

    def test_tensor_type_2d(self, mugraph_to_mlir_module):
        """Test 2D tensor type generation."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        converter = MuGraphToMLIR()
        mlir_type = converter._get_mlir_type((32, 64), "fp32")

        assert mlir_type == "tensor<32x64xf32>"

    def test_tensor_type_3d(self, mugraph_to_mlir_module):
        """Test 3D tensor type generation."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        converter = MuGraphToMLIR()
        mlir_type = converter._get_mlir_type((8, 32, 64), "fp16")

        assert mlir_type == "tensor<8x32x64xf16>"

    def test_tensor_type_bf16(self, mugraph_to_mlir_module):
        """Test BF16 tensor type generation."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        converter = MuGraphToMLIR()
        mlir_type = converter._get_mlir_type((1024, 1024), "bf16")

        assert mlir_type == "tensor<1024x1024xbf16>"


# =============================================================================
# YirageCompiler Tests
# =============================================================================

class TestYirageCompiler:
    """Tests for YirageCompiler class."""

    def test_compiler_class_exists(self, yirage_compiler_module):
        """Test YirageCompiler class exists."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        assert hasattr(yirage_compiler_module, "YirageCompiler")

    def test_compiler_creation(self, yirage_compiler_module):
        """Test YirageCompiler can be created."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        YirageCompiler = getattr(yirage_compiler_module, "YirageCompiler", None)
        if YirageCompiler is None:
            pytest.skip("YirageCompiler not found")

        compiler = YirageCompiler(target='cpu')
        assert compiler is not None
        assert compiler.target == 'cpu'

    def test_compiler_targets(self, yirage_compiler_module):
        """Test available compiler targets."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        YirageCompiler = getattr(yirage_compiler_module, "YirageCompiler", None)
        if YirageCompiler is None:
            pytest.skip("YirageCompiler not found")

        assert hasattr(YirageCompiler, "TARGETS")
        targets = YirageCompiler.TARGETS

        # Check expected targets
        expected_targets = ['cuda', 'rocm', 'cpu', 'mps', 'ascend', 'tpu', 'fpga', 'gpu']
        for target in expected_targets:
            assert target in targets, f"Missing target: {target}"

    def test_compiler_opt_levels(self, yirage_compiler_module):
        """Test optimization levels."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        YirageCompiler = getattr(yirage_compiler_module, "YirageCompiler", None)
        if YirageCompiler is None:
            pytest.skip("YirageCompiler not found")

        for opt_level in [0, 1, 2, 3]:
            compiler = YirageCompiler(target='cpu', opt_level=opt_level)
            assert compiler.opt_level == opt_level

    def test_get_pipeline_info(self, yirage_compiler_module):
        """Test get_pipeline_info method."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        YirageCompiler = getattr(yirage_compiler_module, "YirageCompiler", None)
        if YirageCompiler is None:
            pytest.skip("YirageCompiler not found")

        compiler = YirageCompiler(target='cpu')
        info = compiler.get_pipeline_info()

        assert isinstance(info, dict)
        assert len(info) > 0


# =============================================================================
# CompiledKernel Tests
# =============================================================================

class TestCompiledKernel:
    """Tests for CompiledKernel class."""

    def test_compiled_kernel_class_exists(self, yirage_compiler_module):
        """Test CompiledKernel class exists."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        assert hasattr(yirage_compiler_module, "CompiledKernel")

    def test_compiled_kernel_creation(self, yirage_compiler_module):
        """Test CompiledKernel can be created."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        CompiledKernel = getattr(yirage_compiler_module, "CompiledKernel", None)
        if CompiledKernel is None:
            pytest.skip("CompiledKernel not found")

        kernel = CompiledKernel(
            mlir_source="module {}",
            lowered_mlir="module {}",
            target="cpu",
            entry_func="main",
            opt_level=2
        )

        assert kernel is not None
        assert kernel.target == "cpu"
        assert kernel.entry_func == "main"
        assert kernel.opt_level == 2

    def test_compiled_kernel_get_source(self, yirage_compiler_module):
        """Test get_source method."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        CompiledKernel = getattr(yirage_compiler_module, "CompiledKernel", None)
        if CompiledKernel is None:
            pytest.skip("CompiledKernel not found")

        kernel = CompiledKernel(
            mlir_source="module { func.func @test() {} }",
            lowered_mlir="module {}",
            target="cpu",
            entry_func="test",
            opt_level=2
        )

        source = kernel.get_source()
        assert "func.func @test()" in source

    def test_compiled_kernel_get_lowered(self, yirage_compiler_module):
        """Test get_lowered method."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        CompiledKernel = getattr(yirage_compiler_module, "CompiledKernel", None)
        if CompiledKernel is None:
            pytest.skip("CompiledKernel not found")

        kernel = CompiledKernel(
            mlir_source="module {}",
            lowered_mlir="module { llvm.func @lowered() {} }",
            target="cpu",
            entry_func="main",
            opt_level=2
        )

        lowered = kernel.get_lowered()
        assert "llvm.func @lowered()" in lowered

    def test_compiled_kernel_save(self, yirage_compiler_module):
        """Test save method."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        CompiledKernel = getattr(yirage_compiler_module, "CompiledKernel", None)
        if CompiledKernel is None:
            pytest.skip("CompiledKernel not found")

        kernel = CompiledKernel(
            mlir_source="module { func.func @test() {} }",
            lowered_mlir="module { llvm.func @test() {} }",
            target="cuda",
            entry_func="test",
            opt_level=3
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            output_path = f.name

        try:
            kernel.save(output_path)

            with open(output_path, 'r') as f:
                content = f.read()

            assert "Target: cuda" in content
            assert "Entry: test" in content
            assert "Opt Level: 3" in content
        finally:
            Path(output_path).unlink(missing_ok=True)


# =============================================================================
# MLIR Operation Tests
# =============================================================================

class TestMLIROperations:
    """Tests for MLIR operation mapping."""

    SUPPORTED_OPS = [
        ("kn_matmul_op", "yirage.matmul"),
        ("kn_silu_op", "yirage.silu"),
        ("kn_gelu_op", "yirage.gelu"),
        ("kn_relu_op", "yirage.relu"),
        ("kn_rms_norm_op", "yirage.rms_norm"),
        ("kn_add_op", "arith.addf"),
        ("kn_mul_op", "arith.mulf"),
        ("kn_div_op", "arith.divf"),
        ("kn_exp_op", "math.exp"),
        ("kn_sqrt_op", "math.sqrt"),
        ("kn_log_op", "math.log"),
    ]

    @pytest.mark.parametrize("kn_op,mlir_op", SUPPORTED_OPS)
    def test_operation_mapping(self, mugraph_to_mlir_module, kn_op: str, mlir_op: str):
        """Test operation mapping is correct."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        MuGraphToMLIR = getattr(mugraph_to_mlir_module, "MuGraphToMLIR", None)
        if MuGraphToMLIR is None:
            pytest.skip("MuGraphToMLIR not found")

        assert kn_op in MuGraphToMLIR.OP_MAPPING
        assert MuGraphToMLIR.OP_MAPPING[kn_op] == mlir_op


# =============================================================================
# MLIR File Tests
# =============================================================================

class TestMLIRFiles:
    """Tests for existing MLIR test files."""

    MLIR_TEST_FILES = [
        "simple_matmul.mlir",
        "llama_mlp.mlir",
        "llm_basic_ops.mlir",
        "llm_complete_ops.mlir",
    ]

    @pytest.mark.parametrize("mlir_file", MLIR_TEST_FILES)
    def test_mlir_file_exists(self, mlir_file: str):
        """Test MLIR test files exist."""
        mlir_path = PROJECT_ROOT / "mlir" / "test" / mlir_file
        assert mlir_path.exists(), f"MLIR test file not found: {mlir_file}"

    @pytest.mark.parametrize("mlir_file", MLIR_TEST_FILES)
    def test_mlir_file_syntax(self, mlir_file: str):
        """Test MLIR test files have valid syntax structure."""
        mlir_path = PROJECT_ROOT / "mlir" / "test" / mlir_file
        if not mlir_path.exists():
            pytest.skip(f"MLIR file not found: {mlir_file}")

        content = mlir_path.read_text()

        # Basic syntax checks
        # Should have func.func or module
        has_func = "func.func" in content or "func @" in content
        has_module = "module" in content

        assert has_func or has_module, f"{mlir_file} missing function or module"


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_convert_mugraph_to_mlir_function(self, mugraph_to_mlir_module):
        """Test convert_mugraph_to_mlir function."""
        if mugraph_to_mlir_module is None:
            pytest.skip("MuGraph to MLIR module not available")

        convert_fn = getattr(mugraph_to_mlir_module, "convert_mugraph_to_mlir", None)
        if convert_fn is None:
            pytest.skip("convert_mugraph_to_mlir not found")

        # Create temp JSON file
        graph_data = {
            "operators": [{"type": "matmul", "inputs": [0, 1], "outputs": [{"id": 2}]}],
            "inputs": [{"id": 0, "dims": [32, 64]}, {"id": 1, "dims": [64, 128]}],
            "outputs": [{"id": 2, "dims": [32, 128]}]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_data, f)
            json_path = f.name

        try:
            mlir_text = convert_fn(json_path)
            assert mlir_text is not None
            assert "module" in mlir_text
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_compile_graph_function(self, yirage_compiler_module):
        """Test compile_graph function."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        compile_fn = getattr(yirage_compiler_module, "compile_graph", None)
        if compile_fn is None:
            pytest.skip("compile_graph not found")

        # Function should exist and be callable
        assert callable(compile_fn)

    def test_compile_mlir_function(self, yirage_compiler_module):
        """Test compile_mlir function."""
        if yirage_compiler_module is None:
            pytest.skip("YiRage compiler module not available")

        compile_fn = getattr(yirage_compiler_module, "compile_mlir", None)
        if compile_fn is None:
            pytest.skip("compile_mlir not found")

        # Function should exist and be callable
        assert callable(compile_fn)
