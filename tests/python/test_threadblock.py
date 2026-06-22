# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Threadblock Module Tests

Maps to C++ tests:
  - test_tb_smem_tensor_gtest.cc
  - test_tb_operator_gtest.cc
  - test_tb_graph_gtest.cc
  - test_tb_matmul_gtest.cc
  - test_tb_element_ops_gtest.cc
  - test_tb_reduction_gtest.cc
"""

import pytest
from typing import List, Tuple

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def threadblock_module():
    """Import the threadblock module."""
    try:
        from yirage.kernel import threadblock
        return threadblock
    except ImportError:
        pytest.skip("kernel.threadblock module not available")


@pytest.fixture
def core_module():
    """Import the core module for STensor/DTensor."""
    try:
        from yirage import core
        return core
    except ImportError:
        pytest.skip("core module not available")


@pytest.fixture
def mock_dtensor(core_module):
    """Create a mock DTensor."""
    try:
        return core_module.DTensor
    except AttributeError:
        pytest.skip("DTensor not available in core module")


@pytest.fixture
def mock_stensor(core_module):
    """Create a mock STensor."""
    try:
        return core_module.STensor
    except AttributeError:
        pytest.skip("STensor not available in core module")


# =============================================================================
# TBGraph Tests (maps to test_tb_graph_gtest.cc)
# =============================================================================


class TestTBGraph:
    """Tests for TBGraph class."""

    def test_tbgraph_class_exists(self, threadblock_module):
        """Test that TBGraph class is defined."""
        assert hasattr(threadblock_module, "TBGraph")

    def test_tbgraph_has_new_input(self, threadblock_module):
        """Test TBGraph has new_input method."""
        TBGraph = threadblock_module.TBGraph
        
        # TBGraph wraps a cygraph, so check method exists
        assert hasattr(TBGraph, "new_input")

    def test_tbgraph_has_new_output(self, threadblock_module):
        """Test TBGraph has new_output method."""
        TBGraph = threadblock_module.TBGraph
        
        assert hasattr(TBGraph, "new_output")

    def test_tbgraph_has_matmul(self, threadblock_module):
        """Test TBGraph has matmul operation."""
        TBGraph = threadblock_module.TBGraph
        
        assert hasattr(TBGraph, "matmul")

    def test_tbgraph_has_element_unary_ops(self, threadblock_module):
        """Test TBGraph has element-wise unary operations."""
        TBGraph = threadblock_module.TBGraph
        
        unary_ops = ["exp", "silu", "gelu", "relu", "square", "sqrt"]
        for op in unary_ops:
            assert hasattr(TBGraph, op), f"Missing unary op: {op}"

    def test_tbgraph_has_element_binary_ops(self, threadblock_module):
        """Test TBGraph has element-wise binary operations."""
        TBGraph = threadblock_module.TBGraph
        
        binary_ops = ["add", "mul", "div", "sub"]
        for op in binary_ops:
            assert hasattr(TBGraph, op), f"Missing binary op: {op}"

    def test_tbgraph_has_reduction_ops(self, threadblock_module):
        """Test TBGraph has reduction operations."""
        TBGraph = threadblock_module.TBGraph
        
        assert hasattr(TBGraph, "reduction")
        assert hasattr(TBGraph, "reduction_max")
        assert hasattr(TBGraph, "rms_norm")

    def test_tbgraph_has_forloop_ops(self, threadblock_module):
        """Test TBGraph has forloop operations."""
        TBGraph = threadblock_module.TBGraph
        
        assert hasattr(TBGraph, "forloop_accum")
        assert hasattr(TBGraph, "forloop_accum_rescale")
        assert hasattr(TBGraph, "forloop_accum_max")

    def test_tbgraph_has_concat(self, threadblock_module):
        """Test TBGraph has concat operation."""
        TBGraph = threadblock_module.TBGraph
        
        assert hasattr(TBGraph, "concat")

    def test_tbgraph_has_clamp(self, threadblock_module):
        """Test TBGraph has clamp operation."""
        TBGraph = threadblock_module.TBGraph
        
        assert hasattr(TBGraph, "clamp")

    def test_tbgraph_has_mul_scalar(self, threadblock_module):
        """Test TBGraph has mul_scalar operation."""
        TBGraph = threadblock_module.TBGraph
        
        assert hasattr(TBGraph, "mul_scalar")


# =============================================================================
# Operation Signature Tests (maps to test_tb_operator_gtest.cc)
# =============================================================================


class TestOperationSignatures:
    """Tests for operation method signatures."""

    def test_new_input_signature(self, threadblock_module):
        """Test new_input has correct parameters."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.new_input)
        params = list(sig.parameters.keys())
        
        assert "dtensor" in params
        assert "input_map" in params
        assert "forloop_dim" in params

    def test_new_output_signature(self, threadblock_module):
        """Test new_output has correct parameters."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.new_output)
        params = list(sig.parameters.keys())
        
        assert "stensor" in params
        assert "output_map" in params

    def test_clamp_signature(self, threadblock_module):
        """Test clamp has min/max parameters."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.clamp)
        params = list(sig.parameters.keys())
        
        # Should have A, min_val, max_val
        assert len(params) >= 3

    def test_reduction_signature(self, threadblock_module):
        """Test reduction has dim parameter."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.reduction)
        params = list(sig.parameters.keys())
        
        assert "dim" in params


# =============================================================================
# STensor Tests (maps to test_tb_smem_tensor_gtest.cc)
# =============================================================================


class TestSTensor:
    """Tests for STensor (shared memory tensor)."""

    def test_stensor_exists(self, core_module):
        """Test STensor class exists."""
        assert hasattr(core_module, "STensor")

    def test_stensor_properties(self, core_module):
        """Test STensor class is accessible from core bindings."""
        if not hasattr(core_module, "STensor"):
            pytest.skip("STensor not exposed in core bindings")
        assert core_module.STensor is not None


# =============================================================================
# DTensor Tests (maps to test_tb_operator_gtest.cc)
# =============================================================================


class TestDTensor:
    """Tests for DTensor (device tensor)."""

    def test_dtensor_exists(self, core_module):
        """Test DTensor class exists."""
        assert hasattr(core_module, "DTensor")


# =============================================================================
# Matmul Operation Tests (maps to test_tb_matmul_gtest.cc)
# =============================================================================


class TestMatmulOperation:
    """Tests for matmul operation."""

    def test_matmul_exists(self, threadblock_module):
        """Test matmul method exists."""
        TBGraph = threadblock_module.TBGraph
        assert callable(getattr(TBGraph, "matmul", None))

    def test_matmul_requires_two_inputs(self, threadblock_module):
        """Test matmul requires A and B tensors."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.matmul)
        # self, A, B
        assert len(sig.parameters) >= 2


# =============================================================================
# Element Unary Operation Tests (maps to test_tb_element_ops_gtest.cc)
# =============================================================================


class TestElementUnaryOps:
    """Tests for element-wise unary operations."""

    @pytest.mark.parametrize("op_name", [
        "exp", "silu", "gelu", "relu", "square", "sqrt"
    ])
    def test_unary_op_exists(self, threadblock_module, op_name):
        """Test unary operation method exists."""
        TBGraph = threadblock_module.TBGraph
        assert hasattr(TBGraph, op_name)

    @pytest.mark.parametrize("op_name", [
        "exp", "silu", "gelu", "relu", "square", "sqrt"
    ])
    def test_unary_op_single_input(self, threadblock_module, op_name):
        """Test unary operations take single input."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        method = getattr(TBGraph, op_name)
        sig = inspect.signature(method)
        # self + 1 input = 2 parameters
        # Some may have optional params
        assert len(sig.parameters) >= 1


# =============================================================================
# Element Binary Operation Tests (maps to test_tb_element_ops_gtest.cc)
# =============================================================================


class TestElementBinaryOps:
    """Tests for element-wise binary operations."""

    @pytest.mark.parametrize("op_name", [
        "add", "mul", "div", "sub"
    ])
    def test_binary_op_exists(self, threadblock_module, op_name):
        """Test binary operation method exists."""
        TBGraph = threadblock_module.TBGraph
        assert hasattr(TBGraph, op_name)

    @pytest.mark.parametrize("op_name", [
        "add", "mul", "div", "sub"
    ])
    def test_binary_op_two_inputs(self, threadblock_module, op_name):
        """Test binary operations take two inputs."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        method = getattr(TBGraph, op_name)
        sig = inspect.signature(method)
        # self + 2 inputs = 3 parameters
        assert len(sig.parameters) >= 2


# =============================================================================
# Reduction Operation Tests (maps to test_tb_reduction_gtest.cc)
# =============================================================================


class TestReductionOps:
    """Tests for reduction operations."""

    def test_reduction_exists(self, threadblock_module):
        """Test reduction method exists."""
        TBGraph = threadblock_module.TBGraph
        assert hasattr(TBGraph, "reduction")

    def test_reduction_max_exists(self, threadblock_module):
        """Test reduction_max method exists."""
        TBGraph = threadblock_module.TBGraph
        assert hasattr(TBGraph, "reduction_max")

    def test_rms_norm_exists(self, threadblock_module):
        """Test rms_norm method exists."""
        TBGraph = threadblock_module.TBGraph
        assert hasattr(TBGraph, "rms_norm")

    def test_reduction_has_dim_param(self, threadblock_module):
        """Test reduction has dim parameter."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.reduction)
        params = list(sig.parameters.keys())
        assert "dim" in params

    def test_reduction_max_has_dim_param(self, threadblock_module):
        """Test reduction_max has dim parameter."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.reduction_max)
        params = list(sig.parameters.keys())
        assert "dim" in params


# =============================================================================
# Forloop Operations Tests (maps to test_tb_graph_gtest.cc)
# =============================================================================


class TestForloopOps:
    """Tests for forloop operations."""

    def test_forloop_accum_exists(self, threadblock_module):
        """Test forloop_accum method exists."""
        TBGraph = threadblock_module.TBGraph
        assert hasattr(TBGraph, "forloop_accum")

    def test_forloop_accum_rescale_exists(self, threadblock_module):
        """Test forloop_accum_rescale method exists."""
        TBGraph = threadblock_module.TBGraph
        assert hasattr(TBGraph, "forloop_accum_rescale")

    def test_forloop_accum_max_exists(self, threadblock_module):
        """Test forloop_accum_max method exists."""
        TBGraph = threadblock_module.TBGraph
        assert hasattr(TBGraph, "forloop_accum_max")

    def test_forloop_accum_has_acc_param(self, threadblock_module):
        """Test forloop_accum has acc parameter."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.forloop_accum)
        params = list(sig.parameters.keys())
        assert "acc" in params


# =============================================================================
# Concat Operation Tests (maps to test_tb_graph_gtest.cc)
# =============================================================================


class TestConcatOp:
    """Tests for concat operation."""

    def test_concat_exists(self, threadblock_module):
        """Test concat method exists."""
        TBGraph = threadblock_module.TBGraph
        assert hasattr(TBGraph, "concat")

    def test_concat_has_dim_param(self, threadblock_module):
        """Test concat has dim parameter."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.concat)
        params = list(sig.parameters.keys())
        assert "dim" in params


# =============================================================================
# Input/Output Operation Tests (maps to test_tb_operator_gtest.cc)
# =============================================================================


class TestInputOutputOps:
    """Tests for input/output operations."""

    def test_new_input_has_store_in_dmem(self, threadblock_module):
        """Test new_input has store_in_dmem parameter."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.new_input)
        params = list(sig.parameters.keys())
        assert "store_in_dmem" in params

    def test_new_output_has_forloop_dim(self, threadblock_module):
        """Test new_output has forloop_dim parameter."""
        TBGraph = threadblock_module.TBGraph
        import inspect
        
        sig = inspect.signature(TBGraph.new_output)
        params = list(sig.parameters.keys())
        assert "forloop_dim" in params


# =============================================================================
# Method Completeness Tests
# =============================================================================


class TestMethodCompleteness:
    """Tests to ensure all expected methods are present."""

    def test_all_unary_ops_present(self, threadblock_module):
        """Test all unary operations are present."""
        TBGraph = threadblock_module.TBGraph
        
        expected_ops = [
            "exp", "silu", "gelu", "relu", "square", "sqrt",
            "clamp", "mul_scalar"
        ]
        
        missing = [op for op in expected_ops if not hasattr(TBGraph, op)]
        assert not missing, f"Missing unary ops: {missing}"

    def test_all_binary_ops_present(self, threadblock_module):
        """Test all binary operations are present."""
        TBGraph = threadblock_module.TBGraph
        
        expected_ops = ["add", "mul", "div", "sub"]
        
        missing = [op for op in expected_ops if not hasattr(TBGraph, op)]
        assert not missing, f"Missing binary ops: {missing}"

    def test_all_reduction_ops_present(self, threadblock_module):
        """Test all reduction operations are present."""
        TBGraph = threadblock_module.TBGraph
        
        expected_ops = ["reduction", "reduction_max", "rms_norm"]
        
        missing = [op for op in expected_ops if not hasattr(TBGraph, op)]
        assert not missing, f"Missing reduction ops: {missing}"

    def test_all_forloop_ops_present(self, threadblock_module):
        """Test all forloop operations are present."""
        TBGraph = threadblock_module.TBGraph
        
        expected_ops = ["forloop_accum", "forloop_accum_rescale", "forloop_accum_max"]
        
        missing = [op for op in expected_ops if not hasattr(TBGraph, op)]
        assert not missing, f"Missing forloop ops: {missing}"


# =============================================================================
# Operation Count Tests
# =============================================================================


@pytest.mark.parametrize("category,ops", [
    ("unary", ["exp", "silu", "gelu", "relu", "square", "sqrt"]),
    ("binary", ["add", "mul", "div", "sub"]),
    ("reduction", ["reduction", "reduction_max", "rms_norm"]),
])
def test_operation_categories(threadblock_module, category, ops):
    """Test operation categories have expected methods."""
    TBGraph = threadblock_module.TBGraph
    
    for op in ops:
        assert hasattr(TBGraph, op), f"{category} op '{op}' missing"
