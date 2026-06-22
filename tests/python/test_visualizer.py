# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for yirage.utils.visualizer module.

Tests data structures, op-label mappings, and helper functions.
The module requires graphviz, so tests skip gracefully when unavailable.
"""

import importlib.util
from pathlib import Path

import pytest

_PYTHON_ROOT = Path(__file__).parent.parent.parent / "python"


def _load_visualizer():
    """Load visualizer module directly."""
    path = _PYTHON_ROOT / "yirage" / "utils" / "visualizer.py"
    spec = importlib.util.spec_from_file_location("yirage_visualizer_test", path)
    if spec is None or spec.loader is None:
        pytest.skip("visualizer.py not found")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        pytest.skip("graphviz not installed")
    return mod


@pytest.fixture(scope="module")
def viz():
    """Load and return the visualizer module."""
    return _load_visualizer()


# =============================================================================
# Color Map Tests
# =============================================================================


class TestColorsMap:
    """Test color map definitions."""

    def test_colors_map_has_kernel(self, viz):
        """Test kernel colors are defined."""
        assert "kernel" in viz.colors_map
        assert "node" in viz.colors_map["kernel"]
        assert "bg" in viz.colors_map["kernel"]
        assert "edge" in viz.colors_map["kernel"]

    def test_colors_map_has_block(self, viz):
        """Test block colors are defined."""
        assert "block" in viz.colors_map
        assert "node" in viz.colors_map["block"]
        assert "bg" in viz.colors_map["block"]

    def test_colors_map_has_thread(self, viz):
        """Test thread colors are defined."""
        assert "thread" in viz.colors_map


# =============================================================================
# Op-Label Mapping Tests
# =============================================================================


class TestOpNodeLabelMapping:
    """Test operator-to-label mapping completeness."""

    def test_mapping_exists(self, viz):
        """Test mapping dict is defined."""
        assert isinstance(viz.op_nodelabel_mapping, dict)
        assert len(viz.op_nodelabel_mapping) > 0

    def test_kernel_ops_present(self, viz):
        """Test kernel-level ops are in the mapping."""
        kernel_ops = [
            "kn_input_op",
            "kn_output_op",
            "kn_matmul_op",
            "kn_add_op",
            "kn_mul_op",
            "kn_exp_op",
            "kn_silu_op",
            "kn_relu_op",
            "kn_gelu_op",
            "kn_rms_norm_op",
            "kn_allreduce_op",
        ]
        for op in kernel_ops:
            assert op in viz.op_nodelabel_mapping, f"Missing kernel op: {op}"

    def test_block_ops_present(self, viz):
        """Test block-level (threadblock) ops are in the mapping."""
        block_ops = [
            "tb_input_op",
            "tb_output_op",
            "tb_matmul_op",
            "tb_add_op",
            "tb_mul_op",
            "tb_exp_op",
            "tb_silu_op",
            "tb_relu_op",
            "tb_gelu_op",
            "tb_rms_norm_op",
        ]
        for op in block_ops:
            assert op in viz.op_nodelabel_mapping, f"Missing block op: {op}"

    def test_all_labels_are_strings(self, viz):
        """Test all label values are non-empty strings."""
        for key, label in viz.op_nodelabel_mapping.items():
            assert isinstance(label, str), f"{key}: label is not str"
            assert len(label) > 0, f"{key}: label is empty"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestLetterSequence:
    """Test letter_sequence generator."""

    def test_generates_uppercase(self, viz):
        """Test generator yields uppercase letters."""
        gen = viz.letter_sequence()
        first = next(gen)
        assert first == "A"

    def test_sequential_letters(self, viz):
        """Test first 26 letters are A-Z."""
        gen = viz.letter_sequence()
        letters = [next(gen) for _ in range(26)]
        assert letters == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_wraps_around(self, viz):
        """Test generator wraps around after Z."""
        gen = viz.letter_sequence()
        for _ in range(26):
            next(gen)
        assert next(gen) == "A"  # 27th should wrap


class TestIsGraphData:
    """Test is_graph_data helper."""

    def test_valid_graph_data(self, viz):
        """Test recognizes valid graph data dict."""
        assert viz.is_graph_data({"op_type": "matmul"})

    def test_dict_without_op_type(self, viz):
        """Test rejects dict without op_type."""
        assert not viz.is_graph_data({"key": "value"})

    def test_non_dict(self, viz):
        """Test rejects non-dict inputs."""
        assert not viz.is_graph_data([1, 2, 3])
        assert not viz.is_graph_data("string")
        assert not viz.is_graph_data(42)
        assert not viz.is_graph_data(None)


class TestGetFormatStr:
    """Test get_format_str helper."""

    def test_positive_forloop_dim(self, viz):
        """Test format string with positive forloop_dim."""
        result = viz.get_format_str({"forloop_dim": 2})
        assert "2" in result
        assert "fmap" in result

    def test_negative_forloop_dim(self, viz):
        """Test format string with negative forloop_dim uses phi symbol."""
        result = viz.get_format_str({"forloop_dim": -1})
        assert "\u2205" in result  # φ symbol
        assert "fmap" in result


# =============================================================================
# Node Class Tests
# =============================================================================


class TestNode:
    """Test node base class."""

    def test_node_creation(self, viz):
        """Test creating a basic node."""
        n = viz.node("test_node", "kn_matmul_op", 1, "MatMul", "#ffffff")
        assert n.name == "test_node"
        assert n.op_type == "kn_matmul_op"
        assert n.id == 1
        assert n.label == "MatMul"

    def test_is_customized_node(self, viz):
        """Test is_customized_node detection."""
        n = viz.node("n", "kn_customized_op", 1, "Custom", "#fff")
        assert n.is_customized_node()

    def test_is_not_customized_node(self, viz):
        """Test non-customized node."""
        n = viz.node("n", "kn_matmul_op", 1, "MatMul", "#fff")
        assert not n.is_customized_node()

    def test_is_input_node(self, viz):
        """Test is_input_node detection."""
        n = viz.node("n", "kn_input_op", 1, "Input", "#fff")
        assert n.is_input_node()

    def test_is_output_node(self, viz):
        """Test is_output_node detection."""
        n = viz.node("n", "kn_output_op", 1, "Output", "#fff")
        assert n.is_output_node()

    def test_input_output_tensors_empty(self, viz):
        """Test tensors lists start empty."""
        n = viz.node("n", "kn_matmul_op", 1, "MatMul", "#fff")
        assert n.input_tensors == []
        assert n.output_tensors == []


class TestKernelNode:
    """Test kernel_node subclass."""

    def test_creation(self, viz):
        """Test creating a kernel node."""
        kn = viz.kernel_node("kn1", "kn_matmul_op", 1, "MatMul")
        assert kn.color == viz.colors_map["kernel"]["node"]

    def test_is_kernel_output_node_with_no_outputs(self, viz):
        """Test is_kernel_output_node when no outputs exist."""
        kn = viz.kernel_node("kn1", "kn_matmul_op", 1, "MatMul")
        assert kn.is_kernel_output_node()

    def test_is_not_kernel_output_node_with_outputs(self, viz):
        """Test is_kernel_output_node when outputs exist."""
        kn = viz.kernel_node("kn1", "kn_matmul_op", 1, "MatMul")
        kn.output_tensors.append("dummy")
        assert not kn.is_kernel_output_node()


class TestBlockNode:
    """Test block_node subclass."""

    def test_input_node_creation(self, viz):
        """Test creating a block input node."""
        bn = viz.block_node(
            "bn1",
            "tb_input_op",
            1,
            "Input",
            iomap={"x": 0, "y": 1},
            forloop_dim=2,
            forloop_range=8,
        )
        assert bn.color == viz.colors_map["block"]["node"]
        assert "imap" in bn.iomap_str
        assert bn.forloop_dim == 2

    def test_output_node_creation(self, viz):
        """Test creating a block output node."""
        bn = viz.block_node(
            "bn2",
            "tb_output_op",
            2,
            "Output",
            iomap={"x": 0, "y": -1},
        )
        assert "omap" in bn.iomap_str
        assert "\u2205" in bn.iomap_str  # -1 maps to phi

    def test_regular_node_creation(self, viz):
        """Test creating a regular block node (not input/output)."""
        bn = viz.block_node("bn3", "tb_matmul_op", 3, "MatMul")
        assert bn.name == "bn3"
        assert bn.label == "MatMul"

    def test_formap_str_positive_dim(self, viz):
        """Test formap string with positive forloop_dim."""
        bn = viz.block_node(
            "bn",
            "tb_input_op",
            1,
            "Input",
            iomap={"x": 0},
            forloop_dim=1,
            forloop_range=4,
        )
        assert "1" in bn.formap_str
        assert "fmap" in bn.formap_str

    def test_formap_str_negative_dim(self, viz):
        """Test formap string with negative forloop_dim."""
        bn = viz.block_node(
            "bn",
            "tb_input_op",
            1,
            "Input",
            iomap={"x": 0},
            forloop_dim=-1,
            forloop_range=4,
        )
        assert "\u2205" in bn.formap_str


# =============================================================================
# Kernel Graph Class Tests
# =============================================================================


class TestKernelGraph:
    """Test kernel_graph class structure."""

    def test_creation(self, viz):
        """Test creating a kernel graph."""
        kg = viz.kernel_graph("Test Graph")
        assert kg.label == "Test Graph"
        assert kg.nodes == []
        assert kg.tensors == []
        assert kg.block_graphs == []

    def test_bg_color(self, viz):
        """Test background color is set from colors_map."""
        kg = viz.kernel_graph("Test")
        assert kg.bg_color == viz.colors_map["kernel"]["bg"]


class TestBlockGraph:
    """Test block_graph class structure."""

    def test_creation(self, viz):
        """Test creating a block graph."""
        kg = viz.kernel_graph("Parent")
        grid_dim = {"x": 4, "y": 1, "z": 1}
        bg = viz.block_graph("Block 1", grid_dim, 8, kg)
        assert bg.label == "Block 1"
        assert bg.grid_dim == grid_dim
        assert bg.forloop_range == 8

    def test_get_grid_size_and_forloop(self, viz):
        """Test grid size and forloop formatting."""
        kg = viz.kernel_graph("Parent")
        bg = viz.block_graph(
            "Block",
            {"x": 4, "y": 2, "z": 1},
            16,
            kg,
        )
        result = bg.get_grid_size_and_forloop()
        assert "grid size" in result
        assert "forloop" in result
        assert "16" in result


# =============================================================================
# Visualizer Class Tests
# =============================================================================


class TestVisualizerClass:
    """Test the top-level visualizer class."""

    def test_creation(self, viz, tmp_path):
        """Test creating a visualizer instance."""
        output = str(tmp_path / "test_graph")
        v = viz.visualizer(output)
        assert v.output_filename == output
        assert v.G is not None
        assert v.new_kernel_graph is not None
