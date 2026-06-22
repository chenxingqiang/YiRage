#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Custom Operator Registry Unit Tests

Tests for yirage/kernel/op_registry.py:
  - OpRegistry CRUD operations
  - register_op() / custom_op() public API
  - call_op() on KNGraph (mocked and real paths)
  - Arity checking
  - Duplicate-registration warning/overwrite
  - Introspection helpers (list_ops, get_op)
  - Isolation via fresh OpRegistry instances

Run with:  pytest tests/python/test_op_registry.py -v
"""

import sys
import os
import importlib.util
import warnings
from pathlib import Path
from typing import List
import pytest

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))


# ---------------------------------------------------------------------------
# Import op_registry directly from its file so we don't trigger
# yirage.kernel.__init__ (which needs torch / C++ core).
# ---------------------------------------------------------------------------
_REGISTRY_PATH = PROJECT_ROOT / "python" / "yirage" / "kernel" / "op_registry.py"
_spec = importlib.util.spec_from_file_location("yirage.kernel.op_registry", _REGISTRY_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("yirage.kernel.op_registry", _mod)
_spec.loader.exec_module(_mod)

CustomOpSpec = _mod.CustomOpSpec
OpRegistry = _mod.OpRegistry
global_registry = _mod.global_registry
register_op = _mod.register_op
custom_op = _mod.custom_op
list_ops = _mod.list_ops
get_op = _mod.get_op


# ===========================================================================
# Helpers
# ===========================================================================

def _noop_builder(kgraph, inputs, **kwargs):
    """Minimal builder that just returns inputs unchanged (for pure registry tests)."""
    return inputs


def _counting_builder(counter: list):
    """Return a builder that appends its call signature to *counter*."""
    def _builder(kgraph, inputs, **kwargs):
        counter.append({"kgraph": kgraph, "inputs": inputs, "kwargs": kwargs})
        return inputs
    return _builder


# ===========================================================================
# CustomOpSpec
# ===========================================================================

class TestCustomOpSpec:
    """Unit tests for CustomOpSpec data class and __call__."""

    def test_basic_construction(self):
        spec = CustomOpSpec(name="my_op", builder=_noop_builder)
        assert spec.name == "my_op"
        assert spec.builder is _noop_builder
        assert spec.n_inputs == -1
        assert spec.description == ""
        assert spec.tags == []

    def test_construction_with_all_fields(self):
        spec = CustomOpSpec(
            name="op",
            builder=_noop_builder,
            n_inputs=2,
            description="test op",
            tags=["a", "b"],
        )
        assert spec.n_inputs == 2
        assert spec.description == "test op"
        assert spec.tags == ["a", "b"]

    def test_call_returns_list(self):
        spec = CustomOpSpec(name="op", builder=_noop_builder)
        result = spec("kgraph", ["t1", "t2"])
        assert isinstance(result, list)
        assert result == ["t1", "t2"]

    def test_call_wraps_single_return_in_list(self):
        def _single(kgraph, inputs, **kwargs):
            return "single_tensor"

        spec = CustomOpSpec(name="op", builder=_single)
        result = spec("kgraph", [])
        assert result == ["single_tensor"]

    def test_call_handles_none_return(self):
        def _none(kgraph, inputs, **kwargs):
            return None

        spec = CustomOpSpec(name="op", builder=_none)
        result = spec("kgraph", [])
        assert result == []

    def test_call_arity_check_passes(self):
        spec = CustomOpSpec(name="op", builder=_noop_builder, n_inputs=2)
        result = spec("kg", ["a", "b"])
        assert result == ["a", "b"]

    def test_call_arity_check_fails(self):
        spec = CustomOpSpec(name="op", builder=_noop_builder, n_inputs=2)
        with pytest.raises(ValueError, match="expects 2 input"):
            spec("kg", ["a"])  # only 1 input

    def test_call_arity_minus_one_unchecked(self):
        spec = CustomOpSpec(name="op", builder=_noop_builder, n_inputs=-1)
        # Should not raise even with 0 or 5 inputs
        spec("kg", [])
        spec("kg", ["a", "b", "c", "d", "e"])

    def test_call_forwards_kwargs(self):
        received = {}

        def _capturing(kgraph, inputs, **kwargs):
            received.update(kwargs)
            return inputs

        spec = CustomOpSpec(name="op", builder=_capturing)
        spec("kg", [], grid_dim=(1, 1, 1), reduction_dimx=64)
        assert received["grid_dim"] == (1, 1, 1)
        assert received["reduction_dimx"] == 64


# ===========================================================================
# OpRegistry
# ===========================================================================

class TestOpRegistry:
    """Unit tests for OpRegistry class."""

    def setup_method(self):
        """Use a fresh registry for each test to ensure isolation."""
        self.reg = OpRegistry()

    # ------------------------------------------------------------------
    # register()
    # ------------------------------------------------------------------

    def test_register_basic(self):
        spec = self.reg.register("op1", _noop_builder)
        assert isinstance(spec, CustomOpSpec)
        assert spec.name == "op1"

    def test_register_returns_spec(self):
        spec = self.reg.register("op1", _noop_builder, n_inputs=3, description="d")
        assert spec.n_inputs == 3
        assert spec.description == "d"

    def test_register_with_tags(self):
        spec = self.reg.register("op1", _noop_builder, tags=["fused", "attention"])
        assert "fused" in spec.tags

    def test_register_duplicate_warns(self):
        self.reg.register("op1", _noop_builder)
        with pytest.warns(UserWarning, match="already registered"):
            self.reg.register("op1", _noop_builder)

    def test_register_duplicate_keeps_original(self):
        def _v1(kgraph, inputs, **kwargs):
            return ["v1"]

        def _v2(kgraph, inputs, **kwargs):
            return ["v2"]

        self.reg.register("op1", _v1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.reg.register("op1", _v2)  # duplicate, no overwrite

        result = self.reg.get("op1")("kg", [])
        assert result == ["v1"]  # original preserved

    def test_register_duplicate_overwrite(self):
        def _v1(kgraph, inputs, **kwargs):
            return ["v1"]

        def _v2(kgraph, inputs, **kwargs):
            return ["v2"]

        self.reg.register("op1", _v1)
        self.reg.register("op1", _v2, overwrite=True)

        result = self.reg.get("op1")("kg", [])
        assert result == ["v2"]

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self.reg.register("", _noop_builder)

    def test_register_non_callable_raises(self):
        with pytest.raises(TypeError, match="callable"):
            self.reg.register("op1", "not_a_function")

    # ------------------------------------------------------------------
    # get()
    # ------------------------------------------------------------------

    def test_get_registered(self):
        self.reg.register("op1", _noop_builder)
        spec = self.reg.get("op1")
        assert spec.name == "op1"

    def test_get_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="op_xyz"):
            self.reg.get("op_xyz")

    def test_get_key_error_lists_registered(self):
        self.reg.register("op1", _noop_builder)
        with pytest.raises(KeyError, match="op1"):
            self.reg.get("unknown")

    # ------------------------------------------------------------------
    # __contains__
    # ------------------------------------------------------------------

    def test_contains_true(self):
        self.reg.register("op1", _noop_builder)
        assert "op1" in self.reg

    def test_contains_false(self):
        assert "op_missing" not in self.reg

    # ------------------------------------------------------------------
    # list() / names() / len()
    # ------------------------------------------------------------------

    def test_list_empty(self):
        assert self.reg.list() == []

    def test_list_returns_specs(self):
        self.reg.register("a", _noop_builder)
        self.reg.register("b", _noop_builder)
        names = [s.name for s in self.reg.list()]
        assert set(names) == {"a", "b"}

    def test_names(self):
        self.reg.register("x", _noop_builder)
        self.reg.register("y", _noop_builder)
        assert set(self.reg.names()) == {"x", "y"}

    def test_len(self):
        assert len(self.reg) == 0
        self.reg.register("a", _noop_builder)
        assert len(self.reg) == 1
        self.reg.register("b", _noop_builder)
        assert len(self.reg) == 2

    # ------------------------------------------------------------------
    # unregister() / clear()
    # ------------------------------------------------------------------

    def test_unregister(self):
        self.reg.register("op1", _noop_builder)
        self.reg.unregister("op1")
        assert "op1" not in self.reg

    def test_unregister_missing_raises(self):
        with pytest.raises(KeyError):
            self.reg.unregister("not_there")

    def test_clear(self):
        self.reg.register("a", _noop_builder)
        self.reg.register("b", _noop_builder)
        self.reg.clear()
        assert len(self.reg) == 0

    # ------------------------------------------------------------------
    # decorator()
    # ------------------------------------------------------------------

    def test_decorator_registers(self):
        @self.reg.decorator("my_op", n_inputs=1, description="doc")
        def _builder(kgraph, inputs, **kwargs):
            return inputs

        assert "my_op" in self.reg
        assert self.reg.get("my_op").n_inputs == 1

    def test_decorator_preserves_function(self):
        @self.reg.decorator("my_op2")
        def _builder(kgraph, inputs, **kwargs):
            """Original docstring."""
            return inputs

        # The decorator must return the original callable unmodified
        assert callable(_builder)
        assert _builder.__doc__ == "Original docstring."

    def test_decorator_uses_docstring_as_description(self):
        @self.reg.decorator("doc_op")
        def _builder(kgraph, inputs, **kwargs):
            """This is the description."""
            return inputs

        spec = self.reg.get("doc_op")
        assert "description" in spec.description

    def test_decorator_explicit_description_wins(self):
        @self.reg.decorator("desc_op", description="explicit")
        def _builder(kgraph, inputs, **kwargs):
            """docstring."""
            return inputs

        assert self.reg.get("desc_op").description == "explicit"

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def test_repr(self):
        self.reg.register("op_a", _noop_builder)
        r = repr(self.reg)
        assert "OpRegistry" in r
        assert "op_a" in r


# ===========================================================================
# Module-level helpers (register_op / custom_op / list_ops / get_op)
# Use a private registry to avoid polluting the global_registry.
# ===========================================================================

class TestModuleLevelHelpers:
    """Tests for the module-level convenience functions."""

    def setup_method(self):
        self.reg = OpRegistry()

    def test_register_op_functional(self):
        spec = register_op("fn_op", _noop_builder, n_inputs=2, registry=self.reg)
        assert spec.name == "fn_op"
        assert "fn_op" in self.reg

    def test_custom_op_decorator(self):
        @custom_op("dec_op", n_inputs=1, registry=self.reg)
        def _b(kgraph, inputs, **kwargs):
            return inputs

        assert "dec_op" in self.reg

    def test_list_ops(self):
        register_op("lo1", _noop_builder, registry=self.reg)
        register_op("lo2", _noop_builder, registry=self.reg)
        names = [s.name for s in list_ops(registry=self.reg)]
        assert set(names) == {"lo1", "lo2"}

    def test_get_op(self):
        register_op("g1", _noop_builder, registry=self.reg)
        spec = get_op("g1", registry=self.reg)
        assert spec.name == "g1"

    def test_get_op_missing_raises(self):
        with pytest.raises(KeyError):
            get_op("not_there", registry=self.reg)

    def test_global_registry_is_opregistry(self):
        assert isinstance(global_registry, OpRegistry)


# ===========================================================================
# KNGraph.call_op  (no C++ build needed — uses a mock kgraph)
# ===========================================================================

class _MockKNGraph:
    """Minimal stand-in for KNGraph that exposes call_op via direct import."""

    def call_op(self, name: str, inputs: list, *, registry=None, **kwargs) -> list:
        from yirage.kernel.op_registry import global_registry, OpRegistry
        reg = registry if registry is not None else global_registry
        spec = reg.get(name)
        return spec(self, inputs, **kwargs)


class TestCallOp:
    """Tests for KNGraph.call_op dispatch via registry."""

    def setup_method(self):
        self.reg = OpRegistry()
        self.kgraph = _MockKNGraph()

    def test_call_op_invokes_builder(self):
        calls: list = []

        def _b(kgraph, inputs, **kwargs):
            calls.append(inputs)
            return inputs

        self.reg.register("my_op", _b)
        result = self.kgraph.call_op("my_op", ["t1"], registry=self.reg)
        assert calls == [["t1"]]
        assert result == ["t1"]

    def test_call_op_forwards_kwargs(self):
        received = {}

        def _b(kgraph, inputs, **kwargs):
            received.update(kwargs)
            return inputs

        self.reg.register("kw_op", _b)
        self.kgraph.call_op("kw_op", [], registry=self.reg, grid_dim=(2, 1, 1))
        assert received["grid_dim"] == (2, 1, 1)

    def test_call_op_unknown_raises(self):
        with pytest.raises(KeyError, match="no_such_op"):
            self.kgraph.call_op("no_such_op", [], registry=self.reg)

    def test_call_op_arity_error_propagates(self):
        self.reg.register("ar_op", _noop_builder, n_inputs=3)
        with pytest.raises(ValueError, match="expects 3 input"):
            self.kgraph.call_op("ar_op", ["only_one"], registry=self.reg)

    def test_call_op_passes_kgraph_reference(self):
        received_kgraph = []

        def _b(kgraph, inputs, **kwargs):
            received_kgraph.append(kgraph)
            return inputs

        self.reg.register("kg_op", _b)
        self.kgraph.call_op("kg_op", [], registry=self.reg)
        assert received_kgraph[0] is self.kgraph


# ===========================================================================
# Real KNGraph.call_op (requires native yirage.core)
# ===========================================================================

class TestCallOpWithRealKNGraph:
    """Integration smoke-test: call_op on a real KNGraph (requires yirage.core)."""

    @pytest.fixture(autouse=True)
    def _skip_without_core(self):
        try:
            import yirage as mi
        except ImportError:
            pytest.skip("yirage package not available")
        # The test_rl conftest may install a bare ``yirage`` namespace stub
        # in environments without the native runtime; that stub does not
        # provide ``new_kernel_graph`` etc. and is unusable here.
        if getattr(mi, "_is_test_shim", False):
            pytest.skip("yirage package not available (test shim only)")
        self.mi = mi
        # Use the same registry objects we loaded in this test module
        self.CustomOpSpec = CustomOpSpec
        self.custom_op = custom_op
        self.OpRegistry = OpRegistry

    def test_call_op_uses_customized_path(self):
        """Registered builder runs via call_op, exercising the full dispatch path.

        The builder creates a fused threadblock graph via ``kgraph.customized()``.
        A ``forloop_accum()`` (no-reduction) step is inserted between input and
        output because the threadblock output saver requires ``after_accum == true``
        (:file:`src/threadblock/output.cc:54`); the accumulator sets this flag.
        """
        mi = self.mi
        reg = OpRegistry()

        @custom_op("identity_fused", n_inputs=1, registry=reg)
        def _identity_builder(kgraph, inputs, **kwargs):
            A = inputs[0]
            bgraph = mi.new_threadblock_graph(
                grid_dim=kwargs.get("grid_dim", (1, 1, 1)),
                block_dim=kwargs.get("block_dim", (32, 1, 1)),
                forloop_range=kwargs.get("forloop_range", 1),
                reduction_dimx=kwargs.get(
                    "reduction_dimx", A.dim(A.num_dims - 1)
                ),
            )
            a_smem = bgraph.new_input(A, input_map=(0, -1, -1), forloop_dim=-1)
            a_accum = bgraph.forloop_accum(a_smem)  # None → NO_RED, sets after_accum=true
            bgraph.new_output(a_accum, output_map=(0, -1, -1))
            return kgraph.customized(inputs, bgraph)

        kgraph = mi.new_kernel_graph()
        A = kgraph.new_input((32, 32), dtype=mi.float16)
        outputs = kgraph.call_op(
            "identity_fused",
            [A],
            registry=reg,
            grid_dim=(1, 1, 1),
            block_dim=(32, 1, 1),
            forloop_range=1,
            reduction_dimx=32,
        )
        assert len(outputs) > 0


# ===========================================================================
# Global registry isolation guard
# ===========================================================================

class TestGlobalRegistryIsolation:
    """Ensure tests that use isolated registries do NOT affect global_registry."""

    def test_isolated_registry_does_not_pollute_global(self):
        isolated = OpRegistry()
        register_op("isolated_op_xyz", _noop_builder, registry=isolated)
        # Should NOT be in the global registry
        assert "isolated_op_xyz" not in global_registry


# ===========================================================================
# Edge cases & misc
# ===========================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_register_same_builder_under_different_names(self):
        reg = OpRegistry()
        reg.register("alias_a", _noop_builder)
        reg.register("alias_b", _noop_builder)
        assert "alias_a" in reg
        assert "alias_b" in reg

    def test_register_lambda(self):
        reg = OpRegistry()
        reg.register("lambda_op", lambda kg, inp, **kw: inp)
        assert "lambda_op" in reg

    def test_spec_call_with_tuple_return(self):
        def _tuple_builder(kgraph, inputs, **kwargs):
            return ("out1", "out2")

        spec = CustomOpSpec(name="t", builder=_tuple_builder)
        result = spec("kg", [])
        assert result == ["out1", "out2"]

    def test_register_op_returns_spec_with_tags(self):
        reg = OpRegistry()
        spec = register_op("tagged", _noop_builder, tags=["matmul", "fused"], registry=reg)
        assert "matmul" in spec.tags

    def test_clear_then_reregister(self):
        reg = OpRegistry()
        reg.register("op", _noop_builder)
        reg.clear()
        assert len(reg) == 0
        reg.register("op", _noop_builder)  # should not warn after clear
        assert "op" in reg
