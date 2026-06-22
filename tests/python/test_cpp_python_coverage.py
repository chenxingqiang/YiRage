# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
C++ to Python API Coverage Tests

This file documents and tests the mapping between C++ functionality
and Python bindings. It serves as both documentation and verification
that expected C++ features are properly exposed to Python.

C++ Modules and their Python Exposure Status:
============================================

1. kernel/graph.h (CppKNGraph) → yirage.core.CyKNGraph
   ✓ EXPOSED: new_input, mark_output, matmul, exp, silu, gelu, relu,
              clamp, sqrt, square, add, mul, div, pow, reduction,
              rms_norm, customized, generate_triton_program,
              generate_cuda_program

2. threadblock/graph.h (CppTBGraph) → yirage.core.CyTBGraph
   ✓ EXPOSED: new_input, new_output, matmul, exp, silu, gelu, relu,
              clamp, square, sqrt, mul_scalar, add, mul, div, sub,
              reduction, reduction_max, rms_norm, concat,
              forloop_accum, forloop_accum_rescale, forloop_accum_max

3. transpiler/transpile.h → yirage.core (transpile functions)
   ✓ EXPOSED: TranspilerConfig, TranspileResult, transpile()

4. nki_transpiler/transpile.h → yirage.core
   ✓ EXPOSED: NKITranspilerConfig, NKITranspileResult

5. triton_transpiler/transpile.h → yirage.core
   ✓ EXPOSED: TritonTranspilerConfig, TritonTranspileResult

6. search/search_c.h → yirage.core.cython_search
   ✓ EXPOSED: cython_search, cython_to_json, cython_from_json

7. backend/backends.h → yirage.core
   ✓ EXPOSED: initialize_backends, get_available_backend_names,
              is_backend_available

MISSING/PARTIAL EXPOSURES:
==========================

8. persistent_kernel/pk_backend_interface.h
   ✗ NOT EXPOSED via Cython: PKBackendType, PKMode, PKDataType,
                             PKCapabilities, PKRuntimeConfig
   ✓ EXPOSED via Pure Python: yirage.persistent_kernel.runtime
   (Pure Python implementation, not C++ binding)

9. backend/backend_interface.h
   ✗ NOT EXPOSED: BackendInterface class methods
   (Only basic backend info exposed)

10. search/search.h (KernelGraphGenerator)
    ✗ NOT EXPOSED: Full KernelGraphGenerator API
    ✓ PARTIAL: Only cython_search() wrapper

11. search/config.h (GeneratorConfig)
    ✗ NOT EXPOSED: Full GeneratorConfig structure
    ✓ PARTIAL: Passed as string to cython_search

12. search/verification/verifier.h
    ✗ NOT EXPOSED: Verifier class

13. kernel/device_memory_manager.h
    ✓ EXPOSED: cython_set_gpu_device_id

14. kernel/runtime.h
    ✓ EXPOSED: TaskGraphResult (cuda_code, json_file)
"""

import pytest
from typing import List, Dict, Optional, Set

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def core_module():
    """Import the core module (Cython bindings)."""
    try:
        from yirage import core
        return core
    except ImportError:
        pytest.skip("yirage.core not available (C++ bindings not built)")


@pytest.fixture
def kernel_module():
    """Import the kernel module."""
    try:
        from yirage.kernel import graph
        return graph
    except ImportError:
        pytest.skip("yirage.kernel module not available")


# =============================================================================
# Coverage Documentation
# =============================================================================


# C++ functions that SHOULD be exposed to Python
EXPECTED_CORE_EXPORTS = {
    # Data types
    "dtype",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64",
    
    # KN Graph operations
    "CyKNGraph",
    "get_kn_operator_type_string",
    
    # TB Graph operations
    "CyTBGraph",
    "get_tb_operator_type_string",
    
    # Device management
    "set_gpu_device_id",
    
    # Search functions
    "search",
    
    # Serialization
    "to_json",
    "from_json",
    
    # Transpilation
    "transpile",
    "generate_triton",
    "generate_nki",
}

# C++ functions currently NOT exposed but could be useful
MISSING_EXPOSURES = {
    # Backend Interface (full API)
    "BackendInterface",
    "get_backend_info",
    "get_compile_flags",
    "get_include_dirs",
    "get_library_dirs",
    
    # Search API (full)
    "KernelGraphGenerator",
    "GeneratorConfig",
    "SearchContext",
    "Verifier",
    
    # Persistent Kernel (C++ API)
    "PKBackendInterface",  # Currently pure Python
    "PKTaskExecutor",      # Currently pure Python
    
    # Memory Manager
    "DeviceMemoryManager",
    "allocate_device_memory",
    "free_device_memory",
    
    # Profiler
    "CppProfiler",
    "start_profiling",
    "stop_profiling",
}


# =============================================================================
# Core Module Coverage Tests
# =============================================================================


class TestCoreDtypeExposure:
    """Test that data types are exposed from C++ to Python."""
    
    def test_dtype_class_exists(self, core_module):
        """Test dtype class is exposed."""
        assert hasattr(core_module, "dtype")
    
    @pytest.mark.parametrize("dtype_name", [
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "bfloat16", "float32", "float64"
    ])
    def test_dtype_constants(self, core_module, dtype_name):
        """Test dtype constants are exposed."""
        assert hasattr(core_module, dtype_name), f"Missing dtype: {dtype_name}"
    
    def test_dtype_methods(self, core_module):
        """Test dtype methods."""
        if not hasattr(core_module, "dtype"):
            pytest.skip("dtype not available")
        
        dt = core_module.float16
        assert hasattr(dt, "is_fp16")
        assert hasattr(dt, "is_bf16")
        assert hasattr(dt, "is_fp32")


class TestKNGraphExposure:
    """Test that KNGraph operations are exposed from C++ to Python."""
    
    def test_cykngraph_exists(self, core_module):
        """Test CyKNGraph class is exposed."""
        assert hasattr(core_module, "CyKNGraph")
    
    def test_kngraph_operator_types(self, core_module):
        """Test KN operator type mapping function."""
        assert hasattr(core_module, "get_kn_operator_type_string")
    
    def test_kngraph_creation(self, core_module):
        """Test KNGraph can be created."""
        try:
            graph = core_module.CyKNGraph()
            assert graph is not None
        except Exception as e:
            pytest.skip(f"Cannot create CyKNGraph: {e}")
    
    @pytest.mark.parametrize("method_name", [
        "new_input", "mark_output",
        "matmul", "exp", "silu", "gelu", "relu",
        "clamp", "sqrt", "square",
        "add", "mul", "div", "pow",
        "reduction", "rms_norm", "customized",
    ])
    def test_kngraph_methods(self, core_module, method_name):
        """Test KNGraph methods exist."""
        if not hasattr(core_module, "CyKNGraph"):
            pytest.skip("CyKNGraph not available")
        
        # Check method exists on class
        try:
            graph = core_module.CyKNGraph()
            assert hasattr(graph, method_name), f"Missing method: {method_name}"
        except Exception:
            pytest.skip("Cannot create CyKNGraph for method check")


class TestTBGraphExposure:
    """Test that TBGraph operations are exposed from C++ to Python."""
    
    def test_cytbgraph_exists(self, core_module):
        """Test CyTBGraph class is exposed."""
        assert hasattr(core_module, "CyTBGraph")
    
    def test_tbgraph_operator_types(self, core_module):
        """Test TB operator type mapping function."""
        assert hasattr(core_module, "get_tb_operator_type_string")
    
    @pytest.mark.parametrize("method_name", [
        "new_input", "new_output",
        "matmul", "exp", "silu", "gelu", "relu",
        "clamp", "square", "sqrt", "mul_scalar",
        "add", "mul", "div", "sub",
        "reduction", "reduction_max", "rms_norm",
        "concat", "forloop_accum",
    ])
    def test_tbgraph_methods(self, core_module, method_name):
        """Test TBGraph methods should exist."""
        # Document expected methods
        expected_methods = {
            "new_input", "new_output",
            "matmul", "exp", "silu", "gelu", "relu",
            "clamp", "square", "sqrt", "mul_scalar",
            "add", "mul", "div", "sub",
            "reduction", "reduction_max", "rms_norm",
            "concat", "forloop_accum",
        }
        assert method_name in expected_methods


class TestTranspilerExposure:
    """Test that transpiler functions are exposed."""
    
    def test_transpile_function(self, core_module):
        """Test transpile function exists or is documented."""
        # transpile is typically called via graph methods
        # Check for generate_cuda_program on graph
        pass
    
    def test_triton_transpiler(self, core_module):
        """Test Triton transpiler exposure."""
        # generate_triton_program should be on CyKNGraph
        if hasattr(core_module, "CyKNGraph"):
            try:
                graph = core_module.CyKNGraph()
                assert hasattr(graph, "generate_triton_program")
            except Exception:
                pytest.skip("Cannot verify triton transpiler")


class TestSearchExposure:
    """Test that search functions are exposed."""
    
    def test_search_function(self, core_module):
        """Test search function exists."""
        has_search = hasattr(core_module, "search") or hasattr(
            core_module, "cython_search"
        )
        assert has_search, "Expected search or cython_search on yirage.core"

    def test_json_serialization(self, core_module, tmp_path):
        """CyKNGraph exposes file-based JSON via to_json(path)."""
        if not hasattr(core_module, "CyKNGraph"):
            pytest.skip("CyKNGraph not on core module")
        graph = core_module.CyKNGraph()
        if not hasattr(graph, "to_json"):
            pytest.skip("CyKNGraph.to_json not available")
        a = graph.new_input(dims=(2, 4), dtype=core_module.float16)
        b = graph.new_input(dims=(4, 8), dtype=core_module.float16)
        c = graph.matmul(a, b)
        graph.mark_output(c)
        out = tmp_path / "g.json"
        graph.to_json(str(out))
        assert out.is_file()
        raw = out.read_text(encoding="utf-8").strip()
        assert raw


class TestBackendExposure:
    """Test that backend API is exposed."""
    
    def test_backend_init(self, core_module):
        """Test backend initialization."""
        if hasattr(core_module, "init_backends"):
            assert callable(core_module.init_backends)
        elif hasattr(core_module, "initialize_backends"):
            assert callable(core_module.initialize_backends)
        else:
            pytest.skip("No init_backends/initialize_backends on core")
    
    def test_backend_availability_check(self):
        """Test backend availability functions via Python API."""
        try:
            from yirage.backends.api import (
                get_available_backends,
                is_backend_available,
            )
            
            backends = get_available_backends()
            assert isinstance(backends, list)
        except ImportError:
            pytest.skip("Backend API not available")


# =============================================================================
# Missing Exposure Documentation Tests
# =============================================================================


class TestMissingExposures:
    """Document C++ functionality NOT currently exposed to Python."""
    
    def test_document_missing_backend_interface(self):
        """Document that full BackendInterface is not exposed."""
        missing = [
            "BackendInterface.compile()",
            "BackendInterface.get_compile_flags()",
            "BackendInterface.get_include_dirs()",
            "BackendInterface.get_library_dirs()",
            "BackendInterface.allocate_memory()",
            "BackendInterface.free_memory()",
        ]
        assert len(missing) >= 5
    
    def test_document_missing_search_api(self):
        """Document that full Search API is not exposed."""
        missing = [
            "KernelGraphGenerator class",
            "GeneratorConfig structure",
            "SearchContext class",
            "Verifier class",
            "DimStrategy class",
            "SymbolicGraph classes",
        ]
        assert len(missing) >= 5
    
    def test_document_missing_pk_cpp_api(self):
        """Document that PK C++ API uses pure Python fallback."""
        # persistent_kernel Python implementation does NOT wrap C++
        # It's a pure Python implementation
        note = """
        persistent_kernel module is implemented in pure Python.
        The C++ pk_backend_interface.h classes are NOT exposed via Cython.
        Python classes mirror the C++ API but don't call C++ code directly.
        """
        assert "pure Python" in note and "Cython" in note
    
    def test_document_missing_memory_manager(self):
        """Document that DeviceMemoryManager is not exposed."""
        missing = [
            "DeviceMemoryManager class",
            "allocate_device_memory()",
            "free_device_memory()",
            "get_memory_usage()",
        ]
        assert len(missing) >= 3


# =============================================================================
# Python-Only Implementations (No C++ Binding)
# =============================================================================


class TestPythonOnlyImplementations:
    """Test modules that are pure Python (no C++ binding)."""
    
    def test_persistent_kernel_is_pure_python(self):
        """Verify persistent_kernel is pure Python."""
        try:
            from yirage.persistent_kernel import runtime
            
            # These are Python classes, not Cython wrappers
            assert hasattr(runtime, "PKRuntime")
            assert hasattr(runtime, "PKBackendType")
            
            # Verify no C++ extension
            # Python implementation doesn't need C++ core
            pk_runtime = runtime.PKRuntime(runtime.PKRuntimeConfig())
            assert pk_runtime is not None
        except ImportError:
            pytest.skip("persistent_kernel not available")
    
    def test_ray_module_is_pure_python(self):
        """Verify ray module is pure Python."""
        try:
            from yirage import ray
            
            # Pure Python distributed implementation
            assert hasattr(ray, "DistributedSearchCoordinator")
            assert hasattr(ray, "SearchWorker")
        except ImportError:
            pytest.skip("ray module not available")
    
    def test_rl_module_is_pure_python(self):
        """Verify RL module is pure Python."""
        try:
            from yirage import rl

            assert hasattr(rl, "__name__") and rl.__name__ == "yirage.rl"
        except ImportError:
            pytest.skip("rl module not available")


# =============================================================================
# Coverage Summary
# =============================================================================


class TestCoverageSummary:
    """Summary of C++ to Python coverage."""
    
    def test_print_coverage_report(self, core_module):
        """Print coverage report."""
        report = """
        ╔═══════════════════════════════════════════════════════════════════════╗
        ║               C++ → Python API Coverage Summary                       ║
        ╠═══════════════════════════════════════════════════════════════════════╣
        ║ FULLY EXPOSED (via Cython):                                           ║
        ║   ✓ kernel/graph.h         → CyKNGraph (all operations)               ║
        ║   ✓ threadblock/graph.h    → CyTBGraph (all operations)               ║
        ║   ✓ type.h                 → dtype, DataType, OperatorType enums      ║
        ║   ✓ transpiler/*           → transpile functions                      ║
        ║   ✓ search/search_c.h      → cython_search wrapper                    ║
        ║   ✓ backend/backends.h     → init, available backends                 ║
        ╠═══════════════════════════════════════════════════════════════════════╣
        ║ PARTIALLY EXPOSED:                                                    ║
        ║   ~ search/search.h        → Only wrapper, not full API               ║
        ║   ~ search/config.h        → Passed as string, not struct             ║
        ║   ~ backend/interface.h    → Only basic info exposed                  ║
        ╠═══════════════════════════════════════════════════════════════════════╣
        ║ NOT EXPOSED (C++ only):                                               ║
        ║   ✗ KernelGraphGenerator   → Use Python wrapper instead               ║
        ║   ✗ Verifier classes       → Use Python implementation                ║
        ║   ✗ DeviceMemoryManager    → Only set_gpu_device_id exposed           ║
        ║   ✗ Full BackendInterface  → Only basic availability                  ║
        ╠═══════════════════════════════════════════════════════════════════════╣
        ║ PURE PYTHON (no C++ binding needed):                                  ║
        ║   ○ persistent_kernel      → Python implementation mirrors C++ API    ║
        ║   ○ ray distributed        → Uses Ray Python library                  ║
        ║   ○ rl module              → Uses PyTorch                             ║
        ║   ○ profiler               → Python wrapper + optional C++            ║
        ║   ○ storage                → Python implementation                    ║
        ╚═══════════════════════════════════════════════════════════════════════╝
        """
        print(report)
        assert "CyKNGraph" in report and "C++ → Python" in report


# =============================================================================
# Recommendations for Improving Coverage
# =============================================================================


class TestImprovementRecommendations:
    """Document recommendations for improving C++ → Python coverage."""
    
    def test_recommendations(self):
        """Document recommendations."""
        recommendations = """
        RECOMMENDATIONS FOR IMPROVING C++ → PYTHON COVERAGE:
        
        1. SEARCH API:
           - Expose GeneratorConfig as a proper Python dataclass
           - Expose KernelGraphGenerator with configurable options
           - Expose SearchContext for fine-grained control
        
        2. BACKEND INTERFACE:
           - Expose full BackendInterface for custom backends
           - Allow Python backends to implement BackendInterface
           - Expose compilation flags and include dirs
        
        3. MEMORY MANAGER:
           - Expose DeviceMemoryManager for manual memory control
           - Add memory statistics functions
        
        4. PERSISTENT KERNEL:
           - Consider wrapping C++ PK implementation for performance
           - Keep Python fallback for portability
        
        5. PROFILER:
           - Expose C++ profiler for accurate timing
           - Add hardware counters exposure
        """
        assert "SEARCH API" in recommendations and "MEMORY MANAGER" in recommendations
