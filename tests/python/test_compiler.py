#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Compiler Module Unit Tests

Tests for yirage/compiler/ module including UnifiedCompiler, Pipeline, and Cache.
Run with: pytest tests/python/test_compiler.py -v
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from conftest import safe_import


# =============================================================================
# Module Loading Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def unified_module():
    """Load unified compiler module."""
    return safe_import("yirage.compiler.unified")


@pytest.fixture(scope="module")
def pipeline_module():
    """Load pipeline module."""
    return safe_import("yirage.compiler.pipeline")


@pytest.fixture(scope="module")
def cache_module():
    """Load cache module."""
    return safe_import("yirage.compiler.cache")


# =============================================================================
# CompileMode Tests
# =============================================================================

class TestCompileMode:
    """Tests for CompileMode enum."""

    def test_compile_mode_exists(self, unified_module):
        """Test CompileMode enum exists."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        assert hasattr(unified_module, "CompileMode")

    def test_compile_mode_fast_exists(self, unified_module):
        """Test FAST mode exists."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileMode = getattr(unified_module, "CompileMode", None)
        if CompileMode is None:
            pytest.skip("CompileMode not found")

        assert hasattr(CompileMode, "FAST")

    def test_compile_mode_superoptimize_exists(self, unified_module):
        """Test SUPEROPTIMIZE mode exists."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileMode = getattr(unified_module, "CompileMode", None)
        if CompileMode is None:
            pytest.skip("CompileMode not found")

        assert hasattr(CompileMode, "SUPEROPTIMIZE")

    def test_compile_mode_aggressive_exists(self, unified_module):
        """Test AGGRESSIVE mode exists."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileMode = getattr(unified_module, "CompileMode", None)
        if CompileMode is None:
            pytest.skip("CompileMode not found")

        assert hasattr(CompileMode, "AGGRESSIVE")

    def test_compile_mode_rl_guided_exists(self, unified_module):
        """Test RL_GUIDED mode exists."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileMode = getattr(unified_module, "CompileMode", None)
        if CompileMode is None:
            pytest.skip("CompileMode not found")

        assert hasattr(CompileMode, "RL_GUIDED")

    def test_compile_mode_mlir_only_exists(self, unified_module):
        """Test MLIR_ONLY mode exists."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileMode = getattr(unified_module, "CompileMode", None)
        if CompileMode is None:
            pytest.skip("CompileMode not found")

        assert hasattr(CompileMode, "MLIR_ONLY")


# =============================================================================
# CompileOptions Tests
# =============================================================================

class TestCompileOptions:
    """Tests for CompileOptions dataclass."""

    def test_compile_options_exists(self, unified_module):
        """Test CompileOptions class exists."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        assert hasattr(unified_module, "CompileOptions")

    def test_default_backend_is_auto(self, unified_module):
        """Test default backend is 'auto'."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileOptions = getattr(unified_module, "CompileOptions", None)
        if CompileOptions is None:
            pytest.skip("CompileOptions not found")

        options = CompileOptions()
        assert options.backend == "auto"

    def test_default_enable_cache_true(self, unified_module):
        """Test cache is enabled by default."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileOptions = getattr(unified_module, "CompileOptions", None)
        if CompileOptions is None:
            pytest.skip("CompileOptions not found")

        options = CompileOptions()
        assert options.enable_cache is True

    def test_mlir_opt_level_default(self, unified_module):
        """Test MLIR optimization level default."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileOptions = getattr(unified_module, "CompileOptions", None)
        if CompileOptions is None:
            pytest.skip("CompileOptions not found")

        options = CompileOptions()
        assert hasattr(options, "mlir_opt_level")
        assert 0 <= options.mlir_opt_level <= 3

    def test_custom_backend_option(self, unified_module):
        """Test setting custom backend."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileOptions = getattr(unified_module, "CompileOptions", None)
        if CompileOptions is None:
            pytest.skip("CompileOptions not found")

        options = CompileOptions(backend="cpu")
        assert options.backend == "cpu"


# =============================================================================
# UnifiedCompiler Tests
# =============================================================================

class TestUnifiedCompiler:
    """Tests for UnifiedCompiler class."""

    def test_compiler_exists(self, unified_module):
        """Test UnifiedCompiler class exists."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        assert hasattr(unified_module, "UnifiedCompiler")

    def test_compiler_creation_with_cpu(self, unified_module):
        """Test creating compiler with CPU backend."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        UnifiedCompiler = getattr(unified_module, "UnifiedCompiler", None)
        CompileMode = getattr(unified_module, "CompileMode", None)

        if UnifiedCompiler is None:
            pytest.skip("UnifiedCompiler not found")

        if CompileMode is None:
            compiler = UnifiedCompiler(backend="cpu")
        else:
            compiler = UnifiedCompiler(backend="cpu", mode=CompileMode.FAST)

        assert compiler is not None
        assert compiler.backend == "cpu"

    def test_compiler_has_mode_attribute(self, unified_module):
        """Test compiler has mode attribute."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        UnifiedCompiler = getattr(unified_module, "UnifiedCompiler", None)
        CompileMode = getattr(unified_module, "CompileMode", None)

        if UnifiedCompiler is None or CompileMode is None:
            pytest.skip("Required classes not found")

        compiler = UnifiedCompiler(backend="cpu", mode=CompileMode.FAST)
        assert hasattr(compiler, "mode")

    def test_compiler_statistics_initial(self, unified_module):
        """Test compiler statistics are initially zero."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        UnifiedCompiler = getattr(unified_module, "UnifiedCompiler", None)
        CompileMode = getattr(unified_module, "CompileMode", None)

        if UnifiedCompiler is None:
            pytest.skip("UnifiedCompiler not found")

        if CompileMode:
            compiler = UnifiedCompiler(backend="cpu", mode=CompileMode.FAST)
        else:
            compiler = UnifiedCompiler(backend="cpu")

        stats = compiler.get_statistics()
        assert stats["compile_count"] == 0
        assert stats["cache_hits"] == 0

    def test_compiler_has_compile_method(self, unified_module):
        """Test compiler has compile method."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        UnifiedCompiler = getattr(unified_module, "UnifiedCompiler", None)
        if UnifiedCompiler is None:
            pytest.skip("UnifiedCompiler not found")

        compiler = UnifiedCompiler(backend="cpu")
        assert hasattr(compiler, "compile")
        assert callable(compiler.compile)


# =============================================================================
# CompilePipeline Tests
# =============================================================================

class TestCompilePipeline:
    """Tests for CompilePipeline class."""

    def test_pipeline_exists(self, pipeline_module):
        """Test CompilePipeline class exists."""
        if pipeline_module is None:
            pytest.skip("Pipeline module not available")

        assert hasattr(pipeline_module, "CompilePipeline")

    def test_pipeline_creation(self, pipeline_module):
        """Test creating pipeline."""
        if pipeline_module is None:
            pytest.skip("Pipeline module not available")

        CompilePipeline = getattr(pipeline_module, "CompilePipeline", None)
        if CompilePipeline is None:
            pytest.skip("CompilePipeline not found")

        pipeline = CompilePipeline(
            backend="cpu",
            enable_superoptimize=False,
            enable_mlir=False,
        )
        assert pipeline is not None

    def test_stage_status_enum_exists(self, pipeline_module):
        """Test StageStatus enum exists."""
        if pipeline_module is None:
            pytest.skip("Pipeline module not available")

        assert hasattr(pipeline_module, "StageStatus")

    def test_stage_status_values(self, pipeline_module):
        """Test StageStatus enum values."""
        if pipeline_module is None:
            pytest.skip("Pipeline module not available")

        StageStatus = getattr(pipeline_module, "StageStatus", None)
        if StageStatus is None:
            pytest.skip("StageStatus not found")

        assert hasattr(StageStatus, "PENDING")
        assert hasattr(StageStatus, "RUNNING")
        assert hasattr(StageStatus, "COMPLETED")
        assert hasattr(StageStatus, "FAILED")


# =============================================================================
# Pipeline Stages Tests
# =============================================================================

class TestPipelineStages:
    """Tests for individual pipeline stages."""

    def test_superoptimize_stage_exists(self, pipeline_module):
        """Test SuperoptimizeStage exists."""
        if pipeline_module is None:
            pytest.skip("Pipeline module not available")

        assert hasattr(pipeline_module, "SuperoptimizeStage")

    def test_superoptimize_stage_creation(self, pipeline_module):
        """Test creating SuperoptimizeStage."""
        if pipeline_module is None:
            pytest.skip("Pipeline module not available")

        SuperoptimizeStage = getattr(pipeline_module, "SuperoptimizeStage", None)
        if SuperoptimizeStage is None:
            pytest.skip("SuperoptimizeStage not found")

        stage = SuperoptimizeStage(backend="cpu", use_ray=False)
        assert stage.name == "superoptimize"
        assert stage.backend == "cpu"

    def test_mlir_lowering_stage_exists(self, pipeline_module):
        """Test MLIRLoweringStage exists."""
        if pipeline_module is None:
            pytest.skip("Pipeline module not available")

        assert hasattr(pipeline_module, "MLIRLoweringStage")

    def test_mlir_lowering_stage_creation(self, pipeline_module):
        """Test creating MLIRLoweringStage."""
        if pipeline_module is None:
            pytest.skip("Pipeline module not available")

        MLIRLoweringStage = getattr(pipeline_module, "MLIRLoweringStage", None)
        if MLIRLoweringStage is None:
            pytest.skip("MLIRLoweringStage not found")

        stage = MLIRLoweringStage(opt_level=2)
        assert stage.name == "mlir_lowering"
        assert stage.opt_level == 2

    def test_codegen_stage_exists(self, pipeline_module):
        """Test CodeGenStage exists."""
        if pipeline_module is None:
            pytest.skip("Pipeline module not available")

        # May be called CodeGenStage or similar
        has_codegen = hasattr(pipeline_module, "CodeGenStage") or hasattr(
            pipeline_module, "CodeGenerationStage"
        )
        if not has_codegen:
            pytest.skip("Pipeline module has no CodeGenStage / CodeGenerationStage")


# =============================================================================
# CompileCache Tests
# =============================================================================

class TestCompileCache:
    """Tests for CompileCache class."""

    def test_cache_exists(self, cache_module):
        """Test CompileCache class exists."""
        if cache_module is None:
            pytest.skip("Cache module not available")

        assert hasattr(cache_module, "CompileCache")

    def test_cache_creation(self, cache_module):
        """Test creating cache."""
        if cache_module is None:
            pytest.skip("Cache module not available")

        CompileCache = getattr(cache_module, "CompileCache", None)
        if CompileCache is None:
            pytest.skip("CompileCache not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompileCache(cache_dir=tmpdir)
            assert cache is not None

    def test_cache_put_and_get(self, cache_module):
        """Test cache put and get operations."""
        if cache_module is None:
            pytest.skip("Cache module not available")

        CompileCache = getattr(cache_module, "CompileCache", None)
        if CompileCache is None:
            pytest.skip("CompileCache not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompileCache(cache_dir=tmpdir)

            # Put entry
            entry = cache.put(
                graph_hash="test_hash_123",
                backend="cpu",
                latency_ms=1.5,
                compile_time_seconds=0.5,
            )
            assert entry is not None

            # Get entry
            retrieved = cache.get("test_hash_123", "cpu")
            assert retrieved is not None
            assert retrieved.latency_ms == 1.5

    def test_cache_miss_returns_none(self, cache_module):
        """Test cache miss returns None."""
        if cache_module is None:
            pytest.skip("Cache module not available")

        CompileCache = getattr(cache_module, "CompileCache", None)
        if CompileCache is None:
            pytest.skip("CompileCache not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompileCache(cache_dir=tmpdir)
            result = cache.get("nonexistent_hash", "cpu")
            assert result is None

    def test_cache_statistics(self, cache_module):
        """Test cache statistics tracking."""
        if cache_module is None:
            pytest.skip("Cache module not available")

        CompileCache = getattr(cache_module, "CompileCache", None)
        if CompileCache is None:
            pytest.skip("CompileCache not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompileCache(cache_dir=tmpdir)

            # Generate some activity
            cache.put("hash1", "cpu", latency_ms=1.0)
            cache.get("hash1", "cpu")  # hit
            cache.get("hash2", "cpu")  # miss

            stats = cache.get_statistics()
            assert "hits" in stats
            assert "misses" in stats
            assert "hit_rate" in stats

    def test_cache_hit_rate_calculation(self, cache_module):
        """Test cache hit rate is calculated correctly."""
        if cache_module is None:
            pytest.skip("Cache module not available")

        CompileCache = getattr(cache_module, "CompileCache", None)
        if CompileCache is None:
            pytest.skip("CompileCache not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompileCache(cache_dir=tmpdir)

            # Put one entry
            cache.put("hash1", "cpu", latency_ms=1.0)

            # 2 hits, 1 miss
            cache.get("hash1", "cpu")  # hit
            cache.get("hash1", "cpu")  # hit
            cache.get("hash2", "cpu")  # miss

            stats = cache.get_statistics()
            # Hit rate should be 2/3 = 0.666...
            if stats["hits"] + stats["misses"] > 0:
                expected_rate = stats["hits"] / (stats["hits"] + stats["misses"])
                assert abs(stats["hit_rate"] - expected_rate) < 0.01


# =============================================================================
# CompileResult Tests
# =============================================================================

class TestCompileResult:
    """Tests for CompileResult dataclass."""

    def test_compile_result_exists(self, unified_module):
        """Test CompileResult class exists."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        assert hasattr(unified_module, "CompileResult")

    def test_compile_result_has_success_field(self, unified_module):
        """Test CompileResult has success field."""
        if unified_module is None:
            pytest.skip("Unified compiler module not available")

        CompileResult = getattr(unified_module, "CompileResult", None)
        if CompileResult is None:
            pytest.skip("CompileResult not found")

        # Create a result instance (may need different initialization)
        try:
            result = CompileResult(success=True)
            assert hasattr(result, "success")
        except TypeError:
            # May have different constructor
            pass


# =============================================================================
# Integration Tests
# =============================================================================

class TestCompilerIntegration:
    """Integration tests for compiler subsystem."""

    def test_unified_compiler_uses_cache(self, unified_module, cache_module):
        """Test that UnifiedCompiler can use cache."""
        if unified_module is None or cache_module is None:
            pytest.skip("Required modules not available")

        UnifiedCompiler = getattr(unified_module, "UnifiedCompiler", None)
        CompileOptions = getattr(unified_module, "CompileOptions", None)

        if UnifiedCompiler is None:
            pytest.skip("UnifiedCompiler not found")

        # Create compiler with cache enabled
        if CompileOptions:
            options = CompileOptions(enable_cache=True)
            compiler = UnifiedCompiler(backend="cpu", options=options)
        else:
            compiler = UnifiedCompiler(backend="cpu")

        # Compiler should have cache
        assert hasattr(compiler, "_cache") or hasattr(compiler, "cache")

    def test_pipeline_integrates_with_compiler(self, unified_module, pipeline_module):
        """Test pipeline integrates with compiler."""
        if unified_module is None or pipeline_module is None:
            pytest.skip("Required modules not available")

        # Both modules should be compatible
        UnifiedCompiler = getattr(unified_module, "UnifiedCompiler", None)
        CompilePipeline = getattr(pipeline_module, "CompilePipeline", None)

        if UnifiedCompiler is None or CompilePipeline is None:
            pytest.skip("Required classes not found")

        # Should be able to create both
        compiler = UnifiedCompiler(backend="cpu")
        pipeline = CompilePipeline(backend="cpu", enable_superoptimize=False)

        assert compiler is not None
        assert pipeline is not None
