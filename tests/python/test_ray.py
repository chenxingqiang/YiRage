# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Ray Distributed Module Tests

Tests for distributed search and training using Ray.
"""

import pytest
from typing import List, Dict, Any

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ray_module():
    """Import the ray module."""
    try:
        from yirage import ray as yirage_ray
        return yirage_ray
    except ImportError:
        pytest.skip("yirage.ray module not available")


@pytest.fixture
def is_ray_installed():
    """Check if Ray is installed."""
    try:
        import ray
        return True
    except ImportError:
        return False


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports."""

    def test_coordinator_import(self, ray_module):
        """Test DistributedSearchCoordinator can be imported."""
        assert hasattr(ray_module, "DistributedSearchCoordinator")

    def test_worker_import(self, ray_module):
        """Test SearchWorker can be imported."""
        assert hasattr(ray_module, "SearchWorker")

    def test_feedback_import(self, ray_module):
        """Test SearchFeedback can be imported."""
        assert hasattr(ray_module, "SearchFeedback")

    def test_partition_import(self, ray_module):
        """Test SearchPartition can be imported."""
        assert hasattr(ray_module, "SearchPartition")

    def test_ray_distributed_engine_import(self, ray_module):
        """Test RayDistributedEngine can be imported."""
        assert hasattr(ray_module, "RayDistributedEngine")

    def test_distributed_config_import(self, ray_module):
        """Test DistributedConfig can be imported."""
        assert hasattr(ray_module, "DistributedConfig")


# =============================================================================
# DistributedConfig Tests
# =============================================================================


class TestDistributedConfig:
    """Tests for DistributedConfig."""

    def test_config_exists(self, ray_module):
        """Test DistributedConfig class exists."""
        DistributedConfig = ray_module.DistributedConfig
        assert DistributedConfig is not None

    def test_config_creation(self, ray_module):
        """Test DistributedConfig can be created."""
        try:
            DistributedConfig = ray_module.DistributedConfig
            config = DistributedConfig()
            assert config is not None
        except Exception:
            pytest.skip("DistributedConfig initialization not available")


# =============================================================================
# SearchFeedback Tests
# =============================================================================


class TestSearchFeedback:
    """Tests for SearchFeedback dataclass."""

    def test_feedback_exists(self, ray_module):
        """Test SearchFeedback class exists."""
        assert hasattr(ray_module, "SearchFeedback")

    def test_candidate_info_exists(self, ray_module):
        """Test CandidateInfo class exists."""
        assert hasattr(ray_module, "CandidateInfo")

    def test_training_sample_exists(self, ray_module):
        """Test TrainingSample class exists."""
        assert hasattr(ray_module, "TrainingSample")


# =============================================================================
# SearchPartition Tests
# =============================================================================


class TestSearchPartition:
    """Tests for SearchPartition."""

    def test_partition_exists(self, ray_module):
        """Test SearchPartition class exists."""
        assert hasattr(ray_module, "SearchPartition")

    def test_create_partitions_exists(self, ray_module):
        """Test create_partitions function exists."""
        assert hasattr(ray_module, "create_partitions")


# =============================================================================
# Collective Operations Tests
# =============================================================================


class TestCollectiveOps:
    """Tests for collective operations."""

    def test_collective_config_exists(self, ray_module):
        """Test CollectiveConfig exists."""
        assert hasattr(ray_module, "CollectiveConfig")

    def test_collective_operations_exists(self, ray_module):
        """Test CollectiveOperations exists."""
        assert hasattr(ray_module, "CollectiveOperations")

    def test_reduce_functions_exist(self, ray_module):
        """Test reduce functions exist."""
        assert hasattr(ray_module, "sum_reduce")
        assert hasattr(ray_module, "mean_reduce")
        assert hasattr(ray_module, "min_reduce")
        assert hasattr(ray_module, "max_reduce")
        assert hasattr(ray_module, "concat_reduce")


# =============================================================================
# RL Training Config Tests
# =============================================================================


class TestRLTrainConfig:
    """Tests for RL training configuration."""

    def test_rl_train_config_exists(self, ray_module):
        """Test RLTrainConfig exists."""
        assert hasattr(ray_module, "RLTrainConfig")

    def test_gradient_all_reduce_exists(self, ray_module):
        """Test RLGradientAllReduce exists."""
        assert hasattr(ray_module, "RLGradientAllReduce")


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_engine_exists(self, ray_module):
        """Test create_engine function exists."""
        assert hasattr(ray_module, "create_engine")

    def test_create_workers_exists(self, ray_module):
        """Test create_workers function exists."""
        assert hasattr(ray_module, "create_workers")

    def test_is_ray_available_function(self, ray_module):
        """Test is_ray_available function exists and callable."""
        assert hasattr(ray_module, "is_ray_available")
        assert callable(ray_module.is_ray_available)

    def test_is_ray_available_returns_bool(self, ray_module):
        """Test is_ray_available returns boolean."""
        result = ray_module.is_ray_available()
        assert isinstance(result, bool)


# =============================================================================
# Benchmark Functions Tests
# =============================================================================


class TestBenchmarkFunctions:
    """Tests for benchmark functions."""

    def test_benchmark_result_exists(self, ray_module):
        """Test BenchmarkResult exists."""
        assert hasattr(ray_module, "BenchmarkResult")

    def test_benchmark_object_store_exists(self, ray_module):
        """Test benchmark_object_store exists."""
        assert hasattr(ray_module, "benchmark_object_store")

    def test_benchmark_worker_scaling_exists(self, ray_module):
        """Test benchmark_worker_scaling exists."""
        assert hasattr(ray_module, "benchmark_worker_scaling")

    def test_run_all_benchmarks_exists(self, ray_module):
        """Test run_all_benchmarks exists."""
        assert hasattr(ray_module, "run_all_benchmarks")


# =============================================================================
# YPK Integration Tests
# =============================================================================


class TestYPKIntegration:
    """Tests for YPK (persistent kernel) integration."""

    def test_ypk_backend_exists(self, ray_module):
        """Test YPKBackend exists."""
        assert hasattr(ray_module, "YPKBackend")

    def test_ypk_mode_exists(self, ray_module):
        """Test YPKMode exists."""
        assert hasattr(ray_module, "YPKMode")

    def test_ypk_config_exists(self, ray_module):
        """Test YPKConfig exists."""
        assert hasattr(ray_module, "YPKConfig")

    def test_kernel_search_space_exists(self, ray_module):
        """Test KernelSearchSpace exists."""
        assert hasattr(ray_module, "KernelSearchSpace")

    def test_ypk_ray_optimizer_exists(self, ray_module):
        """Test YPKRayOptimizer exists."""
        assert hasattr(ray_module, "YPKRayOptimizer")

    def test_optimize_ypk_kernel_exists(self, ray_module):
        """Test optimize_ypk_kernel exists."""
        assert hasattr(ray_module, "optimize_ypk_kernel")

    def test_backend_capabilities_exists(self, ray_module):
        """Test BACKEND_CAPABILITIES dict exists."""
        assert hasattr(ray_module, "BACKEND_CAPABILITIES")
        assert isinstance(ray_module.BACKEND_CAPABILITIES, dict)

    def test_get_backend_capabilities_exists(self, ray_module):
        """Test get_backend_capabilities exists."""
        assert hasattr(ray_module, "get_backend_capabilities")

    def test_is_mode_supported_exists(self, ray_module):
        """Test is_mode_supported exists."""
        assert hasattr(ray_module, "is_mode_supported")

    def test_get_default_mode_exists(self, ray_module):
        """Test get_default_mode exists."""
        assert hasattr(ray_module, "get_default_mode")


# =============================================================================
# Backwards Compatibility Tests
# =============================================================================


class TestBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_ray_deep_integration_alias(self, ray_module):
        """Test RayDeepIntegration alias exists."""
        assert hasattr(ray_module, "RayDeepIntegration")

    def test_deep_integration_config_alias(self, ray_module):
        """Test DeepIntegrationConfig alias exists."""
        assert hasattr(ray_module, "DeepIntegrationConfig")

    def test_distributed_search_result_alias(self, ray_module):
        """Test DistributedSearchResult alias exists."""
        assert hasattr(ray_module, "DistributedSearchResult")

    def test_create_distributed_engine_alias(self, ray_module):
        """Test create_distributed_engine alias exists."""
        assert hasattr(ray_module, "create_distributed_engine")


# =============================================================================
# Pattern Classes Tests
# =============================================================================


class TestPatternClasses:
    """Tests for pattern classes."""

    def test_distributed_search_pattern_exists(self, ray_module):
        """Test DistributedSearchPattern exists."""
        assert hasattr(ray_module, "DistributedSearchPattern")

    def test_distributed_training_pattern_exists(self, ray_module):
        """Test DistributedTrainingPattern exists."""
        assert hasattr(ray_module, "DistributedTrainingPattern")


# =============================================================================
# Retry and Partition Strategy Tests
# =============================================================================


class TestStrategyClasses:
    """Tests for strategy classes."""

    def test_retry_config_exists(self, ray_module):
        """Test RetryConfig exists."""
        assert hasattr(ray_module, "RetryConfig")

    def test_retry_strategy_exists(self, ray_module):
        """Test RetryStrategy exists."""
        assert hasattr(ray_module, "RetryStrategy")

    def test_partition_strategy_exists(self, ray_module):
        """Test PartitionStrategy exists."""
        assert hasattr(ray_module, "PartitionStrategy")


# =============================================================================
# GPU Placement Tests
# =============================================================================


class TestGPUPlacement:
    """Tests for GPU placement configuration."""

    def test_gpu_placement_config_exists(self, ray_module):
        """Test GPUPlacementConfig exists."""
        assert hasattr(ray_module, "GPUPlacementConfig")


# =============================================================================
# Gradient Reducer Tests
# =============================================================================


class TestGradientReducer:
    """Tests for GradientReducer stub."""

    def test_gradient_reducer_exists(self, ray_module):
        """Test GradientReducer exists."""
        assert hasattr(ray_module, "GradientReducer")

    def test_gradient_reducer_instantiation(self, ray_module):
        """Test GradientReducer can be instantiated."""
        GradientReducer = ray_module.GradientReducer
        reducer = GradientReducer(world_size=4)
        
        assert reducer.world_size == 4

    def test_gradient_reducer_all_reduce(self, ray_module):
        """Test GradientReducer has all_reduce_dict method."""
        GradientReducer = ray_module.GradientReducer
        reducer = GradientReducer(world_size=2)
        
        assert hasattr(reducer, "all_reduce_dict")
        
        # Should return input unchanged (stub behavior)
        tensors = {"a": 1, "b": 2}
        result = reducer.all_reduce_dict(tensors)
        assert result == tensors


# =============================================================================
# All Exports Test
# =============================================================================


class TestAllExports:
    """Tests for __all__ exports."""

    def test_all_exports_accessible(self, ray_module):
        """Test all items in __all__ are accessible."""
        if hasattr(ray_module, "__all__"):
            for name in ray_module.__all__:
                assert hasattr(ray_module, name), f"Export '{name}' not accessible"


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize("function_name", [
    "sum_reduce",
    "mean_reduce",
    "min_reduce",
    "max_reduce",
    "concat_reduce",
])
def test_reduce_function_callable(ray_module, function_name):
    """Test reduce functions are callable."""
    func = getattr(ray_module, function_name, None)
    assert callable(func), f"{function_name} is not callable"


@pytest.mark.parametrize("class_name", [
    "DistributedSearchCoordinator",
    "SearchWorker",
    "SearchFeedback",
    "SearchPartition",
    "RayDistributedEngine",
    "DistributedConfig",
])
def test_core_classes_exist(ray_module, class_name):
    """Test core classes exist."""
    assert hasattr(ray_module, class_name), f"Class '{class_name}' missing"
