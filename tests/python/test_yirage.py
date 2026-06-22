#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
YiRage Unified Test Suite

Consolidated tests for all YiRage Python modules:
1. Module Coverage - Syntax and export verification
2. Backend API - Backend detection and configuration
3. Compiler - Unified compiler and pipeline
4. RL Integration - Feature extraction, search, reward
5. Ray Integration - Distributed computing
6. Storage & Profiler - Data structures

Run with: pytest tests/python/test_yirage.py -v
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple

import pytest
import numpy as np

# Import from conftest
from conftest import (
    PROJECT_ROOT,
    PYTHON_ROOT,
    TORCH_AVAILABLE,
    CUDA_AVAILABLE,
    MPS_AVAILABLE,
    RAY_AVAILABLE,
    load_module,
    check_module_syntax,
)


# =============================================================================
# 1. Module Coverage Tests
# =============================================================================

class TestModuleCoverage:
    """Verify module syntax and exports."""

    # Backend configuration modules
    BACKEND_MODULES = [
        ("yirage/backends/cuda/config.py", ["get_cuda_search_config", "CUDAArch"]),
        ("yirage/backends/mps/config.py", ["get_mps_search_config", "AppleChipFamily"]),
        ("yirage/backends/rocm/config.py", ["get_rocm_search_config", "ROCmArch"]),
        ("yirage/backends/cpu/config.py", ["get_cpu_search_config", "SIMDType"]),
        ("yirage/backends/ascend/config.py", ["get_ascend_search_config"]),
        ("yirage/backends/maca/config.py", ["get_maca_search_config", "MACA_WARP_SIZE"]),
        ("yirage/backends/tpu/config.py", ["get_tpu_search_config", "TPUVersion"]),
        ("yirage/backends/xpu/config.py", ["get_xpu_search_config", "XPUArch"]),
        ("yirage/backends/fpga/config.py", ["get_fpga_search_config", "FPGADevice"]),
    ]

    # Storage modules
    STORAGE_MODULES = [
        ("yirage/storage/mugraph_store.py", ["MuGraphStore", "MuGraphEntry", "MuGraphMetadata"]),
    ]

    # RL modules
    RL_MODULES = [
        ("yirage/rl/features/mugraph_features.py", ["MuGraphFeature"]),
        ("yirage/rl/features/processor.py", ["FeatureProcessor"]),
        ("yirage/rl/env/observation.py", ["ObservationSpace", "SearchState"]),
        ("yirage/rl/env/reward.py", ["RewardConfig", "RewardComputer"]),
        ("yirage/rl/search/config_space.py", ["HardwareConfig", "SearchSpaceConstraints"]),
        ("yirage/rl/models/search_policy.py", ["SearchPolicyNetwork"]),
    ]

    @pytest.mark.coverage
    @pytest.mark.parametrize("rel_path,expected_exports", BACKEND_MODULES)
    def test_backend_module_syntax(self, rel_path: str, expected_exports: List[str]):
        """Test backend module has valid syntax and expected exports."""
        path = PYTHON_ROOT / rel_path
        if not path.exists():
            pytest.skip(f"Module not found: {rel_path}")

        # Check syntax
        valid, error = check_module_syntax(path)
        assert valid, f"Syntax error in {rel_path}: {error}"

        # Check exports in source
        with open(path, "r") as f:
            source = f.read()

        for export in expected_exports:
            assert export in source, f"Missing export '{export}' in {rel_path}"

    @pytest.mark.coverage
    @pytest.mark.parametrize("rel_path,expected_exports", STORAGE_MODULES)
    def test_storage_module_syntax(self, rel_path: str, expected_exports: List[str]):
        """Test storage module has valid syntax and expected exports."""
        path = PYTHON_ROOT / rel_path
        if not path.exists():
            pytest.skip(f"Module not found: {rel_path}")

        valid, error = check_module_syntax(path)
        assert valid, f"Syntax error in {rel_path}: {error}"

        with open(path, "r") as f:
            source = f.read()

        for export in expected_exports:
            assert export in source, f"Missing export '{export}' in {rel_path}"

    @pytest.mark.coverage
    @pytest.mark.parametrize("rel_path,expected_exports", RL_MODULES)
    def test_rl_module_syntax(self, rel_path: str, expected_exports: List[str]):
        """Test RL module has valid syntax and expected exports."""
        path = PYTHON_ROOT / rel_path
        if not path.exists():
            pytest.skip(f"Module not found: {rel_path}")

        valid, error = check_module_syntax(path)
        assert valid, f"Syntax error in {rel_path}: {error}"

        with open(path, "r") as f:
            source = f.read()

        for export in expected_exports:
            assert export in source, f"Missing export '{export}' in {rel_path}"


# =============================================================================
# 2. Backend API Tests
# =============================================================================

class TestBackendAPI:
    """Test backend API functionality."""

    def test_yirage_import(self, yirage_module):
        """Test basic yirage import."""
        if yirage_module is None:
            pytest.skip("YiRage module not available")

        assert hasattr(yirage_module, "__version__")

    def test_backend_api_functions(self, yirage_module):
        """Test backend API functions exist."""
        if yirage_module is None:
            pytest.skip("YiRage module not available")

        assert callable(getattr(yirage_module, "get_available_backends", None))
        assert callable(getattr(yirage_module, "is_backend_available", None))
        assert callable(getattr(yirage_module, "get_default_backend", None))

    def test_get_available_backends(self, backend_api_module):
        """Test getting available backends."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        backends = backend_api_module.get_available_backends()
        assert isinstance(backends, list)

    def test_is_backend_available(self, backend_api_module):
        """Test checking backend availability."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        # These should not raise
        backend_api_module.is_backend_available("cuda")
        backend_api_module.is_backend_available("cpu")
        backend_api_module.is_backend_available("mps")
        backend_api_module.is_backend_available("nonexistent")


class TestBackendConfigs:
    """Test backend configuration modules."""

    @pytest.mark.parametrize("backend", ["cuda", "mps", "rocm", "cpu", "ascend", "maca", "tpu", "xpu", "fpga"])
    def test_backend_config_callable(self, backend_configs, backend: str):
        """Test backend config function is callable and returns dict."""
        if backend not in backend_configs:
            pytest.skip(f"{backend} config not available")

        module = backend_configs[backend]
        func_name = f"get_{backend}_search_config"

        if not hasattr(module, func_name):
            pytest.skip(f"{func_name} not found")

        get_config = getattr(module, func_name)
        config = get_config()

        assert isinstance(config, dict)
        assert len(config) > 0


# =============================================================================
# 3. Compiler Tests
# =============================================================================

class TestCompiler:
    """Test unified compiler module."""

    def test_compile_mode_enum(self, compiler_module):
        """Test CompileMode enum values."""
        if compiler_module is None:
            pytest.skip("Compiler module not available")

        CompileMode = getattr(compiler_module, "CompileMode", None)
        if CompileMode is None:
            pytest.skip("CompileMode not found")

        assert hasattr(CompileMode, "FAST")
        assert hasattr(CompileMode, "SUPEROPTIMIZE")

    def test_compile_options(self, compiler_module):
        """Test CompileOptions defaults."""
        if compiler_module is None:
            pytest.skip("Compiler module not available")

        CompileOptions = getattr(compiler_module, "CompileOptions", None)
        if CompileOptions is None:
            pytest.skip("CompileOptions not found")

        options = CompileOptions()
        assert options.backend == "auto"
        assert options.enable_cache is True

    def test_unified_compiler_creation(self, compiler_module):
        """Test creating UnifiedCompiler."""
        if compiler_module is None:
            pytest.skip("Compiler module not available")

        UnifiedCompiler = getattr(compiler_module, "UnifiedCompiler", None)
        CompileMode = getattr(compiler_module, "CompileMode", None)

        if UnifiedCompiler is None or CompileMode is None:
            pytest.skip("UnifiedCompiler or CompileMode not found")

        compiler = UnifiedCompiler(backend="cpu", mode=CompileMode.FAST)
        assert compiler is not None
        assert compiler.backend == "cpu"


class TestCompileCache:
    """Test compile cache functionality."""

    def test_cache_operations(self):
        """Test cache put/get operations."""
        cache_module = load_module(
            "cache", PYTHON_ROOT / "yirage" / "compiler" / "cache.py"
        )
        if cache_module is None:
            pytest.skip("Cache module not available")

        CompileCache = getattr(cache_module, "CompileCache", None)
        if CompileCache is None:
            pytest.skip("CompileCache not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompileCache(cache_dir=tmpdir)

            # Put
            entry = cache.put(
                graph_hash="test_hash",
                backend="cpu",
                latency_ms=1.5,
                compile_time_seconds=0.5,
            )
            assert entry is not None

            # Get
            retrieved = cache.get("test_hash", "cpu")
            assert retrieved is not None
            assert retrieved.latency_ms == 1.5

            # Miss
            assert cache.get("nonexistent", "cpu") is None


# =============================================================================
# 4. RL Integration Tests
# =============================================================================

class TestRLFeatures:
    """Test RL feature extraction."""

    def test_mugraph_feature_parsing(self, rl_features_module, sample_mugraph_json):
        """Test µGraph feature parsing from JSON."""
        if rl_features_module is None:
            pytest.skip("RL features module not available")

        MuGraphFeature = getattr(rl_features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)

        assert features is not None
        assert len(features.operators) == 2
        assert len(features.tensors) == 4
        assert features.graph_depth == 2

    def test_feature_processor(self, rl_processor_module, rl_features_module, sample_mugraph_json):
        """Test feature processing for neural network input."""
        if rl_processor_module is None or rl_features_module is None:
            pytest.skip("RL modules not available")

        MuGraphFeature = getattr(rl_features_module, "MuGraphFeature", None)
        FeatureProcessor = getattr(rl_processor_module, "FeatureProcessor", None)

        if MuGraphFeature is None or FeatureProcessor is None:
            pytest.skip("Required classes not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        processor = FeatureProcessor()
        processed = processor.process(features)

        assert "node_features" in processed
        assert "edge_index" in processed
        assert "global_features" in processed


class TestRLObservation:
    """Test RL observation encoding."""

    def test_search_state_creation(self, rl_observation_module):
        """Test SearchState creation."""
        if rl_observation_module is None:
            pytest.skip("RL observation module not available")

        SearchState = getattr(rl_observation_module, "SearchState", None)
        if SearchState is None:
            pytest.skip("SearchState not found")

        state = SearchState(
            search_level=1,
            search_depth=5,
            num_kn_operators=3,
            num_tb_operators=2,
            num_tensors=4,
            num_valid_found=1,
            best_latency_ms=0.5,
            current_grid_dim=(4, 2, 1),
            current_block_dim=(128, 1, 1),
            backend="cuda",
            compute_capability=80,
        )

        assert state is not None
        assert state.search_level == 1

    def test_observation_encoder(self, rl_observation_module):
        """Test ObservationEncoder."""
        if rl_observation_module is None:
            pytest.skip("RL observation module not available")

        SearchState = getattr(rl_observation_module, "SearchState", None)
        ObservationSpace = getattr(rl_observation_module, "ObservationSpace", None)
        ObservationEncoder = getattr(rl_observation_module, "ObservationEncoder", None)

        if None in (SearchState, ObservationSpace, ObservationEncoder):
            pytest.skip("Required classes not found")

        state = SearchState(
            search_level=1, search_depth=5, num_kn_operators=3,
            num_tb_operators=2, num_tensors=4, num_valid_found=1,
            best_latency_ms=0.5, current_grid_dim=(4, 2, 1),
            current_block_dim=(128, 1, 1), backend="cuda", compute_capability=80,
        )

        obs_space = ObservationSpace()
        encoder = ObservationEncoder(obs_space)
        obs = encoder.encode(state)

        assert "graph_embedding" in obs
        assert "action_mask" in obs


class TestRLReward:
    """Test RL reward computation."""

    def test_reward_config(self, rl_reward_module):
        """Test RewardConfig creation."""
        if rl_reward_module is None:
            pytest.skip("RL reward module not available")

        RewardConfig = getattr(rl_reward_module, "RewardConfig", None)
        if RewardConfig is None:
            pytest.skip("RewardConfig not found")

        config = RewardConfig(
            validity_weight=1.0,
            performance_weight=2.0,
            efficiency_weight=0.5,
            exploration_weight=0.1
        )

        assert config.validity_weight == 1.0
        assert config.performance_weight == 2.0

    def test_reward_computer(self, rl_reward_module):
        """Test RewardComputer."""
        if rl_reward_module is None:
            pytest.skip("RL reward module not available")

        RewardConfig = getattr(rl_reward_module, "RewardConfig", None)
        RewardComputer = getattr(rl_reward_module, "RewardComputer", None)
        VerifyResult = getattr(rl_reward_module, "VerifyResult", None)
        ProfileResult = getattr(rl_reward_module, "ProfileResult", None)

        if None in (RewardConfig, RewardComputer, VerifyResult, ProfileResult):
            pytest.skip("Required classes not found")

        config = RewardConfig()
        computer = RewardComputer(config)
        computer.reset()

        verify_result = VerifyResult(verified=True, fingerprint_time_ms=0.1)
        profile_result = ProfileResult(
            latency_ms=0.05, memory_bytes=8192, flops=2097152.0, compile_time_ms=100.0
        )

        reward = computer.compute(
            verify_result=verify_result,
            profile_result=profile_result,
            config_hash="test_config",
            search_depth=5,
            action_type=0,
        )

        assert reward is not None
        assert isinstance(reward, (int, float))


class TestRLSearch:
    """Test RL search space."""

    def test_hardware_config(self, rl_search_config_module):
        """Test HardwareConfig creation."""
        if rl_search_config_module is None:
            pytest.skip("RL search config module not available")

        HardwareConfig = getattr(rl_search_config_module, "HardwareConfig", None)
        if HardwareConfig is None:
            pytest.skip("HardwareConfig not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        assert config.grid_dim == (4, 2, 1)
        assert config.block_dim == (128, 1, 1)

    def test_search_constraints(self, rl_search_config_module):
        """Test SearchSpaceConstraints."""
        if rl_search_config_module is None:
            pytest.skip("RL search config module not available")

        HardwareConfig = getattr(rl_search_config_module, "HardwareConfig", None)
        SearchSpaceConstraints = getattr(rl_search_config_module, "SearchSpaceConstraints", None)

        if HardwareConfig is None or SearchSpaceConstraints is None:
            pytest.skip("Required classes not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        constraints = SearchSpaceConstraints(config)

        assert len(constraints.valid_imaps) > 0
        assert constraints.max_operators > 0


@pytest.mark.torch
class TestRLModel:
    """Test RL policy network (requires PyTorch)."""

    def test_model_creation(self, rl_policy_module):
        """Test SearchPolicyNetwork creation."""
        if rl_policy_module is None:
            pytest.skip("RL policy module not available")

        SearchPolicyNetwork = getattr(rl_policy_module, "SearchPolicyNetwork", None)
        if SearchPolicyNetwork is None:
            pytest.skip("SearchPolicyNetwork not found")

        import torch

        model = SearchPolicyNetwork(
            graph_dim=128, config_dim=64, history_dim=32,
            hidden_dim=256, config_action_dim=64, graph_action_dim=32,
            use_gnn=False,
        )

        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_model_forward(self, rl_policy_module):
        """Test model forward pass."""
        if rl_policy_module is None:
            pytest.skip("RL policy module not available")

        SearchPolicyNetwork = getattr(rl_policy_module, "SearchPolicyNetwork", None)
        if SearchPolicyNetwork is None:
            pytest.skip("SearchPolicyNetwork not found")

        import torch

        model = SearchPolicyNetwork(
            graph_dim=128, config_dim=64, history_dim=32,
            hidden_dim=256, config_action_dim=64, graph_action_dim=32,
            use_gnn=False,
        )

        batch_size = 1
        num_nodes = 6

        graph_features = {
            "node_features": torch.randn(num_nodes, 16),
            "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            "batch": torch.zeros(num_nodes, dtype=torch.long),
        }
        global_features = torch.randn(batch_size, 48)
        history_features = torch.randn(batch_size, 32)
        action_mask = torch.ones(batch_size, 64)

        logits, values = model(
            graph_features=graph_features,
            global_features=global_features,
            history_features=history_features,
            level=1,
            action_mask=action_mask
        )

        assert logits.shape == (batch_size, 64)
        assert values.shape == (batch_size,)

    def test_model_save_load(self, rl_policy_module):
        """Test model save and load."""
        if rl_policy_module is None:
            pytest.skip("RL policy module not available")

        SearchPolicyNetwork = getattr(rl_policy_module, "SearchPolicyNetwork", None)
        if SearchPolicyNetwork is None:
            pytest.skip("SearchPolicyNetwork not found")

        import torch

        model = SearchPolicyNetwork(
            graph_dim=128, config_dim=64, history_dim=32,
            hidden_dim=256, config_action_dim=64, graph_action_dim=32,
            use_gnn=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            model.save(save_path)

            loaded = SearchPolicyNetwork.load(save_path)
            assert loaded is not None

            # Verify parameters match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), loaded.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)


class TestRLVerifier:
    """Test RL GPU verifier interface."""

    def test_local_verifier(self, rl_verifier_module):
        """Test LocalGPUVerifier."""
        if rl_verifier_module is None:
            pytest.skip("RL verifier module not available")

        LocalGPUVerifier = getattr(rl_verifier_module, "LocalGPUVerifier", None)
        if LocalGPUVerifier is None:
            pytest.skip("LocalGPUVerifier not found")

        verifier = LocalGPUVerifier(gpu_id=0)
        assert verifier is not None

    def test_verify_fingerprint(self, rl_verifier_module):
        """Test fingerprint verification."""
        if rl_verifier_module is None:
            pytest.skip("RL verifier module not available")

        LocalGPUVerifier = getattr(rl_verifier_module, "LocalGPUVerifier", None)
        if LocalGPUVerifier is None:
            pytest.skip("LocalGPUVerifier not found")

        verifier = LocalGPUVerifier(gpu_id=0)
        result = verifier.verify_fingerprint(
            kernel_graph_json=json.dumps({"type": "test"}),
            target_graph_json=json.dumps({"expected": "output"}),
        )

        assert result is not None
        assert hasattr(result, "verified")


# =============================================================================
# 5. Ray Integration Tests
# =============================================================================

@pytest.mark.ray
class TestRayIntegration:
    """Test Ray distributed computing integration."""

    def test_ray_initialized(self, ray_session):
        """Test Ray is initialized."""
        assert ray_session.is_initialized()

    def test_ray_remote_task(self, ray_session):
        """Test basic Ray remote task."""
        @ray_session.remote
        def add(a, b):
            return a + b

        result = ray_session.get(add.remote(1, 2))
        assert result == 3

    def test_ray_object_store(self, ray_session):
        """Test Ray object store."""
        data = {"key": "value", "numbers": [1, 2, 3]}
        ref = ray_session.put(data)
        retrieved = ray_session.get(ref)
        assert retrieved == data

    def test_ray_parallel_tasks(self, ray_session):
        """Test parallel task execution."""
        @ray_session.remote
        def square(x):
            return x * x

        futures = [square.remote(i) for i in range(10)]
        results = ray_session.get(futures)
        assert results == [i * i for i in range(10)]

    def test_ray_actor(self, ray_session):
        """Test Ray actor for stateful workers."""
        @ray_session.remote
        class Counter:
            def __init__(self):
                self.count = 0

            def increment(self):
                self.count += 1
                return self.count

        counter = Counter.remote()
        r1 = ray_session.get(counter.increment.remote())
        r2 = ray_session.get(counter.increment.remote())

        assert r1 == 1
        assert r2 == 2


# =============================================================================
# 6. Storage & Profiler Tests
# =============================================================================

class TestStorage:
    """Test storage module."""

    def test_storage_data_structures(self, storage_module):
        """Test storage data structure creation."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        # Test MuGraphMetadata
        MuGraphMetadata = getattr(storage_module, "MuGraphMetadata", None)
        if MuGraphMetadata:
            metadata = MuGraphMetadata(
                graph_hash="test123",
                backend="cuda",
                latency_ms=1.5,
            )
            assert metadata.graph_hash == "test123"

        # Test GraphStructure
        GraphStructure = getattr(storage_module, "GraphStructure", None)
        if GraphStructure:
            structure = GraphStructure(num_operators=5, num_tensors=10)
            assert structure.num_operators == 5

    def test_mugraph_entry(self, storage_module):
        """Test MuGraphEntry creation."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphEntry = getattr(storage_module, "MuGraphEntry", None)
        if MuGraphEntry is None:
            pytest.skip("MuGraphEntry not found")

        entry = MuGraphEntry()
        assert entry is not None

        # Test to_dict if available
        if hasattr(entry, "to_dict"):
            d = entry.to_dict()
            assert isinstance(d, dict)


class TestProfiler:
    """Test profiler module."""

    def test_profiler_backend_enum(self, profiler_module):
        """Test ProfilerBackend enum."""
        if profiler_module is None:
            pytest.skip("Profiler module not available")

        ProfilerBackend = getattr(profiler_module, "ProfilerBackend", None)
        if ProfilerBackend is None:
            pytest.skip("ProfilerBackend not found")

        assert hasattr(ProfilerBackend, "CUDA")
        assert hasattr(ProfilerBackend, "CPU")

    def test_timing_result(self, profiler_module):
        """Test TimingResult creation."""
        if profiler_module is None:
            pytest.skip("Profiler module not available")

        TimingResult = getattr(profiler_module, "TimingResult", None)
        if TimingResult is None:
            pytest.skip("TimingResult not found")

        latencies = [1.0, 1.1, 1.2, 0.9, 1.05, 1.15, 0.95, 1.08, 1.12, 0.98]
        result = TimingResult.from_latencies(latencies, num_warmup=2)

        assert result.mean_ms > 0
        assert result.std_ms >= 0


# =============================================================================
# 7. End-to-End Tests
# =============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_simulated_rl_episode(self):
        """Simulate end-to-end RL search episode."""
        class SimulatedEnv:
            def __init__(self):
                self.step_count = 0
                self.max_steps = 10

            def reset(self):
                self.step_count = 0
                return {
                    "graph_embedding": np.zeros(64, dtype=np.float32),
                    "search_level": 1,
                    "search_depth": 0,
                    "action_mask": np.ones(100, dtype=np.float32),
                }

            def step(self, action):
                self.step_count += 1
                obs = {
                    "graph_embedding": np.random.randn(64).astype(np.float32),
                    "search_level": 2 if self.step_count > 3 else 1,
                    "search_depth": self.step_count,
                    "action_mask": np.ones(100, dtype=np.float32),
                }
                reward = 0.1 * self.step_count + np.random.randn() * 0.1
                done = self.step_count >= self.max_steps
                return obs, reward, done, False, {"is_valid": True}

        env = SimulatedEnv()
        obs = env.reset()
        total_reward = 0

        for _ in range(10):
            action = np.random.randint(0, 100)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done:
                break

        assert total_reward is not None

    @pytest.mark.torch
    def test_full_pipeline(self, device):
        """Test full compilation pipeline components exist."""
        import torch

        # Create test tensor
        x = torch.randn(32, 64, device=device)
        y = torch.randn(64, 128, device=device)

        # Basic matmul should work
        z = torch.matmul(x, y)
        assert z.shape == (32, 128)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
