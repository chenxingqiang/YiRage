# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for COMET Search Strategy.

Tests pattern detection, cost model, and search optimization.
"""

import pytest
import sys
from pathlib import Path

# Add python directory to path
python_dir = Path(__file__).parent.parent.parent / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))


class TestCOMETSearchEnums:
    """Test COMET search enums."""
    
    def test_compound_op_types(self):
        """Test compound operation type enum."""
        from yirage.search.comet_search import CompoundOpType
        
        assert CompoundOpType.NONE.value == 0
        assert CompoundOpType.GEMM_SOFTMAX.value == 1
        assert CompoundOpType.GEMM_LAYERNORM.value == 2
        assert CompoundOpType.SELF_ATTENTION.value == 3
        assert CompoundOpType.GATED_MLP.value == 4
        assert CompoundOpType.RMS_NORM_LINEAR.value == 5
    
    def test_scheduling_strategies(self):
        """Test scheduling strategy enum."""
        from yirage.search.comet_search import SchedulingStrategy
        
        assert SchedulingStrategy.SEQUENTIAL.value == 0
        assert SchedulingStrategy.PIPELINED.value == 1
        assert SchedulingStrategy.PARALLEL.value == 2
    
    def test_collective_op_types(self):
        """Test collective operation types."""
        from yirage.search.comet_search import CollectiveOpType
        
        assert CollectiveOpType.NONE.value == 0
        assert CollectiveOpType.ALLREDUCE.value == 1
        assert CollectiveOpType.ALLGATHER.value == 2
        assert CollectiveOpType.REDUCESCATTER.value == 3
        assert CollectiveOpType.BROADCAST.value == 4
    
    def test_memory_levels(self):
        """Test memory level enum."""
        from yirage.search.comet_search import MemoryLevel
        
        assert MemoryLevel.REGISTER.value == 0
        assert MemoryLevel.L1_CACHE.value == 1
        assert MemoryLevel.DRAM.value == 4


class TestCOMETSearchConfig:
    """Test COMET search configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from yirage.search.comet_search import COMETSearchConfig
        
        config = COMETSearchConfig()
        
        assert config.max_iterations == 1000
        assert config.max_fusion_depth == 5
        assert config.enable_fusion_search is True
        assert 128 in config.tile_sizes
        assert config.num_devices == 1
        assert config.objective == "minimize_latency"
    
    def test_custom_config(self):
        """Test custom configuration."""
        from yirage.search.comet_search import COMETSearchConfig, SchedulingStrategy
        
        config = COMETSearchConfig(
            max_iterations=500,
            tile_sizes=[64, 128],
            num_devices=8,
            objective="minimize_energy"
        )
        
        assert config.max_iterations == 500
        assert config.tile_sizes == [64, 128]
        assert config.num_devices == 8
        assert config.objective == "minimize_energy"


class TestCOMETCostModel:
    """Test COMET cost model."""
    
    def test_cost_model_creation(self):
        """Test cost model creation."""
        from yirage.search.comet_search import COMETCostModel, COMETSearchConfig
        
        model = COMETCostModel()
        assert model.dram_bandwidth_gbps == 900.0
        assert model.peak_tflops == 312.0
        
        config = COMETSearchConfig(dram_bandwidth_gbps=1200.0, peak_tflops=400.0)
        model2 = COMETCostModel(config)
        assert model2.dram_bandwidth_gbps == 1200.0
        assert model2.peak_tflops == 400.0
    
    def test_compute_latency_estimation(self):
        """Test compute latency estimation."""
        from yirage.search.comet_search import COMETCostModel
        
        model = COMETCostModel()
        
        flops = 1e12  # 1 TFLOP
        latency = model.estimate_compute_latency_ns(flops)
        
        # With 312 TFLOPS * 0.7 utilization = ~218 TFLOPS effective
        # 1 TFLOP / 218 TFLOPS = ~4.6 ms = 4.6e6 ns
        assert latency > 0
        assert latency < 1e9  # Less than 1 second
    
    def test_memory_latency_estimation(self):
        """Test memory latency estimation."""
        from yirage.search.comet_search import COMETCostModel, MemoryLevel
        
        model = COMETCostModel()
        
        data_bytes = 1024 * 1024  # 1 MB
        
        # DRAM transfer
        dram_latency = model.estimate_memory_latency_ns(
            data_bytes, MemoryLevel.DRAM, MemoryLevel.REGISTER
        )
        
        # On-chip transfer
        onchip_latency = model.estimate_memory_latency_ns(
            data_bytes, MemoryLevel.L1_CACHE, MemoryLevel.REGISTER
        )
        
        # On-chip should be faster
        assert onchip_latency < dram_latency
        assert dram_latency > 0
    
    def test_collective_latency_estimation(self):
        """Test collective communication latency estimation."""
        from yirage.search.comet_search import COMETCostModel, CollectiveOpType
        
        model = COMETCostModel()
        data_bytes = 1024 * 1024  # 1 MB
        
        # Single device should have zero latency
        single = model.estimate_collective_latency_ns(
            CollectiveOpType.ALLREDUCE, data_bytes, 1
        )
        assert single == 0.0
        
        # Multi-device should have positive latency
        multi = model.estimate_collective_latency_ns(
            CollectiveOpType.ALLREDUCE, data_bytes, 8
        )
        assert multi > 0
        
        # All-reduce typically more expensive than broadcast
        bcast = model.estimate_collective_latency_ns(
            CollectiveOpType.BROADCAST, data_bytes, 8
        )
        assert bcast > 0
    
    def test_scheduling_overhead_estimation(self):
        """Test scheduling overhead estimation."""
        from yirage.search.comet_search import COMETCostModel, SchedulingStrategy
        
        model = COMETCostModel()
        
        seq = model.estimate_scheduling_overhead_ns(SchedulingStrategy.SEQUENTIAL, 5)
        pipe = model.estimate_scheduling_overhead_ns(SchedulingStrategy.PIPELINED, 5)
        par = model.estimate_scheduling_overhead_ns(SchedulingStrategy.PARALLEL, 5)
        
        # Sequential has least overhead
        assert seq < pipe < par
    
    def test_gemm_softmax_latency(self):
        """Test GEMM-Softmax latency estimation."""
        from yirage.search.comet_search import COMETCostModel, TileConfig
        
        model = COMETCostModel()
        tile_config = TileConfig(128, 128, 64)
        
        fused = model.estimate_gemm_softmax_latency_ns(1024, 1024, 1024, tile_config, fused=True)
        unfused = model.estimate_gemm_softmax_latency_ns(1024, 1024, 1024, tile_config, fused=False)
        
        # Fused should be faster (less memory traffic)
        assert fused <= unfused
        assert fused > 0
    
    def test_self_attention_latency(self):
        """Test self-attention latency estimation."""
        from yirage.search.comet_search import COMETCostModel, TileConfig
        
        model = COMETCostModel()
        tile_config = TileConfig(64, 64, 64)
        
        fused = model.estimate_self_attention_latency_ns(
            1, 8, 512, 64, tile_config, fused=True
        )
        unfused = model.estimate_self_attention_latency_ns(
            1, 8, 512, 64, tile_config, fused=False
        )
        
        assert fused > 0
        assert unfused > 0
    
    def test_score_computation(self):
        """Test optimization score computation."""
        from yirage.search.comet_search import COMETCostModel
        
        model = COMETCostModel()
        
        # Lower latency should give higher score
        score_fast = model.compute_score(1e6, 1e9, "minimize_latency")
        score_slow = model.compute_score(1e7, 1e9, "minimize_latency")
        assert score_fast > score_slow
        
        # Balance should consider both
        score_balance = model.compute_score(1e6, 1e9, "balance", energy_weight=0.5)
        assert score_balance > 0


class TestPatternDetection:
    """Test compound pattern detection."""
    
    def test_detect_gemm_softmax(self):
        """Test GEMM-Softmax pattern detection."""
        from yirage.search.comet_search import detect_compound_patterns, CompoundOpType
        
        op_types = ["matmul", "exp", "reduction", "div"]
        patterns = detect_compound_patterns(op_types)
        
        assert len(patterns) >= 1
        assert any(p.op_type == CompoundOpType.GEMM_SOFTMAX for p in patterns)
    
    def test_detect_self_attention(self):
        """Test self-attention pattern detection."""
        from yirage.search.comet_search import detect_compound_patterns, CompoundOpType
        
        op_types = ["matmul", "exp", "reduction", "div", "matmul"]
        patterns = detect_compound_patterns(op_types)
        
        assert len(patterns) >= 1
        # Should detect self-attention (has two matmuls with softmax)
        attention_patterns = [p for p in patterns if p.op_type == CompoundOpType.SELF_ATTENTION]
        assert len(attention_patterns) >= 1
    
    def test_detect_gated_mlp(self):
        """Test Gated MLP pattern detection."""
        from yirage.search.comet_search import detect_compound_patterns, CompoundOpType
        
        op_types = ["matmul", "silu", "mul", "matmul"]
        patterns = detect_compound_patterns(op_types)
        
        assert len(patterns) >= 1
        gated_patterns = [p for p in patterns if p.op_type == CompoundOpType.GATED_MLP]
        assert len(gated_patterns) >= 1
    
    def test_no_patterns(self):
        """Test when no compound patterns exist."""
        from yirage.search.comet_search import detect_compound_patterns
        
        op_types = ["add", "sub", "relu"]
        patterns = detect_compound_patterns(op_types)
        
        # No compound patterns in simple element-wise ops
        assert len(patterns) == 0
    
    def test_pattern_fusion_benefit(self):
        """Test pattern fusion benefit calculation."""
        from yirage.search.comet_search import CompoundPattern, CompoundOpType
        
        pattern = CompoundPattern(
            op_type=CompoundOpType.GEMM_SOFTMAX,
            op_indices=[0, 1, 2, 3],
            memory_reduction_ratio=0.5,
            latency_reduction_ratio=0.3
        )
        
        benefit = pattern.get_fusion_benefit()
        assert benefit == 0.6 * 0.5 + 0.4 * 0.3
        assert benefit > 0


class TestCOMETSearchStrategy:
    """Test COMET search strategy."""
    
    def test_strategy_creation(self):
        """Test search strategy creation."""
        from yirage.search.comet_search import COMETSearchStrategy, COMETSearchConfig
        
        strategy = COMETSearchStrategy()
        assert strategy.config is not None
        
        config = COMETSearchConfig(max_iterations=100)
        strategy2 = COMETSearchStrategy(config)
        assert strategy2.config.max_iterations == 100
    
    def test_search_gemm_softmax(self):
        """Test search for GEMM-Softmax optimization."""
        from yirage.search.comet_search import COMETSearchStrategy, CompoundOpType
        
        strategy = COMETSearchStrategy()
        
        op_types = ["matmul", "exp", "reduction", "div"]
        problem_dims = {"M": 1024, "K": 1024, "N": 1024}
        
        result = strategy.search(op_types, problem_dims)
        
        assert result is not None
        assert result.score > 0
        assert result.pattern.op_type == CompoundOpType.GEMM_SOFTMAX
        assert result.latency_ns > 0
    
    def test_search_self_attention(self):
        """Test search for self-attention optimization."""
        from yirage.search.comet_search import COMETSearchStrategy
        
        strategy = COMETSearchStrategy()
        
        op_types = ["matmul", "exp", "reduction", "div", "matmul"]
        problem_dims = {
            "M": 512, "K": 512, "N": 512,
            "batch": 1, "heads": 8, "seq_len": 512, "head_dim": 64
        }
        
        result = strategy.search(op_types, problem_dims)
        
        assert result is not None
        assert result.score > 0
    
    def test_search_statistics(self):
        """Test search statistics collection."""
        from yirage.search.comet_search import COMETSearchStrategy
        
        strategy = COMETSearchStrategy()
        
        op_types = ["matmul", "exp", "reduction", "div"]
        problem_dims = {"M": 256, "K": 256, "N": 256}
        
        strategy.search(op_types, problem_dims)
        stats = strategy.get_statistics()
        
        assert stats["patterns_detected"] > 0
        assert stats["candidates_generated"] > 0
        assert stats["candidates_evaluated"] > 0
        assert stats["best_score"] > 0
    
    def test_tile_config_generation(self):
        """Test tile configuration generation."""
        from yirage.search.comet_search import COMETSearchStrategy, COMETSearchConfig
        
        config = COMETSearchConfig(tile_sizes=[64, 128])
        strategy = COMETSearchStrategy(config)
        
        problem_dims = {"M": 256, "K": 256, "N": 256}
        tiles = strategy._generate_tile_configs(problem_dims)
        
        # Should generate combinations of valid tile sizes
        assert len(tiles) > 0
        for tile in tiles:
            assert tile.tile_m in [64, 128]
            assert tile.tile_n in [64, 128]
            assert tile.tile_k in [64, 128]


class TestOptimizeCompoundGraph:
    """Test high-level optimization API."""
    
    def test_optimize_gemm_softmax(self):
        """Test high-level optimization for GEMM-Softmax."""
        from yirage.search.comet_search import optimize_compound_graph
        
        op_types = ["matmul", "exp", "reduction", "div"]
        problem_dims = {"M": 1024, "K": 1024, "N": 1024}
        
        result = optimize_compound_graph(op_types, problem_dims)
        
        assert result["success"] is True
        assert "tile_config" in result
        assert "scheduling" in result
        assert result["latency_ns"] > 0
    
    def test_optimize_with_custom_config(self):
        """Test optimization with custom configuration."""
        from yirage.search.comet_search import optimize_compound_graph, COMETSearchConfig
        
        config = COMETSearchConfig(
            tile_sizes=[128, 256],
            objective="balance",
            energy_weight=0.5
        )
        
        op_types = ["matmul", "exp", "reduction", "div"]
        problem_dims = {"M": 512, "K": 512, "N": 512}
        
        result = optimize_compound_graph(op_types, problem_dims, config)
        
        assert result["success"] is True
        assert result["tile_config"]["tile_m"] in [128, 256]
    
    def test_optimize_no_patterns(self):
        """Test optimization when no patterns are found."""
        from yirage.search.comet_search import optimize_compound_graph
        
        op_types = ["add", "relu", "sub"]
        problem_dims = {"M": 1024, "K": 1024, "N": 1024}
        
        result = optimize_compound_graph(op_types, problem_dims)
        
        # Should return default/empty result
        assert result is not None


class TestTileConfig:
    """Test tile configuration dataclass."""
    
    def test_tile_config_creation(self):
        """Test tile configuration creation."""
        from yirage.search.comet_search import TileConfig
        
        config = TileConfig(128, 256, 64)
        assert config.tile_m == 128
        assert config.tile_n == 256
        assert config.tile_k == 64
    
    def test_tile_config_iteration(self):
        """Test tile configuration iteration."""
        from yirage.search.comet_search import TileConfig
        
        config = TileConfig(128, 256, 64)
        values = list(config)
        
        assert values == [128, 256, 64]
    
    def test_tile_config_hash(self):
        """Test tile configuration hashing."""
        from yirage.search.comet_search import TileConfig
        
        config1 = TileConfig(128, 128, 64)
        config2 = TileConfig(128, 128, 64)
        config3 = TileConfig(256, 128, 64)
        
        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)


class TestDistributedSearch:
    """Test distributed search with collectives."""
    
    def test_multi_device_search(self):
        """Test search with multiple devices."""
        from yirage.search.comet_search import (
            COMETSearchStrategy, COMETSearchConfig
        )
        
        config = COMETSearchConfig(
            num_devices=8,
            optimize_collectives=True
        )
        strategy = COMETSearchStrategy(config)
        
        op_types = ["matmul", "exp", "reduction", "div"]
        problem_dims = {"M": 2048, "K": 2048, "N": 2048}
        
        result = strategy.search(op_types, problem_dims)
        
        assert result is not None
        # Distributed execution might add collective overhead
        assert result.latency_ns > 0


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_empty_tile_sizes_raises(self):
        """Test that empty tile_sizes raises ValueError."""
        from yirage.search.comet_search import COMETSearchStrategy, COMETSearchConfig
        
        config = COMETSearchConfig(tile_sizes=[])
        
        with pytest.raises(ValueError, match="tile_sizes cannot be empty"):
            COMETSearchStrategy(config)
    
    def test_invalid_num_devices_raises(self):
        """Test that invalid num_devices raises ValueError."""
        from yirage.search.comet_search import COMETSearchStrategy, COMETSearchConfig
        
        config = COMETSearchConfig(num_devices=0)
        
        with pytest.raises(ValueError, match="num_devices must be >= 1"):
            COMETSearchStrategy(config)
    
    def test_invalid_bandwidth_raises(self):
        """Test that invalid bandwidth raises ValueError."""
        from yirage.search.comet_search import COMETSearchStrategy, COMETSearchConfig
        
        config = COMETSearchConfig(dram_bandwidth_gbps=-100)
        
        with pytest.raises(ValueError, match="dram_bandwidth_gbps must be positive"):
            COMETSearchStrategy(config)
    
    def test_invalid_energy_weight_raises(self):
        """Test that invalid energy_weight raises ValueError."""
        from yirage.search.comet_search import COMETSearchStrategy, COMETSearchConfig
        
        config = COMETSearchConfig(energy_weight=1.5)
        
        with pytest.raises(ValueError, match="energy_weight must be in"):
            COMETSearchStrategy(config)


class TestP2PCollectives:
    """Test P2P collective operations."""
    
    def test_p2p_send_latency(self):
        """Test P2P send latency estimation."""
        from yirage.search.comet_search import COMETCostModel, CollectiveOpType
        
        model = COMETCostModel()
        
        latency = model.estimate_collective_latency_ns(
            CollectiveOpType.P2P_SEND, 1024 * 1024, 2
        )
        
        assert latency > 0
    
    def test_p2p_recv_latency(self):
        """Test P2P recv latency estimation."""
        from yirage.search.comet_search import COMETCostModel, CollectiveOpType
        
        model = COMETCostModel()
        
        latency = model.estimate_collective_latency_ns(
            CollectiveOpType.P2P_RECV, 1024 * 1024, 2
        )
        
        assert latency > 0


class TestTimeoutHandling:
    """Test timeout handling in search."""
    
    def test_search_respects_timeout(self):
        """Test that search respects timeout."""
        from yirage.search.comet_search import COMETSearchStrategy, COMETSearchConfig
        import time
        
        # Very short timeout
        config = COMETSearchConfig(
            timeout_seconds=0.001,  # 1ms timeout
            tile_sizes=[32, 64, 128, 256, 512]  # Many combinations
        )
        strategy = COMETSearchStrategy(config)
        
        op_types = ["matmul", "exp", "reduction", "div"]
        problem_dims = {"M": 1024, "K": 1024, "N": 1024}
        
        start = time.time()
        result = strategy.search(op_types, problem_dims)
        elapsed = time.time() - start
        
        # Should return quickly (well under 1 second)
        assert elapsed < 1.0
        # Should still return a valid result
        assert result is not None


class TestPatternDeduplication:
    """Test that patterns don't overlap."""
    
    def test_no_duplicate_ops_in_patterns(self):
        """Test that the same op isn't in multiple patterns."""
        from yirage.search.comet_search import detect_compound_patterns
        
        # This sequence could match multiple patterns without proper deduplication
        op_types = ["matmul", "exp", "reduction", "div", "matmul", "silu", "mul"]
        
        patterns = detect_compound_patterns(op_types)
        
        # Collect all used indices
        all_indices = []
        for p in patterns:
            all_indices.extend(p.op_indices)
        
        # Check for duplicates
        assert len(all_indices) == len(set(all_indices)), \
            "Patterns should not share operator indices"


class TestLayerNormPatternFix:
    """Test fixed LayerNorm pattern detection."""
    
    def test_layernorm_requires_scale(self):
        """Test that LayerNorm detection requires scale op."""
        from yirage.search.comet_search import detect_compound_patterns, CompoundOpType
        
        # Only reduction, no scale - should NOT match LayerNorm
        op_types = ["matmul", "reduction"]
        patterns = detect_compound_patterns(op_types)
        
        layernorm_patterns = [p for p in patterns if p.op_type == CompoundOpType.GEMM_LAYERNORM]
        assert len(layernorm_patterns) == 0
        
        # With reduction AND scale - should match
        op_types = ["matmul", "reduction", "mul", "add"]
        patterns = detect_compound_patterns(op_types)
        
        layernorm_patterns = [p for p in patterns if p.op_type == CompoundOpType.GEMM_LAYERNORM]
        assert len(layernorm_patterns) >= 1


class TestBackendConfig:
    """Test backend-specific COMET configurations for ALL backends."""
    
    # =========================================================================
    # NVIDIA CUDA Backends
    # =========================================================================
    def test_cuda_h100_config(self):
        """Test CUDA H100 (Hopper) backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("cuda", "h100")
        
        assert config.dram_bandwidth_gbps == 3350.0  # HBM3
        assert config.peak_tflops == 989.0  # FP16 Tensor Core
        assert config.noc_bandwidth_gbps == 900.0  # NVLink 4.0
        assert 128 in config.tile_sizes
        assert config.optimize_collectives is True
    
    def test_cuda_a100_config(self):
        """Test CUDA A100 (Ampere) backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("cuda", "a100")
        
        assert config.dram_bandwidth_gbps == 2039.0  # HBM2e
        assert config.peak_tflops == 312.0
        assert config.noc_bandwidth_gbps == 600.0  # NVLink 3.0
        assert 128 in config.tile_sizes
    
    def test_cuda_v100_config(self):
        """Test CUDA V100 (Volta) backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("cuda", "v100")
        
        assert config.dram_bandwidth_gbps == 900.0  # HBM2
        assert config.peak_tflops == 125.0
        assert 64 in config.tile_sizes
    
    def test_cuda_default_config(self):
        """Test CUDA default configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("cuda")
        
        # Default should be A100
        assert config.peak_tflops == 312.0
    
    # =========================================================================
    # AMD ROCm Backends
    # =========================================================================
    def test_rocm_mi300x_config(self):
        """Test ROCm MI300X backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("rocm", "mi300x")
        
        assert config.dram_bandwidth_gbps == 5300.0  # HBM3 (8 stacks)
        assert config.peak_tflops == 1307.0  # FP16
        assert config.noc_bandwidth_gbps == 896.0  # Infinity Fabric
        assert config.optimize_collectives is True
    
    def test_rocm_mi250x_config(self):
        """Test ROCm MI250X backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("rocm", "mi250x")
        
        assert config.dram_bandwidth_gbps == 3200.0
        assert config.peak_tflops == 383.0
    
    # =========================================================================
    # Intel XPU Backends
    # =========================================================================
    def test_xpu_pvc_config(self):
        """Test Intel Ponte Vecchio (XPU) backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("xpu", "pvc")
        
        assert config.dram_bandwidth_gbps == 3200.0  # HBM2e
        assert config.peak_tflops == 420.0  # FP16 with XMX
        assert config.noc_bandwidth_gbps == 200.0  # Xe Link
        assert config.optimize_collectives is True
    
    def test_xpu_default_config(self):
        """Test XPU default configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("xpu")
        
        assert config.peak_tflops == 420.0
    
    # =========================================================================
    # Huawei Ascend Backends
    # =========================================================================
    def test_ascend_910b_config(self):
        """Test Huawei Ascend 910B backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("ascend", "910b")
        
        assert config.dram_bandwidth_gbps == 1600.0  # HBM2e
        assert config.peak_tflops == 320.0  # FP16
        assert config.noc_bandwidth_gbps == 392.0  # HCCS
        assert config.optimize_collectives is True
        assert config.max_fusion_depth >= 5
    
    # =========================================================================
    # Google TPU Backends
    # =========================================================================
    def test_tpu_v5e_config(self):
        """Test Google TPU v5e backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("tpu", "v5e")
        
        assert config.dram_bandwidth_gbps == 1600.0
        assert config.peak_tflops == 197.0  # BF16
        assert config.noc_bandwidth_gbps == 1600.0  # ICI
        # TPU prefers larger tiles for systolic array
        assert 128 in config.tile_sizes or 256 in config.tile_sizes
        assert config.max_fusion_depth >= 8  # XLA aggressive fusion
    
    def test_tpu_v4_config(self):
        """Test Google TPU v4 backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("tpu", "v4")
        
        assert config.peak_tflops == 275.0
        assert config.noc_bandwidth_gbps == 1200.0
    
    # =========================================================================
    # MetaX MACA Backends
    # =========================================================================
    def test_maca_mxc500_config(self):
        """Test MetaX MXC500 (MACA) backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("maca", "mxc500")
        
        assert config.dram_bandwidth_gbps == 2000.0  # HBM2e
        assert config.peak_tflops == 256.0  # FP16
        assert config.noc_bandwidth_gbps == 400.0
        assert config.optimize_collectives is True
    
    def test_maca_default_config(self):
        """Test MACA default configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("maca")
        
        assert config.peak_tflops == 256.0
    
    # =========================================================================
    # Apple MPS Backends
    # =========================================================================
    def test_mps_m3_max_config(self):
        """Test Apple M3 Max (MPS) backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("mps", "m3_max")
        
        assert config.dram_bandwidth_gbps == 400.0  # Unified memory
        assert config.peak_tflops == 14.2  # FP32
        # Apple Silicon is single device
        assert config.optimize_collectives is False
        assert config.max_fusion_depth == 4
    
    def test_mps_m2_ultra_config(self):
        """Test Apple M2 Ultra (MPS) backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("mps", "m2_ultra")
        
        assert config.dram_bandwidth_gbps == 800.0
        assert config.peak_tflops == 27.2
    
    # =========================================================================
    # CPU Backends
    # =========================================================================
    def test_cpu_xeon_config(self):
        """Test Intel Xeon CPU backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("cpu", "xeon")
        
        assert config.dram_bandwidth_gbps == 200.0  # DDR5
        assert config.peak_tflops == 4.0  # AVX-512
        assert config.optimize_collectives is False
        # Cache-friendly tiles
        assert 64 in config.tile_sizes
    
    def test_cpu_epyc_config(self):
        """Test AMD EPYC CPU backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("cpu", "epyc")
        
        assert config.dram_bandwidth_gbps == 460.0  # Higher memory bandwidth
        assert config.peak_tflops == 5.0
    
    def test_cpu_default_config(self):
        """Test CPU default configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("cpu")
        
        assert config.dram_bandwidth_gbps < 500  # CPU has lower bandwidth
        assert config.optimize_collectives is False
    
    # =========================================================================
    # FPGA Backends
    # =========================================================================
    def test_fpga_alveo_config(self):
        """Test Xilinx Alveo FPGA backend configuration."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("fpga", "alveo")
        
        assert config.dram_bandwidth_gbps == 77.0  # DDR4
        assert config.peak_tflops == 4.0
        # FPGA can do very deep fusion (dataflow)
        assert config.max_fusion_depth >= 10
        # Smaller tiles for BRAM
        assert 16 in config.tile_sizes or 32 in config.tile_sizes
    
    # =========================================================================
    # Utility Tests
    # =========================================================================
    def test_list_all_backends(self):
        """Test listing all supported backends."""
        from yirage.search.backend_config import list_supported_backends, BACKEND_PROFILES
        
        backends = list_supported_backends()
        
        # Verify all 9 backend families
        assert "cuda" in backends
        assert "rocm" in backends
        assert "xpu" in backends
        assert "ascend" in backends
        assert "tpu" in backends
        assert "maca" in backends
        assert "mps" in backends
        assert "cpu" in backends
        assert "fpga" in backends
        
        # Verify variants
        assert len(backends["cuda"]) >= 3  # h100, a100, v100
        assert len(backends["rocm"]) >= 2  # mi300x, mi250x
        assert len(backends["tpu"]) >= 2   # v4, v5e
        assert len(backends["mps"]) >= 2   # m2_ultra, m3_max
        assert len(backends["cpu"]) >= 2   # xeon, epyc
        
        # Verify total count
        assert len(BACKEND_PROFILES) >= 14
    
    def test_unknown_backend_returns_default(self):
        """Test that unknown backend returns default config."""
        from yirage.search.backend_config import get_backend_config
        
        config = get_backend_config("unknown_backend")
        
        # Should return default config without error
        assert config is not None
        assert config.max_iterations == 1000  # Default value
    
    def test_all_backends_have_valid_configs(self):
        """Test that all backend profiles produce valid configs."""
        from yirage.search.backend_config import BACKEND_PROFILES, get_backend_config
        
        for profile_key in BACKEND_PROFILES.keys():
            parts = profile_key.split("_", 1)
            backend = parts[0]
            variant = parts[1] if len(parts) > 1 else None
            
            config = get_backend_config(backend, variant)
            
            # Validate config
            assert config.dram_bandwidth_gbps > 0, f"{profile_key}: invalid DRAM bandwidth"
            assert config.peak_tflops > 0, f"{profile_key}: invalid peak TFLOPS"
            assert len(config.tile_sizes) > 0, f"{profile_key}: no tile sizes"
            assert config.max_fusion_depth > 0, f"{profile_key}: invalid fusion depth"


class TestCOMETBackendIntegration:
    """Test COMET search with ALL backend configs."""
    
    def _run_search_test(self, backend: str, variant: str = None):
        """Helper to run search test for a backend."""
        from yirage.search import COMETSearchStrategy, get_backend_config
        
        config = get_backend_config(backend, variant)
        strategy = COMETSearchStrategy(config)
        
        op_types = ["matmul", "exp", "reduction", "div"]
        problem_dims = {"M": 1024, "K": 1024, "N": 1024}
        
        result = strategy.search(op_types, problem_dims)
        
        assert result is not None, f"{backend}/{variant}: search returned None"
        assert result.score > 0, f"{backend}/{variant}: score is zero"
        return result
    
    # =========================================================================
    # CUDA Backend Tests
    # =========================================================================
    def test_cuda_h100_search(self):
        """Test COMET search with CUDA H100 config."""
        result = self._run_search_test("cuda", "h100")
        assert result.latency_ns > 0
    
    def test_cuda_a100_search(self):
        """Test COMET search with CUDA A100 config."""
        result = self._run_search_test("cuda", "a100")
        assert result.latency_ns > 0
    
    def test_cuda_v100_search(self):
        """Test COMET search with CUDA V100 config."""
        result = self._run_search_test("cuda", "v100")
        assert result.latency_ns > 0
    
    # =========================================================================
    # ROCm Backend Tests
    # =========================================================================
    def test_rocm_mi300x_search(self):
        """Test COMET search with ROCm MI300X config."""
        result = self._run_search_test("rocm", "mi300x")
        assert result.latency_ns > 0
    
    def test_rocm_mi250x_search(self):
        """Test COMET search with ROCm MI250X config."""
        result = self._run_search_test("rocm", "mi250x")
        assert result.latency_ns > 0
    
    # =========================================================================
    # XPU Backend Tests
    # =========================================================================
    def test_xpu_pvc_search(self):
        """Test COMET search with Intel XPU (Ponte Vecchio) config."""
        result = self._run_search_test("xpu", "pvc")
        assert result.latency_ns > 0
    
    # =========================================================================
    # Ascend Backend Tests
    # =========================================================================
    def test_ascend_910b_search(self):
        """Test COMET search with Huawei Ascend 910B config."""
        result = self._run_search_test("ascend", "910b")
        assert result.latency_ns > 0
    
    # =========================================================================
    # TPU Backend Tests
    # =========================================================================
    def test_tpu_v5e_search(self):
        """Test COMET search with Google TPU v5e config."""
        result = self._run_search_test("tpu", "v5e")
        assert result.latency_ns > 0
    
    def test_tpu_v4_search(self):
        """Test COMET search with Google TPU v4 config."""
        result = self._run_search_test("tpu", "v4")
        assert result.latency_ns > 0
    
    # =========================================================================
    # MACA Backend Tests
    # =========================================================================
    def test_maca_mxc500_search(self):
        """Test COMET search with MetaX MXC500 (MACA) config."""
        result = self._run_search_test("maca", "mxc500")
        assert result.latency_ns > 0
    
    # =========================================================================
    # MPS Backend Tests
    # =========================================================================
    def test_mps_m3_max_search(self):
        """Test COMET search with Apple M3 Max (MPS) config."""
        result = self._run_search_test("mps", "m3_max")
        assert result.latency_ns > 0
    
    def test_mps_m2_ultra_search(self):
        """Test COMET search with Apple M2 Ultra (MPS) config."""
        result = self._run_search_test("mps", "m2_ultra")
        assert result.latency_ns > 0
    
    # =========================================================================
    # CPU Backend Tests
    # =========================================================================
    def test_cpu_xeon_search(self):
        """Test COMET search with Intel Xeon CPU config."""
        result = self._run_search_test("cpu", "xeon")
        assert result.latency_ns > 0
    
    def test_cpu_epyc_search(self):
        """Test COMET search with AMD EPYC CPU config."""
        result = self._run_search_test("cpu", "epyc")
        assert result.latency_ns > 0
    
    # =========================================================================
    # FPGA Backend Tests
    # =========================================================================
    def test_fpga_alveo_search(self):
        """Test COMET search with Xilinx Alveo FPGA config."""
        result = self._run_search_test("fpga", "alveo")
        assert result.latency_ns > 0
    
    # =========================================================================
    # Cross-Backend Comparison Tests
    # =========================================================================
    def test_different_backends_different_tiles(self):
        """Test that different backends produce different tile configs."""
        from yirage.search import get_backend_config
        
        cuda_config = get_backend_config("cuda", "h100")
        fpga_config = get_backend_config("fpga")
        
        # CUDA H100 should have different tile sizes than FPGA
        assert cuda_config.tile_sizes != fpga_config.tile_sizes
        
        # FPGA uses smaller tiles
        assert min(fpga_config.tile_sizes) < min(cuda_config.tile_sizes)
    
    def test_gpu_faster_than_cpu(self):
        """Test that GPU configs estimate lower latency than CPU for large problems."""
        from yirage.search import COMETSearchStrategy, get_backend_config
        
        cuda_config = get_backend_config("cuda", "a100")
        cpu_config = get_backend_config("cpu", "xeon")
        
        op_types = ["matmul", "exp", "reduction", "div"]
        problem_dims = {"M": 4096, "K": 4096, "N": 4096}
        
        cuda_strategy = COMETSearchStrategy(cuda_config)
        cpu_strategy = COMETSearchStrategy(cpu_config)
        
        cuda_result = cuda_strategy.search(op_types, problem_dims)
        cpu_result = cpu_strategy.search(op_types, problem_dims)
        
        # GPU should have lower latency for large GEMM
        assert cuda_result.latency_ns < cpu_result.latency_ns
    
    def test_all_backends_search_self_attention(self):
        """Test that all backends can optimize self-attention pattern."""
        from yirage.search import COMETSearchStrategy, get_backend_config, BACKEND_PROFILES
        
        op_types = ["matmul", "exp", "reduction", "div", "matmul"]
        problem_dims = {
            "M": 512, "K": 512, "N": 512,
            "batch": 1, "heads": 8, "seq_len": 512, "head_dim": 64
        }
        
        results = {}
        for profile_key in BACKEND_PROFILES.keys():
            parts = profile_key.split("_", 1)
            backend = parts[0]
            variant = parts[1] if len(parts) > 1 else None
            
            config = get_backend_config(backend, variant)
            strategy = COMETSearchStrategy(config)
            result = strategy.search(op_types, problem_dims)
            
            assert result is not None, f"{profile_key}: search failed"
            assert result.score > 0, f"{profile_key}: zero score"
            results[profile_key] = result.latency_ns
        
        # Verify we tested all backends
        assert len(results) == len(BACKEND_PROFILES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
