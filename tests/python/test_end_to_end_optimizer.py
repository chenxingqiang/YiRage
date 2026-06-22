#!/usr/bin/env python3
"""
Tests for End-to-End Optimizer

Tests the complete pipeline:
muGraph → Pattern Detection → COMET Search → Optimized MLIR
"""

import sys
import pytest
import json

sys.path.insert(0, '/workspace/python')
sys.path.insert(0, '/workspace/mlir/python')

from yirage.optimizer.end_to_end import (
    YirageOptimizer, OptimizationConfig, OptimizationResult,
    MuGraph, GraphOp, GraphTensor, PatternDetector,
    OptimizedMLIRGenerator, optimize_graph, get_supported_targets
)


#==============================================================================
# Test Fixtures
#==============================================================================

@pytest.fixture
def simple_matmul_graph():
    """Simple matmul graph."""
    return {
        "inputs": [
            {"id": 0, "dims": [32, 64], "dtype": "f16"},
            {"id": 1, "dims": [64, 128], "dtype": "f16"},
        ],
        "operators": [
            {
                "id": 0, "type": "matmul",
                "inputs": [0, 1],
                "outputs": [{"id": 2, "dims": [32, 128], "dtype": "f16"}]
            }
        ],
        "outputs": [
            {"id": 2, "dims": [32, 128], "dtype": "f16"}
        ]
    }


@pytest.fixture
def attention_graph():
    """Self-attention graph (Q@K → softmax → @V)."""
    return {
        "inputs": [
            {"id": 0, "dims": [1, 32, 2048, 128], "dtype": "f16"},  # Q
            {"id": 1, "dims": [1, 32, 2048, 128], "dtype": "f16"},  # K
            {"id": 2, "dims": [1, 32, 2048, 128], "dtype": "f16"},  # V
        ],
        "operators": [
            {
                "id": 0, "type": "matmul",
                "inputs": [0, 1],
                "outputs": [{"id": 3, "dims": [1, 32, 2048, 2048], "dtype": "f16"}]
            },
            {
                "id": 1, "type": "softmax",
                "inputs": [3],
                "outputs": [{"id": 4, "dims": [1, 32, 2048, 2048], "dtype": "f16"}]
            },
            {
                "id": 2, "type": "matmul",
                "inputs": [4, 2],
                "outputs": [{"id": 5, "dims": [1, 32, 2048, 128], "dtype": "f16"}]
            }
        ],
        "outputs": [
            {"id": 5, "dims": [1, 32, 2048, 128], "dtype": "f16"}
        ]
    }


@pytest.fixture
def gated_mlp_graph():
    """Gated MLP graph (SwiGLU)."""
    return {
        "inputs": [
            {"id": 0, "dims": [2048, 4096], "dtype": "f16"},
            {"id": 1, "dims": [4096, 11008], "dtype": "f16"},  # gate weight
            {"id": 2, "dims": [4096, 11008], "dtype": "f16"},  # up weight
            {"id": 3, "dims": [11008, 4096], "dtype": "f16"},  # down weight
        ],
        "operators": [
            {
                "id": 0, "type": "matmul",  # gate proj
                "inputs": [0, 1],
                "outputs": [{"id": 4, "dims": [2048, 11008], "dtype": "f16"}]
            },
            {
                "id": 1, "type": "matmul",  # up proj
                "inputs": [0, 2],
                "outputs": [{"id": 5, "dims": [2048, 11008], "dtype": "f16"}]
            },
            {
                "id": 2, "type": "silu",
                "inputs": [4],
                "outputs": [{"id": 6, "dims": [2048, 11008], "dtype": "f16"}]
            },
            {
                "id": 3, "type": "mul",
                "inputs": [6, 5],
                "outputs": [{"id": 7, "dims": [2048, 11008], "dtype": "f16"}]
            },
            {
                "id": 4, "type": "matmul",  # down proj
                "inputs": [7, 3],
                "outputs": [{"id": 8, "dims": [2048, 4096], "dtype": "f16"}]
            }
        ],
        "outputs": [
            {"id": 8, "dims": [2048, 4096], "dtype": "f16"}
        ]
    }


@pytest.fixture
def rms_norm_linear_graph():
    """RMS norm followed by linear."""
    return {
        "inputs": [
            {"id": 0, "dims": [2048, 4096], "dtype": "f16"},
            {"id": 1, "dims": [4096], "dtype": "f16"},  # gamma
            {"id": 2, "dims": [4096, 4096], "dtype": "f16"},  # weight
        ],
        "operators": [
            {
                "id": 0, "type": "rms_norm",
                "inputs": [0, 1],
                "outputs": [{"id": 3, "dims": [2048, 4096], "dtype": "f16"}]
            },
            {
                "id": 1, "type": "matmul",
                "inputs": [3, 2],
                "outputs": [{"id": 4, "dims": [2048, 4096], "dtype": "f16"}]
            }
        ],
        "outputs": [
            {"id": 4, "dims": [2048, 4096], "dtype": "f16"}
        ]
    }


#==============================================================================
# MuGraph Tests
#==============================================================================

class TestMuGraph:
    """Tests for MuGraph class."""
    
    def test_from_json_dict(self, simple_matmul_graph):
        """Test creating MuGraph from JSON dict."""
        mugraph = MuGraph.from_json(simple_matmul_graph)
        
        assert mugraph.num_ops() == 1
        assert mugraph.num_tensors() >= 2
        assert len(mugraph.input_tensor_ids) == 2
        assert len(mugraph.output_tensor_ids) == 1
    
    def test_from_json_string(self, simple_matmul_graph):
        """Test creating MuGraph from JSON string."""
        json_str = json.dumps(simple_matmul_graph)
        mugraph = MuGraph.from_json(json_str)
        
        assert mugraph.num_ops() == 1
    
    def test_to_json(self, simple_matmul_graph):
        """Test converting MuGraph back to JSON."""
        mugraph = MuGraph.from_json(simple_matmul_graph)
        json_data = mugraph.to_json()
        
        assert 'inputs' in json_data
        assert 'operators' in json_data
        assert 'outputs' in json_data
        assert len(json_data['operators']) == 1
    
    def test_attention_graph_structure(self, attention_graph):
        """Test attention graph parsing."""
        mugraph = MuGraph.from_json(attention_graph)
        
        assert mugraph.num_ops() == 3
        assert len(mugraph.input_tensor_ids) == 3


#==============================================================================
# Pattern Detection Tests
#==============================================================================

class TestPatternDetector:
    """Tests for PatternDetector class."""
    
    def test_detect_self_attention(self, attention_graph):
        """Test self-attention pattern detection."""
        mugraph = MuGraph.from_json(attention_graph)
        detector = PatternDetector(mugraph)
        
        patterns = detector.detect_all_patterns()
        
        # Should detect self-attention pattern
        attention_patterns = [p for p in patterns if p['type'] == 'SELF_ATTENTION']
        assert len(attention_patterns) >= 1
    
    def test_detect_gated_mlp(self, gated_mlp_graph):
        """Test gated MLP pattern detection."""
        mugraph = MuGraph.from_json(gated_mlp_graph)
        detector = PatternDetector(mugraph)
        
        patterns = detector.detect_all_patterns()
        
        # Should detect gated MLP pattern
        mlp_patterns = [p for p in patterns if p['type'] == 'GATED_MLP']
        assert len(mlp_patterns) >= 1
    
    def test_detect_rms_norm_linear(self, rms_norm_linear_graph):
        """Test RMS norm + linear pattern detection."""
        mugraph = MuGraph.from_json(rms_norm_linear_graph)
        detector = PatternDetector(mugraph)
        
        patterns = detector.detect_all_patterns()
        
        # Should detect RMS norm + linear pattern
        rms_patterns = [p for p in patterns if p['type'] == 'RMS_NORM_LINEAR']
        assert len(rms_patterns) >= 1
    
    def test_no_overlap_patterns(self, attention_graph):
        """Test that patterns don't overlap."""
        mugraph = MuGraph.from_json(attention_graph)
        detector = PatternDetector(mugraph)
        
        patterns = detector.detect_all_patterns()
        
        # Collect all used op indices
        all_indices = []
        for pattern in patterns:
            all_indices.extend(pattern['op_indices'])
        
        # No duplicates
        assert len(all_indices) == len(set(all_indices))


#==============================================================================
# MLIR Generation Tests
#==============================================================================

class TestOptimizedMLIRGenerator:
    """Tests for OptimizedMLIRGenerator class."""
    
    def test_generate_simple_matmul(self, simple_matmul_graph):
        """Test MLIR generation for simple matmul."""
        mugraph = MuGraph.from_json(simple_matmul_graph)
        config = OptimizationConfig(target="cuda-h100")
        
        generator = OptimizedMLIRGenerator(mugraph, [], config)
        mlir = generator.generate()
        
        assert "module {" in mlir
        assert "func.func" in mlir
        assert "yirage.matmul" in mlir
        assert "return" in mlir
    
    def test_generate_with_fusion(self, attention_graph):
        """Test MLIR generation with fused patterns."""
        mugraph = MuGraph.from_json(attention_graph)
        detector = PatternDetector(mugraph)
        patterns = detector.detect_all_patterns()
        
        config = OptimizationConfig(target="cuda-h100")
        generator = OptimizedMLIRGenerator(mugraph, patterns, config)
        mlir = generator.generate()
        
        # Should contain fused attention
        assert "yirage.attention" in mlir or "Fused pattern" in mlir
    
    def test_mlir_has_target_comment(self, simple_matmul_graph):
        """Test that generated MLIR includes target info."""
        mugraph = MuGraph.from_json(simple_matmul_graph)
        config = OptimizationConfig(target="cuda-h100")
        
        generator = OptimizedMLIRGenerator(mugraph, [], config)
        mlir = generator.generate()
        
        assert "cuda-h100" in mlir


#==============================================================================
# End-to-End Optimizer Tests
#==============================================================================

class TestYirageOptimizer:
    """Tests for YirageOptimizer class."""
    
    def test_create_optimizer(self):
        """Test creating optimizer."""
        optimizer = YirageOptimizer(target="cuda-h100")
        
        assert optimizer.target == "cuda-h100"
        assert optimizer.config is not None
    
    def test_optimize_simple_graph(self, simple_matmul_graph):
        """Test optimizing simple graph."""
        optimizer = YirageOptimizer(target="cuda-h100")
        result = optimizer.optimize(simple_matmul_graph)
        
        assert isinstance(result, OptimizationResult)
        assert result.input_graph_ops == 1
        assert len(result.optimized_mlir) > 0
        assert result.target == "cuda-h100"
    
    def test_optimize_attention_graph(self, attention_graph):
        """Test optimizing attention graph."""
        optimizer = YirageOptimizer(target="cuda-h100")
        result = optimizer.optimize(attention_graph)
        
        assert result.input_graph_ops == 3
        assert len(result.patterns_detected) >= 1
        assert "pattern_fusion" in result.optimizations_applied
    
    def test_optimize_with_custom_config(self, simple_matmul_graph):
        """Test optimization with custom config."""
        config = OptimizationConfig(
            target="rocm-mi300x",
            enable_search=False,
            mlir_opt_level=2
        )
        optimizer = YirageOptimizer(config=config)
        result = optimizer.optimize(simple_matmul_graph)
        
        assert result.target == "rocm-mi300x"
    
    def test_result_to_dict(self, attention_graph):
        """Test converting result to dict."""
        optimizer = YirageOptimizer(target="cuda-h100")
        result = optimizer.optimize(attention_graph)
        
        result_dict = result.to_dict()
        
        assert 'input_graph_ops' in result_dict
        assert 'patterns_detected' in result_dict
        assert 'target' in result_dict


#==============================================================================
# Multi-Backend Tests
#==============================================================================

class TestMultiBackend:
    """Tests for multi-backend optimization."""
    
    @pytest.mark.parametrize("target", [
        "cuda-h100", "cuda-a100", "cuda-v100",
        "rocm-mi300x", "rocm-mi250",
        "tpu-v5e", "tpu-v4",
        "ascend-910b",
        "cpu-avx512",
    ])
    def test_optimize_for_target(self, target, attention_graph):
        """Test optimization for each target."""
        optimizer = YirageOptimizer(target=target)
        result = optimizer.optimize(attention_graph)
        
        assert result.target == target
        assert len(result.optimized_mlir) > 0
        assert target in result.optimized_mlir or target.split('-')[0] in result.optimized_mlir.lower()
    
    def test_supported_targets(self):
        """Test getting supported targets."""
        targets = get_supported_targets()
        
        assert len(targets) > 10
        assert "cuda-h100" in targets
        assert "rocm-mi300x" in targets
        assert "tpu-v5e" in targets


#==============================================================================
# Convenience Function Tests
#==============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_optimize_graph_function(self, attention_graph):
        """Test optimize_graph convenience function."""
        result = optimize_graph(attention_graph, target="cuda-h100")
        
        assert isinstance(result, OptimizationResult)
        assert len(result.optimized_mlir) > 0


#==============================================================================
# Integration Tests
#==============================================================================

class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline_attention(self, attention_graph):
        """Test full pipeline for attention."""
        # 1. Parse graph
        mugraph = MuGraph.from_json(attention_graph)
        
        # 2. Detect patterns
        detector = PatternDetector(mugraph)
        patterns = detector.detect_all_patterns()
        
        assert len(patterns) >= 1
        
        # 3. Generate optimized MLIR
        config = OptimizationConfig(target="cuda-h100")
        generator = OptimizedMLIRGenerator(mugraph, patterns, config)
        mlir = generator.generate()
        
        # 4. Verify MLIR structure
        assert "module {" in mlir
        assert "func.func" in mlir
        assert "return" in mlir
        
        # 5. Verify fusion
        assert "Fused" in mlir or "yirage.attention" in mlir
    
    def test_full_pipeline_mlp(self, gated_mlp_graph):
        """Test full pipeline for gated MLP."""
        result = optimize_graph(gated_mlp_graph, target="cuda-h100")
        
        assert result.input_graph_ops == 5
        assert len(result.patterns_detected) >= 1
        
        # Verify MLIR contains MLP ops
        mlir = result.optimized_mlir
        assert "yirage" in mlir
    
    def test_transformer_block(self):
        """Test optimization of a complete transformer block."""
        # Simplified transformer block
        transformer_graph = {
            "inputs": [
                {"id": 0, "dims": [1, 2048, 4096], "dtype": "f16"},
                {"id": 1, "dims": [4096], "dtype": "f16"},  # norm weight
                {"id": 2, "dims": [4096, 4096], "dtype": "f16"},  # wq
            ],
            "operators": [
                {
                    "id": 0, "type": "rms_norm",
                    "inputs": [0, 1],
                    "outputs": [{"id": 3, "dims": [1, 2048, 4096], "dtype": "f16"}]
                },
                {
                    "id": 1, "type": "matmul",
                    "inputs": [3, 2],
                    "outputs": [{"id": 4, "dims": [1, 2048, 4096], "dtype": "f16"}]
                },
            ],
            "outputs": [
                {"id": 4, "dims": [1, 2048, 4096], "dtype": "f16"}
            ]
        }
        
        result = optimize_graph(transformer_graph, target="cuda-h100")
        
        assert result.input_graph_ops == 2
        assert "rms_norm" in result.optimized_mlir.lower() or "RMS_NORM_LINEAR" in str(result.patterns_detected)


#==============================================================================
# Performance Tests
#==============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_large_graph_optimization(self):
        """Test optimization of a larger graph."""
        # Create a graph with many ops
        graph = {
            "inputs": [
                {"id": 0, "dims": [2048, 4096], "dtype": "f16"},
                {"id": 1, "dims": [4096, 4096], "dtype": "f16"},
            ],
            "operators": [],
            "outputs": []
        }
        
        # Add 50 matmul operations
        for i in range(50):
            graph["operators"].append({
                "id": i,
                "type": "matmul",
                "inputs": [0, 1],
                "outputs": [{"id": i + 100, "dims": [2048, 4096], "dtype": "f16"}]
            })
        
        graph["outputs"] = [{"id": 149, "dims": [2048, 4096], "dtype": "f16"}]
        
        config = OptimizationConfig(
            target="cuda-h100",
            enable_search=False,  # Disable search for speed
        )
        optimizer = YirageOptimizer(config=config)
        result = optimizer.optimize(graph)
        
        assert result.input_graph_ops == 50


#==============================================================================
# Main
#==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
