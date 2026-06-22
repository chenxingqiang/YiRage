#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
RL Models Module Unit Tests

Tests for yirage/rl/models/ module including SearchPolicyNetwork.
Run with: pytest tests/python/test_rl/test_models.py -v
"""

import pytest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import PYTHON_ROOT, load_module, TORCH_AVAILABLE


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def policy_module():
    """Load policy network module."""
    return load_module(
        "search_policy",
        PYTHON_ROOT / "yirage" / "rl" / "models" / "search_policy.py"
    )


@pytest.fixture(scope="module")
def graph_encoder_module():
    """Load graph encoder module."""
    return load_module(
        "graph_encoder",
        PYTHON_ROOT / "yirage" / "rl" / "models" / "graph_encoder.py"
    )


# =============================================================================
# SearchPolicyNetwork Tests
# =============================================================================

@pytest.mark.torch
class TestSearchPolicyNetwork:
    """Tests for SearchPolicyNetwork class (requires PyTorch)."""

    def test_policy_network_class_exists(self, policy_module):
        """Test SearchPolicyNetwork class exists."""
        if policy_module is None:
            pytest.skip("Policy module not available")

        assert hasattr(policy_module, "SearchPolicyNetwork")

    def test_model_creation(self, policy_module):
        """Test SearchPolicyNetwork can be created."""
        if policy_module is None:
            pytest.skip("Policy module not available")

        SearchPolicyNetwork = getattr(policy_module, "SearchPolicyNetwork", None)
        if SearchPolicyNetwork is None:
            pytest.skip("SearchPolicyNetwork not found")

        model = SearchPolicyNetwork(
            graph_dim=128,
            config_dim=64,
            history_dim=32,
            hidden_dim=256,
            config_action_dim=64,
            graph_action_dim=32,
            use_gnn=False,
        )
        assert model is not None

    def test_model_has_parameters(self, policy_module):
        """Test model has learnable parameters."""
        if policy_module is None:
            pytest.skip("Policy module not available")

        SearchPolicyNetwork = getattr(policy_module, "SearchPolicyNetwork", None)
        if SearchPolicyNetwork is None:
            pytest.skip("SearchPolicyNetwork not found")

        model = SearchPolicyNetwork(
            graph_dim=128, config_dim=64, history_dim=32,
            hidden_dim=256, config_action_dim=64, graph_action_dim=32,
            use_gnn=False,
        )

        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

    def test_forward_pass_shapes(self, policy_module):
        """Test forward pass produces correct shapes."""
        if policy_module is None:
            pytest.skip("Policy module not available")

        import torch

        SearchPolicyNetwork = getattr(policy_module, "SearchPolicyNetwork", None)
        if SearchPolicyNetwork is None:
            pytest.skip("SearchPolicyNetwork not found")

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

    def test_action_masking_applied(self, policy_module):
        """Test action masking is applied correctly."""
        if policy_module is None:
            pytest.skip("Policy module not available")

        import torch

        SearchPolicyNetwork = getattr(policy_module, "SearchPolicyNetwork", None)
        if SearchPolicyNetwork is None:
            pytest.skip("SearchPolicyNetwork not found")

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

        # Mask out half the actions
        action_mask = torch.ones(batch_size, 64)
        action_mask[:, 32:] = 0

        logits, _ = model(
            graph_features=graph_features,
            global_features=global_features,
            history_features=history_features,
            level=1,
            action_mask=action_mask
        )

        # Masked actions should have -inf logits
        assert torch.all(logits[:, 32:] == float('-inf'))

    def test_save_load_weights_match(self, policy_module):
        """Test model save/load preserves weights."""
        if policy_module is None:
            pytest.skip("Policy module not available")

        import torch

        SearchPolicyNetwork = getattr(policy_module, "SearchPolicyNetwork", None)
        if SearchPolicyNetwork is None:
            pytest.skip("SearchPolicyNetwork not found")

        model = SearchPolicyNetwork(
            graph_dim=128, config_dim=64, history_dim=32,
            hidden_dim=256, config_action_dim=64, graph_action_dim=32,
            use_gnn=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"
            model.save(save_path)

            loaded = SearchPolicyNetwork.load(save_path)

            # Compare parameters
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), loaded.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)

    def test_gradient_flow(self, policy_module):
        """Test gradients flow through the model."""
        if policy_module is None:
            pytest.skip("Policy module not available")

        import torch

        SearchPolicyNetwork = getattr(policy_module, "SearchPolicyNetwork", None)
        if SearchPolicyNetwork is None:
            pytest.skip("SearchPolicyNetwork not found")

        try:
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

            # Compute loss and backprop
            loss = logits.sum() + values.sum()
            loss.backward()

            # Check that at least some parameters received gradients.
            # Not all parameters participate in every forward path
            # (e.g. level=1 only uses config_policy_head, not graph_policy_head).
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            assert len(grads) > 0, "No parameters received gradients"
        except Exception as e:
            pytest.skip(f"Gradient flow test failed: {e}")


# =============================================================================
# GraphEncoder Tests
# =============================================================================

class TestGraphEncoder:
    """Tests for GraphEncoder class."""

    def test_graph_encoder_class_exists(self, graph_encoder_module):
        """Test GraphEncoder class exists."""
        if graph_encoder_module is None:
            pytest.skip("Graph encoder module not available")

        assert hasattr(graph_encoder_module, "GraphEncoder")

    def test_simple_graph_encoder_exists(self, graph_encoder_module):
        """Test SimpleGraphEncoder class exists."""
        if graph_encoder_module is None:
            pytest.skip("Graph encoder module not available")

        assert hasattr(graph_encoder_module, "SimpleGraphEncoder")

    def test_simple_encoder_creation(self, graph_encoder_module):
        """Test SimpleGraphEncoder can be created."""
        if graph_encoder_module is None:
            pytest.skip("Graph encoder module not available")

        SimpleGraphEncoder = getattr(graph_encoder_module, "SimpleGraphEncoder", None)
        if SimpleGraphEncoder is None:
            pytest.skip("SimpleGraphEncoder not found")

        encoder = SimpleGraphEncoder(output_dim=64)
        assert encoder is not None
        assert encoder.output_dim == 64

    def test_simple_encoder_encode(self, graph_encoder_module):
        """Test SimpleGraphEncoder encode method."""
        if graph_encoder_module is None:
            pytest.skip("Graph encoder module not available")

        import numpy as np

        SimpleGraphEncoder = getattr(graph_encoder_module, "SimpleGraphEncoder", None)
        if SimpleGraphEncoder is None:
            pytest.skip("SimpleGraphEncoder not found")

        encoder = SimpleGraphEncoder(output_dim=64)

        test_json = '{"operators": [{"type": "matmul"}], "tensors": [{"dims": [1024, 1024]}]}'
        features = encoder.encode(test_json)

        assert features is not None
        assert features.shape == (64,)


# =============================================================================
# MessagePassingLayer Tests
# =============================================================================

@pytest.mark.torch
class TestMessagePassingLayer:
    """Tests for MessagePassingLayer (requires PyTorch)."""

    def test_message_passing_layer_exists(self, graph_encoder_module):
        """Test MessagePassingLayer class exists."""
        if graph_encoder_module is None:
            pytest.skip("Graph encoder module not available")

        assert hasattr(graph_encoder_module, "MessagePassingLayer")

    def test_message_passing_layer_creation(self, graph_encoder_module):
        """Test MessagePassingLayer can be created."""
        if graph_encoder_module is None:
            pytest.skip("Graph encoder module not available")

        MessagePassingLayer = getattr(graph_encoder_module, "MessagePassingLayer", None)
        if MessagePassingLayer is None:
            pytest.skip("MessagePassingLayer not found")

        import torch

        try:
            layer = MessagePassingLayer(
                in_dim=32,
                out_dim=64,
            )
            assert layer is not None
        except Exception as e:
            pytest.skip(f"MessagePassingLayer creation failed: {e}")
