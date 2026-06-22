# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Search Policy Network for RL-guided kernel search.

Features flow from C++ µGraph through FeatureProcessor to this model:

    C++ µGraph → FeatureProcessor → SearchPolicyNetwork → Action

Supports:
- GNN or MLP graph encoding
- Model save/load
- ONNX export
- Integration with RLlib
"""

from typing import Dict, Optional, Tuple, List, Any, Union
from pathlib import Path
import json
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class GraphEncoder(nn.Module):
        """
        Encodes µGraph features into embedding vector.

        Supports:
        - MLP: Simple node feature aggregation
        - GNN: Message passing on graph structure
        """

        def __init__(
            self,
            node_dim: int = 16,
            hidden_dim: int = 128,
            output_dim: int = 128,
            num_layers: int = 3,
            use_gnn: bool = False,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.node_dim = node_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            self.use_gnn = use_gnn

            if use_gnn:
                self._build_gnn()
            else:
                self._build_mlp(dropout)

        def _build_mlp(self, dropout: float):
            """Build MLP encoder."""
            layers = []

            layers.append(nn.Linear(self.node_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            for _ in range(self.num_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(self.hidden_dim, self.output_dim))

            self.encoder = nn.Sequential(*layers)
            self.output_transform = nn.Linear(self.output_dim, self.output_dim)

        def _build_gnn(self):
            """Build GNN encoder."""
            try:
                from torch_geometric.nn import GCNConv, global_mean_pool

                self.node_embed = nn.Linear(self.node_dim, self.hidden_dim)

                self.convs = nn.ModuleList(
                    [GCNConv(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
                )

                self.output_transform = nn.Linear(self.hidden_dim, self.output_dim)
                self._pool = global_mean_pool
                self._gnn_available = True

            except ImportError:
                print("torch_geometric not available, using MLP")
                self.use_gnn = False
                self._build_mlp(0.1)

        def forward(
            self,
            node_features: torch.Tensor,
            edge_index: Optional[torch.Tensor] = None,
            batch: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                node_features: [num_nodes, node_dim]
                edge_index: [2, num_edges] (for GNN)
                batch: [num_nodes] node-to-graph assignment

            Returns:
                Graph embedding: [batch_size, output_dim]
            """
            if self.use_gnn and edge_index is not None:
                return self._forward_gnn(node_features, edge_index, batch)
            else:
                return self._forward_mlp(node_features, batch)

        def _forward_mlp(
            self,
            node_features: torch.Tensor,
            batch: Optional[torch.Tensor],
        ) -> torch.Tensor:
            """MLP forward with pooling."""
            x = self.encoder(node_features)

            if batch is None:
                # Single graph - global mean pooling
                graph_emb = x.mean(dim=0, keepdim=True)
            else:
                # Batch of graphs - scatter pooling
                num_graphs = batch.max().item() + 1
                graph_emb = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
                for i in range(num_graphs):
                    mask = batch == i
                    if mask.any():
                        graph_emb[i] = x[mask].mean(dim=0)

            return self.output_transform(graph_emb)

        def _forward_gnn(
            self,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            batch: Optional[torch.Tensor],
        ) -> torch.Tensor:
            """GNN forward."""
            x = self.node_embed(node_features)

            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)

            if batch is None:
                graph_emb = x.mean(dim=0, keepdim=True)
            else:
                graph_emb = self._pool(x, batch)

            return self.output_transform(graph_emb)

    class ConfigEncoder(nn.Module):
        """Encodes hardware configuration features."""

        def __init__(
            self,
            input_dim: int = 32,
            hidden_dim: int = 64,
            output_dim: int = 64,
        ):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class SearchPolicyNetwork(nn.Module):
        """
        Policy network for RL-guided kernel search.

        Architecture:
            graph_features → GraphEncoder ─┐
            config_features → ConfigEncoder ├─→ FusionMLP → Policy Head → π(a|s)
            history_features → HistoryMLP ──┘              → Value Head → V(s)

        Input features come from C++ µGraph through FeatureProcessor.

        Supports:
        - save(): Save model checkpoint
        - load(): Load model checkpoint
        - export_onnx(): Export to ONNX format
        """

        def __init__(
            self,
            # Encoder dimensions
            graph_dim: int = 128,
            config_dim: int = 64,
            history_dim: int = 32,
            # Network dimensions
            hidden_dim: int = 256,
            # Output dimensions
            config_action_dim: int = 64,  # Level 1 action
            graph_action_dim: int = 32,  # Level 2 action
            # Options
            use_gnn: bool = False,
            dropout: float = 0.1,
        ):
            super().__init__()

            # Save config for serialization
            self.config = {
                "graph_dim": graph_dim,
                "config_dim": config_dim,
                "history_dim": history_dim,
                "hidden_dim": hidden_dim,
                "config_action_dim": config_action_dim,
                "graph_action_dim": graph_action_dim,
                "use_gnn": use_gnn,
                "dropout": dropout,
            }

            # Encoders
            self.graph_encoder = GraphEncoder(
                node_dim=16,
                output_dim=graph_dim,
                use_gnn=use_gnn,
                dropout=dropout,
            )

            self.config_encoder = ConfigEncoder(
                input_dim=48,  # Global feature dim from FeatureProcessor
                output_dim=config_dim,
            )

            self.history_encoder = nn.Sequential(
                nn.Linear(history_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )

            # Fusion layer
            fusion_input_dim = graph_dim + config_dim + 64
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Policy heads (separate for Level 1 and Level 2)
            self.config_policy_head = nn.Linear(hidden_dim, config_action_dim)
            self.graph_policy_head = nn.Linear(hidden_dim, graph_action_dim)

            # Value head (shared)
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Initialize network weights."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(
            self,
            graph_features: Dict[str, torch.Tensor],
            global_features: torch.Tensor,
            history_features: torch.Tensor,
            level: int = 1,
            action_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Args:
                graph_features: {
                    "node_features": [num_nodes, 16],
                    "edge_index": [2, num_edges] (optional),
                    "batch": [num_nodes] (optional),
                }
                global_features: [batch, 48] from FeatureProcessor
                history_features: [batch, history_dim]
                level: 1 for config policy, 2 for graph policy
                action_mask: [batch, action_dim] binary mask

            Returns:
                (action_logits, value)
            """
            # Graph encoding
            graph_emb = self.graph_encoder(
                graph_features["node_features"],
                graph_features.get("edge_index"),
                graph_features.get("batch"),
            )

            # Config encoding
            config_emb = self.config_encoder(global_features)

            # History encoding
            history_emb = self.history_encoder(history_features)

            # Fusion
            fused = torch.cat([graph_emb, config_emb, history_emb], dim=-1)
            hidden = self.fusion(fused)

            # Select policy head based on level
            if level == 1:
                logits = self.config_policy_head(hidden)
            else:
                logits = self.graph_policy_head(hidden)

            # Apply action mask
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask.bool(), float("-inf"))

            # Value
            value = self.value_head(hidden)

            return logits, value.squeeze(-1)

        def get_action(
            self,
            graph_features: Dict[str, torch.Tensor],
            global_features: torch.Tensor,
            history_features: torch.Tensor,
            level: int = 1,
            action_mask: Optional[torch.Tensor] = None,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Sample action from policy.

            Returns:
                (action, log_prob, value)
            """
            logits, value = self.forward(
                graph_features, global_features, history_features, level, action_mask
            )

            # Sample action
            if deterministic:
                action = logits.argmax(dim=-1)
                probs = F.softmax(logits, dim=-1)
                log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            return action, log_prob, value

        def evaluate_actions(
            self,
            graph_features: Dict[str, torch.Tensor],
            global_features: torch.Tensor,
            history_features: torch.Tensor,
            actions: torch.Tensor,
            level: int = 1,
            action_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate actions for PPO training.

            Returns:
                (log_prob, entropy, value)
            """
            logits, value = self.forward(
                graph_features, global_features, history_features, level, action_mask
            )

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            return log_prob, entropy, value

        # =============================================
        # Model Save/Load
        # =============================================

        def save(
            self,
            path: str,
            optimizer: Optional[Any] = None,
            scheduler: Optional[Any] = None,
            epoch: int = 0,
            extra_info: Optional[Dict[str, Any]] = None,
        ):
            """
            Save model checkpoint.

            Args:
                path: Save path
                optimizer: Optimizer to save (optional)
                scheduler: LR scheduler to save (optional)
                epoch: Current epoch number
                extra_info: Additional info to save
            """
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_config": self.config,
                "model_state_dict": self.state_dict(),
                "epoch": epoch,
            }

            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            if extra_info is not None:
                checkpoint["extra_info"] = extra_info

            torch.save(checkpoint, path)
            print(f"Model saved to {path}")

        @classmethod
        def load(
            cls,
            path: str,
            device: str = "cpu",
            load_optimizer: bool = False,
            optimizer_class: Optional[type] = None,
        ) -> Union["SearchPolicyNetwork", Tuple["SearchPolicyNetwork", Any]]:
            """
            Load model from checkpoint.

            Args:
                path: Checkpoint path
                device: Target device
                load_optimizer: Whether to return optimizer
                optimizer_class: Optimizer class (e.g., torch.optim.Adam)

            Returns:
                model or (model, optimizer) if load_optimizer=True
            """
            checkpoint = torch.load(path, map_location=device)

            # Create model from config
            config = checkpoint["model_config"]
            model = cls(**config)

            # Load weights
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            print(f"Model loaded from {path} (epoch {checkpoint.get('epoch', 'unknown')})")

            if load_optimizer and optimizer_class is not None:
                optimizer = optimizer_class(model.parameters())
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                return model, optimizer

            return model

        def export_onnx(
            self,
            path: str,
            batch_size: int = 1,
            num_nodes: int = 10,
        ):
            """
            Export model to ONNX format.

            Args:
                path: ONNX file path
                batch_size: Batch size for export
                num_nodes: Number of nodes in dummy graph
            """
            self.eval()

            # Create dummy inputs
            device = next(self.parameters()).device

            dummy_node_features = torch.randn(num_nodes, 16, device=device)
            dummy_global_features = torch.randn(batch_size, 48, device=device)
            dummy_history = torch.randn(batch_size, self.config["history_dim"], device=device)
            dummy_batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

            # Export
            try:
                torch.onnx.export(
                    self,
                    (
                        {
                            "node_features": dummy_node_features,
                            "batch": dummy_batch,
                        },
                        dummy_global_features,
                        dummy_history,
                        1,  # level
                        None,  # action_mask
                    ),
                    path,
                    opset_version=14,
                    input_names=["graph_features", "global_features", "history"],
                    output_names=["logits", "value"],
                    dynamic_axes={
                        "graph_features": {0: "num_nodes"},
                    },
                )
                print(f"ONNX model exported to {path}")
            except Exception as e:
                print(f"ONNX export failed: {e}")
                print("Consider using torch.jit.save instead")

        def save_jit(self, path: str):
            """Save as TorchScript for deployment."""
            self.eval()

            # Note: May need to trace with example inputs
            try:
                scripted = torch.jit.script(self)
                torch.jit.save(scripted, path)
                print(f"TorchScript model saved to {path}")
            except Exception as e:
                print(f"TorchScript save failed: {e}")
                print("Using trace instead...")

                # Fallback to trace
                device = next(self.parameters()).device
                example_graph = {
                    "node_features": torch.randn(10, 16, device=device),
                }
                example_global = torch.randn(1, 48, device=device)
                example_history = torch.randn(1, self.config["history_dim"], device=device)

                traced = torch.jit.trace(
                    self,
                    (example_graph, example_global, example_history),
                )
                torch.jit.save(traced, path)
                print(f"Traced model saved to {path}")

    class ModelCheckpoint:
        """
        Utility class for managing model checkpoints during training.
        """

        def __init__(
            self,
            save_dir: str,
            save_freq: int = 100,
            max_checkpoints: int = 5,
            monitor: str = "reward",
            mode: str = "max",
        ):
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

            self.save_freq = save_freq
            self.max_checkpoints = max_checkpoints
            self.monitor = monitor
            self.mode = mode

            self.best_value = float("-inf") if mode == "max" else float("inf")
            self.checkpoints: List[Path] = []

        def save_if_needed(
            self,
            model: SearchPolicyNetwork,
            epoch: int,
            metrics: Dict[str, float],
            optimizer: Optional[Any] = None,
        ):
            """Save checkpoint if conditions are met."""

            # Save at frequency
            if epoch % self.save_freq == 0:
                self._save_checkpoint(model, epoch, optimizer, "periodic")

            # Save if best
            current_value = metrics.get(self.monitor, 0)
            is_best = (self.mode == "max" and current_value > self.best_value) or (
                self.mode == "min" and current_value < self.best_value
            )

            if is_best:
                self.best_value = current_value
                self._save_checkpoint(model, epoch, optimizer, "best")

        def _save_checkpoint(
            self,
            model: SearchPolicyNetwork,
            epoch: int,
            optimizer: Optional[Any],
            prefix: str,
        ):
            """Save a checkpoint."""
            path = self.save_dir / f"{prefix}_epoch_{epoch}.pt"

            model.save(
                str(path),
                optimizer=optimizer,
                epoch=epoch,
                extra_info={"best_value": self.best_value},
            )

            if prefix == "periodic":
                self.checkpoints.append(path)

                # Remove old checkpoints
                while len(self.checkpoints) > self.max_checkpoints:
                    old_path = self.checkpoints.pop(0)
                    if old_path.exists():
                        old_path.unlink()

        def get_best_checkpoint(self) -> Optional[Path]:
            """Get path to best checkpoint."""
            best_path = self.save_dir / "best_epoch_*.pt"
            matches = list(self.save_dir.glob("best_epoch_*.pt"))

            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)
            return None

else:
    # Stub when PyTorch not available
    class SearchPolicyNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SearchPolicyNetwork")

    class GraphEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required")

    class ModelCheckpoint:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required")
