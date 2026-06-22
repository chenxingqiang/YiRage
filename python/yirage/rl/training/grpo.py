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
GRPO (Group Relative Policy Optimization) for µGraph Search.

GRPO improves upon PPO by:
1. Computing advantages relative to groups of samples
2. Better credit assignment for sparse rewards
3. More stable training for large action spaces

This is particularly effective for kernel search where:
- Rewards are sparse (only valid kernels get positive reward)
- Action space is large (many possible operators and configs)
- Credit assignment is difficult (final performance depends on sequence of actions)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import json
from pathlib import Path

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Group sampling
    group_size: int = 8  # Number of samples per group
    num_groups_per_batch: int = 16

    # PPO-like parameters
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # Learning rate
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine"  # "constant", "linear", "cosine"
    warmup_steps: int = 100

    # Training
    max_grad_norm: float = 0.5
    epochs_per_update: int = 4
    mini_batch_size: int = 64

    # GRPO specific
    advantage_normalization: str = "group"  # "group", "batch", "none"
    reward_baseline: str = "group_mean"  # "group_mean", "group_median", "moving_avg"

    # Regularization
    kl_target: float = 0.01
    kl_coef: float = 0.1

    # Fine-tuning
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1


@dataclass
class GRPOBatch:
    """Batch of grouped samples for GRPO update."""

    # States: (num_groups, group_size, state_dim)
    states: Any  # torch.Tensor

    # Actions: (num_groups, group_size)
    actions: Any

    # Rewards: (num_groups, group_size)
    rewards: Any

    # Old log probs: (num_groups, group_size)
    old_log_probs: Any

    # Group info
    group_ids: List[int] = field(default_factory=list)

    # Optional: hardware features for each group
    hardware_features: Optional[Any] = None


class GroupedReplayBuffer:
    """
    Replay buffer that stores samples in groups.

    Each group contains multiple rollouts for the same initial state,
    allowing for group-relative advantage computation.
    """

    def __init__(self, max_groups: int = 1000, group_size: int = 8):
        self.max_groups = max_groups
        self.group_size = group_size
        self.groups: List[Dict[str, Any]] = []

    def add_group(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        log_probs: List[float],
        hardware_features: Optional[np.ndarray] = None,
    ):
        """Add a group of samples."""
        if len(states) != self.group_size:
            return

        group = {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "log_probs": np.array(log_probs),
            "hardware_features": hardware_features,
        }

        self.groups.append(group)

        # Trim if over capacity
        if len(self.groups) > self.max_groups:
            self.groups = self.groups[-self.max_groups :]

    def sample_batch(self, num_groups: int) -> Optional[GRPOBatch]:
        """Sample a batch of groups."""
        if not HAS_TORCH:
            return None

        if len(self.groups) < num_groups:
            num_groups = len(self.groups)

        if num_groups == 0:
            return None

        indices = np.random.choice(len(self.groups), num_groups, replace=False)

        states = np.stack([self.groups[i]["states"] for i in indices])
        actions = np.stack([self.groups[i]["actions"] for i in indices])
        rewards = np.stack([self.groups[i]["rewards"] for i in indices])
        log_probs = np.stack([self.groups[i]["log_probs"] for i in indices])

        hw_features = None
        if self.groups[0].get("hardware_features") is not None:
            hw_features = np.stack([self.groups[i]["hardware_features"] for i in indices])
            hw_features = torch.FloatTensor(hw_features)

        return GRPOBatch(
            states=torch.FloatTensor(states),
            actions=torch.LongTensor(actions),
            rewards=torch.FloatTensor(rewards),
            old_log_probs=torch.FloatTensor(log_probs),
            group_ids=list(indices),
            hardware_features=hw_features,
        )

    def __len__(self):
        return len(self.groups)

    def clear(self):
        self.groups.clear()


class GRPOTrainer:
    """
    GRPO Trainer for µGraph Search Policy.

    Key innovations:
    1. Group-relative advantages for better credit assignment
    2. Hardware-aware policy conditioning
    3. Support for large model fine-tuning (LoRA)
    """

    def __init__(
        self,
        policy: Any,  # nn.Module
        config: GRPOConfig,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for GRPO training")

        self.policy = policy
        self.config = config
        self.device = device

        # Move policy to device
        self.policy = self.policy.to(device)

        # Setup optimizer
        if config.use_lora:
            # Only optimize LoRA parameters
            lora_params = [p for n, p in policy.named_parameters() if "lora" in n]
            self.optimizer = torch.optim.AdamW(lora_params, lr=config.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Replay buffer
        self.buffer = GroupedReplayBuffer(
            max_groups=1000,
            group_size=config.group_size,
        )

        # Training stats
        self.update_count = 0
        self.stats_history: List[Dict[str, float]] = []

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_schedule == "constant":
            return None
        elif self.config.lr_schedule == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=10000,
            )
        elif self.config.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=10000,
                eta_min=1e-6,
            )
        return None

    def compute_group_advantages(
        self,
        rewards: "torch.Tensor",  # (num_groups, group_size)
    ) -> "torch.Tensor":
        """
        Compute group-relative advantages.

        For each sample in a group, advantage = reward - baseline
        where baseline is computed from the group.

        Args:
            rewards: Rewards for each sample in each group

        Returns:
            Advantages tensor of same shape
        """
        cfg = self.config

        if cfg.reward_baseline == "group_mean":
            # Advantage = r_i - mean(r_group)
            baseline = rewards.mean(dim=1, keepdim=True)
            advantages = rewards - baseline

        elif cfg.reward_baseline == "group_median":
            # Advantage = r_i - median(r_group)
            baseline = rewards.median(dim=1, keepdim=True).values
            advantages = rewards - baseline

        else:
            advantages = rewards

        # Normalize advantages
        if cfg.advantage_normalization == "group":
            # Normalize within each group
            std = advantages.std(dim=1, keepdim=True) + 1e-8
            advantages = advantages / std

        elif cfg.advantage_normalization == "batch":
            # Normalize across entire batch
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def compute_policy_loss(
        self,
        batch: GRPOBatch,
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        """
        Compute GRPO policy loss.

        Returns:
            loss: Scalar loss tensor
            stats: Dictionary of training statistics
        """
        cfg = self.config

        # Flatten batch for policy forward pass
        batch_size = batch.states.shape[0] * batch.states.shape[1]
        states_flat = batch.states.view(batch_size, -1).to(self.device)
        actions_flat = batch.actions.view(batch_size).to(self.device)
        old_log_probs_flat = batch.old_log_probs.view(batch_size).to(self.device)

        # Forward pass
        if batch.hardware_features is not None:
            hw_flat = (
                batch.hardware_features.unsqueeze(1)
                .expand(-1, cfg.group_size, -1)
                .reshape(batch_size, -1)
                .to(self.device)
            )
            logits, values = self.policy(states_flat, hardware_features=hw_flat)
        else:
            logits, values = self.policy(states_flat)

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions_flat.unsqueeze(1)).squeeze(1)

        # Compute advantages (grouped)
        advantages = self.compute_group_advantages(batch.rewards.to(self.device))
        advantages_flat = advantages.view(batch_size)

        # PPO clipped objective
        ratio = torch.exp(action_log_probs - old_log_probs_flat)
        clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_param, 1 + cfg.clip_param)

        policy_loss1 = -advantages_flat * ratio
        policy_loss2 = -advantages_flat * clipped_ratio
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()

        # Value loss
        returns = batch.rewards.view(batch_size).to(self.device)
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Entropy bonus
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        # Total loss
        total_loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy

        # KL divergence (for monitoring)
        with torch.no_grad():
            kl = (old_log_probs_flat - action_log_probs).mean()

        stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl_divergence": kl.item(),
            "clip_fraction": ((ratio - 1).abs() > cfg.clip_param).float().mean().item(),
        }

        return total_loss, stats

    def update(self) -> Dict[str, float]:
        """
        Perform one GRPO update.

        Returns:
            Dictionary of training statistics
        """
        cfg = self.config

        # Sample batch
        batch = self.buffer.sample_batch(cfg.num_groups_per_batch)
        if batch is None:
            return {}

        all_stats = []

        for epoch in range(cfg.epochs_per_update):
            self.optimizer.zero_grad()

            loss, stats = self.compute_policy_loss(batch)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                cfg.max_grad_norm,
            )

            self.optimizer.step()

            all_stats.append(stats)

        # Update scheduler
        if self.scheduler:
            self.scheduler.step()

        self.update_count += 1

        # Average stats
        avg_stats = {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}
        avg_stats["update_count"] = self.update_count
        avg_stats["buffer_size"] = len(self.buffer)

        self.stats_history.append(avg_stats)

        return avg_stats

    def collect_group(
        self,
        env: Any,
        initial_state: np.ndarray,
        hardware_features: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Collect a group of rollouts from the same initial state.

        Args:
            env: Environment with reset() and step()
            initial_state: Initial state for all rollouts
            hardware_features: Hardware features for conditioning

        Returns:
            best_reward: Best reward in the group
            info: Collection statistics
        """
        cfg = self.config

        states = []
        actions = []
        rewards = []
        log_probs = []

        for _ in range(cfg.group_size):
            # Reset to initial state
            obs = initial_state.copy()

            # Run episode
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                if hardware_features is not None:
                    hw_tensor = torch.FloatTensor(hardware_features).unsqueeze(0).to(self.device)
                    logits, _ = self.policy(obs_tensor, hardware_features=hw_tensor)
                else:
                    logits, _ = self.policy(obs_tensor)

                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Execute action in environment
            _, reward, done, _, info = env.step(action.item())

            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())

        # Add to buffer
        self.buffer.add_group(states, actions, rewards, log_probs, hardware_features)

        return max(rewards), {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "group_size": len(rewards),
        }

    def save(self, path: Path):
        """Save trainer state."""
        if not HAS_TORCH:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "update_count": self.update_count,
                "stats_history": self.stats_history,
            },
            path,
        )

    def load(self, path: Path):
        """Load trainer state."""
        if not HAS_TORCH:
            return

        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint.get("update_count", 0)
        self.stats_history = checkpoint.get("stats_history", [])


# LoRA (Low-Rank Adaptation) implementation for fine-tuning

if HAS_TORCH:

    class LoRALinear(nn.Module):
        """
        Linear layer with LoRA adaptation.

        LoRA adds trainable low-rank matrices to frozen pretrained weights:
        W' = W + BA where B: (out, r), A: (r, in)
        """

        def __init__(
            self,
            original_layer: nn.Linear,
            rank: int = 8,
            alpha: float = 32.0,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.original = original_layer
            self.rank = rank
            self.alpha = alpha

            # Freeze original weights
            for param in self.original.parameters():
                param.requires_grad = False

            # LoRA matrices
            self.lora_A = nn.Parameter(torch.zeros(rank, original_layer.in_features))
            self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))

            # Scaling factor
            self.scaling = alpha / rank

            # Dropout
            self.dropout = nn.Dropout(dropout)

            # Initialize A with Kaiming, B with zeros
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Original forward
            result = self.original(x)

            # LoRA forward: BA * x
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

            return result + lora_out * self.scaling

    def apply_lora(
        model: nn.Module,
        target_modules: List[str] = ["q_proj", "v_proj"],
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ) -> nn.Module:
        """
        Apply LoRA to specified modules in a model.

        Args:
            model: PyTorch model
            target_modules: Names of modules to apply LoRA to
            rank: LoRA rank
            alpha: LoRA alpha (scaling)
            dropout: LoRA dropout

        Returns:
            Modified model with LoRA layers
        """
        for name, module in model.named_modules():
            if any(t in name for t in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA version
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]

                    parent = model
                    for part in parent_name.split("."):
                        if part:
                            parent = getattr(parent, part)

                    lora_layer = LoRALinear(module, rank, alpha, dropout)
                    setattr(parent, child_name, lora_layer)

        return model

    def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
        """Get only LoRA parameters for optimization."""
        params = []
        for name, param in model.named_parameters():
            if "lora_" in name:
                params.append(param)
        return params

    def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

else:
    # Stubs when PyTorch not available
    class LoRALinear:
        pass

    def apply_lora(*args, **kwargs):
        raise ImportError("PyTorch required")

    def get_lora_parameters(*args, **kwargs):
        raise ImportError("PyTorch required")

    def count_trainable_parameters(*args, **kwargs):
        raise ImportError("PyTorch required")
