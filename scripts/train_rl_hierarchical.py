#!/usr/bin/env python3
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
Hierarchical RL Training for Kernel Search.

This script demonstrates the complete hierarchical RL training pipeline:
- Level 1: Hardware configuration policy (grid_dim, block_dim, etc.)
- Level 2: µGraph construction policy (operators, mappings)

The training uses curriculum learning to progressively increase difficulty.
"""

import sys
import os
import json
import argparse
import time
import types
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Setup paths
WORKSPACE_ROOT = Path(__file__).parent.parent
PYTHON_ROOT = WORKSPACE_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))


def load_module_directly(module_name: str, file_path: Path):
    """Load a Python module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def setup_modules():
    """Setup fake yirage package and load RL modules."""
    # Create fake yirage package
    fake_yirage = types.ModuleType("yirage")
    fake_yirage.__path__ = [str(PYTHON_ROOT / "yirage")]
    sys.modules["yirage"] = fake_yirage

    # Create rl subpackage
    fake_rl = types.ModuleType("yirage.rl")
    fake_rl.__path__ = [str(PYTHON_ROOT / "yirage" / "rl")]
    sys.modules["yirage.rl"] = fake_rl

    # Create subpackages
    for subpkg in ["env", "verifier", "training", "models", "search", "features"]:
        subpkg_path = PYTHON_ROOT / "yirage" / "rl" / subpkg
        if subpkg_path.exists():
            fake_subpkg = types.ModuleType(f"yirage.rl.{subpkg}")
            fake_subpkg.__path__ = [str(subpkg_path)]
            sys.modules[f"yirage.rl.{subpkg}"] = fake_subpkg

    # Load modules
    modules = {}

    # Config space
    modules["config_space"] = load_module_directly(
        "yirage.rl.search.config_space",
        PYTHON_ROOT / "yirage" / "rl" / "search" / "config_space.py",
    )

    # Graph space
    modules["graph_space"] = load_module_directly(
        "yirage.rl.search.graph_space", PYTHON_ROOT / "yirage" / "rl" / "search" / "graph_space.py"
    )

    # Features
    modules["mugraph_features"] = load_module_directly(
        "yirage.rl.features.mugraph_features",
        PYTHON_ROOT / "yirage" / "rl" / "features" / "mugraph_features.py",
    )

    modules["processor"] = load_module_directly(
        "yirage.rl.features.processor", PYTHON_ROOT / "yirage" / "rl" / "features" / "processor.py"
    )

    # Reward
    modules["reward"] = load_module_directly(
        "yirage.rl.env.reward", PYTHON_ROOT / "yirage" / "rl" / "env" / "reward.py"
    )

    return modules


@dataclass
class TrainingConfig:
    """Configuration for hierarchical training."""

    # Training parameters
    num_episodes: int = 100
    max_steps_per_episode: int = 50
    learning_rate: float = 3e-4
    gamma: float = 0.99

    # Curriculum parameters
    use_curriculum: bool = True
    initial_difficulty: float = 0.2
    max_difficulty: float = 1.0
    curriculum_episodes: int = 50

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995

    # Logging
    log_interval: int = 10
    save_interval: int = 50

    # Output
    output_dir: str = "checkpoints"


class SimulatedEnvironment:
    """
    Simulated hierarchical search environment for training.

    This environment simulates the two-level search process:
    - Level 1: Config selection
    - Level 2: Graph construction
    """

    def __init__(
        self,
        modules: Dict[str, Any],
        difficulty: float = 0.5,
        target_graph: Optional[Dict] = None,
    ):
        self.modules = modules
        self.difficulty = difficulty
        self.target_graph = target_graph or self._create_default_target()

        # State
        self.current_config = None
        self.current_constraints = None
        self.current_graph = {"operators": [], "tensors": [], "edges": []}
        self.step_count = 0
        self.level = 1  # 1 = config, 2 = graph

        # Action spaces
        self.config_action_space = modules["config_space"].ConfigActionSpace()
        self.graph_action_space = None  # Created after config selection

        # Reward
        self.reward_computer = modules["reward"].RewardComputer()

    def _create_default_target(self) -> Dict:
        """Create a default target graph."""
        return {
            "inputs": [
                {"dims": [8, 4096], "dtype": "float16"},
                {"dims": [4096, 4096], "dtype": "float16"},
            ],
            "operators": [
                {"type": "matmul", "inputs": [0, 1], "outputs": [2]},
                {"type": "silu", "inputs": [2], "outputs": [3]},
            ],
            "outputs": [3],
        }

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment for new episode."""
        self.current_config = None
        self.current_constraints = None
        self.current_graph = {"operators": [], "tensors": [], "edges": []}
        self.step_count = 0
        self.level = 1
        self.reward_computer.reset()

        return self._get_observation()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        obs = {
            "level": np.array([self.level], dtype=np.float32),
            "step": np.array([self.step_count], dtype=np.float32),
            "num_operators": np.array([len(self.current_graph["operators"])], dtype=np.float32),
            "graph_features": np.random.randn(64).astype(np.float32),  # Simulated
        }

        if self.current_config is not None:
            config = self.current_config
            obs["config_features"] = np.array(
                [
                    config.grid_dim_x / 128,
                    config.grid_dim_y / 128,
                    config.block_dim_x / 1024,
                    config.forloop_range / 64,
                ],
                dtype=np.float32,
            )
        else:
            obs["config_features"] = np.zeros(4, dtype=np.float32)

        # Action mask
        if self.level == 1:
            obs["action_mask"] = np.ones(10, dtype=np.float32)  # Config actions
        else:
            obs["action_mask"] = np.ones(20, dtype=np.float32)  # Graph actions
            if len(self.current_graph["operators"]) >= 5:
                obs["action_mask"][0:10] = 0  # Disable add operator

        return obs

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action.

        Args:
            action: Action index

        Returns:
            (observation, reward, done, truncated, info)
        """
        self.step_count += 1
        info = {}

        if self.level == 1:
            # Level 1: Config selection
            reward, done = self._step_config(action)
            if not done:
                self.level = 2  # Transition to Level 2
                info["level_transition"] = True
        else:
            # Level 2: Graph construction
            reward, done = self._step_graph(action)

        truncated = self.step_count >= 50
        obs = self._get_observation()

        return obs, reward, done, truncated, info

    def _step_config(self, action: int) -> Tuple[float, bool]:
        """Execute Level 1 config action."""
        config_mod = self.modules["config_space"]

        # Decode action to config
        # Simplified: action selects from pre-defined configs
        configs = [
            (1, 1, 128, 1),  # Small
            (4, 1, 128, 4),  # Medium
            (8, 2, 256, 8),  # Large
            (16, 4, 512, 16),  # XL
        ]

        idx = action % len(configs)
        g, gy, b, f = configs[idx]

        self.current_config = config_mod.HardwareConfig(
            grid_dim_x=g,
            grid_dim_y=gy,
            grid_dim_z=1,
            block_dim_x=b,
            block_dim_y=1,
            block_dim_z=1,
            forloop_range=f,
        )

        # Create constraints for Level 2
        self.current_constraints = config_mod.SearchSpaceConstraints(self.current_config)

        # Small reward for valid config
        reward = 0.1

        return reward, False

    def _step_graph(self, action: int) -> Tuple[float, bool]:
        """Execute Level 2 graph action."""
        operators = ["matmul", "add", "mul", "silu", "reduction", "softmax"]

        if action < len(operators):
            # Add operator
            op_type = operators[action % len(operators)]
            op_id = len(self.current_graph["operators"])

            self.current_graph["operators"].append(
                {
                    "op_id": op_id,
                    "type": op_type,
                    "inputs": [max(0, op_id - 1)],
                    "outputs": [op_id + 1],
                }
            )

            reward = 0.05
            done = False

        elif action == len(operators):
            # Finish action
            reward = self._compute_final_reward()
            done = True

        else:
            # Invalid action
            reward = -0.1
            done = False

        return reward, done

    def _compute_final_reward(self) -> float:
        """Compute reward for completed graph."""
        reward = 0.0

        num_ops = len(self.current_graph["operators"])

        # Reward for having operators
        reward += min(num_ops * 0.1, 0.5)

        # Bonus for matching target complexity
        target_ops = len(self.target_graph.get("operators", []))
        if num_ops == target_ops:
            reward += 0.5
        elif abs(num_ops - target_ops) <= 1:
            reward += 0.3

        # Simulate verification
        verified = np.random.random() > (0.3 + 0.4 * self.difficulty)
        if verified:
            reward += 1.0

            # Simulate profiling
            latency = 0.1 + 0.9 * np.random.random() * self.difficulty
            speedup = 1.0 / max(latency, 0.01)
            reward += min(np.log(speedup + 1), 2.0)
        else:
            reward -= 0.5

        return reward


class SimplePolicy:
    """Simple epsilon-greedy policy for demonstration."""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.q_values = {}  # State -> action -> value

    def select_action(self, obs: Dict[str, np.ndarray], num_actions: int) -> int:
        """Select action using epsilon-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, num_actions)

        # Get state key
        state_key = (
            int(obs["level"][0]),
            int(obs["num_operators"][0]),
        )

        if state_key not in self.q_values:
            return np.random.randint(0, num_actions)

        q = self.q_values[state_key]
        return max(q, key=q.get)

    def update(
        self,
        obs: Dict,
        action: int,
        reward: float,
        next_obs: Dict,
        done: bool,
        lr: float = 0.1,
        gamma: float = 0.99,
    ):
        """Simple Q-learning update."""
        state_key = (
            int(obs["level"][0]),
            int(obs["num_operators"][0]),
        )

        next_state_key = (
            int(next_obs["level"][0]),
            int(next_obs["num_operators"][0]),
        )

        if state_key not in self.q_values:
            self.q_values[state_key] = {}

        if action not in self.q_values[state_key]:
            self.q_values[state_key][action] = 0.0

        # Compute target
        if done:
            target = reward
        else:
            next_q = self.q_values.get(next_state_key, {})
            max_next_q = max(next_q.values()) if next_q else 0.0
            target = reward + gamma * max_next_q

        # Update
        current = self.q_values[state_key][action]
        self.q_values[state_key][action] = current + lr * (target - current)

    def decay_epsilon(self, decay: float):
        """Decay exploration."""
        self.epsilon = max(0.1, self.epsilon * decay)


def train(config: TrainingConfig):
    """Run training loop."""
    print("=" * 60)
    print("Hierarchical RL Training for Kernel Search")
    print("=" * 60)
    print(f"Episodes: {config.num_episodes}")
    print(f"Max steps: {config.max_steps_per_episode}")
    print(f"Curriculum: {config.use_curriculum}")
    print()

    # Setup
    modules = setup_modules()

    # Create environment and policy
    env = SimulatedEnvironment(modules, difficulty=config.initial_difficulty)
    policy = SimplePolicy(epsilon=config.epsilon_start)

    # Training stats
    episode_rewards = []
    episode_lengths = []
    best_reward = float("-inf")

    for episode in range(config.num_episodes):
        # Curriculum: increase difficulty
        if config.use_curriculum:
            progress = min(episode / config.curriculum_episodes, 1.0)
            difficulty = (
                config.initial_difficulty
                + (config.max_difficulty - config.initial_difficulty) * progress
            )
            env.difficulty = difficulty

        # Run episode
        obs = env.reset()
        episode_reward = 0
        transitions = []

        for step in range(config.max_steps_per_episode):
            # Select action
            num_actions = 10 if obs["level"][0] == 1 else 20
            action = policy.select_action(obs, num_actions)

            # Execute action
            next_obs, reward, done, truncated, info = env.step(action)

            # Store transition
            transitions.append((obs, action, reward, next_obs, done))

            episode_reward += reward
            obs = next_obs

            if done or truncated:
                break

        # Update policy
        for obs, action, reward, next_obs, done in transitions:
            policy.update(
                obs,
                action,
                reward,
                next_obs,
                done,
                lr=config.learning_rate,
                gamma=config.gamma,
            )

        # Decay exploration
        policy.decay_epsilon(config.epsilon_decay)

        # Record stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        if episode_reward > best_reward:
            best_reward = episode_reward

        # Log
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval :])
            avg_length = np.mean(episode_lengths[-config.log_interval :])

            print(f"Episode {episode + 1}/{config.num_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Best: {best_reward:.3f}")
            print(f"  Epsilon: {policy.epsilon:.3f}")
            print(f"  Difficulty: {env.difficulty:.2f}")
            print()

    # Summary
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Total Episodes: {config.num_episodes}")
    print(f"Final Avg Reward: {np.mean(episode_rewards[-10:]):.3f}")
    print(f"Best Reward: {best_reward:.3f}")
    print(f"States Learned: {len(policy.q_values)}")

    return policy, episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Hierarchical RL Training")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument(
        "--curriculum", action="store_true", default=True, help="Use curriculum learning"
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N episodes")

    args = parser.parse_args()

    config = TrainingConfig(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        learning_rate=args.lr,
        use_curriculum=args.curriculum,
        log_interval=args.log_interval,
    )

    policy, rewards = train(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
