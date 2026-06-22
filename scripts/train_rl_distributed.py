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
Distributed RL Training using Ray.

This script demonstrates scalable RL training for kernel search using Ray:
- Multiple CPU workers for parallel episode collection
- GPU workers for verification (when available)
- Centralized policy updates
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Check for Ray
try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    print("Ray not installed. Install with: pip install ray")

# Setup paths
WORKSPACE_ROOT = Path(__file__).parent.parent
PYTHON_ROOT = WORKSPACE_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    # Ray config
    num_workers: int = 4
    num_gpus: int = 0  # For verification

    # Training
    num_iterations: int = 100
    episodes_per_worker: int = 10
    batch_size: int = 1024

    # Algorithm
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2

    # Logging
    log_interval: int = 10


def create_simulated_env():
    """Create a simulated environment (no dependencies)."""

    class SimEnv:
        def __init__(self):
            self.state = np.zeros(16, dtype=np.float32)
            self.step_count = 0
            self.max_steps = 20

        def reset(self):
            self.state = np.random.randn(16).astype(np.float32) * 0.1
            self.step_count = 0
            return self.state.copy(), {}

        def step(self, action):
            self.step_count += 1

            # Update state
            self.state += np.random.randn(16).astype(np.float32) * 0.1

            # Compute reward
            reward = np.random.randn() * 0.1 + 0.05 * self.step_count

            # Check done
            done = self.step_count >= self.max_steps or np.random.random() < 0.1
            truncated = self.step_count >= self.max_steps

            info = {
                "step": self.step_count,
                "verified": np.random.random() > 0.5,
            }

            return self.state.copy(), reward, done, truncated, info

        @property
        def action_space_n(self):
            return 10

    return SimEnv()


if HAS_RAY:

    @ray.remote
    class RolloutWorker:
        """
        Worker for collecting rollouts (episodes).

        Runs on CPU, collects experience using the current policy.
        """

        def __init__(self, worker_id: int, config: Dict):
            self.worker_id = worker_id
            self.config = config
            self.env = create_simulated_env()

            # Simple policy weights (would be neural network in production)
            self.policy_weights = np.random.randn(16, 10).astype(np.float32) * 0.1

        def set_weights(self, weights: np.ndarray):
            """Update policy weights."""
            self.policy_weights = weights

        def collect_episodes(self, num_episodes: int) -> List[Dict]:
            """
            Collect episodes using current policy.

            Returns:
                List of episode data dictionaries
            """
            episodes = []

            for _ in range(num_episodes):
                episode = self._collect_one_episode()
                episodes.append(episode)

            return episodes

        def _collect_one_episode(self) -> Dict:
            """Collect single episode."""
            obs, _ = self.env.reset()

            observations = []
            actions = []
            rewards = []
            dones = []

            done = False
            while not done:
                # Select action (softmax policy)
                logits = obs @ self.policy_weights
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
                action = np.random.choice(len(probs), p=probs)

                # Step environment
                next_obs, reward, done, truncated, info = self.env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done or truncated)

                obs = next_obs

            return {
                "observations": np.array(observations),
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "dones": np.array(dones),
                "total_reward": sum(rewards),
                "length": len(rewards),
            }

        def get_worker_id(self) -> int:
            return self.worker_id

    @ray.remote(num_gpus=0.25)
    class GPUVerifierWorker:
        """
        GPU worker for kernel verification.

        Runs on GPU, verifies kernel correctness and profiles performance.
        """

        def __init__(self, gpu_id: int):
            self.gpu_id = gpu_id
            self._initialized = False

        def verify_batch(self, kernels: List[str]) -> List[Dict]:
            """
            Verify batch of kernels.

            Args:
                kernels: List of kernel JSON strings

            Returns:
                List of verification results
            """
            results = []

            for kernel in kernels:
                # Simulate verification
                verified = np.random.random() > 0.3
                latency = np.random.exponential(0.1) if verified else float("inf")

                results.append(
                    {
                        "verified": verified,
                        "latency_ms": latency,
                        "gpu_id": self.gpu_id,
                    }
                )

            return results


def train_distributed(config: DistributedConfig):
    """Run distributed training."""

    if not HAS_RAY:
        print("Ray not available. Running single-process training.")
        return train_single_process(config)

    print("=" * 60)
    print("Distributed RL Training with Ray")
    print("=" * 60)
    print(f"Workers: {config.num_workers}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Episodes per worker: {config.episodes_per_worker}")
    print()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Create workers
    workers = [RolloutWorker.remote(i, vars(config)) for i in range(config.num_workers)]

    # Initial policy weights
    policy_weights = np.random.randn(16, 10).astype(np.float32) * 0.1

    # Training stats
    all_rewards = []
    all_lengths = []

    for iteration in range(config.num_iterations):
        # Broadcast weights to workers
        ray.get([w.set_weights.remote(policy_weights) for w in workers])

        # Collect episodes in parallel
        episode_futures = [w.collect_episodes.remote(config.episodes_per_worker) for w in workers]

        all_episodes = ray.get(episode_futures)

        # Flatten episodes
        episodes = []
        for worker_episodes in all_episodes:
            episodes.extend(worker_episodes)

        # Compute statistics
        rewards = [ep["total_reward"] for ep in episodes]
        lengths = [ep["length"] for ep in episodes]

        all_rewards.extend(rewards)
        all_lengths.extend(lengths)

        # Simple policy gradient update
        # Concatenate all observations and actions
        all_obs = np.concatenate([ep["observations"] for ep in episodes])
        all_actions = np.concatenate([ep["actions"] for ep in episodes])
        all_returns = []

        for ep in episodes:
            # Compute returns
            returns = []
            G = 0
            for r in reversed(ep["rewards"]):
                G = r + config.gamma * G
                returns.insert(0, G)
            all_returns.extend(returns)

        all_returns = np.array(all_returns)

        # Normalize returns
        all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

        # Policy gradient
        for i in range(len(all_obs)):
            obs = all_obs[i]
            action = all_actions[i]
            ret = all_returns[i]

            # Compute gradient (simplified)
            logits = obs @ policy_weights
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()

            grad = np.outer(obs, -probs)
            grad[:, action] += obs

            policy_weights += config.learning_rate * ret * grad

        # Log
        if (iteration + 1) % config.log_interval == 0:
            avg_reward = np.mean(rewards)
            avg_length = np.mean(lengths)

            print(f"Iteration {iteration + 1}/{config.num_iterations}")
            print(f"  Episodes: {len(episodes)}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print()

    # Cleanup
    ray.shutdown()

    # Summary
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Total Iterations: {config.num_iterations}")
    print(f"Total Episodes: {len(all_rewards)}")
    print(f"Final Avg Reward: {np.mean(all_rewards[-100:]):.3f}")

    return policy_weights


def train_single_process(config: DistributedConfig):
    """Fallback single-process training."""
    print("=" * 60)
    print("Single-Process RL Training")
    print("=" * 60)

    env = create_simulated_env()
    policy_weights = np.random.randn(16, 10).astype(np.float32) * 0.1

    all_rewards = []

    for iteration in range(config.num_iterations):
        episodes = []

        for _ in range(config.episodes_per_worker):
            obs, _ = env.reset()
            episode_reward = 0

            while True:
                # Select action
                logits = obs @ policy_weights
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
                action = np.random.choice(len(probs), p=probs)

                # Step
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward

                if done or truncated:
                    break

            episodes.append({"total_reward": episode_reward})

        rewards = [ep["total_reward"] for ep in episodes]
        all_rewards.extend(rewards)

        if (iteration + 1) % config.log_interval == 0:
            print(f"Iteration {iteration + 1}: Avg Reward = {np.mean(rewards):.3f}")

    print(f"\nFinal Avg Reward: {np.mean(all_rewards[-50:]):.3f}")
    return policy_weights


def main():
    parser = argparse.ArgumentParser(description="Distributed RL Training")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations")
    parser.add_argument(
        "--episodes-per-worker",
        type=int,
        default=5,
        help="Episodes collected per worker per iteration",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations")

    args = parser.parse_args()

    config = DistributedConfig(
        num_workers=args.workers,
        num_iterations=args.iterations,
        episodes_per_worker=args.episodes_per_worker,
        learning_rate=args.lr,
        log_interval=args.log_interval,
    )

    train_distributed(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
