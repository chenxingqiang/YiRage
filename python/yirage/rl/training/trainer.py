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
RL Training entry point for YiRage search.

Provides high-level API for training RL policies to guide kernel search.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..env import YiRageSearchEnv, EnvConfig


@dataclass
class TrainingConfig:
    """Configuration for RL training."""

    # Algorithm
    algorithm: str = "PPO"  # PPO, SAC, APPO

    # Environment
    env_config: EnvConfig = field(default_factory=EnvConfig)

    # Resources
    num_workers: int = 4
    num_envs_per_worker: int = 1
    num_gpus_for_trainer: int = 0  # Policy training on CPU

    # Training
    train_batch_size: int = 256
    sgd_minibatch_size: int = 64
    num_sgd_iter: int = 10
    lr: float = 3e-4

    # Stopping
    max_iterations: int = 1000
    target_valid_rate: float = 0.8
    target_latency_improvement: float = 2.0

    # Checkpointing
    checkpoint_freq: int = 50
    checkpoint_dir: str = "./checkpoints"

    # Evaluation
    evaluation_interval: int = 10
    evaluation_num_episodes: int = 10


def train_rl_search(
    config: Optional[TrainingConfig] = None,
    target_graphs: Optional[List[str]] = None,
    resume_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train RL policy for YiRage kernel search.

    This is the main entry point for training.

    Args:
        config: Training configuration
        target_graphs: List of target graph JSONs for training
        resume_checkpoint: Path to checkpoint to resume from

    Returns:
        Training results including best checkpoint path

    Example:
        >>> config = TrainingConfig(
        ...     algorithm="PPO",
        ...     num_workers=8,
        ...     max_iterations=500,
        ... )
        >>> results = train_rl_search(config, target_graphs=[graph_json])
        >>> print(results["best_checkpoint"])
    """
    try:
        import ray
        from ray import tune
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.algorithms.sac import SACConfig
    except ImportError:
        raise ImportError(
            "Ray and RLlib are required for training. " "Install with: pip install 'ray[rllib]'"
        )

    from .callbacks import YiRageCallbacks

    if config is None:
        config = TrainingConfig()

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()

    # Update env config with target graphs
    env_config = config.env_config
    if target_graphs and len(target_graphs) > 0:
        env_config.target_graph_json = target_graphs[0]

    # Build algorithm config
    if config.algorithm == "PPO":
        algo_config = (
            PPOConfig()
            .environment(
                env=YiRageSearchEnv,
                env_config=vars(env_config),
            )
            .framework("torch")
            .resources(
                num_gpus=config.num_gpus_for_trainer,
                num_cpus_per_worker=1,
            )
            .rollouts(
                num_rollout_workers=config.num_workers,
                num_envs_per_worker=config.num_envs_per_worker,
            )
            .training(
                train_batch_size=config.train_batch_size,
                sgd_minibatch_size=config.sgd_minibatch_size,
                num_sgd_iter=config.num_sgd_iter,
                lr=config.lr,
            )
            .callbacks(YiRageCallbacks)
            .evaluation(
                evaluation_interval=config.evaluation_interval,
                evaluation_num_workers=1,
                evaluation_config={"explore": False},
            )
        )
    elif config.algorithm == "SAC":
        algo_config = (
            SACConfig()
            .environment(
                env=YiRageSearchEnv,
                env_config=vars(env_config),
            )
            .framework("torch")
            .resources(
                num_gpus=config.num_gpus_for_trainer,
                num_cpus_per_worker=1,
            )
            .rollouts(
                num_rollout_workers=config.num_workers,
                num_envs_per_worker=config.num_envs_per_worker,
            )
            .training(
                train_batch_size=config.train_batch_size,
            )
            .callbacks(YiRageCallbacks)
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    # Define stopping conditions
    def stop_condition(trial_id: str, result: Dict) -> bool:
        if result["training_iteration"] >= config.max_iterations:
            return True

        custom = result.get("custom_metrics", {})
        valid_rate = custom.get("total_valid_mean", 0)

        if valid_rate >= config.target_valid_rate:
            # Check latency improvement
            best_latency = custom.get("best_latency_ms_mean", float("inf"))
            if best_latency < float("inf"):
                return True

        return False

    # Run training
    checkpoint_config = tune.CheckpointConfig(
        checkpoint_frequency=config.checkpoint_freq,
        checkpoint_at_end=True,
    )

    run_config = tune.RunConfig(
        stop=stop_condition,
        checkpoint_config=checkpoint_config,
        storage_path=config.checkpoint_dir,
    )

    tuner = tune.Tuner(
        config.algorithm,
        param_space=algo_config.to_dict(),
        run_config=run_config,
    )

    # Resume if checkpoint provided
    if resume_checkpoint:
        tuner = tune.Tuner.restore(
            resume_checkpoint,
            trainable=config.algorithm,
        )

    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result()

    return {
        "best_checkpoint": best_result.checkpoint.path if best_result.checkpoint else None,
        "best_metrics": best_result.metrics if best_result else {},
        "all_results": results,
    }


def load_trained_policy(checkpoint_path: str):
    """
    Load a trained policy from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        RLlib Algorithm for inference
    """
    try:
        from ray.rllib.algorithms.algorithm import Algorithm
    except ImportError:
        raise ImportError("Ray RLlib required")

    return Algorithm.from_checkpoint(checkpoint_path)


def search_with_policy(
    policy,
    target_graph_json: str,
    env_config: Optional[EnvConfig] = None,
    max_episodes: int = 100,
) -> Dict[str, Any]:
    """
    Use trained policy to search for optimal kernels.

    This is the inference-time usage of trained RL policy.

    Args:
        policy: Trained RLlib algorithm
        target_graph_json: Target computation graph
        env_config: Environment configuration
        max_episodes: Maximum search episodes

    Returns:
        Dictionary with found kernels and statistics
    """
    if env_config is None:
        env_config = EnvConfig()

    env_config.target_graph_json = target_graph_json
    env = YiRageSearchEnv(vars(env_config))

    best_kernels = []
    best_latency = float("inf")

    for episode in range(max_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = policy.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info.get("verified"):
                latency = info.get("latency_ms")
                if latency is not None and latency < best_latency:
                    best_latency = latency
                    best_kernels.append(
                        {
                            "kernel_graph": info.get("kernel_graph"),
                            "latency_ms": latency,
                            "episode": episode,
                        }
                    )

    env.close()

    return {
        "best_kernels": sorted(best_kernels, key=lambda x: x["latency_ms"])[:10],
        "best_latency_ms": best_latency,
        "total_episodes": max_episodes,
    }
