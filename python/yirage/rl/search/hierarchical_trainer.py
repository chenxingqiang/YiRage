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
Hierarchical Search Trainer

Trains both Level 1 (Config) and Level 2 (Graph) policies.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

from .config_space import HardwareConfig, SearchSpaceConstraints
from .hierarchical_env import (
    HierarchicalSearchEnv,
    ConfigEnv,
    ConstrainedGraphEnv,
    HierarchicalEnvConfig,
)


@dataclass
class HierarchicalTrainingConfig:
    """Configuration for hierarchical training."""

    # Training mode
    mode: str = "joint"  # "joint", "hierarchical", "config_only", "graph_only", "codesign"

    # Level 0 (Accelerator) training — AccelForge co-design
    accelerator_enabled: bool = False
    accelerator_episodes_per_iter: int = 5
    area_budget_mm2: float = 100.0
    power_budget_mw: float = 5000.0

    # Level 1 (Config) training
    config_algorithm: str = "PPO"
    config_learning_rate: float = 3e-4
    config_episodes_per_iter: int = 10

    # Level 2 (Graph) training
    graph_algorithm: str = "PPO"
    graph_learning_rate: float = 3e-4
    graph_steps_per_config: int = 50

    # General
    max_iterations: int = 1000
    checkpoint_freq: int = 50
    checkpoint_dir: str = "./checkpoints"

    # Resources
    num_workers: int = 4
    num_gpus: int = 1


class HierarchicalTrainer:
    """
    Trainer for hierarchical search.

    Two training modes:
    1. Joint: Train both levels simultaneously
    2. Hierarchical: Train Level 2 first (with fixed configs), then Level 1
    """

    def __init__(self, config: Optional[HierarchicalTrainingConfig] = None):
        self.config = config or HierarchicalTrainingConfig()

        # Policies (would be RLlib algorithms)
        self.config_policy = None
        self.graph_policy = None

        # Statistics
        self.training_stats = {
            "config_rewards": [],
            "graph_rewards": [],
            "valid_kernels_found": 0,
            "best_latency": float("inf"),
        }

    def train(
        self,
        target_graphs: List[str],
        env_config: Optional[HierarchicalEnvConfig] = None,
    ) -> Dict[str, Any]:
        """
        Train hierarchical policies.

        Args:
            target_graphs: List of target graph JSONs
            env_config: Environment configuration

        Returns:
            Training results
        """
        if self.config.mode == "joint":
            return self._train_joint(target_graphs, env_config)
        elif self.config.mode == "hierarchical":
            return self._train_hierarchical(target_graphs, env_config)
        elif self.config.mode == "codesign":
            return self._train_codesign(target_graphs, env_config)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def _train_joint(
        self,
        target_graphs: List[str],
        env_config: Optional[HierarchicalEnvConfig],
    ) -> Dict[str, Any]:
        """
        Joint training of both levels.

        Each iteration:
        1. Level 1 selects config
        2. Level 2 runs graph search
        3. Both policies receive reward
        """
        print("=== Joint Hierarchical Training ===")

        if env_config is None:
            env_config = HierarchicalEnvConfig()

        for iteration in range(self.config.max_iterations):
            iteration_stats = {
                "config_rewards": [],
                "graph_rewards": [],
                "valid_found": 0,
            }

            for target in target_graphs:
                env_config.target_graph_json = target
                env = HierarchicalSearchEnv(vars(env_config))

                obs, info = env.reset()

                for config_ep in range(self.config.config_episodes_per_iter):
                    # Level 1: Select config (random for now)
                    action = env.action_space.sample()

                    obs, reward, done, truncated, info = env.step(action)

                    iteration_stats["config_rewards"].append(reward)

                    if info.get("level2_result", {}).get("verified", False):
                        iteration_stats["valid_found"] += 1
                        latency = info["level2_result"].get("latency_ms", float("inf"))
                        if latency < self.training_stats["best_latency"]:
                            self.training_stats["best_latency"] = latency

                    if done:
                        break

            # Update policies (would call RLlib here)
            # self.config_policy.learn(...)
            # self.graph_policy.learn(...)

            avg_config_reward = np.mean(iteration_stats["config_rewards"])
            self.training_stats["config_rewards"].append(avg_config_reward)
            self.training_stats["valid_kernels_found"] += iteration_stats["valid_found"]

            if (iteration + 1) % 10 == 0:
                print(
                    f"Iter {iteration + 1}: "
                    f"config_reward={avg_config_reward:.3f}, "
                    f"valid={iteration_stats['valid_found']}, "
                    f"best_latency={self.training_stats['best_latency']:.3f}"
                )

        return {
            "mode": "joint",
            "iterations": self.config.max_iterations,
            "stats": self.training_stats,
        }

    def _train_hierarchical(
        self,
        target_graphs: List[str],
        env_config: Optional[HierarchicalEnvConfig],
    ) -> Dict[str, Any]:
        """
        Hierarchical training: Level 2 first, then Level 1.

        Phase 1: Train Level 2 with canonical configs
        Phase 2: Train Level 1 with frozen Level 2
        """
        print("=== Hierarchical Training ===")

        if env_config is None:
            env_config = HierarchicalEnvConfig()

        # Phase 1: Train Level 2 with fixed configs
        print("\nPhase 1: Training Graph Policy (Level 2)...")
        canonical_configs = self._get_canonical_configs()

        for cfg_idx, hw_config in enumerate(canonical_configs):
            constraints = SearchSpaceConstraints(hw_config)

            for target in target_graphs:
                env_config.target_graph_json = target
                graph_env = ConstrainedGraphEnv(constraints, vars(env_config))

                obs, _ = graph_env.reset()

                for step in range(self.config.graph_steps_per_config):
                    action = graph_env.action_space.sample()
                    obs, reward, done, truncated, info = graph_env.step(action)

                    self.training_stats["graph_rewards"].append(reward)

                    if done or truncated:
                        if info.get("verified", False):
                            self.training_stats["valid_kernels_found"] += 1
                        break

            if (cfg_idx + 1) % 2 == 0:
                avg_reward = np.mean(self.training_stats["graph_rewards"][-100:])
                print(
                    f"  Config {cfg_idx + 1}/{len(canonical_configs)}: "
                    f"avg_reward={avg_reward:.3f}"
                )

        # Phase 2: Train Level 1 with trained Level 2
        print("\nPhase 2: Training Config Policy (Level 1)...")

        for iteration in range(self.config.max_iterations // 2):
            for target in target_graphs:
                env_config.target_graph_json = target
                env = HierarchicalSearchEnv(vars(env_config))

                obs, _ = env.reset()

                for _ in range(self.config.config_episodes_per_iter):
                    action = env.action_space.sample()
                    obs, reward, done, truncated, info = env.step(action)

                    self.training_stats["config_rewards"].append(reward)

                    if done:
                        break

            if (iteration + 1) % 10 == 0:
                avg_reward = np.mean(self.training_stats["config_rewards"][-50:])
                print(f"  Iter {iteration + 1}: avg_reward={avg_reward:.3f}")

        return {
            "mode": "hierarchical",
            "iterations": self.config.max_iterations,
            "stats": self.training_stats,
        }

    def _train_codesign(
        self,
        target_graphs: List[str],
        env_config: Optional[HierarchicalEnvConfig],
    ) -> Dict[str, Any]:
        """
        Three-level co-design training: Level 0 + Level 1 + Level 2.

        Level 0 (AccelForge): Explore accelerator architectures
        Level 1 (Config): Select kernel configurations constrained by Level 0
        Level 2 (Graph): Build µGraph constrained by Level 1

        Uses Pareto front tracking for multi-objective optimization.
        """
        from .accelerator_space import AcceleratorEnv

        print("=== Hardware-Software Co-Design Training ===")

        if env_config is None:
            env_config = HierarchicalEnvConfig()

        # Initialize Level 0 environment
        accel_env_config = {
            "target_workload_json": "{}",
            "area_budget_mm2": self.config.area_budget_mm2,
            "power_budget_mw": self.config.power_budget_mw,
            "max_design_episodes": self.config.accelerator_episodes_per_iter,
        }
        accel_env = AcceleratorEnv(accel_env_config)

        codesign_stats = {
            "design_rewards": [],
            "config_rewards": [],
            "graph_rewards": [],
            "pareto_front_size": [],
            "best_latency": float("inf"),
            "best_energy": float("inf"),
        }

        for iteration in range(self.config.max_iterations):
            iter_stats = {"design_rewards": [], "config_rewards": [], "valid_found": 0}

            for target in target_graphs:
                # Level 0: Explore accelerator designs
                accel_obs, _ = accel_env.reset(
                    options={"target_workload_json": target}
                )

                for design_ep in range(self.config.accelerator_episodes_per_iter):
                    # Sample accelerator design
                    accel_action = accel_env.action_space.sample()
                    accel_obs, accel_reward, accel_done, _, accel_info = accel_env.step(
                        accel_action
                    )
                    iter_stats["design_rewards"].append(accel_reward)

                    # Run Level 1+2 under this design
                    # **Coupling**: propagate Level 0 design into Level 1+2
                    env_config.target_graph_json = target
                    env_config.backend = "accelforge"

                    # Propagate design point so Level 1+2 use it
                    design_dict = accel_info.get("design", {})
                    metrics_dict = accel_info.get("metrics", {})
                    constraints_from_l0 = accel_info.get("constraints", None)

                    env_config.accelforge_design = design_dict
                    env_config.accelerator_constraints = constraints_from_l0

                    # Build hardware profile from Level 0 design
                    from ..hardware.accelforge_bridge import (
                        AccelForgeBridge,
                        AccelForgeDesignPoint,
                        AccelForgeMetrics,
                    )
                    bridge = AccelForgeBridge()
                    l0_design = AccelForgeDesignPoint.from_dict(design_dict)
                    l0_metrics = (
                        AccelForgeMetrics.from_dict(metrics_dict)
                        if metrics_dict
                        else None
                    )
                    env_config.hardware_profile = bridge.to_hardware_profile(
                        l0_design, l0_metrics
                    )

                    env = HierarchicalSearchEnv(vars(env_config))
                    obs, info = env.reset()

                    for config_ep in range(self.config.config_episodes_per_iter):
                        action = env.action_space.sample()
                        obs, reward, done, truncated, info = env.step(action)
                        iter_stats["config_rewards"].append(reward)

                        level2_result = info.get("level2_result", {})
                        if level2_result.get("verified", False):
                            iter_stats["valid_found"] += 1
                            latency = level2_result.get("latency_ms", float("inf"))

                            # Add AccelForge metrics to result
                            if accel_info.get("metrics"):
                                metrics = accel_info["metrics"]
                                level2_result["energy_pj"] = metrics.get(
                                    "energy_per_op_pj", 0.0
                                )
                                level2_result["area_mm2"] = metrics.get("area_mm2", 0.0)
                                level2_result["power_mw"] = metrics.get(
                                    "total_power_mw", 0.0
                                )

                            # Report to Level 0
                            accel_env.set_level1_result(level2_result)

                            if latency < codesign_stats["best_latency"]:
                                codesign_stats["best_latency"] = latency

                        if done:
                            break

                    if accel_done:
                        break

            # Update stats
            if iter_stats["design_rewards"]:
                codesign_stats["design_rewards"].append(
                    np.mean(iter_stats["design_rewards"])
                )
            if iter_stats["config_rewards"]:
                codesign_stats["config_rewards"].append(
                    np.mean(iter_stats["config_rewards"])
                )
            codesign_stats["pareto_front_size"].append(accel_env.pareto_tracker.size())

            if (iteration + 1) % 10 == 0:
                avg_design = np.mean(codesign_stats["design_rewards"][-10:])
                avg_config = np.mean(codesign_stats["config_rewards"][-10:])
                print(
                    f"Iter {iteration + 1}: "
                    f"design_reward={avg_design:.3f}, "
                    f"config_reward={avg_config:.3f}, "
                    f"valid={iter_stats['valid_found']}, "
                    f"pareto_size={accel_env.pareto_tracker.size()}, "
                    f"best_latency={codesign_stats['best_latency']:.3f}"
                )

        return {
            "mode": "codesign",
            "iterations": self.config.max_iterations,
            "stats": codesign_stats,
            "pareto_front": accel_env.get_pareto_front(),
        }

    def _get_canonical_configs(self) -> List[HardwareConfig]:
        """Get canonical hardware configs for Phase 1 training."""
        return [
            # Small configs
            HardwareConfig(grid_dim_x=1, grid_dim_y=1, block_dim_x=128, forloop_range=1),
            HardwareConfig(grid_dim_x=4, grid_dim_y=1, block_dim_x=128, forloop_range=4),
            # Medium configs
            HardwareConfig(grid_dim_x=8, grid_dim_y=8, block_dim_x=256, forloop_range=8),
            HardwareConfig(grid_dim_x=16, grid_dim_y=4, block_dim_x=128, forloop_range=16),
            # Large configs
            HardwareConfig(grid_dim_x=32, grid_dim_y=8, block_dim_x=256, forloop_range=32),
            HardwareConfig(grid_dim_x=64, grid_dim_y=16, block_dim_x=512, forloop_range=64),
        ]


class HierarchicalSearchCoordinator:
    """
    Coordinates distributed hierarchical search with Ray.
    """

    def __init__(
        self,
        num_workers: int = 4,
        num_gpus: int = 1,
    ):
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self._ray_available = False

        try:
            import ray

            self._ray_available = True
        except ImportError:
            pass

    def parallel_search(
        self,
        target_graph_json: str,
        num_config_trials: int = 100,
        graph_steps_per_config: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Parallel hierarchical search.

        Each worker explores different configs in parallel.
        """
        if not self._ray_available:
            return self._local_search(
                target_graph_json,
                num_config_trials,
                graph_steps_per_config,
            )

        import ray

        if not ray.is_initialized():
            ray.init()

        # Create config search workers
        @ray.remote(num_cpus=1)
        def search_config(
            config_action: np.ndarray,
            target_graph: str,
            max_graph_steps: int,
        ) -> Dict[str, Any]:
            """Search one config."""
            from .config_space import ConfigActionSpace, SearchSpaceConstraints
            from .hierarchical_env import ConstrainedGraphEnv, HierarchicalEnvConfig

            # Decode config
            action_space = ConfigActionSpace()
            hw_config = action_space.decode_flat(config_action)
            constraints = SearchSpaceConstraints(hw_config)

            # Run graph search
            env_config = HierarchicalEnvConfig(target_graph_json=target_graph)
            graph_env = ConstrainedGraphEnv(constraints, vars(env_config))

            obs, _ = graph_env.reset()

            best_result = {"verified": False, "latency_ms": float("inf")}

            for step in range(max_graph_steps):
                action = graph_env.action_space.sample()
                obs, reward, done, truncated, info = graph_env.step(action)

                if done or truncated:
                    if info.get("verified", False):
                        latency = info.get("latency_ms", float("inf"))
                        if latency < best_result["latency_ms"]:
                            best_result = {
                                "verified": True,
                                "latency_ms": latency,
                                "config": hw_config.to_dict(),
                            }
                    break

            return best_result

        # Sample configs and search in parallel
        action_space = ConfigActionSpace()
        futures = []

        for _ in range(num_config_trials):
            config_action = action_space.flat_space.sample()
            future = search_config.remote(
                config_action,
                target_graph_json,
                graph_steps_per_config,
            )
            futures.append(future)

        results = ray.get(futures)

        # Filter and sort
        valid_results = [r for r in results if r.get("verified", False)]
        valid_results.sort(key=lambda r: r.get("latency_ms", float("inf")))

        return valid_results

    def _local_search(
        self,
        target_graph_json: str,
        num_config_trials: int,
        graph_steps_per_config: int,
    ) -> List[Dict[str, Any]]:
        """Local (non-Ray) search."""
        from .config_space import ConfigActionSpace, SearchSpaceConstraints
        from .hierarchical_env import ConstrainedGraphEnv, HierarchicalEnvConfig

        action_space = ConfigActionSpace()
        results = []

        for _ in range(num_config_trials):
            # Sample config
            config_action = action_space.flat_space.sample()
            hw_config = action_space.decode_flat(config_action)
            constraints = SearchSpaceConstraints(hw_config)

            # Run graph search
            env_config = HierarchicalEnvConfig(target_graph_json=target_graph_json)
            graph_env = ConstrainedGraphEnv(constraints, vars(env_config))

            obs, _ = graph_env.reset()

            for step in range(graph_steps_per_config):
                action = graph_env.action_space.sample()
                obs, reward, done, truncated, info = graph_env.step(action)

                if done or truncated:
                    if info.get("verified", False):
                        results.append(
                            {
                                "verified": True,
                                "latency_ms": info.get("latency_ms", float("inf")),
                                "config": hw_config.to_dict(),
                            }
                        )
                    break

        # Sort by latency
        results.sort(key=lambda r: r.get("latency_ms", float("inf")))

        return results
