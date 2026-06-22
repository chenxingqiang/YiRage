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
RLlib callbacks for YiRage search training.

Provides hooks for:
- Episode statistics collection
- Curriculum learning
- Checkpoint management
"""

from typing import Dict, Any, Optional


try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.env import BaseEnv
    from ray.rllib.evaluation import Episode
    from ray.rllib.evaluation.episode_v2 import EpisodeV2
    from ray.rllib.policy import Policy

    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False

    # Stub for when RLlib is not available
    class DefaultCallbacks:
        pass


class YiRageCallbacks(DefaultCallbacks):
    """
    Custom callbacks for YiRage RL search training.

    Monitors:
    - Valid kernel discovery rate
    - Best performance achieved
    - Search depth statistics
    - Curriculum progression
    """

    def __init__(self):
        if RLLIB_AVAILABLE:
            super().__init__()

        # Curriculum state
        self.curriculum_level = 0
        self.total_valid_kernels = 0
        self.total_episodes = 0

    def on_episode_start(
        self,
        *,
        worker,
        base_env: "BaseEnv",
        policies: Dict[str, "Policy"],
        episode: "Episode",
        **kwargs,
    ) -> None:
        """Called at episode start."""
        # Initialize episode custom data
        episode.custom_metrics["search_steps"] = 0
        episode.custom_metrics["valid_found"] = 0
        episode.custom_metrics["best_latency"] = float("inf")

    def on_episode_step(
        self,
        *,
        worker,
        base_env: "BaseEnv",
        policies: Dict[str, "Policy"],
        episode: "Episode",
        **kwargs,
    ) -> None:
        """Called on each episode step."""
        # Get last info
        info = episode.last_info_for()
        if info is None:
            return

        episode.custom_metrics["search_steps"] = info.get("episode_step", 0)

        if info.get("verified", False):
            episode.custom_metrics["valid_found"] = info.get("num_valid_found", 0)

            latency = info.get("latency_ms")
            if latency is not None:
                current_best = episode.custom_metrics.get("best_latency", float("inf"))
                episode.custom_metrics["best_latency"] = min(current_best, latency)

    def on_episode_end(
        self,
        *,
        worker,
        base_env: "BaseEnv",
        policies: Dict[str, "Policy"],
        episode: "Episode",
        **kwargs,
    ) -> None:
        """Called at episode end."""
        info = episode.last_info_for()
        if info is None:
            return

        # Final statistics
        episode.custom_metrics["final_search_depth"] = info.get("search_depth", 0)
        episode.custom_metrics["total_valid"] = info.get("num_valid_found", 0)

        best_latency = info.get("best_latency_ms", float("inf"))
        if best_latency < float("inf"):
            episode.custom_metrics["best_latency_ms"] = best_latency

        # Track global statistics
        self.total_episodes += 1
        self.total_valid_kernels += info.get("num_valid_found", 0)

    def on_train_result(
        self,
        *,
        algorithm,
        result: Dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Called after each training iteration.

        Use for:
        - Logging custom metrics
        - Curriculum advancement
        - Dynamic resource allocation
        """
        # Get average valid rate
        custom = result.get("custom_metrics", {})
        valid_mean = custom.get("total_valid_mean", 0)

        # Curriculum progression
        if valid_mean > 0.5 and self.curriculum_level < 3:
            self.curriculum_level += 1
            print(f"Curriculum advanced to level {self.curriculum_level}")

            # Could update env config here
            # algorithm.workers.foreach_env(
            #     lambda env: env.update_curriculum(self.curriculum_level)
            # )

        # Log custom stats
        result["yirage_total_valid_kernels"] = self.total_valid_kernels
        result["yirage_curriculum_level"] = self.curriculum_level


class CurriculumCallback(YiRageCallbacks):
    """
    Extended callbacks with curriculum learning support.

    Progressively increases problem difficulty:
    Level 0: Small graphs, simple ops
    Level 1: Medium graphs, more ops
    Level 2: Large graphs, full op set
    Level 3: Target production workloads
    """

    CURRICULUM_THRESHOLDS = [
        0.3,  # Level 0 -> 1: 30% valid rate
        0.5,  # Level 1 -> 2: 50% valid rate
        0.7,  # Level 2 -> 3: 70% valid rate
    ]

    def __init__(self):
        super().__init__()
        self.level_episodes = [0, 0, 0, 0]
        self.level_valid = [0, 0, 0, 0]

    def on_episode_end(
        self,
        *,
        worker,
        base_env: "BaseEnv",
        policies: Dict[str, "Policy"],
        episode: "Episode",
        **kwargs,
    ) -> None:
        super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs,
        )

        info = episode.last_info_for() or {}

        # Track per-level statistics
        self.level_episodes[self.curriculum_level] += 1
        self.level_valid[self.curriculum_level] += info.get("num_valid_found", 0)

    def on_train_result(
        self,
        *,
        algorithm,
        result: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        # Check curriculum progression
        if self.curriculum_level < len(self.CURRICULUM_THRESHOLDS):
            episodes = self.level_episodes[self.curriculum_level]

            if episodes >= 100:  # Minimum episodes before advancing
                valid_rate = self.level_valid[self.curriculum_level] / max(episodes, 1)

                threshold = self.CURRICULUM_THRESHOLDS[self.curriculum_level]
                if valid_rate >= threshold:
                    self.curriculum_level += 1
                    print(
                        f"Curriculum: Level {self.curriculum_level} "
                        f"(valid_rate={valid_rate:.2f})"
                    )
