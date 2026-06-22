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
Hierarchical Search Environments

Two-level environment structure:
- Level 1 (ConfigEnv): Select hardware configuration
- Level 2 (ConstrainedGraphEnv): Build µGraph within config constraints
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import json
import numpy as np

from .config_space import (
    HardwareConfig,
    SearchSpaceConstraints,
    ConfigActionSpace,
    ConfigObservationSpace,
    GYM_AVAILABLE,
)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        # Use stubs from config_space
        from .config_space import spaces, gym

from .graph_space import (
    ConstrainedGraphActionSpace,
    GraphObservationSpace,
    GraphAction,
    GraphState,
)


def _get_positive_int_attr(obj: Any, name: str, default: int = 1) -> int:
    try:
        return max(1, int(getattr(obj, name, default)))
    except (TypeError, ValueError):
        return default


@dataclass
class HierarchicalEnvConfig:
    """Configuration for hierarchical search environment."""

    # Target
    target_graph_json: str = "{}"
    backend: str = "cuda"

    # Level 1 (Config)
    max_config_episodes: int = 10

    # Level 2 (Graph)
    max_graph_steps: int = 50
    max_kn_operators: int = 10
    max_tb_operators: int = 15
    max_tensors: int = 16

    # GPU
    num_gpus: int = 1

    # Multi-objective reward weights (AccelForge integration)
    reward_weight_latency: float = 0.5
    reward_weight_energy: float = 0.2
    reward_weight_area: float = 0.15
    reward_weight_power: float = 0.15

    # Budgets for area/power constraints
    area_budget_mm2: float = 100.0
    power_budget_mw: float = 5000.0
    baseline_energy_pj: float = 10.0

    # Hardware coupling — populated by Level 0 (AccelForge) or by user
    hardware_profile: Optional[Any] = None  # HardwareProfile from Level 0 or detection
    accelforge_design: Optional[Any] = None  # AccelForgeDesignPoint from Level 0
    accelerator_constraints: Optional[Any] = None  # AcceleratorDesignConstraints from Level 0


class ConfigEnv(gym.Env):
    """
    Level 1 Environment: Hardware Configuration Selection

    State: Target graph features + hardware capabilities
    Action: Hardware config parameters
    Reward: Based on Level 2 search results

    This env controls the search space for Level 2.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        if config is None:
            config = {}

        if isinstance(config, HierarchicalEnvConfig):
            self.config = config
        else:
            self.config = HierarchicalEnvConfig(**config)

        # Action/observation spaces
        self.action_space_helper = ConfigActionSpace()
        self.obs_space_helper = ConfigObservationSpace()

        self.action_space = self.action_space_helper.flat_space
        self.observation_space = self.obs_space_helper.space

        # State
        self.current_config: Optional[HardwareConfig] = None
        self.episode_results: List[Dict] = []
        self.episode_idx = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset for new config search episode."""
        if GYM_AVAILABLE:
            super().reset(seed=seed)

        if options and "target_graph_json" in options:
            self.config.target_graph_json = options["target_graph_json"]

        self.current_config = None
        self.episode_results = []
        self.episode_idx = 0

        obs = self._get_observation()

        return obs, {"episode_idx": 0}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Select a configuration.

        This doesn't immediately give reward - reward comes after
        Level 2 search is complete.
        """
        # Decode action to config
        self.current_config = self.action_space_helper.decode_flat(action)
        self._couple_config_with_hardware_profile()

        # Get constraints for Level 2
        constraints = self.get_current_constraints()

        self.episode_idx += 1
        done = self.episode_idx >= self.config.max_config_episodes

        obs = self._get_observation()

        # Reward is 0 here - will be set externally after Level 2 runs
        reward = 0.0

        info = {
            "config": self.current_config.to_dict(),
            "constraints": {
                "valid_imaps": len(constraints.valid_imaps),
                "valid_franges": len(constraints.valid_franges),
                "max_operators": constraints.max_operators,
                "max_shared_memory": constraints.max_shared_memory,
                "max_tensor_elements": constraints.max_tensor_elements,
                "supported_precisions": constraints.supported_precisions,
            },
            "episode_idx": self.episode_idx,
        }

        return obs, reward, done, False, info

    def set_level2_result(self, result: Dict[str, Any]):
        """
        Receive result from Level 2 search.

        This allows computing the true reward for the config choice.
        """
        self.episode_results.append(result)

    def get_config_reward(self) -> float:
        """
        Compute multi-objective reward for this config based on Level 2 results.

        Combines latency, energy, area, and power into a single scalar reward.
        Weights are configurable via HierarchicalEnvConfig.
        """
        if not self.episode_results:
            return -1.0

        # Best result under this config
        valid_results = [r for r in self.episode_results if r.get("verified", False)]

        if not valid_results:
            return -0.5

        best = min(valid_results, key=lambda r: r.get("latency_ms", float("inf")))
        latency = best.get("latency_ms", float("inf"))

        if latency == float("inf"):
            return -0.5

        # Latency reward (always present)
        w_lat = self.config.reward_weight_latency
        reward = w_lat * np.log(10.0 / latency + 1)

        # Energy reward (from AccelForge)
        energy_pj = best.get("energy_pj", 0.0)
        if energy_pj > 0:
            w_energy = self.config.reward_weight_energy
            baseline = self.config.baseline_energy_pj
            reward += w_energy * np.log(baseline / energy_pj + 1)

        # Area reward (from AccelForge)
        area_mm2 = best.get("area_mm2", 0.0)
        if area_mm2 > 0:
            w_area = self.config.reward_weight_area
            budget = self.config.area_budget_mm2
            reward += w_area * max(0.0, 1.0 - area_mm2 / budget)

        # Power reward (from AccelForge)
        power_mw = best.get("power_mw", 0.0)
        if power_mw > 0:
            w_power = self.config.reward_weight_power
            budget = self.config.power_budget_mw
            reward += w_power * np.log(budget / power_mw + 1)

        return reward

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get Level 1 observation."""
        graph_features = self.obs_space_helper.encode_target_graph(self.config.target_graph_json)
        hw_features = self.obs_space_helper.encode_hardware(
            self.config.backend,
            self._get_hardware_profile(),
        )

        # History features (from previous results)
        history = np.zeros(self.obs_space_helper.HISTORY_FEATURE_DIM, dtype=np.float32)
        if self.episode_results:
            latencies = [r.get("latency_ms", float("inf")) for r in self.episode_results]
            valid_latencies = [l for l in latencies if l < float("inf")]
            if valid_latencies:
                history[0] = np.mean(valid_latencies) / 10.0
                history[1] = np.min(valid_latencies) / 10.0
            history[2] = len(self.episode_results) / 10.0
            history[3] = (
                len(valid_latencies) / len(self.episode_results) if self.episode_results else 0
            )

        return {
            "target_graph_features": graph_features,
            "hardware_features": hw_features,
            "history_features": history,
        }

    def get_current_constraints(self) -> Optional[SearchSpaceConstraints]:
        """Get constraints from current config for Level 2."""
        if self.current_config is None:
            return None
        return (
            SearchSpaceConstraints(self.current_config)
            .apply_hardware_profile(self._get_hardware_profile())
            .apply_accelerator_constraints(self.config.accelerator_constraints)
        )

    def _get_hardware_profile(self) -> Optional[Any]:
        """Return the unified HardwareProfile if one is provided or derivable."""
        if self.config.hardware_profile is not None:
            if isinstance(self.config.hardware_profile, dict):
                try:
                    from ..hardware.profile import HardwareProfile

                    self.config.hardware_profile = HardwareProfile.from_dict(
                        self.config.hardware_profile
                    )
                except Exception:
                    return None
            return self.config.hardware_profile

        if self.config.accelforge_design is None:
            return None

        try:
            from ..hardware.profile import HardwareProfile

            self.config.hardware_profile = HardwareProfile.from_accelforge(
                self.config.accelforge_design
            )
            return self.config.hardware_profile
        except Exception:
            return None

    def _couple_config_with_hardware_profile(self) -> None:
        """Merge the Level 1 config action with hardware-derived limits."""
        hardware_profile = self._get_hardware_profile()
        if self.current_config is None or hardware_profile is None:
            return

        try:
            from ..hardware.config_coupling import ConfigGenerator

            generated = ConfigGenerator(hardware_profile).generate()
        except Exception:
            generated = None

        if getattr(hardware_profile, "backend", "") == "accelforge" and generated is not None:
            self.current_config = generated
            return

        max_threads = _get_positive_int_attr(hardware_profile, "max_threads_per_block")
        max_shared_memory = _get_positive_int_attr(hardware_profile, "max_shared_memory_per_block")
        max_registers_per_thread = _get_positive_int_attr(
            hardware_profile,
            "max_registers_per_thread",
        )
        max_grid = getattr(hardware_profile, "max_grid_dim", (1, 1, 1))

        self.current_config.grid_dim_x = min(self.current_config.grid_dim_x, int(max_grid[0]))
        self.current_config.grid_dim_y = min(self.current_config.grid_dim_y, int(max_grid[1]))
        self.current_config.grid_dim_z = min(self.current_config.grid_dim_z, int(max_grid[2]))
        self.current_config.block_dim_x = min(self.current_config.block_dim_x, max_threads)
        self.current_config.shared_memory_size = min(
            self.current_config.shared_memory_size,
            max_shared_memory,
        )
        self.current_config.num_registers = min(
            self.current_config.num_registers,
            max_registers_per_thread,
        )


class ConstrainedGraphEnv(gym.Env):
    """
    Level 2 Environment: µGraph Construction within Constraints

    State: Current graph + config constraints
    Action: Graph operations (constrained by Level 1 config)
    Reward: GPU verification + performance

    This env operates WITHIN the constraints set by Level 1.
    """

    def __init__(
        self,
        constraints: SearchSpaceConstraints,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.constraints = constraints

        if config is None:
            config = {}
        if isinstance(config, HierarchicalEnvConfig):
            self.env_config = config
        else:
            self.env_config = HierarchicalEnvConfig(**config)

        # Action/observation spaces (CONSTRAINED)
        self.action_space_helper = ConstrainedGraphActionSpace(
            constraints,
            max_tensors=self.env_config.max_tensors,
        )
        self.obs_space_helper = GraphObservationSpace(
            constraints,
            max_tensors=self.env_config.max_tensors,
            max_operators=constraints.max_operators,
        )

        self.action_space = self.action_space_helper.flat_space
        self.observation_space = self.obs_space_helper.space

        # State
        self.state: Optional[GraphState] = None
        self.kernel_graph_json: str = "{}"
        self.step_count: int = 0

    @property
    def current_kernel_graph_json(self) -> str:
        """Backward-compatible alias for ``kernel_graph_json``."""
        return self.kernel_graph_json

    @current_kernel_graph_json.setter
    def current_kernel_graph_json(self, value: str) -> None:
        self.kernel_graph_json = value

    def _coerce_graph_json(self, value: Any) -> str:
        if not value or value == "{}":
            return "{}"
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def _resolve_kernel_graph_json(self) -> str:
        """Return the best available µGraph JSON for AccelForge / profiling."""
        if self.kernel_graph_json and self.kernel_graph_json != "{}":
            return self.kernel_graph_json

        target = self.env_config.target_graph_json
        if target and target != "{}":
            return self._coerce_graph_json(target)
        return "{}"

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset for new graph search episode."""
        if GYM_AVAILABLE:
            super().reset(seed=seed)

        # Update constraints if provided
        if options and "constraints" in options:
            self.constraints = options["constraints"]
            self._update_spaces()

        self.state = GraphState(
            search_level=0,
            num_kn_operators=0,
            num_tb_operators=0,
            num_tensors=2,  # Initial inputs
        )
        if options and "kernel_graph_json" in options:
            self.kernel_graph_json = self._coerce_graph_json(options["kernel_graph_json"])
        elif self.env_config.target_graph_json and self.env_config.target_graph_json != "{}":
            self.kernel_graph_json = self._coerce_graph_json(self.env_config.target_graph_json)
        else:
            self.kernel_graph_json = "{}"
        self.step_count = 0

        obs = self.obs_space_helper.encode(self.state, self.kernel_graph_json)

        return obs, {"step": 0, "kernel_graph_json_set": self.kernel_graph_json != "{}"}

    def _update_spaces(self):
        """Update action/obs spaces for new constraints."""
        self.action_space_helper = ConstrainedGraphActionSpace(
            self.constraints,
            max_tensors=self.env_config.max_tensors,
        )
        self.obs_space_helper = GraphObservationSpace(
            self.constraints,
            max_tensors=self.env_config.max_tensors,
            max_operators=self.constraints.max_operators,
        )
        self.action_space = self.action_space_helper.flat_space
        self.observation_space = self.obs_space_helper.space

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one graph construction step.

        Actions are constrained by Level 1 config through:
        - Valid imap choices
        - Valid frange choices
        - Resource limits
        """
        self.step_count += 1

        # Decode action (respects constraints)
        graph_action = self.action_space_helper.decode_flat(action)

        terminated = False
        truncated = False
        reward = 0.0
        info = {
            "action_type": graph_action.action_type,
            "step": self.step_count,
        }

        # Apply action
        if graph_action.action_type == GraphAction.ADD_KN_OP:
            self.state.num_kn_operators += 1
            self.state.num_tensors += 1
            info["operator"] = graph_action.operator

        elif graph_action.action_type == GraphAction.CREATE_TB:
            self.state.search_level = 1
            info["imap"] = graph_action.imap
            info["frange"] = graph_action.frange

        elif graph_action.action_type == GraphAction.ADD_TB_OP:
            self.state.num_tb_operators += 1
            info["operator"] = graph_action.operator

        elif graph_action.action_type == GraphAction.FINISH:
            terminated = True
            # GPU verification would happen here
            info["verified"] = self._simulate_verification()
            if info["verified"]:
                info["latency_ms"] = self._simulate_profiling()
                # Multi-objective reward
                reward = 1.0 + np.log(10.0 / info["latency_ms"] + 1)

                # AccelForge metrics (if available from hardware)
                af_metrics = self._get_accelforge_metrics()
                if af_metrics:
                    info["accelforge_metrics"] = af_metrics
                    info["energy_pj"] = af_metrics.get("energy_per_op_pj", 0.0)
                    info["area_mm2"] = af_metrics.get("area_mm2", 0.0)
                    info["power_mw"] = af_metrics.get("total_power_mw", 0.0)
                    info["latency_ms_af"] = af_metrics.get("latency_ms", 0.0)
                    # Bonus for energy efficiency
                    if info["energy_pj"] > 0:
                        reward += 0.2 * np.log(10.0 / info["energy_pj"] + 1)
            else:
                reward = -0.5

        # Check limits
        if self.step_count >= self.env_config.max_graph_steps:
            truncated = True

        total_ops = self.state.num_kn_operators + self.state.num_tb_operators
        if total_ops >= self.constraints.max_operators:
            truncated = True

        # Small step penalty
        reward -= 0.01

        obs = self.obs_space_helper.encode(self.state, self.kernel_graph_json)

        return obs, reward, terminated, truncated, info

    def _simulate_verification(self) -> bool:
        """Simulate GPU verification (placeholder)."""
        # Real implementation would call C++ fingerprint verification
        has_kn = self.state.num_kn_operators > 0
        has_tb = self.state.num_tb_operators > 0
        return has_kn and has_tb

    def _simulate_profiling(self) -> float:
        """Simulate GPU profiling (placeholder)."""
        # Real implementation would run actual kernel
        base = 1.0
        # More ops = slightly slower
        total_ops = self.state.num_kn_operators + self.state.num_tb_operators
        return base + 0.1 * total_ops + np.random.random() * 0.5

    def _get_accelforge_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get AccelForge metrics using the coupled hardware design.

        Uses the design point from env_config (propagated from Level 0)
        instead of creating a standalone default.  The workload is derived
        from the current µGraph via ``mugraph_to_workload()`` so AccelForge
        models the actual kernel structure, not a synthetic proxy.

        Returns None for non-AccelForge backends or when no design is
        available.
        """
        if self.env_config.backend != "accelforge":
            return None

        try:
            from ..hardware.accelforge_bridge import (
                AccelForgeBridge,
                AccelForgeDesignPoint,
                mugraph_to_workload,
            )

            bridge = AccelForgeBridge()

            # Use the coupled design from Level 0 / env_config, not a standalone default
            design = None
            if self.env_config.accelforge_design is not None:
                if isinstance(self.env_config.accelforge_design, dict):
                    design = AccelForgeDesignPoint.from_dict(
                        self.env_config.accelforge_design
                    )
                else:
                    design = self.env_config.accelforge_design
            elif (
                self.env_config.hardware_profile is not None
                and hasattr(self.env_config.hardware_profile, "extensions")
            ):
                af_design_dict = self.env_config.hardware_profile.extensions.get(
                    "accelforge_design", {}
                )
                if af_design_dict:
                    design = AccelForgeDesignPoint.from_dict(af_design_dict)

            if design is None:
                design = AccelForgeDesignPoint()  # Fallback to default

            # Build workload from the actual µGraph (priority-0 path: real M/K/N)
            # Fall back to operator-count heuristic if graph JSON is not yet set.
            graph_json = self._resolve_kernel_graph_json()
            if graph_json and graph_json != "{}":
                workload = mugraph_to_workload(graph_json)
            else:
                workload = {
                    "estimated_flops": (
                        self.state.num_kn_operators + self.state.num_tb_operators
                    ) * 1e6,
                }

            metrics = bridge.evaluate(design, workload)
            return metrics.to_dict()
        except Exception:
            return None


class HierarchicalSearchEnv(gym.Env):
    """
    Combined Hierarchical Environment

    Manages both Level 1 (Config) and Level 2 (Graph) environments.

    Episode structure:
    1. Level 1 selects config
    2. Level 2 runs graph search within config constraints
    3. GPU verification
    4. Reward to both levels
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        if config is None:
            config = {}
        if isinstance(config, HierarchicalEnvConfig):
            self.config = config
        else:
            self.config = HierarchicalEnvConfig(**config)

        # Default config for initial action space
        default_hw = HardwareConfig()
        default_constraints = SearchSpaceConstraints(default_hw)

        # Level 1 env
        self.config_env = ConfigEnv(config)

        # Level 2 env (will be reset with constraints from Level 1)
        self.graph_env = ConstrainedGraphEnv(default_constraints, config)

        # State
        self.current_level: int = 1  # 1 or 2
        self.current_hw_config: Optional[HardwareConfig] = None
        self.current_constraints: Optional[SearchSpaceConstraints] = None

        # For combined action space, we use Level 1 action space
        # and internally handle Level 2
        self.action_space = self.config_env.action_space
        self.observation_space = self._build_combined_obs_space()

    def _build_combined_obs_space(self) -> spaces.Dict:
        """Build combined observation space."""
        return spaces.Dict(
            {
                # Current level indicator
                "level": spaces.Discrete(2),
                # Level 1 observations
                "config_obs": self.config_env.observation_space,
                # Level 2 observations (when in Level 2)
                "graph_obs": self.graph_env.observation_space,
            }
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset hierarchical environment."""
        if GYM_AVAILABLE:
            super().reset(seed=seed)

        self.current_level = 1

        # Reset Level 1
        config_obs, config_info = self.config_env.reset(seed=seed, options=options)

        obs = {
            "level": 0,  # In config selection phase
            "config_obs": config_obs,
            "graph_obs": self._empty_graph_obs(),
        }

        return obs, {"level": 1, **config_info}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute hierarchical step.

        If in Level 1: Select config, then auto-run Level 2.
        Level 0 accelerator constraints (if present) flow through to Level 2.
        """
        if self.current_level == 1:
            # Level 1: Config selection
            config_obs, _, config_done, _, config_info = self.config_env.step(action)

            # Get constraints for Level 2
            self.current_hw_config = self.config_env.current_config
            self.current_constraints = self.config_env.get_current_constraints()

            # Run Level 2 search with these constraints
            level2_result = self._run_level2_search()

            # Report result to Level 1
            self.config_env.set_level2_result(level2_result)

            # Get reward (based on Level 2 result)
            if config_done:
                reward = self.config_env.get_config_reward()
            else:
                reward = 0.1 if level2_result.get("verified", False) else -0.1

            obs = {
                "level": 0,
                "config_obs": config_obs,
                "graph_obs": self._empty_graph_obs(),
            }

            info = {
                "level": 1,
                "config": config_info.get("config"),
                "level2_result": level2_result,
            }

            return obs, reward, config_done, False, info

        else:
            raise ValueError("Unexpected level")

    def _run_level2_search(self, max_steps: int = 30) -> Dict[str, Any]:
        """
        Run Level 2 search with current constraints.

        This is an internal loop - Level 2 runs to completion.
        The graph env inherits the AccelForge design from the config
        so metrics are computed for the coupled hardware.
        """
        if self.current_constraints is None:
            return {"verified": False, "reason": "no_constraints"}

        # Reset Level 2 with new constraints
        self.graph_env.constraints = self.current_constraints
        self.graph_env._update_spaces()

        # Propagate AccelForge design into graph env so _get_accelforge_metrics
        # uses the coupled design, not a standalone default
        if self.config.accelforge_design is not None:
            self.graph_env.env_config.accelforge_design = self.config.accelforge_design
        if self.config.hardware_profile is not None:
            self.graph_env.env_config.hardware_profile = self.config.hardware_profile
        if self.config.backend == "accelforge":
            self.graph_env.env_config.backend = "accelforge"

        reset_options: Dict[str, Any] = {}
        if self.config.target_graph_json and self.config.target_graph_json != "{}":
            reset_options["kernel_graph_json"] = self.config.target_graph_json
        graph_obs, _ = self.graph_env.reset(
            options=reset_options if reset_options else None
        )

        best_result = {"verified": False, "latency_ms": float("inf")}

        for step in range(max_steps):
            # Random action for now (would be from Level 2 policy)
            action = self.graph_env.action_space.sample()

            graph_obs, reward, terminated, truncated, info = self.graph_env.step(action)

            if terminated or truncated:
                if info.get("verified", False):
                    latency = info.get("latency_ms", float("inf"))
                    if latency < best_result.get("latency_ms", float("inf")):
                        best_result = {
                            "verified": True,
                            "latency_ms": latency,
                            "steps": step + 1,
                        }
                        # Propagate AccelForge metrics from graph env
                        for key in ("energy_pj", "area_mm2", "power_mw", "latency_ms_af"):
                            if key in info:
                                best_result[key] = info[key]
                        if "accelforge_metrics" in info:
                            best_result["accelforge_metrics"] = info["accelforge_metrics"]
                break

        return best_result

    def _empty_graph_obs(self) -> Dict[str, np.ndarray]:
        """Empty observation for Level 2 when not active."""
        return {
            "graph_embedding": np.zeros(128, dtype=np.float32),
            "search_level": 0,
            "num_kn_operators": np.array([0], dtype=np.float32),
            "num_tb_operators": np.array([0], dtype=np.float32),
            "num_tensors": np.array([0], dtype=np.float32),
            "config_embedding": np.zeros(32, dtype=np.float32),
            "valid_imap_mask": np.ones(27, dtype=np.int8),
            "valid_frange_mask": np.ones(7, dtype=np.int8),
            "valid_action_type_mask": np.ones(4, dtype=np.int8),
            "remaining_operators": np.array([30], dtype=np.float32),
        }
