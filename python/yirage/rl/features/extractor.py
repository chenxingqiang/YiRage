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
Graph feature extractor.

Interfaces with C++ to extract features from µGraph.
"""

from typing import Optional, Dict, Any
import json

from .mugraph_features import MuGraphFeature


class GraphFeatureExtractor:
    """
    Extracts features from µGraph.

    Two modes:
    1. C++ mode: Uses C++ interface to extract real features
    2. Fallback mode: Parses graph JSON for basic features
    """

    def __init__(self, use_cpp: bool = True):
        self.use_cpp = use_cpp
        self._cpp_extractor = None

        if use_cpp:
            self._init_cpp_extractor()

    def _init_cpp_extractor(self):
        """Initialize C++ feature extractor."""
        try:
            # Try to import C++ bindings
            from yirage._cython.rl_core import RLSearchContext

            self._cpp_available = True
        except ImportError:
            self._cpp_available = False
            self.use_cpp = False

    def extract(
        self,
        context: Optional[Any] = None,
        graph_json: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> MuGraphFeature:
        """
        Extract features from µGraph.

        Args:
            context: C++ RLSearchContext (if available)
            graph_json: Graph JSON string (fallback)
            config: Hardware configuration dict

        Returns:
            MuGraphFeature with all extracted features
        """
        if self.use_cpp and context is not None and self._cpp_available:
            return self._extract_from_cpp(context, config)
        elif graph_json is not None:
            return self._extract_from_json(graph_json, config)
        else:
            return MuGraphFeature()

    def _extract_from_cpp(
        self,
        context: Any,
        config: Optional[Dict[str, Any]],
    ) -> MuGraphFeature:
        """
        Extract features from C++ context.

        Calls C++ GraphFeatureExtractor through the context.
        """
        try:
            # Call C++ feature extraction
            if hasattr(context, "extract_features"):
                features_json = context.extract_features()
                return MuGraphFeature.from_json(features_json)
            elif hasattr(context, "get_state"):
                # Fallback: use state as features
                state = context.get_state()
                return self._state_to_features(state, config)
            else:
                return MuGraphFeature()
        except Exception as e:
            print(f"C++ feature extraction failed: {e}")
            return MuGraphFeature()

    def _extract_from_json(
        self,
        graph_json: str,
        config: Optional[Dict[str, Any]],
    ) -> MuGraphFeature:
        """
        Extract features from graph JSON (fallback mode).
        """
        features = MuGraphFeature.from_graph_json(graph_json)

        # Add config features if provided
        if config:
            grid = config.get("grid_dim", {})
            block = config.get("block_dim", {})

            features.grid_dim = (
                grid.get("x", 1),
                grid.get("y", 1),
                grid.get("z", 1),
            )
            features.block_dim = (
                block.get("x", 128),
                block.get("y", 1),
                block.get("z", 1),
            )
            features.forloop_range = config.get("forloop_range", 1)
            features.reduction_dimx = config.get("reduction_dimx", 1)

        # Compute derived features
        self._compute_derived_features(features)

        return features

    def _state_to_features(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]],
    ) -> MuGraphFeature:
        """Convert search state to features."""
        features = MuGraphFeature(
            num_operators=state.get("num_kn_operators", 0) + state.get("num_tb_operators", 0),
            num_tensors=state.get("num_tensors", 0),
            search_level=state.get("search_level", 0),
            search_depth=state.get("search_depth", 0),
        )

        # Add config
        if config:
            grid = config.get("grid_dim", state.get("current_grid_dim", {}))
            block = config.get("block_dim", state.get("current_block_dim", {}))

            if isinstance(grid, dict):
                features.grid_dim = (grid.get("x", 1), grid.get("y", 1), grid.get("z", 1))
            elif isinstance(grid, (list, tuple)):
                features.grid_dim = tuple(grid[:3]) if len(grid) >= 3 else (1, 1, 1)

            if isinstance(block, dict):
                features.block_dim = (block.get("x", 128), block.get("y", 1), block.get("z", 1))
            elif isinstance(block, (list, tuple)):
                features.block_dim = tuple(block[:3]) if len(block) >= 3 else (128, 1, 1)

        return features

    def _compute_derived_features(self, features: MuGraphFeature):
        """Compute derived features from basic features."""
        # Graph structure estimates
        if features.num_operators > 0:
            features.graph_depth = max(1, features.num_operators // 2)
            features.graph_width = max(1, features.num_operators // features.graph_depth)
            features.critical_path_length = features.graph_depth
            features.parallelism_degree = features.graph_width / max(features.graph_depth, 1)

        # Performance estimates
        total_threads = (
            features.grid_dim[0]
            * features.grid_dim[1]
            * features.grid_dim[2]
            * features.block_dim[0]
            * features.block_dim[1]
            * features.block_dim[2]
        )

        # Rough occupancy estimate (SM utilization)
        max_threads_per_sm = 2048
        threads_per_block = features.block_dim[0] * features.block_dim[1] * features.block_dim[2]
        blocks_per_sm = max(1, max_threads_per_sm // threads_per_block)
        features.occupancy = min(1.0, (blocks_per_sm * threads_per_block) / max_threads_per_sm)

        # Memory estimates
        features.shared_mem_usage = 0.5  # Placeholder
        features.register_usage = 0.5  # Placeholder

    def extract_with_accelforge(
        self,
        context: Optional[Any] = None,
        graph_json: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        hardware_profile: Optional[Any] = None,
    ) -> MuGraphFeature:
        """
        Extract features including AccelForge hardware design metrics.

        Extends standard extraction with energy, area, and power features
        from AccelForge modeling.

        Args:
            context: C++ RLSearchContext (if available)
            graph_json: Graph JSON string (fallback)
            config: Hardware configuration dict
            hardware_profile: HardwareProfile instance (with AccelForge extensions)

        Returns:
            MuGraphFeature with AccelForge hardware metrics included
        """
        # Get base features
        features = self.extract(context, graph_json, config)

        # Enrich with AccelForge metrics if hardware_profile has them
        if hardware_profile is not None:
            af_metrics = None
            if hasattr(hardware_profile, "extensions"):
                af_metrics = hardware_profile.extensions.get("accelforge_metrics")

            if af_metrics:
                features.energy_per_op_pj = af_metrics.get("energy_per_op_pj", 0.0)
                features.area_mm2 = af_metrics.get("area_mm2", 0.0)
                features.total_power_mw = af_metrics.get("total_power_mw", 0.0)
                features.leak_power_mw = af_metrics.get("leak_power_mw", 0.0)
                features.pe_utilization = af_metrics.get("pe_utilization", 0.0)

        return features


# Convenience function
def extract_graph_features(
    context: Optional[Any] = None,
    graph_json: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    use_cpp: bool = True,
) -> MuGraphFeature:
    """
    Extract features from µGraph.

    Convenience function that creates extractor and extracts features.
    """
    extractor = GraphFeatureExtractor(use_cpp=use_cpp)
    return extractor.extract(context, graph_json, config)
