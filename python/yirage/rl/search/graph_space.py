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
Level 2: µGraph Search Space (Constrained by Config)

The graph search operates within constraints defined by Level 1 config.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .config_space import (
    ALL_IMAPS,
    FORLOOP_RANGE_CHOICES,
    GYM_AVAILABLE,
    HardwareConfig,
    NumpyMultiDiscreteSpace,
    SearchSpaceConstraints,
)

# Import spaces from config_space to ensure consistent behavior
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


# All possible operators
KN_OPERATORS = [
    "MATMUL",
    "ADD",
    "MUL",
    "DIV",
    "EXP",
    "SILU",
    "GELU",
    "RELU",
    "REDUCTION",
    "RMS_NORM",
    "SOFTMAX",
]

TB_OPERATORS = [
    "MATMUL",
    "ADD",
    "MUL",
    "DIV",
    "EXP",
    "SILU",
    "REDUCTION_0",
    "REDUCTION_1",
    "REDUCTION_2",
    "CONCAT_0",
    "CONCAT_1",
    "FORLOOP_ACCUM",
    "SQUARE",
    "SQRT",
    "RMS_NORM",
]

ALL_OPERATORS = list(set(KN_OPERATORS + TB_OPERATORS))


@dataclass
class GraphAction:
    """
    Action for Level 2 graph construction.
    """

    action_type: int  # 0=ADD_KN, 1=CREATE_TB, 2=ADD_TB, 3=FINISH
    operator: Optional[str] = None
    inputs: Optional[List[int]] = None
    imap: Optional[Tuple[int, int, int]] = None
    omap: Optional[Tuple[int, int, int]] = None
    frange: Optional[int] = None

    # Action type constants
    ADD_KN_OP = 0
    CREATE_TB = 1
    ADD_TB_OP = 2
    FINISH = 3


class ConstrainedGraphActionSpace:
    """
    Level 2 Action Space: Graph construction within config constraints.

    The action space is DYNAMIC based on the Level 1 config:
    - Valid imaps depend on config.grid_dim
    - Valid franges depend on config.forloop_range
    - Max operators depend on resources
    """

    def __init__(
        self,
        constraints: SearchSpaceConstraints,
        max_tensors: int = 16,
    ):
        self.constraints = constraints
        self.max_tensors = max_tensors

        # Valid choices (constrained by Level 1)
        self.valid_imaps = constraints.valid_imaps
        self.valid_franges = constraints.valid_franges
        self.max_operators = constraints.max_operators

        # Build action space
        self.space = spaces.Dict(
            {
                # Action type
                "action_type": spaces.Discrete(4),
                # Operator selection
                "kn_operator": spaces.Discrete(len(KN_OPERATORS)),
                "tb_operator": spaces.Discrete(len(TB_OPERATORS)),
                # Input tensor indices
                "input_0": spaces.Discrete(max_tensors),
                "input_1": spaces.Discrete(max_tensors),
                # imap: index into VALID imaps (not all imaps!)
                "imap_idx": spaces.Discrete(max(1, len(self.valid_imaps))),
                # omap: index into VALID imaps
                "omap_idx": spaces.Discrete(max(1, len(self.valid_imaps))),
                # frange: index into VALID franges (not all!)
                "frange_idx": spaces.Discrete(max(1, len(self.valid_franges))),
            }
        )

        flat_nvec = [
            4,  # action_type
            len(KN_OPERATORS),
            len(TB_OPERATORS),
            max_tensors,
            max_tensors,
            max(1, len(self.valid_imaps)),
            max(1, len(self.valid_imaps)),
            max(1, len(self.valid_franges)),
        ]
        if GYM_AVAILABLE:
            self.flat_space = spaces.MultiDiscrete(flat_nvec)
        else:
            self.flat_space = NumpyMultiDiscreteSpace(flat_nvec)

    def sample(self) -> Dict[str, int]:
        return self.space.sample()

    def decode(self, action: Dict[str, int]) -> GraphAction:
        """
        Decode action to GraphAction.

        The decoding ensures the action respects constraints
        by selecting from valid choices only.
        """
        action_type = action["action_type"]

        # Select operator based on action type
        if action_type == GraphAction.ADD_KN_OP:
            operator = KN_OPERATORS[action["kn_operator"] % len(KN_OPERATORS)]
        elif action_type in [GraphAction.CREATE_TB, GraphAction.ADD_TB_OP]:
            operator = TB_OPERATORS[action["tb_operator"] % len(TB_OPERATORS)]
        else:
            operator = None

        # imap from VALID choices (constrained!)
        imap_idx = action["imap_idx"] % len(self.valid_imaps)
        imap = self.valid_imaps[imap_idx]

        # omap from VALID choices
        omap_idx = action["omap_idx"] % len(self.valid_imaps)
        omap = self.valid_imaps[omap_idx]

        # frange from VALID choices (constrained!)
        frange_idx = action["frange_idx"] % len(self.valid_franges)
        frange = self.valid_franges[frange_idx]

        return GraphAction(
            action_type=action_type,
            operator=operator,
            inputs=[action["input_0"], action["input_1"]],
            imap=imap,
            omap=omap,
            frange=frange,
        )

    def decode_flat(self, action: np.ndarray) -> GraphAction:
        """Decode flattened action."""
        action_dict = {
            "action_type": int(action[0]),
            "kn_operator": int(action[1]),
            "tb_operator": int(action[2]),
            "input_0": int(action[3]),
            "input_1": int(action[4]),
            "imap_idx": int(action[5]),
            "omap_idx": int(action[6]),
            "frange_idx": int(action[7]),
        }
        return self.decode(action_dict)

    def get_action_mask(self, state: "GraphState") -> np.ndarray:
        """
        Compute which actions are valid in current state.

        Based on:
        - Search level (KN or TB)
        - Number of operators added
        - Resource constraints
        """
        # For now, simple mask for action_type
        mask = np.ones(4, dtype=np.int8)

        # Can't add TB ops at KN level
        if state.search_level == 0:
            mask[2] = 0  # ADD_TB_OP

        # Can't add KN ops at TB level
        if state.search_level == 1:
            mask[0] = 0  # ADD_KN_OP

        # Limit operators
        total_ops = state.num_kn_operators + state.num_tb_operators
        if total_ops >= self.max_operators:
            mask[0] = 0
            mask[2] = 0

        return mask


@dataclass
class GraphState:
    """
    State for Level 2 graph construction.
    """

    search_level: int = 0  # 0=KN, 1=TB
    num_kn_operators: int = 0
    num_tb_operators: int = 0
    num_tensors: int = 0

    # Current graph embedding (from GNN or features)
    graph_embedding: Optional[np.ndarray] = None


class GraphObservationSpace:
    """
    Level 2 Observation Space.

    Includes:
    - Current graph state
    - Config constraints (from Level 1)
    - Valid action masks
    """

    GRAPH_EMBEDDING_DIM = 128
    CONFIG_EMBEDDING_DIM = 32

    def __init__(
        self,
        constraints: SearchSpaceConstraints,
        max_tensors: int = 16,
        max_operators: int = 30,
    ):
        self.constraints = constraints
        self.max_tensors = max_tensors
        self.max_operators = max_operators

        self.space = spaces.Dict(
            {
                # Current graph embedding
                "graph_embedding": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.GRAPH_EMBEDDING_DIM,),
                    dtype=np.float32,
                ),
                # Search state
                "search_level": spaces.Discrete(2),
                "num_kn_operators": spaces.Box(0, max_operators, (1,), dtype=np.float32),
                "num_tb_operators": spaces.Box(0, max_operators, (1,), dtype=np.float32),
                "num_tensors": spaces.Box(0, max_tensors, (1,), dtype=np.float32),
                # Config constraints (from Level 1)
                "config_embedding": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.CONFIG_EMBEDDING_DIM,),
                    dtype=np.float32,
                ),
                # Valid action masks (based on constraints)
                "valid_imap_mask": spaces.MultiBinary(len(ALL_IMAPS)),
                "valid_frange_mask": spaces.MultiBinary(len(FORLOOP_RANGE_CHOICES)),
                "valid_action_type_mask": spaces.MultiBinary(4),
                # Remaining resources
                "remaining_operators": spaces.Box(0, max_operators, (1,), dtype=np.float32),
            }
        )

    def encode(
        self,
        state: GraphState,
        graph_json: str = "{}",
    ) -> Dict[str, np.ndarray]:
        """
        Encode current state to observation dict.
        """
        # Graph embedding (simple features for now)
        if state.graph_embedding is not None:
            graph_embedding = state.graph_embedding
        else:
            graph_embedding = self._extract_graph_features(graph_json)

        # Config embedding
        config_embedding = self.constraints.encode()

        # Masks
        imap_mask = self.constraints.get_imap_mask()
        frange_mask = self.constraints.get_frange_mask()

        # Action type mask
        action_mask = np.ones(4, dtype=np.int8)
        if state.search_level == 0:
            action_mask[2] = 0  # No TB ops at KN level
        if state.search_level == 1:
            action_mask[0] = 0  # No KN ops at TB level

        total_ops = state.num_kn_operators + state.num_tb_operators
        remaining = self.constraints.max_operators - total_ops

        if remaining <= 0:
            action_mask[0] = 0
            action_mask[2] = 0

        return {
            "graph_embedding": graph_embedding.astype(np.float32),
            "search_level": state.search_level,
            "num_kn_operators": np.array([state.num_kn_operators], dtype=np.float32),
            "num_tb_operators": np.array([state.num_tb_operators], dtype=np.float32),
            "num_tensors": np.array([state.num_tensors], dtype=np.float32),
            "config_embedding": config_embedding.astype(np.float32),
            "valid_imap_mask": imap_mask,
            "valid_frange_mask": frange_mask,
            "valid_action_type_mask": action_mask,
            "remaining_operators": np.array([remaining], dtype=np.float32),
        }

    def _extract_graph_features(self, graph_json: str) -> np.ndarray:
        """Simple feature extraction from graph JSON."""
        import json

        features = np.zeros(self.GRAPH_EMBEDDING_DIM, dtype=np.float32)

        try:
            graph = json.loads(graph_json)

            ops = graph.get("operators", [])
            tensors = graph.get("tensors", [])

            features[0] = len(ops) / 20.0
            features[1] = len(tensors) / 20.0

            # Operator type distribution
            for i, op in enumerate(ops[:10]):
                op_type = op.get("type", "")
                features[10 + i] = hash(op_type) % 100 / 100.0

        except:
            pass

        return features
