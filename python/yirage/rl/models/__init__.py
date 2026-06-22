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
Neural network models for YiRage RL search.

Provides:
- Graph encoders for kernel graph representation
- Custom policy networks with action masking
"""

from .graph_encoder import GraphEncoder, SimpleGraphEncoder

# ActionMaskingModel requires PyTorch + RLlib
try:
    from .policy_network import ActionMaskingModel
except ImportError:
    ActionMaskingModel = None

# Search Policy Network with save/load
try:
    from .search_policy import (
        SearchPolicyNetwork,
        GraphEncoder as NNGraphEncoder,
        ConfigEncoder,
        ModelCheckpoint,
    )

    _SEARCH_POLICY_AVAILABLE = True
except ImportError:
    _SEARCH_POLICY_AVAILABLE = False
    SearchPolicyNetwork = None
    NNGraphEncoder = None
    ConfigEncoder = None
    ModelCheckpoint = None

__all__ = [
    "GraphEncoder",
    "SimpleGraphEncoder",
    "ActionMaskingModel",
    # Search policy
    "SearchPolicyNetwork",
    "NNGraphEncoder",
    "ConfigEncoder",
    "ModelCheckpoint",
]
