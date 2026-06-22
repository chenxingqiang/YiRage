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
Training utilities for YiRage RL search.

Supports multiple training strategies:
- Traditional RL: PPO, GRPO
- LLM Fine-tuning: SFT, DPO, ORPO (via TRL)
- Efficient fine-tuning: LoRA, QLoRA
"""

from .trainer import train_rl_search, TrainingConfig
from .callbacks import YiRageCallbacks
from .grpo import (
    GRPOConfig,
    GRPOTrainer,
    GRPOBatch,
    GroupedReplayBuffer,
)

# TRL integration (optional)
try:
    from .trl_integration import (
        FineTuningConfig,
        MuGraphPolicyTrainer,
        MuGraphDatasetFormatter,
        TRLTrainerFactory,
        create_trainer,
        train_mugraph_policy,
    )

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

__all__ = [
    # Traditional RL
    "train_rl_search",
    "TrainingConfig",
    "YiRageCallbacks",
    # GRPO
    "GRPOConfig",
    "GRPOTrainer",
    "GRPOBatch",
    "GroupedReplayBuffer",
]

if TRL_AVAILABLE:
    __all__.extend(
        [
            "FineTuningConfig",
            "MuGraphPolicyTrainer",
            "MuGraphDatasetFormatter",
            "TRLTrainerFactory",
            "create_trainer",
            "train_mugraph_policy",
        ]
    )
