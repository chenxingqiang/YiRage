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
Policy network with action masking for YiRage search.

Supports masking invalid actions based on current search state.
"""

from typing import Dict, List, Optional, Any

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


try:
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
    from ray.rllib.utils.torch_utils import FLOAT_MIN

    RLLIB_AVAILABLE = True
except ImportError:
    RLLIB_AVAILABLE = False


if TORCH_AVAILABLE and RLLIB_AVAILABLE:

    class ActionMaskingModel(TorchModelV2, nn.Module):
        """
        Policy network with action masking.

        Masks invalid actions to prevent the policy from
        selecting illegal moves during search.
        """

        def __init__(
            self,
            obs_space,
            action_space,
            num_outputs: int,
            model_config: Dict,
            name: str,
        ):
            TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
            nn.Module.__init__(self)

            self.internal_model = FullyConnectedNetwork(
                obs_space,
                action_space,
                num_outputs,
                model_config,
                name + "_internal",
            )

            # Action masking
            self.action_mask_key = "action_mask"

        def forward(
            self,
            input_dict: Dict[str, Any],
            state: List[torch.Tensor],
            seq_lens: torch.Tensor,
        ) -> tuple:
            """Forward pass with action masking."""

            # Get logits from internal model
            logits, state = self.internal_model(input_dict, state, seq_lens)

            # Apply action mask
            obs = input_dict.get("obs", {})
            if isinstance(obs, dict) and self.action_mask_key in obs:
                action_mask = obs[self.action_mask_key]

                # Mask invalid actions with very negative value
                inf_mask = torch.clamp(
                    torch.log(action_mask.float()),
                    min=FLOAT_MIN,
                )

                # Expand mask if needed
                if inf_mask.shape[-1] < logits.shape[-1]:
                    padding = torch.zeros(
                        *inf_mask.shape[:-1],
                        logits.shape[-1] - inf_mask.shape[-1],
                        device=inf_mask.device,
                    )
                    inf_mask = torch.cat([inf_mask, padding], dim=-1)

                logits = logits + inf_mask[:, : logits.shape[-1]]

            return logits, state

        def value_function(self) -> torch.Tensor:
            """Return value estimate."""
            return self.internal_model.value_function()


# Non-RLlib version for standalone use
if TORCH_AVAILABLE:

    class StandalonePolicyNetwork(nn.Module):
        """
        Standalone policy network without RLlib.

        For use in custom training loops.
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dims: List[int] = [256, 256],
        ):
            super().__init__()

            # Policy network
            layers = []
            prev_dim = obs_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)
            self.policy_head = nn.Linear(prev_dim, action_dim)
            self.value_head = nn.Linear(prev_dim, 1)

        def forward(
            self,
            obs: torch.Tensor,
            action_mask: Optional[torch.Tensor] = None,
        ) -> tuple:
            """
            Forward pass.

            Args:
                obs: Observation tensor
                action_mask: Binary mask of valid actions

            Returns:
                (action_logits, value)
            """
            features = self.backbone(obs)
            logits = self.policy_head(features)
            value = self.value_head(features)

            # Apply action mask
            if action_mask is not None:
                mask_value = torch.finfo(logits.dtype).min
                logits = torch.where(
                    action_mask.bool(),
                    logits,
                    torch.full_like(logits, mask_value),
                )

            return logits, value

        def get_action(
            self,
            obs: torch.Tensor,
            action_mask: Optional[torch.Tensor] = None,
            deterministic: bool = False,
        ) -> torch.Tensor:
            """
            Sample action from policy.

            Args:
                obs: Observation tensor
                action_mask: Binary mask of valid actions
                deterministic: Whether to take argmax

            Returns:
                Action tensor
            """
            logits, _ = self.forward(obs, action_mask)

            if deterministic:
                return logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                return torch.multinomial(probs, num_samples=1).squeeze(-1)

else:

    class ActionMaskingModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and RLlib required")

    class StandalonePolicyNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required")
