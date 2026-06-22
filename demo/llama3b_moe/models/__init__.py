# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Package init for llama3b_moe model modules."""

from .configuration_llama3b_moe import LLaMA3BMoEConfig
from .modeling_llama3b_moe import LLaMA3BMoEModel

__all__ = ["LLaMA3BMoEConfig", "LLaMA3BMoEModel"]
