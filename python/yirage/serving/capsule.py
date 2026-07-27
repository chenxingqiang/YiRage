# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""FusionCapsule: engine-schedulable fused block."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, MutableMapping, Optional

from .plan import FusionPlan


class FusionCapsule(ABC):
    """Schedulable fused compute unit selected by RuntimeFusion.step."""

    def __init__(self, plan: FusionPlan):
        self.plan = plan

    @property
    def name(self) -> str:
        return self.plan.name

    @property
    def kind(self) -> str:
        return self.plan.kind

    @abstractmethod
    def execute(
        self,
        inputs: Mapping[str, Any],
        meta: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run fused capsule; return output tensor dict (must include ``hidden``)."""

    def inspect(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "plan": self.plan.to_dict(),
        }
