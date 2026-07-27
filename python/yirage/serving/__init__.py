# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""RuntimeFusion serving surface (FusionPlan / FusionCapsule / RF.step).

Product concepts (not Mirage MPK/µGraph):
- FusionPlan: searched/cached local execution plan
- FusionCapsule: engine-schedulable fused block
- RuntimeFusion: per-step select/orchestrate capsules from engine meta

Legacy symbols (``mugraph``, ``PersistentKernel``, …) may back implementations later;
this package is the public RF identity.
"""

from .capsule import FusionCapsule
from .mlp_capsule import MlpFusionCapsule, mlp_eager_numpy
from .plan import FusionPlan
from .runtime_fusion import RuntimeFusion, StepMeta, StepResult

__all__ = [
    "FusionPlan",
    "FusionCapsule",
    "MlpFusionCapsule",
    "mlp_eager_numpy",
    "RuntimeFusion",
    "StepMeta",
    "StepResult",
]
