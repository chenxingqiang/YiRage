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
from .engine_stub import (
    EngineAttentionMeta,
    EngineDecoderLayerStub,
    EngineModelStub,
    QWEN2_MLP_HF_ATTACH,
)
from .hybrid_model import HybridModelOverride, resolve_rf_mlp_layer_ids
from .kv_meta import (
    PagedKvMeta,
    attach_paged_kv_to_step_meta,
    block_tables_to_paged_kv,
    last_page_len_from_seq,
)
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
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
    "QWEN2_MLP_HF_ATTACH",
    "EngineAttentionMeta",
    "EngineDecoderLayerStub",
    "EngineModelStub",
    "RuntimeFusionMlpLayerOverride",
    "LayerForwardResult",
    "build_layer_mlp_capsule",
    "capsule_name_for_layer",
    "HybridModelOverride",
    "resolve_rf_mlp_layer_ids",
    "PagedKvMeta",
    "block_tables_to_paged_kv",
    "attach_paged_kv_to_step_meta",
    "last_page_len_from_seq",
]
