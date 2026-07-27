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
from .radix_meta import (
    RadixHitMeta,
    apply_radix_shrink,
    attach_radix_to_step_meta,
    parse_radix_hit_mask,
)
from .torch_plugin import TorchDecoderMlpRfHook, build_torch_mlp_rf_hook
from .vllm_plugin import (
    VLLM_QWEN2_MLP_ATTACH,
    VllmMlpWeightView,
    VllmQwen2MlpRfHook,
    build_vllm_qwen2_mlp_rf_hook,
    extract_qwen2_mlp_weights,
    is_vllm_available,
    require_vllm,
)
from .bootstrap import bootstrap_yirage_stub, import_serving, repo_root, require_numpy
from .cpu_cert import (
    CertReport,
    CertStage,
    run_serving_cpu_cert,
    serving_cpu_cert_manifest,
)
from .sm_budget import (
    DEFAULT_CAPSULE_SM_COST,
    DEFAULT_RESERVED_AUX_SMS,
    DEFAULT_TOTAL_SMS,
    SmStepAllocation,
    SmWorkerQuota,
    assert_aux_coresidence,
    capsule_sm_cost,
    resolve_sm_worker_quota,
)
from .exec_backend import (
    BACKEND_NUMPY_REF,
    BACKEND_TORCH,
    BACKEND_YIRAGE_CPU,
    default_serving_backend,
    is_real_backend,
    is_yirage_backend,
)
from .torch_engine import TorchAttentionMeta, TorchDecoderLayer, TorchEngineModel
from .torch_exec import (
    BenchResult,
    bench_forward,
    default_device,
    mlp_torch,
    require_torch,
    to_numpy,
    to_torch,
)
from .capsule_orchestration import (
    build_split_mlp_runtime_fusion,
    pipeline_meta_for_layer,
    resolve_capsule_pipeline,
    split_mlp_down_name,
    split_mlp_gate_up_name,
    split_mlp_pipeline_names,
)
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
from .mlp_capsule import MlpFusionCapsule, mlp_eager_numpy
from .plan import FusionPlan
from .segment_override import (
    DecoderSegmentOverride,
    RuntimeFusionSplitMlpLayerOverride,
    SegmentForwardResult,
    SegmentHybridModelOverride,
    resolve_segment_layer_ids,
)
from .split_mlp_capsule import (
    MlpDownFusionCapsule,
    MlpGateUpFusionCapsule,
    build_layer_split_mlp_capsules,
    split_mlp_matches_fused,
    split_mlp_parity_numpy,
)
from .bench_archive import (
    ServingBenchArchive,
    ServingBenchArchiveRow,
    TorchSegmentForwardResult,
    TorchSegmentHybridModelOverride,
    run_segment_torch_bench_archive,
)
from .runtime_fusion import RuntimeFusion, StepMeta, StepResult
from .yirage_exec import (
    YirageServingMlpRunner,
    bench_superoptimize_down_matmul,
    build_gate_up_seed_graph,
    build_mlp_down_seed_graph,
    is_yirage_core_available,
    require_yirage_core,
    superoptimize_down_matmul_cpu,
)

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
    "build_split_mlp_runtime_fusion",
    "pipeline_meta_for_layer",
    "resolve_capsule_pipeline",
    "split_mlp_gate_up_name",
    "split_mlp_down_name",
    "split_mlp_pipeline_names",
    "MlpGateUpFusionCapsule",
    "MlpDownFusionCapsule",
    "build_layer_split_mlp_capsules",
    "split_mlp_parity_numpy",
    "split_mlp_matches_fused",
    "RuntimeFusionSplitMlpLayerOverride",
    "DecoderSegmentOverride",
    "SegmentHybridModelOverride",
    "SegmentForwardResult",
    "resolve_segment_layer_ids",
    "TorchSegmentHybridModelOverride",
    "TorchSegmentForwardResult",
    "run_segment_torch_bench_archive",
    "ServingBenchArchive",
    "ServingBenchArchiveRow",
    "is_vllm_available",
    "require_vllm",
    "TorchDecoderMlpRfHook",
    "build_torch_mlp_rf_hook",
    "VLLM_QWEN2_MLP_ATTACH",
    "VllmMlpWeightView",
    "VllmQwen2MlpRfHook",
    "build_vllm_qwen2_mlp_rf_hook",
    "extract_qwen2_mlp_weights",
    "HybridModelOverride",
    "resolve_rf_mlp_layer_ids",
    "PagedKvMeta",
    "block_tables_to_paged_kv",
    "attach_paged_kv_to_step_meta",
    "last_page_len_from_seq",
    "RadixHitMeta",
    "parse_radix_hit_mask",
    "attach_radix_to_step_meta",
    "apply_radix_shrink",
    "bootstrap_yirage_stub",
    "import_serving",
    "repo_root",
    "require_numpy",
    "run_serving_cpu_cert",
    "serving_cpu_cert_manifest",
    "CertReport",
    "CertStage",
    "SmWorkerQuota",
    "SmStepAllocation",
    "resolve_sm_worker_quota",
    "capsule_sm_cost",
    "assert_aux_coresidence",
    "DEFAULT_TOTAL_SMS",
    "DEFAULT_RESERVED_AUX_SMS",
    "DEFAULT_CAPSULE_SM_COST",
    "BACKEND_TORCH",
    "BACKEND_YIRAGE_CPU",
    "BACKEND_NUMPY_REF",
    "default_serving_backend",
    "is_real_backend",
    "is_yirage_backend",
    "is_yirage_core_available",
    "require_yirage_core",
    "YirageServingMlpRunner",
    "superoptimize_down_matmul_cpu",
    "bench_superoptimize_down_matmul",
    "build_gate_up_seed_graph",
    "build_mlp_down_seed_graph",
    "TorchEngineModel",
    "TorchDecoderLayer",
    "TorchAttentionMeta",
    "require_torch",
    "default_device",
    "mlp_torch",
    "bench_forward",
    "BenchResult",
    "to_torch",
    "to_numpy",
]
