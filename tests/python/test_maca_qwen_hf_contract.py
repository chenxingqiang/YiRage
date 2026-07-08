"""Contract tests for MACA HF Qwen scaffold and MetaX rebuild script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "python"


def _load_module(name: str, path: Path):
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_qwen_hf_utils_default_dims_match_qwen3_8b():
    hf = _load_module("qwen_hf_utils", _REPO / "demo" / "maca" / "qwen_hf_utils.py")
    dims = hf.default_qwen_dims()
    assert dims.hidden_size == 4096
    assert dims.intermediate_size == 12288
    assert dims.num_heads == 32
    assert dims.num_kv_heads == 8
    assert dims.fused_qkv_outdim == (32 + 2 * 8) * 128


def test_qwen_dims_from_hf_config_object():
    hf = _load_module("qwen_hf_utils", _REPO / "demo" / "maca" / "qwen_hf_utils.py")

    class _Cfg:
        hidden_size = 4096
        intermediate_size = 12288
        num_attention_heads = 32
        num_key_value_heads = 8

    dims = hf.qwen_dims_from_hf_config(_Cfg())
    assert dims == hf.default_qwen_dims()


def test_qwen_inference_demo_exposes_hf_flags():
    demo = _REPO / "demo" / "maca" / "qwen_inference_demo.py"
    text = demo.read_text(encoding="utf-8")
    assert "--model" in text
    assert "--config-only" in text
    assert "--from-pretrained" in text
    assert "qwen_hf_utils" in text


def test_maca_rebuild_core_script_exists_and_documents_smem():
    script = _REPO / "scripts" / "maca_rebuild_core.sh"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "YIRAGE_BACKEND=maca" in text
    assert "get_shared_memory_capacity" in text
    assert "65536" in text
    assert text.startswith("#!/usr/bin/env bash")


def test_qwen_from_pretrained_demo_exists_and_uses_maca_modeling():
    demo = _REPO / "demo" / "maca" / "qwen_from_pretrained_demo.py"
    assert demo.is_file()
    text = demo.read_text(encoding="utf-8")
    assert "modeling_qwen2_maca" in text
    assert "from_pretrained" in text
    assert "superoptimize_kernels" in text
    assert "demo/qwen2.5/demo.py" in text
    assert "--max-layers" in text
    assert "--no-cuda-graph" in text
    assert "qwen_decode_loop" in text


def test_qwen_decode_loop_cuda_graph_contract():
    loop_py = _REPO / "demo" / "maca" / "qwen_decode_loop.py"
    assert loop_py.is_file()
    text = loop_py.read_text(encoding="utf-8")
    assert "torch.cuda.CUDAGraph" in text
    assert "graph.replay" in text
    assert "torch.cuda.graph" in text
    assert "used_cuda_graph" in text


def test_modeling_qwen2_maca_superoptimize_uses_maca_backend():
    modeling = _REPO / "demo" / "maca" / "models" / "modeling_qwen2_maca.py"
    assert modeling.is_file()
    text = modeling.read_text(encoding="utf-8")
    assert "superoptimize_mlp_gate_up" in text
    assert "superoptimize_attn_qkv" in text
    assert "qwen_kernel_utils" in text
    assert "import flashinfer" not in text


def test_qwen_kernel_utils_maca_search_contract():
    utils_path = _REPO / "demo" / "maca" / "qwen_kernel_utils.py"
    text = utils_path.read_text(encoding="utf-8")
    assert 'backend": "maca"' in text or "backend='maca'" in text
    assert "maca_search_kwargs" in text
    assert "superoptimize_mlp_gate_up" in text


def test_qwen3_persistent_kernel_demo_has_compile_modes():
    demo = _REPO / "demo" / "maca" / "qwen3_persistent_kernel_demo.py"
    text = demo.read_text(encoding="utf-8")
    assert "--compile-plan" in text
    assert "--compile-only" in text
    assert "maca_pk_minimal_compile_smoke" in text


def test_qwen3_persistent_kernel_demo_exists_and_aligns_cuda_qwen3():
    demo = _REPO / "demo" / "maca" / "qwen3_persistent_kernel_demo.py"
    assert demo.is_file()
    text = demo.read_text(encoding="utf-8")
    assert "demo/qwen3/demo.py" in text
    assert "--inspect-only" in text
    assert "qwen3_pk_utils" in text
    assert "PKRuntime" in text or "maca_pk_runtime_smoke" in text


def test_qwen3_pk_utils_scaffold_contract():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    report = pk_utils.inspect_qwen3_pk_scaffold()
    assert report["cuda_reference"] == "demo/qwen3/demo.py --use-yirage"
    assert report["mode"] == "offline"
    assert report["hidden_size"] == 4096
    assert report["compile_path"] == "mxcc"


def test_qwen3_pk_compile_plan_contract():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    plan = pk_utils.inspect_maca_pk_compile_plan()
    assert plan["compile_plan_ready"] is True
    assert plan["variant"] == "embed_only"
    assert plan["minimal_task_graph"] == "embed_layer"
    assert plan["hidden_size"] == 4096
    assert "yirage.core" in plan["requires"]


def test_qwen3_pk_one_layer_compile_plan_contract():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    plan = pk_utils.inspect_maca_pk_one_layer_compile_plan()
    assert plan["compile_plan_ready"] is True
    assert plan["variant"] == "one_layer"
    assert "layer[0]" in plan["minimal_task_graph"]
    assert plan["tasks"] == [
        "embedding",
        "rms_norm",
        "linear",
        "paged_attention",
        "linear_with_residual",
        "silu_mul",
    ]


def test_qwen3_pk_stack_compile_plan_contract():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    plan = pk_utils.inspect_maca_pk_stack_compile_plan()
    assert plan["compile_plan_ready"] is True
    assert plan["variant"] == "stack"
    assert "lm_head" in plan["minimal_task_graph"]
    assert plan["pk_compile_layers"] == 2
    assert "argmax_partial" in plan["tasks"]
    assert "argmax_reduce" in plan["tasks"]


def test_qwen3_pk_grid_for_rmsnorm_linear_layer():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    from demo.maca.qwen_hf_utils import default_qwen_dims

    gated_up = 2 * default_qwen_dims().intermediate_size
    assert pk_utils.grid_for_rmsnorm_linear_layer(gated_up) == 96
    assert pk_utils.grid_for_rmsnorm_linear_layer(4096) == 64


def test_qwen3_pk_meta_tensors_shapes():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    import torch

    scaffold = pk_utils.Qwen3PKScaffold()
    meta = pk_utils.build_qwen3_pk_meta_tensors(scaffold, torch.device("cpu"))
    assert meta["tokens"].shape == (scaffold.max_num_batched_requests, scaffold.max_seq_length)
    assert meta["input_tokens"].shape == (scaffold.max_num_batched_tokens, 1)
    assert meta["qo_indptr_buffer"].shape == (scaffold.max_num_batched_requests + 1,)


def test_qwen3_pk_compile_contract():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    contract = pk_utils.inspect_maca_pk_compile_contract()
    assert contract["required_tokens_ok"] is True
    assert contract["compile_ready"] is True
    assert contract["compiler"] == "mxcc"


def test_qwen3_pk_runtime_smoke_offline():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    result = pk_utils.maca_pk_runtime_smoke(num_workers=4, num_schedulers=1)
    assert result["initialized"] is True
    assert result["backend"] == "maca"
    assert result["mode"] == "offline"
    assert result["max_shared_memory"] == 64 * 1024


def test_qwen3_pk_runtime_plan_contract():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    plan = pk_utils.inspect_maca_pk_runtime_plan(variant="stack", num_layers=1)
    assert plan["plan_kind"] == "runtime"
    assert plan["runtime_plan_ready"] is True
    assert "ypk() launch" in plan["runtime_steps"][-1]
    assert "qo_indptr_buffer" in plan["meta_tensors"]


def test_qwen3_pk_hf_runtime_plan_contract():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    plan = pk_utils.inspect_maca_pk_hf_runtime_plan()
    assert plan["hf_runtime_ready"] is True
    assert plan["weight_injection_status"] == "implemented"
    assert plan["maca_pk_runtime_entry"] == "maca_pk_hf_stack_runtime_smoke"
    assert "qwen_from_pretrained_demo.py" in plan["maca_pretrained_demo"]


def test_qwen3_pk_hf_weight_plan_contract():
    hf_utils = _load_module("qwen3_pk_hf_utils", _REPO / "demo" / "maca" / "qwen3_pk_hf_utils.py")
    plan = hf_utils.inspect_maca_pk_hf_weight_plan(max_layers=1)
    assert plan["weight_plan_ready"] is True
    assert plan["attach_map_count"] >= 18
    assert plan["loader"] == "load_maca_pk_hf_weight_bundle"
    plan2 = hf_utils.inspect_maca_pk_hf_weight_plan(max_layers=2)
    assert plan2["weight_plan_ready"] is True
    assert plan2["attach_map_count"] >= 31


def test_qwen3_pk_hf_padded_lm_head_plan_contract():
    hf_utils = _load_module("qwen3_pk_hf_utils", _REPO / "demo" / "maca" / "qwen3_pk_hf_utils.py")
    plan = hf_utils.inspect_maca_pk_hf_padded_lm_head_plan()
    assert plan["padded_lm_head_plan_ready"] is True
    assert plan["pad_vocab_size"] == 153600
    assert plan["lm_head_shape"][0] == 153600


def test_pad_lm_head_weight_shape():
    import torch

    hf_utils = _load_module("qwen3_pk_hf_utils", _REPO / "demo" / "maca" / "qwen3_pk_hf_utils.py")
    weight = torch.randn(128, 64, dtype=torch.bfloat16)
    padded = hf_utils.pad_lm_head_weight(weight, 64)
    assert padded.shape == (153600, 64)


def test_resolve_maca_pk_lm_vocab_size():
    hf_utils = _load_module("qwen3_pk_hf_utils", _REPO / "demo" / "maca" / "qwen3_pk_hf_utils.py")
    assert hf_utils.resolve_maca_pk_lm_vocab_size(vocab_smoke=128) == 128
    assert hf_utils.resolve_maca_pk_lm_vocab_size(use_padded_lm_head=True) == 153600


def test_qwen3_pk_hf_generation_plan_contract():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    plan = pk_utils.inspect_maca_pk_hf_generation_plan()
    assert plan["generation_plan_ready"] is True
    assert plan["generation_ready"] is False
    assert plan["multi_step_decode_ready"] is True
    assert any("run_maca_pk_decode_loop" in step for step in plan["implemented_steps"])


def test_maca_pk_decode_step_contract():
    gen_utils = _load_module(
        "qwen3_pk_generation_utils", _REPO / "demo" / "maca" / "qwen3_pk_generation_utils.py"
    )
    contract = gen_utils.inspect_maca_pk_decode_step_contract()
    assert contract["decode_step_contract_ready"] is True
    assert contract["initial_step"] == "prompt_len - 1"


def test_prepare_and_advance_maca_pk_prompt_meta_cpu():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    gen_utils = _load_module(
        "qwen3_pk_generation_utils", _REPO / "demo" / "maca" / "qwen3_pk_generation_utils.py"
    )
    import torch

    scaffold = pk_utils.Qwen3PKScaffold()
    meta = pk_utils.build_qwen3_pk_meta_tensors(scaffold, torch.device("cpu"))
    prompt_ids = [11, 22, 33, 44]
    summary = gen_utils.prepare_maca_pk_prompt_meta(meta, scaffold, prompt_ids, num_tokens=1)
    assert summary["prompt_len"] == 4
    assert int(meta["step"][0].item()) == 3
    assert int(meta["input_tokens"][0, 0].item()) == 44

    meta["output_tokens"][0, 0] = 99
    advanced = gen_utils.advance_maca_pk_decode_step(meta, scaffold, req=0)
    assert advanced["output_token"] == 99
    assert advanced["new_step"] == 4
    assert int(meta["tokens"][0, 4].item()) == 99
    assert int(meta["input_tokens"][0, 0].item()) == 99


def test_qwen3_pk_prepare_runtime_meta_shapes():
    pk_utils = _load_module("qwen3_pk_utils", _REPO / "demo" / "maca" / "qwen3_pk_utils.py")
    import torch

    scaffold = pk_utils.Qwen3PKScaffold()
    meta = pk_utils.build_qwen3_pk_meta_tensors(scaffold, torch.device("cpu"))
    summary = pk_utils.prepare_maca_pk_runtime_meta(meta, scaffold, prompt_len=4, num_tokens=1)
    assert summary["qo_indptr"] == [0, 1]
    assert int(meta["qo_indptr_buffer"][1].item()) == 1
    assert int(meta["paged_kv_last_page_len_buffer"][0].item()) == 4


def test_qwen3_persistent_kernel_demo_has_runtime_modes():
    demo = _REPO / "demo" / "maca" / "qwen3_persistent_kernel_demo.py"
    text = demo.read_text(encoding="utf-8")
    assert "--runtime-plan" in text
    assert "--runtime-stack" in text
    assert "--hf-runtime-plan" in text
    assert "--hf-weight-plan" in text
    assert "--hf-runtime-stack" in text
    assert "--hf-padded-plan" in text
    assert "--hf-generation-plan" in text
    assert "--hf-decode-step-plan" in text
    assert "--hf-generation-smoke" in text
    assert "maca_pk_stack_runtime_smoke" in text
    assert "maca_pk_hf_stack_runtime_smoke" in text
    assert "maca_pk_hf_generation_smoke" in text
    assert "inspect_maca_pk_hf_generation_plan" in text


@pytest.mark.skipif(
    not (_REPO / "demo" / "maca" / "qwen_hf_utils.py").is_file(),
    reason="qwen_hf_utils missing",
)
def test_resolve_qwen_dims_without_model_uses_builtin():
    hf = _load_module("qwen_hf_utils", _REPO / "demo" / "maca" / "qwen_hf_utils.py")
    assert hf.resolve_qwen_dims(None) == hf.default_qwen_dims()
