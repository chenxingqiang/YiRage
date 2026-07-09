#!/usr/bin/env python3
"""
MACA Qwen3 PersistentKernel scaffold — aligned to ``demo/qwen3/demo.py --use-yirage``.

CUDA reference builds an offline ``yirage.PersistentKernel`` task graph, compiles with
nvcc, and runs LLM serving. This MACA demo closes the **smoke/contract** gap:

  - ``--inspect-only``: Cloud/CI scaffold report (no GPU, no weights).
  - Default / ``--quick``: MetaX mcPytorch device + MACA ``PKRuntime`` offline init +
    worker/scheduler counts from ``get_configurations_from_gpu``.

Full ``ypk.compile()`` task-graph: ``--compile-plan`` (Cloud) /
``--compile-only`` embed (MetaX) / ``--compile-one-layer`` decoder block /
``--compile-stack`` N layers + lm_head/argmax (MetaX). Stack runtime launch:
``--runtime-plan`` (Cloud) / ``--runtime-stack`` (MetaX). HF-weight stack:
``--hf-weight-plan`` / ``--hf-runtime-plan`` (Cloud) / ``--hf-runtime-stack`` (MetaX).
Padded lm_head (153600): ``--hf-padded-lm-head``; generation scaffold: ``--hf-generation-plan``.
Multi-step decode: ``--hf-decode-step-plan`` (Cloud) / ``--hf-generation-smoke`` (MetaX).
Tokenizer full-path: ``--hf-tokenizer-generation-plan`` (Cloud) / ``--hf-tokenizer-generation-smoke`` (MetaX).
Batched decode: ``--hf-batched-decode-plan`` / ``--hf-batched-generation-smoke --active-requests 2``.
Full-layer e2e: ``--hf-full-layer-generation-plan`` / ``--hf-full-layer-generation-smoke``.
Divergent batched prompts: ``--hf-divergent-batch-plan`` / ``--hf-divergent-generation-smoke``.
Full-layer batched padded: ``--hf-full-layer-batched-generation-plan`` / ``--hf-full-layer-batched-generation-smoke``.

MetaX VM:
  export MACA_PATH=/opt/maca YIRAGE_BACKEND=maca PYTHONPATH=.
  python3 demo/maca/qwen3_persistent_kernel_demo.py --inspect-only
  python3 demo/maca/qwen3_persistent_kernel_demo.py --compile-plan
  python3 demo/maca/qwen3_persistent_kernel_demo.py --compile-plan --compile-plan-variant stack
  python3 demo/maca/qwen3_persistent_kernel_demo.py --compile-only
  python3 demo/maca/qwen3_persistent_kernel_demo.py --compile-one-layer
  python3 demo/maca/qwen3_persistent_kernel_demo.py --compile-stack --pk-compile-layers 2
  python3 demo/maca/qwen3_persistent_kernel_demo.py --runtime-plan --runtime-plan-variant stack
  python3 demo/maca/qwen3_persistent_kernel_demo.py --runtime-stack --pk-compile-layers 1
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-weight-plan
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-runtime-stack --pk-compile-layers 1
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-runtime-stack --pk-compile-layers 2
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-runtime-stack --hf-padded-lm-head
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-generation-plan
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-decode-step-plan
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-generation-smoke --decode-steps 4
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-tokenizer-generation-plan
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-tokenizer-generation-smoke --decode-steps 4
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-batched-decode-plan
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-batched-generation-smoke --active-requests 2
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-full-layer-generation-plan
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-full-layer-generation-smoke --decode-steps 4
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-divergent-batch-plan
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-divergent-generation-smoke --chat-prompts "Hello,What is AI?"
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-full-layer-batched-generation-plan
  python3 demo/maca/qwen3_persistent_kernel_demo.py --hf-full-layer-batched-generation-smoke --active-requests 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo.maca.qwen3_pk_utils import (  # noqa: E402
    Qwen3PKScaffold,
    inspect_maca_pk_compile_contract,
    inspect_maca_pk_compile_plan,
    inspect_maca_pk_hf_runtime_plan,
    inspect_maca_pk_hf_generation_plan,
    inspect_maca_pk_runtime_plan,
    inspect_qwen3_pk_scaffold,
    maca_pk_hf_batched_tokenizer_generation_smoke,
    maca_pk_hf_divergent_batched_tokenizer_generation_smoke,
    maca_pk_hf_full_layer_batched_padded_generation_smoke,
    maca_pk_hf_full_layer_tokenizer_generation_smoke,
    maca_pk_hf_generation_smoke,
    maca_pk_hf_tokenizer_generation_smoke,
    maca_pk_hf_stack_runtime_smoke,
    maca_pk_minimal_compile_smoke,
    maca_pk_one_layer_compile_smoke,
    maca_pk_runtime_smoke,
    maca_pk_stack_compile_smoke,
    maca_pk_stack_runtime_smoke,
    resolve_maca_pk_workers_schedulers,
)
from demo.maca.qwen3_pk_hf_utils import (  # noqa: E402
    inspect_maca_pk_hf_padded_lm_head_plan,
    inspect_maca_pk_hf_weight_plan,
)
from demo.maca.qwen3_pk_generation_utils import (  # noqa: E402
    inspect_maca_pk_batched_decode_plan,
    inspect_maca_pk_decode_step_contract,
    inspect_maca_pk_divergent_batch_plan,
    inspect_maca_pk_hf_full_layer_batched_padded_generation_plan,
    inspect_maca_pk_hf_full_layer_generation_plan,
    inspect_maca_pk_hf_tokenizer_generation_plan,
    inspect_maca_pk_multi_request_batch_plan,
)


def _apply_maca_env() -> None:
    os.environ.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    os.environ.setdefault("MACA_PATH", "/opt/maca")
    os.environ.setdefault("YIRAGE_BACKEND", "maca")


def _is_maca_device() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0)
    if "MetaX" in name:
        return True
    return os.environ.get("YIRAGE_MACA_ALLOW_NON_METAX", "") == "1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3 PersistentKernel MACA scaffold (CUDA demo/qwen3/demo.py --use-yirage)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=Qwen3PKScaffold.model,
        help="HuggingFace model id (config shapes only unless full PK lands)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=Qwen3PKScaffold.max_num_batched_tokens,
    )
    parser.add_argument(
        "--max-num-batched-requests",
        type=int,
        default=Qwen3PKScaffold.max_num_batched_requests,
    )
    parser.add_argument("--page-size", type=int, default=Qwen3PKScaffold.page_size)
    parser.add_argument("--max-num-pages", type=int, default=Qwen3PKScaffold.max_num_pages)
    parser.add_argument("--max-seq-length", type=int, default=Qwen3PKScaffold.max_seq_length)
    parser.add_argument(
        "--pk-compile-layers",
        type=int,
        default=Qwen3PKScaffold.pk_compile_layers,
        help="Decoder layers for --compile-stack (default: scaffold pk_compile_layers=2)",
    )
    parser.add_argument(
        "--compile-plan",
        action="store_true",
        help="Validate PK compile plan (no GPU required)",
    )
    parser.add_argument(
        "--compile-plan-variant",
        type=str,
        choices=("embed_only", "one_layer", "stack"),
        default="embed_only",
        help="Compile plan variant for --compile-plan (default: embed_only)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="MetaX: build minimal embed PK graph and ypk.compile() via mxcc",
    )
    parser.add_argument(
        "--compile-one-layer",
        action="store_true",
        help="MetaX: build one-layer Qwen3 PK graph (embed+attn+mlp) and ypk.compile()",
    )
    parser.add_argument(
        "--compile-stack",
        action="store_true",
        help="MetaX: build N-layer stack + lm_head/argmax PK graph and ypk.compile()",
    )
    parser.add_argument(
        "--runtime-plan",
        action="store_true",
        help="Validate PK runtime launch plan (no GPU required)",
    )
    parser.add_argument(
        "--runtime-plan-variant",
        type=str,
        choices=("stack", "one_layer"),
        default="stack",
        help="Runtime plan variant for --runtime-plan (default: stack)",
    )
    parser.add_argument(
        "--hf-runtime-plan",
        action="store_true",
        help="Validate HF-weight PK runtime contract (no GPU required)",
    )
    parser.add_argument(
        "--hf-weight-plan",
        action="store_true",
        help="Validate HF-weight attach_input mapping contract (no GPU required)",
    )
    parser.add_argument(
        "--hf-runtime-stack",
        action="store_true",
        help="MetaX: HF-weight N-layer stack compile + ypk() launch",
    )
    parser.add_argument(
        "--hf-padded-lm-head",
        action="store_true",
        help="With --hf-runtime-stack: pad lm_head to 153600 (CUDA demo aligned)",
    )
    parser.add_argument(
        "--hf-generation-plan",
        action="store_true",
        help="Validate HF PK generation loop contract (no GPU required)",
    )
    parser.add_argument(
        "--hf-decode-step-plan",
        action="store_true",
        help="Validate multi-step decode tensor contract (no GPU required)",
    )
    parser.add_argument(
        "--hf-generation-smoke",
        action="store_true",
        help="MetaX: HF stack compile + multi-step ypk() decode loop",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=1,
        help="Decode steps for --hf-generation-smoke (default: 1)",
    )
    parser.add_argument(
        "--chat-prompt",
        type=str,
        default="Hello",
        help="Chat prompt when --hf-generation-smoke --use-tokenizer",
    )
    parser.add_argument(
        "--use-tokenizer",
        action="store_true",
        help="With --hf-generation-smoke: encode/decode via HF tokenizer",
    )
    parser.add_argument(
        "--hf-tokenizer-generation-plan",
        action="store_true",
        help="Validate tokenizer full-path generation contract (no GPU required)",
    )
    parser.add_argument(
        "--hf-multi-request-batch-plan",
        action="store_true",
        help="Validate multi-request batch meta contract (no GPU required)",
    )
    parser.add_argument(
        "--hf-tokenizer-generation-smoke",
        action="store_true",
        help="MetaX: tokenizer encode + multi-step ypk() + decode + latency",
    )
    parser.add_argument(
        "--active-requests",
        type=int,
        default=2,
        help="Active requests for --hf-batched-generation-smoke (default: 2)",
    )
    parser.add_argument(
        "--hf-batched-decode-plan",
        action="store_true",
        help="Validate multi-request ypk() decode loop contract (no GPU required)",
    )
    parser.add_argument(
        "--hf-batched-generation-smoke",
        action="store_true",
        help="MetaX: batched tokenizer generation with multi-request ypk() loop",
    )
    parser.add_argument(
        "--hf-full-layer-generation-plan",
        action="store_true",
        help="Validate full-layer HF PK generation e2e contract (no GPU required)",
    )
    parser.add_argument(
        "--hf-full-layer-generation-smoke",
        action="store_true",
        help="MetaX: full pk_compile_layers stack tokenizer generation e2e",
    )
    parser.add_argument(
        "--hf-divergent-batch-plan",
        action="store_true",
        help="Validate per-request distinct prompt batch contract (no GPU required)",
    )
    parser.add_argument(
        "--hf-divergent-generation-smoke",
        action="store_true",
        help="MetaX: batched generation with distinct per-request chat prompts",
    )
    parser.add_argument(
        "--chat-prompts",
        type=str,
        default="Hello,What is AI?",
        help="Comma-separated prompts for --hf-divergent-generation-smoke",
    )
    parser.add_argument(
        "--hf-full-layer-batched-generation-plan",
        action="store_true",
        help="Validate full-layer batched padded-lm_head generation contract (no GPU)",
    )
    parser.add_argument(
        "--hf-full-layer-batched-generation-smoke",
        action="store_true",
        help="MetaX: full-layer batched generation with padded lm_head (153600)",
    )
    parser.add_argument(
        "--hf-padded-plan",
        action="store_true",
        help="Validate padded lm_head (153600) argmax contract (no GPU required)",
    )
    parser.add_argument(
        "--runtime-stack",
        action="store_true",
        help="MetaX: compile N-layer stack, fill meta tensors, and ypk() launch",
    )
    parser.add_argument(
        "--compile-inspect",
        action="store_true",
        help="Validate mxcc PK compile contract (no GPU required)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print scaffold JSON and exit (no GPU required)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Tractable smoke: PK runtime init only (default on)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report",
    )
    args = parser.parse_args()

    _apply_maca_env()

    scaffold = Qwen3PKScaffold(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_batched_requests=args.max_num_batched_requests,
        page_size=args.page_size,
        max_num_pages=args.max_num_pages,
        max_seq_length=args.max_seq_length,
        pk_compile_layers=args.pk_compile_layers,
    )

    report: dict = {"scaffold": inspect_qwen3_pk_scaffold(scaffold)}

    if args.compile_plan:
        report["compile_plan"] = inspect_maca_pk_compile_plan(
            scaffold, variant=args.compile_plan_variant
        )
        report["status"] = "compile_plan"
        if not report["compile_plan"]["compile_plan_ready"]:
            print("✗ PK compile plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print(
                "MACA Qwen3 PK compile plan "
                f"(variant={args.compile_plan_variant})"
            )
            print("=" * 70)
            for key, val in report["compile_plan"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — compile-plan (no MetaX GPU required)")
        return 0

    if args.runtime_plan:
        layers = 1 if args.runtime_plan_variant == "one_layer" else args.pk_compile_layers
        report["runtime_plan"] = inspect_maca_pk_runtime_plan(
            scaffold, variant=args.runtime_plan_variant, num_layers=layers
        )
        report["status"] = "runtime_plan"
        if not report["runtime_plan"]["runtime_plan_ready"]:
            print("✗ PK runtime plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print(
                "MACA Qwen3 PK runtime plan "
                f"(variant={args.runtime_plan_variant}, layers={layers})"
            )
            print("=" * 70)
            for key, val in report["runtime_plan"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — runtime-plan (no MetaX GPU required)")
        return 0

    if args.hf_runtime_plan:
        report["hf_runtime_plan"] = inspect_maca_pk_hf_runtime_plan(
            scaffold, max_layers=args.pk_compile_layers
        )
        report["status"] = "hf_runtime_plan"
        if not report["hf_runtime_plan"]["hf_runtime_ready"]:
            print("✗ PK HF runtime plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF runtime plan")
            print("=" * 70)
            for key, val in report["hf_runtime_plan"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — hf-runtime-plan (no MetaX GPU required)")
        return 0

    if args.hf_generation_plan:
        report["hf_generation_plan"] = inspect_maca_pk_hf_generation_plan(scaffold)
        report["status"] = "hf_generation_plan"
        if not report["hf_generation_plan"]["generation_plan_ready"]:
            print("✗ PK HF generation plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF generation plan")
            print("=" * 70)
            for key, val in report["hf_generation_plan"].items():
                if key not in ("hf_runtime_plan", "decode_step_contract"):
                    print(f"  {key}: {val}")
            print()
            print("PASS — hf-generation-plan (no MetaX GPU required)")
        return 0

    if args.hf_decode_step_plan:
        report["hf_decode_step_plan"] = inspect_maca_pk_decode_step_contract(scaffold)
        report["status"] = "hf_decode_step_plan"
        if not report["hf_decode_step_plan"]["decode_step_contract_ready"]:
            print("✗ PK HF decode step plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF decode step contract")
            print("=" * 70)
            for key, val in report["hf_decode_step_plan"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — hf-decode-step-plan (no MetaX GPU required)")
        return 0

    if args.hf_tokenizer_generation_plan:
        report["hf_tokenizer_generation_plan"] = inspect_maca_pk_hf_tokenizer_generation_plan(
            scaffold
        )
        report["status"] = "hf_tokenizer_generation_plan"
        if not report["hf_tokenizer_generation_plan"]["tokenizer_generation_plan_ready"]:
            print("✗ PK HF tokenizer generation plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF tokenizer generation plan")
            print("=" * 70)
            for key, val in report["hf_tokenizer_generation_plan"].items():
                if key != "multi_request_batch_plan":
                    print(f"  {key}: {val}")
            print()
            print("PASS — hf-tokenizer-generation-plan (no MetaX GPU required)")
        return 0

    if args.hf_multi_request_batch_plan:
        report["hf_multi_request_batch_plan"] = inspect_maca_pk_multi_request_batch_plan(scaffold)
        report["status"] = "hf_multi_request_batch_plan"
        if not report["hf_multi_request_batch_plan"]["multi_request_batch_plan_ready"]:
            print("✗ PK HF multi-request batch plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF multi-request batch plan")
            print("=" * 70)
            for key, val in report["hf_multi_request_batch_plan"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — hf-multi-request-batch-plan (no MetaX GPU required)")
        return 0

    if args.hf_batched_decode_plan:
        report["hf_batched_decode_plan"] = inspect_maca_pk_batched_decode_plan(scaffold)
        report["status"] = "hf_batched_decode_plan"
        if not report["hf_batched_decode_plan"]["batched_decode_plan_ready"]:
            print("✗ PK HF batched decode plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF batched decode plan")
            print("=" * 70)
            for key, val in report["hf_batched_decode_plan"].items():
                if key != "multi_request_batch_plan":
                    print(f"  {key}: {val}")
            print()
            print("PASS — hf-batched-decode-plan (no MetaX GPU required)")
        return 0

    if args.hf_full_layer_generation_plan:
        report["hf_full_layer_generation_plan"] = inspect_maca_pk_hf_full_layer_generation_plan(
            scaffold
        )
        report["status"] = "hf_full_layer_generation_plan"
        if not report["hf_full_layer_generation_plan"]["full_layer_generation_plan_ready"]:
            print("✗ PK HF full-layer generation plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF full-layer generation plan")
            print("=" * 70)
            for key, val in report["hf_full_layer_generation_plan"].items():
                if key != "tokenizer_generation_plan":
                    print(f"  {key}: {val}")
            print()
            print("PASS — hf-full-layer-generation-plan (no MetaX GPU required)")
        return 0

    if args.hf_divergent_batch_plan:
        report["hf_divergent_batch_plan"] = inspect_maca_pk_divergent_batch_plan(scaffold)
        report["status"] = "hf_divergent_batch_plan"
        if not report["hf_divergent_batch_plan"]["divergent_batch_plan_ready"]:
            print("✗ PK HF divergent batch plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF divergent batch plan")
            print("=" * 70)
            for key, val in report["hf_divergent_batch_plan"].items():
                if key != "multi_request_batch_plan":
                    print(f"  {key}: {val}")
            print()
            print("PASS — hf-divergent-batch-plan (no MetaX GPU required)")
        return 0

    if args.hf_full_layer_batched_generation_plan:
        report["hf_full_layer_batched_generation_plan"] = (
            inspect_maca_pk_hf_full_layer_batched_padded_generation_plan(scaffold)
        )
        report["status"] = "hf_full_layer_batched_generation_plan"
        plan = report["hf_full_layer_batched_generation_plan"]
        if not plan["full_layer_batched_padded_generation_plan_ready"]:
            print("✗ PK HF full-layer batched padded generation plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF full-layer batched padded generation plan")
            print("=" * 70)
            for key, val in plan.items():
                if key not in ("full_layer_generation_plan", "batched_decode_plan"):
                    print(f"  {key}: {val}")
            print()
            print("PASS — hf-full-layer-batched-generation-plan (no MetaX GPU required)")
        return 0

    if args.hf_padded_plan:
        report["hf_padded_plan"] = inspect_maca_pk_hf_padded_lm_head_plan(scaffold)
        report["status"] = "hf_padded_plan"
        if not report["hf_padded_plan"]["padded_lm_head_plan_ready"]:
            print("✗ PK HF padded lm_head plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF padded lm_head plan (153600)")
            print("=" * 70)
            for key, val in report["hf_padded_plan"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — hf-padded-plan (no MetaX GPU required)")
        return 0

    if args.hf_weight_plan:
        report["hf_weight_plan"] = inspect_maca_pk_hf_weight_plan(
            scaffold, max_layers=args.pk_compile_layers
        )
        report["status"] = "hf_weight_plan"
        if not report["hf_weight_plan"]["weight_plan_ready"]:
            print("✗ PK HF weight plan failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF weight attach plan")
            print("=" * 70)
            for key, val in report["hf_weight_plan"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — hf-weight-plan (no MetaX GPU required)")
        return 0

    if args.compile_inspect:
        report["compile_contract"] = inspect_maca_pk_compile_contract(scaffold)
        report["status"] = "compile_inspect"
        if not report["compile_contract"]["compile_ready"]:
            print("✗ PK compile contract failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK mxcc compile contract")
            print("=" * 70)
            for key, val in report["compile_contract"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — compile-inspect (no MetaX GPU required)")
        return 0

    if args.hf_runtime_stack:
        if not _is_maca_device():
            print("✗ MetaX MACA GPU required for --hf-runtime-stack", file=sys.stderr)
            return 1
        os.environ.setdefault("YIRAGE_HOME", _REPO_ROOT)
        report["hf_runtime"] = maca_pk_hf_stack_runtime_smoke(
            scaffold,
            num_layers=args.pk_compile_layers,
            use_padded_lm_head=args.hf_padded_lm_head,
        )
        report["status"] = "hf_runtime_stack"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF stack runtime smoke")
            print("=" * 70)
            print(f"  hf_runtime: {report['hf_runtime']}")
            print()
            print(
                f"PASS — hf-runtime-stack ({args.pk_compile_layers} layer(s) HF compile + ypk() launch)"
            )
        return 0

    if args.hf_generation_smoke:
        if not _is_maca_device():
            print("✗ MetaX MACA GPU required for --hf-generation-smoke", file=sys.stderr)
            return 1
        os.environ.setdefault("YIRAGE_HOME", _REPO_ROOT)
        report["hf_generation"] = maca_pk_hf_generation_smoke(
            scaffold,
            num_layers=args.pk_compile_layers,
            decode_steps=args.decode_steps,
            use_padded_lm_head=args.hf_padded_lm_head,
            use_tokenizer=args.use_tokenizer,
            chat_prompt=args.chat_prompt,
        )
        report["status"] = "hf_generation_smoke"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF generation smoke")
            print("=" * 70)
            print(f"  hf_generation: {report['hf_generation']}")
            print()
            print(
                f"PASS — hf-generation-smoke "
                f"({args.decode_steps} decode step(s), layers={args.pk_compile_layers})"
            )
        return 0

    if args.hf_tokenizer_generation_smoke:
        if not _is_maca_device():
            print("✗ MetaX MACA GPU required for --hf-tokenizer-generation-smoke", file=sys.stderr)
            return 1
        os.environ.setdefault("YIRAGE_HOME", _REPO_ROOT)
        report["hf_tokenizer_generation"] = maca_pk_hf_tokenizer_generation_smoke(
            scaffold,
            num_layers=args.pk_compile_layers,
            decode_steps=args.decode_steps,
            chat_prompt=args.chat_prompt,
            use_padded_lm_head=args.hf_padded_lm_head,
        )
        report["status"] = "hf_tokenizer_generation_smoke"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF tokenizer generation smoke")
            print("=" * 70)
            print(f"  hf_tokenizer_generation: {report['hf_tokenizer_generation']}")
            print()
            print(
                f"PASS — hf-tokenizer-generation-smoke "
                f"({args.decode_steps} decode step(s), layers={args.pk_compile_layers})"
            )
        return 0

    if args.hf_batched_generation_smoke:
        if not _is_maca_device():
            print("✗ MetaX MACA GPU required for --hf-batched-generation-smoke", file=sys.stderr)
            return 1
        os.environ.setdefault("YIRAGE_HOME", _REPO_ROOT)
        report["hf_batched_generation"] = maca_pk_hf_batched_tokenizer_generation_smoke(
            scaffold,
            num_layers=args.pk_compile_layers,
            active_requests=args.active_requests,
            decode_steps=args.decode_steps,
            chat_prompt=args.chat_prompt,
            use_padded_lm_head=args.hf_padded_lm_head,
        )
        report["status"] = "hf_batched_generation_smoke"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF batched generation smoke")
            print("=" * 70)
            print(f"  hf_batched_generation: {report['hf_batched_generation']}")
            print()
            print(
                f"PASS — hf-batched-generation-smoke "
                f"(requests={args.active_requests}, steps={args.decode_steps})"
            )
        return 0

    if args.hf_full_layer_generation_smoke:
        if not _is_maca_device():
            print("✗ MetaX MACA GPU required for --hf-full-layer-generation-smoke", file=sys.stderr)
            return 1
        os.environ.setdefault("YIRAGE_HOME", _REPO_ROOT)
        report["hf_full_layer_generation"] = maca_pk_hf_full_layer_tokenizer_generation_smoke(
            scaffold,
            decode_steps=args.decode_steps,
            chat_prompt=args.chat_prompt,
            use_padded_lm_head=args.hf_padded_lm_head,
        )
        report["status"] = "hf_full_layer_generation_smoke"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF full-layer generation smoke")
            print("=" * 70)
            print(f"  hf_full_layer_generation: {report['hf_full_layer_generation']}")
            print()
            print(
                f"PASS — hf-full-layer-generation-smoke "
                f"({scaffold.pk_compile_layers} layers, {args.decode_steps} decode step(s))"
            )
        return 0

    if args.hf_divergent_generation_smoke:
        if not _is_maca_device():
            print("✗ MetaX MACA GPU required for --hf-divergent-generation-smoke", file=sys.stderr)
            return 1
        os.environ.setdefault("YIRAGE_HOME", _REPO_ROOT)
        chat_prompts = [p.strip() for p in args.chat_prompts.split(",") if p.strip()]
        report["hf_divergent_generation"] = maca_pk_hf_divergent_batched_tokenizer_generation_smoke(
            scaffold,
            num_layers=args.pk_compile_layers,
            active_requests=len(chat_prompts),
            decode_steps=args.decode_steps,
            chat_prompts=chat_prompts,
            use_padded_lm_head=args.hf_padded_lm_head,
        )
        report["status"] = "hf_divergent_generation_smoke"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF divergent batched generation smoke")
            print("=" * 70)
            print(f"  hf_divergent_generation: {report['hf_divergent_generation']}")
            print()
            print(
                f"PASS — hf-divergent-generation-smoke "
                f"(requests={len(chat_prompts)}, steps={args.decode_steps})"
            )
        return 0

    if args.hf_full_layer_batched_generation_smoke:
        if not _is_maca_device():
            print(
                "✗ MetaX MACA GPU required for --hf-full-layer-batched-generation-smoke",
                file=sys.stderr,
            )
            return 1
        os.environ.setdefault("YIRAGE_HOME", _REPO_ROOT)
        report["hf_full_layer_batched_generation"] = (
            maca_pk_hf_full_layer_batched_padded_generation_smoke(
                scaffold,
                active_requests=args.active_requests,
                decode_steps=args.decode_steps,
                chat_prompt=args.chat_prompt,
            )
        )
        report["status"] = "hf_full_layer_batched_generation_smoke"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK HF full-layer batched padded generation smoke")
            print("=" * 70)
            print(
                f"  hf_full_layer_batched_generation: "
                f"{report['hf_full_layer_batched_generation']}"
            )
            print()
            print(
                f"PASS — hf-full-layer-batched-generation-smoke "
                f"(layers={scaffold.pk_compile_layers}, requests={args.active_requests}, "
                f"steps={args.decode_steps}, padded_lm_head=True)"
            )
        return 0

    if args.runtime_stack:
        if not _is_maca_device():
            print("✗ MetaX MACA GPU required for --runtime-stack", file=sys.stderr)
            return 1
        os.environ.setdefault("YIRAGE_HOME", _REPO_ROOT)
        report["runtime"] = maca_pk_stack_runtime_smoke(
            scaffold, num_layers=args.pk_compile_layers
        )
        report["status"] = "runtime_stack"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK stack runtime smoke")
            print("=" * 70)
            print(f"  runtime: {report['runtime']}")
            print()
            print(
                f"PASS — runtime-stack ({args.pk_compile_layers} layer(s) compile + ypk() launch)"
            )
        return 0

    if args.compile_only or args.compile_one_layer or args.compile_stack:
        if not _is_maca_device():
            if args.compile_stack:
                flag = "--compile-stack"
            elif args.compile_one_layer:
                flag = "--compile-one-layer"
            else:
                flag = "--compile-only"
            print(f"✗ MetaX MACA GPU required for {flag}", file=sys.stderr)
            return 1
        os.environ.setdefault("YIRAGE_HOME", _REPO_ROOT)
        if args.compile_stack:
            report["compile"] = maca_pk_stack_compile_smoke(
                scaffold, num_layers=args.pk_compile_layers
            )
            report["status"] = "compile_stack"
            pass_msg = (
                f"PASS — compile-stack ({args.pk_compile_layers} layers + lm_head/argmax via mxcc)"
            )
            title = "MACA Qwen3 PK stack compile smoke"
        elif args.compile_one_layer:
            report["compile"] = maca_pk_one_layer_compile_smoke(scaffold)
            report["status"] = "compile_one_layer"
            pass_msg = "PASS — compile-one-layer (decoder block via mxcc)"
            title = "MACA Qwen3 PK one-layer compile smoke"
        else:
            report["compile"] = maca_pk_minimal_compile_smoke(scaffold)
            report["status"] = "compile_only"
            pass_msg = "PASS — compile-only (minimal embed task-graph via mxcc)"
            title = "MACA Qwen3 PK minimal compile smoke"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print(title)
            print("=" * 70)
            print(f"  compile: {report['compile']}")
            print()
            print(pass_msg)
        return 0

    if args.inspect_only:
        report["status"] = "inspect_only"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PersistentKernel scaffold (inspect-only)")
            print("=" * 70)
            for key, val in report["scaffold"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — inspect-only (no MetaX GPU required)")
        return 0

    if not _is_maca_device():
        print("✗ MetaX MACA GPU not detected; use --inspect-only on Cloud VM", file=sys.stderr)
        return 1

    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    print(f"✓ Device: {torch.cuda.get_device_name(0)}")

    num_workers, num_schedulers = resolve_maca_pk_workers_schedulers(0)
    report["gpu"] = {
        "device_name": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "num_workers": num_workers,
        "num_schedulers": num_schedulers,
    }
    report["pk_runtime"] = maca_pk_runtime_smoke(
        num_workers=num_workers,
        num_schedulers=num_schedulers,
        device_id=0,
    )

    smem: int | None = None
    try:
        from yirage.utils.common import get_shared_memory_capacity

        smem = get_shared_memory_capacity(70)
    except ImportError:
        pass

    if smem is not None:
        report["smem_capacity"] = smem
        if smem != 65536:
            print(
                f"WARNING: get_shared_memory_capacity(70)={smem}, expected 65536; "
                "run bash scripts/maca_rebuild_core.sh on MetaX VM",
                file=sys.stderr,
            )

    report["status"] = "pass"
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 70)
        print("MACA Qwen3 PersistentKernel smoke")
        print("=" * 70)
        print(f"  workers/schedulers: {num_workers}/{num_schedulers}")
        print(f"  PK runtime: {report['pk_runtime']}")
        if smem is not None:
            print(f"  smem capacity: {smem}")
        print()
        print("PASS — MACA PK runtime smoke (full multi-layer qwen3 e2e backlog)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
