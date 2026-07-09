#!/usr/bin/env python3
"""
MACA Qwen HF from_pretrained demo — aligned to ``demo/qwen2.5/demo.py`` (CUDA).

CUDA reference:
  ``Qwen2ForCausalLM.from_pretrained`` → ``fuse_weights`` → ``superoptimize_kernels()``
  → chat prefill → CUDA Graph decode with YiRage MLP/attn kernels.

MACA path (no flashinfer):
  ``demo/maca/models/modeling_qwen2_maca.py`` with ``backend=maca`` superoptimize.

MetaX VM:
  export MACA_PATH=/opt/maca YIRAGE_BACKEND=maca PYTHONPATH=.
  python3 demo/maca/qwen_from_pretrained_demo.py --model Qwen/Qwen3-8B --max-tokens 32
  python3 demo/maca/qwen_from_pretrained_demo.py --model Qwen/Qwen3-8B --max-layers 1 --quick
  python3 demo/maca/qwen_from_pretrained_demo.py --model Qwen/Qwen3-8B --cuda-graph --max-tokens 32
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo._maca_utils import apply_maca_demo_env  # noqa: E402
from demo.maca.qwen_decode_loop import run_qwen_decode_loop  # noqa: E402
from demo.maca.qwen_hf_utils import DEFAULT_QWEN_MODEL  # noqa: E402


def check_maca_available() -> bool:
    if not torch.cuda.is_available():
        print("✗ mcPytorch CUDA/MACA device not available")
        return False
    name = torch.cuda.get_device_name(0)
    print(f"✓ Device: {name}")
    if "MetaX" not in name and os.environ.get("YIRAGE_MACA_ALLOW_NON_METAX", "") != "1":
        print(
            "WARNING: expected MetaX GPU; set YIRAGE_MACA_ALLOW_NON_METAX=1 for other CUDA devices.",
            file=sys.stderr,
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen HF from_pretrained on MetaX MACA (CUDA qwen2.5/demo.py aligned)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_QWEN_MODEL,
        help="HuggingFace model id (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument("--disable-yirage", action="store_true", help="Skip YiRage superoptimize")
    parser.add_argument("--quick", action="store_true", default=True, help="Tractable MACA search")
    parser.add_argument("--full-search", action="store_true", help="Full get_maca_search_config grid")
    parser.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="Superoptimize only first N decoder layers (0 = all; use 1 for smoke)",
    )
    parser.add_argument("--max-tokens", type=int, default=64, help="Decode tokens after prefill")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup decode steps before timing")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Weight/compute dtype (CUDA demo uses bfloat16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model + fuse only; skip superoptimize and generation",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Eager decode only (default uses torch.cuda.CUDAGraph like CUDA qwen2.5/demo.py)",
    )
    args = parser.parse_args()
    args.cuda_graph = not args.no_cuda_graph

    apply_maca_demo_env()
    os.environ.setdefault("YIRAGE_BACKEND", "maca")

    print("=" * 70)
    print("MACA Qwen from_pretrained (CUDA demo/qwen2.5/demo.py aligned)")
    print("=" * 70)
    print(f"Arguments: {args}")
    print()

    if not check_maca_available():
        return 1

    from transformers import AutoTokenizer

    from demo.maca.models.modeling_qwen2_maca import Qwen2ForCausalLM

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.set_default_dtype(torch_dtype)

    model_name = args.model
    print(f"Loading model: {model_name}")

    with torch.device(device):
        model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
        model.fuse_weights()
        if not args.disable_yirage and not args.dry_run:
            quick = not args.full_search
            max_layers = args.max_layers if args.max_layers > 0 else None
            print(
                f"Superoptimizing kernels (backend=maca, quick={quick}, "
                f"max_layers={max_layers or 'all'})..."
            )
            try:
                model.superoptimize_kernels(
                    quick=quick,
                    dtype_name=args.dtype,
                    max_layers=max_layers,
                )
                print("✓ YiRage MACA kernel optimization complete")
            except Exception as exc:
                print(f"Warning: YiRage MACA optimization failed: {exc}")
                print("Falling back to pure PyTorch execution")
                args.disable_yirage = True

    if args.dry_run:
        print("PASS: dry-run (from_pretrained + fuse_weights, no superoptimize/generation)")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    max_seq_len = 32768
    tokens = torch.full((1, max_seq_len), 0, dtype=torch.long, device=device)
    for i in range(model_inputs.input_ids.shape[-1]):
        tokens[0, i] = model_inputs.input_ids[0, i]
    prompt_len = model_inputs.input_ids.shape[-1]
    positions = torch.arange(max_seq_len).unsqueeze(0).to(device)
    position_embeddings = model.model.rotary_emb(positions)

    warmup = args.warmup
    output_len = args.max_tokens

    mode = "CUDA Graph" if args.cuda_graph else "eager"
    print(f"\nGenerating up to {output_len} tokens (prompt length: {prompt_len}, decode={mode})...")

    decode_result = run_qwen_decode_loop(
        model,
        tokens=tokens,
        prompt_len=prompt_len,
        position_embeddings=position_embeddings,
        max_tokens=output_len,
        warmup=warmup,
        device=device,
        use_cuda_graph=args.cuda_graph,
    )
    cur_pos = decode_result.cur_pos
    run_time = decode_result.run_time_ms

    generated_ids = tokens[:, : prev_pos + 1]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "=" * 70)
    print("Generated Response:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    tokens_generated = decode_result.tokens_generated
    if tokens_generated > 0 and run_time > 0:
        latency_per_token = run_time / tokens_generated
        print(f"\nPerformance:")
        print(f"  Prompt length: {prompt_len} tokens")
        print(f"  Generated: {cur_pos + 1 - prompt_len} tokens")
        print(f"  Decode mode: {mode}")
        print(f"  CUDA Graph captured: {decode_result.used_cuda_graph}")
        print(f"  Per-token latency: {latency_per_token:.2f} ms")
        print(f"  Throughput: {1000 / latency_per_token:.2f} tokens/sec")

    print(f"PASS: MACA Qwen from_pretrained full-chain smoke ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
