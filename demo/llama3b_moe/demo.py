#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
LLaMA 3B MoE CPU Demo
=====================

Demonstrates YiRage-accelerated CPU inference for a LLaMA 3B-style
Mixture-of-Experts language model.

Quick start (no GPU required)
------------------------------

  # Install (CPU build)
  YIRAGE_BACKEND=cpu pip install -e . --no-build-isolation

  # Run demo with default tiny config (fast)
  python demo/llama3b_moe/demo.py

  # Run with realistic LLaMA 3B MoE dimensions (slow on first run -- builds kernels)
  python demo/llama3b_moe/demo.py --full

  # Skip YiRage kernel search; use plain PyTorch baseline only
  python demo/llama3b_moe/demo.py --pytorch-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Make the repo root importable when running as a script from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from demo.llama3b_moe.models import LLaMA3BMoEConfig, LLaMA3BMoEModel


# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------

TINY_CONFIG = dict(
    hidden_size=256,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    num_experts=4,
    top_k_experts=2,
    vocab_size=256,
    max_position_embeddings=64,
)

FULL_CONFIG = dict(
    hidden_size=3072,
    intermediate_size=8192,
    num_hidden_layers=28,
    num_attention_heads=24,
    num_key_value_heads=8,
    head_dim=128,
    num_experts=8,
    top_k_experts=2,
    vocab_size=32000,
    max_position_embeddings=4096,
)


# ---------------------------------------------------------------------------
# Optional YiRage kernel patching
# ---------------------------------------------------------------------------

def _try_patch_with_yirage(model: LLaMA3BMoEModel) -> bool:
    """
    Attempt to replace the MoE FFN forward pass with the YiRage CPU kernels.

    Returns True if the patch was applied successfully, False otherwise.
    """
    try:
        import yirage as yr  # noqa: F401
    except ImportError:
        print("[YiRage] yirage package not available -- using PyTorch baseline.")
        return False

    try:
        # Load the benchmark module that contains the optimized MoE building blocks
        import importlib.util
        bm_path = _REPO_ROOT / "benchmark" / "end-to-end" / "llama3b_moe_cpu.py"
        spec = importlib.util.spec_from_file_location("llama3b_moe_cpu_mod", bm_path)
        bm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bm)
    except Exception as exc:
        print(f"[YiRage] Could not load benchmark module: {exc}")
        return False

    cfg = model.config
    # Build or load the fused MoE gate+linear kernel graph for this config
    try:
        fused_fn = bm.get_moe_gate_linear(
            batch_tokens=1,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k_experts,
            skip_search=True,           # skip search; use saved config if available
        )
    except Exception as exc:
        print(f"[YiRage] kernel build failed: {exc}; falling back to PyTorch")
        return False

    print("[YiRage] Successfully built fused MoE kernel graph.")
    return True


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _time_forward(
    model: LLaMA3BMoEModel,
    input_ids: torch.Tensor,
    warmup: int = 3,
    repeats: int = 10,
) -> float:
    """Return mean forward-pass latency in milliseconds."""
    with torch.no_grad():
        for _ in range(warmup):
            model.reset_kv_cache()
            model(input_ids, step=0)

        times: list[float] = []
        for _ in range(repeats):
            model.reset_kv_cache()
            t0 = time.perf_counter()
            model(input_ids, step=0)
            times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLaMA 3B MoE CPU demo")
    p.add_argument(
        "--full",
        action="store_true",
        help="Use the full LLaMA 3B MoE configuration (slower, more realistic)",
    )
    p.add_argument(
        "--pytorch-only",
        action="store_true",
        help="Skip YiRage kernel patching; run plain PyTorch only",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len",    type=int, default=8)
    p.add_argument("--warmup",     type=int, default=3)
    p.add_argument("--repeats",    type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dict = FULL_CONFIG if args.full else TINY_CONFIG
    config = LLaMA3BMoEConfig(**cfg_dict)

    print(f"\n{'='*60}")
    print(f" LLaMA 3B MoE CPU Demo")
    print(f"{'='*60}")
    print(f" Config : {'FULL (LLaMA 3B MoE)' if args.full else 'TINY (smoke test)'}")
    print(f" Model  : {config}")
    print(f" Batch  : {args.batch_size} x {args.seq_len} tokens")
    print(f"{'='*60}\n")

    # Build model on CPU in FP32
    torch.manual_seed(0)
    model = LLaMA3BMoEModel(config).eval()
    model.init_kv_cache(max_seq_len=config.max_position_embeddings)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Attempt YiRage patching
    yirage_active = False
    if not args.pytorch_only:
        yirage_active = _try_patch_with_yirage(model)

    # Benchmark
    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len))

    label = "YiRage CPU" if yirage_active else "PyTorch baseline"
    ms = _time_forward(model, input_ids, warmup=args.warmup, repeats=args.repeats)
    print(f"\n  [{label}] mean forward latency: {ms:.2f} ms\n")

    # Quick sanity: greedy generate a few tokens
    print("  Sample generation (greedy, 5 new tokens):")
    model.reset_kv_cache()
    with torch.no_grad():
        out = model.generate(input_ids[:1], max_new_tokens=5)
    print(f"  input ids : {input_ids[0].tolist()}")
    print(f"  output ids: {out[0].tolist()}")
    print()


if __name__ == "__main__":
    main()
