#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5 Demo for Huawei Ascend NPU

This demo demonstrates running Qwen2.5/Qwen3 models on Ascend NPU
with YiRage kernel optimization.

Requirements:
- Huawei CANN toolkit installed
- torch_npu (PyTorch for Ascend NPU)
- YiRage compiled with Ascend support (USE_ASCEND=ON)

Usage:
    python demo.py
    python demo.py --disable-yirage  # Run without YiRage optimization
    python demo.py --model Qwen/Qwen2.5-7B-Instruct
"""

from models.modeling_qwen2_ascend import Qwen2ForCausalLM
from transformers import AutoTokenizer
import torch
import argparse
import time


# Check for Ascend NPU availability
def check_npu_available():
    """Check if Ascend NPU is available via torch_npu"""
    try:
        import torch_npu

        if torch.npu.is_available():
            device_count = torch.npu.device_count()
            device_name = torch.npu.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✓ Ascend NPU available: {device_count} device(s)")
            print(f"  Device 0: {device_name}")
            return True
        else:
            print("✗ torch_npu installed but no NPU detected")
            return False
    except ImportError:
        print("✗ torch_npu not installed")
        print("  Install with: pip install torch_npu")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5 Demo for Ascend NPU")
    parser.add_argument("--disable-yirage", action="store_true", help="Disable YiRage kernels")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-8B", help="Model name (default: Qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--warmup", type=int, default=16, help="Warmup iterations before timing (default: 16)"
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device ID (default: 0)")
    parser.add_argument(
        "--cpu-fallback", action="store_true", help="Use CPU if NPU is not available"
    )
    args = parser.parse_args()
    print("=" * 70)
    print("Qwen2.5 Demo for Huawei Ascend NPU")
    print("=" * 70)
    print(f"Arguments: {args}")
    print()

    # Check NPU availability
    npu_available = check_npu_available()

    if npu_available:
        import torch_npu

        device = f"npu:{args.device}"
        torch.npu.set_device(args.device)
    elif args.cpu_fallback:
        print("Using CPU fallback mode")
        device = "cpu"
    else:
        print("\nError: Ascend NPU not available. Use --cpu-fallback for CPU mode.")
        exit(1)

    print(f"\nUsing device: {device}")

    # Set default dtype
    torch.set_default_dtype(torch.bfloat16)

    model_name = args.model
    print(f"Loading model: {model_name}")

    # Load model
    with torch.device(device):
        model = Qwen2ForCausalLM.from_pretrained(model_name).to(device)
        model.fuse_weights()
        if not args.disable_yirage:
            print("Superoptimizing kernels with YiRage (Ascend backend)...")
            try:
                model.superoptimize_kernels()
                print("✓ YiRage kernel optimization complete")
            except Exception as e:
                print(f"Warning: YiRage optimization failed: {e}")
                print("Falling back to pure PyTorch execution on NPU")
                args.disable_yirage = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input
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
    prev_pos = 0

    # Timing setup
    if npu_available:
        starter = torch.npu.Event(enable_timing=True)
        ender = torch.npu.Event(enable_timing=True)
        sync_fn = torch.npu.synchronize
    else:
        starter, ender = None, None
        sync_fn = lambda: None
        start_time = None

    # Generation parameters
    warmup = args.warmup
    output_len = args.max_tokens
    step = torch.tensor([0], dtype=torch.int32, device=device)

    print(f"\nGenerating {output_len} tokens (prompt length: {prompt_len})...")

    # Generation loop
    for cur_pos in range(prompt_len, prompt_len + output_len):
        step.fill_(cur_pos - 1)

        # Prefilling phase
        if cur_pos < prompt_len + 1:
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            logits = model.forward(
                input_ids=input_ids, position_embeddings=(cos_embeddings, sin_embeddings), step=step
            )
        # Decoding phase
        else:
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            logits = model.forward(
                input_ids=input_ids, position_embeddings=(cos_embeddings, sin_embeddings), step=step
            )

        next_token = logits.argmax(dim=-1)
        next_token = next_token[0, -1]
        tokens[0, cur_pos] = next_token
        prev_pos = cur_pos

        if next_token == model.config.eos_token_id:
            break

        # Start timing after warmup
        if cur_pos == prompt_len + warmup:
            sync_fn()
            if npu_available:
                starter.record()
            else:
                start_time = time.perf_counter()

    # End timing
    if npu_available:
        ender.record()
        sync_fn()
        run_time = starter.elapsed_time(ender)
    else:
        sync_fn()
        run_time = (time.perf_counter() - start_time) * 1000 if start_time else 0

    # Decode output
    generated_ids = tokens[:, :prev_pos]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "=" * 70)
    print("Generated Response:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    tokens_generated = cur_pos + 1 - warmup
    if tokens_generated > 0:
        latency_per_token = run_time / tokens_generated
        print(f"\nPerformance:")
        print(f"  Prompt length: {prompt_len} tokens")
        print(f"  Generated: {cur_pos + 1} tokens")
        print(f"  Per-token latency: {latency_per_token:.2f} ms")
        print(f"  Throughput: {1000 / latency_per_token:.2f} tokens/sec")
