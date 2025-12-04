#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) Benchmark for MetaX MACA GPU
Based on the original lora.py, adapted for MACA backend with 64-thread warps.

This benchmark demonstrates YiRage's ability to fuse LoRA computations
with the base model weights for efficient inference.
"""

import yirage as yr
import torch
import argparse
import time

# Try to import flashinfer (may not be available on MACA)
try:
    import flashinfer
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    print("Warning: flashinfer not available, using PyTorch attention fallback")

# Model configuration
n_local_heads = 32
n_local_kv_heads = 8
head_dim = 128
intermediate_size = 14336
num_tokens = 1
num_kv_tokens = 4096
batch_size = 8
lora_rank = 16  # LoRA rank

# MACA uses CUDA device through mcPytorch
device = "cuda"

silu = torch.nn.SiLU()


def get_rms_linear():
    """RMS Norm + Linear fusion for QKV projection"""
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim), dtype=yr.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    return graph.superoptimize(
        backend="maca",
        config="mlp",
        previous_checkpoint=f"lora_rms_linear_maca_bs{batch_size}.json"
    )


def get_lora():
    """
    LoRA fused with base weights for FFN gate/up projection.
    Computes: RMSNorm(X) @ W + X @ A @ B
    where A is the low-rank down projection and B is the up projection.
    
    MACA-optimized with 64-thread warps.
    """
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, intermediate_size * 2), dtype=yr.float16)
    A = graph.new_input(dims=(4096, lora_rank), dtype=yr.float16)
    B = graph.new_input(dims=(lora_rank, intermediate_size * 2), dtype=yr.float16)
    
    # Custom threadblock graph for fused LoRA computation
    # MACA: Using block_dim that's multiple of 64 (warp size)
    tb_graph = yr.new_threadblock_graph(
        grid_dim=(448, 1, 1),
        block_dim=(128, 1, 1),  # 2 warps of 64 threads
        forloop_range=64,
        reduction_dimx=64
    )
    
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tA = tb_graph.new_input(dtensor=A, input_map=(-1, -1, -1), forloop_dim=0)
    tB = tb_graph.new_input(dtensor=B, input_map=(1, -1, -1), forloop_dim=-1)
    
    # Fused computation: RMSNorm(X) @ W + X @ A @ B
    tAccX = tb_graph.forloop_accum(tX, "rms")
    tC = tb_graph.matmul(tX, tW)      # Base: X @ W
    tD = tb_graph.matmul(tX, tA)      # LoRA down: X @ A
    tE = tb_graph.matmul(tD, tB)      # LoRA up: (X @ A) @ B
    tM = tb_graph.add(tC, tE)         # Merge: base + lora
    tAccM = tb_graph.forloop_accum(tM)
    tO = tb_graph.div(tAccM, tAccX)   # RMS normalization
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    
    O = graph.customized([X, W, A, B], tb_graph)
    graph.mark_output(O[0])
    return graph


def get_lora2():
    """
    LoRA for FFN down projection.
    Computes: X @ W + X @ A @ B (without RMS norm)
    
    MACA-optimized with 64-thread warps.
    """
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, intermediate_size), dtype=yr.float16)
    W = graph.new_input(dims=(intermediate_size, 4096), dtype=yr.float16)
    A = graph.new_input(dims=(intermediate_size, lora_rank), dtype=yr.float16)
    B = graph.new_input(dims=(lora_rank, 4096), dtype=yr.float16)
    
    # Custom threadblock graph for fused LoRA computation
    # MACA: Using block_dim that's multiple of 64 (warp size)
    tb_graph = yr.new_threadblock_graph(
        grid_dim=(128, 1, 1),
        block_dim=(128, 1, 1),  # 2 warps of 64 threads
        forloop_range=224,
        reduction_dimx=64
    )
    
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tA = tb_graph.new_input(dtensor=A, input_map=(-1, -1, -1), forloop_dim=0)
    tB = tb_graph.new_input(dtensor=B, input_map=(1, -1, -1), forloop_dim=-1)
    
    # Fused computation: X @ W + X @ A @ B
    tC = tb_graph.matmul(tX, tW)      # Base: X @ W
    tD = tb_graph.matmul(tX, tA)      # LoRA down: X @ A
    tE = tb_graph.matmul(tD, tB)      # LoRA up: (X @ A) @ B
    tO = tb_graph.add(tC, tE)         # Merge: base + lora
    tAccumO = tb_graph.forloop_accum(tO)
    tb_graph.new_output(stensor=tAccumO, output_map=(1, -1, -1))
    
    O = graph.customized([X, W, A, B], tb_graph)
    graph.mark_output(O[0])
    return graph


def pytorch_attention(Xq, Kcache, Vcache):
    """PyTorch fallback attention when flashinfer not available"""
    scale = head_dim ** -0.5
    n_rep = n_local_heads // n_local_kv_heads
    
    q = Xq.transpose(0, 1)
    k = Kcache.permute(1, 2, 0)
    v = Vcache.permute(1, 0, 2)
    
    k = k.repeat_interleave(n_rep, dim=0)
    v = v.repeat_interleave(n_rep, dim=0)
    
    scores = torch.matmul(q, k) * scale
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output.transpose(0, 1)


def yirage_llama_lora(X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A2, B2, kernels):
    """Single LLaMA layer with LoRA forward pass"""
    # QKV projection with RMS norm fusion
    func = kernels[0]
    outputs = func(inputs=[X, Wqkv])
    Xqkv = outputs[0]
    
    # Split QKV
    Xq = Xqkv[:, : (n_local_heads * head_dim)]
    output_shape = Xq.shape
    Xkv = Xqkv[:, (n_local_heads * head_dim) :]
    Xk, Xv = Xkv.chunk(2, 1)
    
    # Reshape for attention
    Xq = Xq.view(Xq.shape[0], n_local_heads, head_dim)
    Xk = Xk.view(Xk.shape[0], n_local_kv_heads, head_dim)
    Xv = Xv.view(Xv.shape[0], n_local_kv_heads, head_dim)
    
    # Attention
    if HAS_FLASHINFER:
        output = flashinfer.single_prefill_with_kv_cache(Xq, Kcache, Vcache, causal=True)
    else:
        output = pytorch_attention(Xq, Kcache, Vcache)
    
    # Output projection
    output = torch.matmul(output.reshape(output_shape), Wo)
    
    # FFN with LoRA1 (gate/up projection)
    X = output
    func = kernels[2]
    outputs = func(inputs=[X, W13, A1, B1])
    X13 = outputs[0]
    X1, X3 = X13.chunk(2, -1)
    
    # SwiGLU activation
    X1 = silu(X1) * X3
    
    # Down projection with LoRA2
    func = kernels[3]
    outputs = func(inputs=[X1, W2, A2, B2])
    output = outputs[0]
    
    return output


def benchmark(kernels, X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A2, B2, warmup=16, repetitions=1000):
    """Run benchmark with proper CUDA timing"""
    # Warmup
    for _ in range(warmup):
        yirage_llama_lora(X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A2, B2, kernels)
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    starter.record()
    for _ in range(repetitions):
        yirage_llama_lora(X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A2, B2, kernels)
    ender.record()
    torch.cuda.synchronize()
    
    total_time = starter.elapsed_time(ender)
    mean_time = total_time / repetitions
    return mean_time


def main():
    parser = argparse.ArgumentParser(description="LoRA MACA Benchmark")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--warmup", type=int, default=16, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=1000, help="Benchmark repetitions")
    parser.add_argument("--skip-search", action="store_true", help="Skip optimization search")
    args = parser.parse_args()
    
    global batch_size, lora_rank
    batch_size = args.batch_size
    lora_rank = args.lora_rank
    
    print("=" * 60)
    print("LoRA Model Benchmark - MACA Backend")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {batch_size}")
    print(f"LoRA rank: {lora_rank}")
    print(f"MACA warp size: 64 threads")
    print()
    
    # Initialize tensors
    X = torch.randn(batch_size * num_tokens, 4096, dtype=torch.float16, device=device)
    Wqkv = torch.randn(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim, dtype=torch.float16, device=device)
    Wo = torch.randn(n_local_heads * head_dim, 4096, dtype=torch.float16, device=device)
    W13 = torch.randn(4096, intermediate_size * 2, dtype=torch.float16, device=device)
    W2 = torch.rand(intermediate_size, 4096, dtype=torch.float16, device=device)
    Kcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device=device)
    Vcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device=device)
    
    # LoRA weights
    A1 = torch.rand(4096, lora_rank, dtype=torch.float16, device=device)
    B1 = torch.rand(lora_rank, intermediate_size * 2, dtype=torch.float16, device=device)
    A2 = torch.rand(intermediate_size, lora_rank, dtype=torch.float16, device=device)
    B2 = torch.rand(lora_rank, 4096, dtype=torch.float16, device=device)
    
    if args.skip_search:
        print("Skipping optimization search...")
        return
    
    # Run optimization search
    print("Searching for optimal kernel fusions...")
    print("(This may take several minutes on first run)")
    print()
    
    k1 = get_rms_linear()
    print("✓ RMS+Linear (QKV) optimized")
    
    k2 = None  # Placeholder
    
    k3 = get_lora()
    print("✓ LoRA1 (FFN gate/up) fused")
    
    k4 = get_lora2()
    print("✓ LoRA2 (FFN down) fused")
    
    kernels = [k1, k2, k3, k4]
    
    # Check if custom graphs are valid (they don't go through superoptimize)
    if k3 is not None and k4 is not None:
        print()
        print("Running benchmark...")
        mean_time = benchmark(kernels, X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A2, B2,
                             warmup=args.warmup, repetitions=args.repeat)
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Mean latency: {mean_time:.4f} ms")
        print(f"Throughput: {1000 / mean_time:.2f} iter/s")
    else:
        print("Warning: Some kernels failed to initialize")


if __name__ == "__main__":
    main()

