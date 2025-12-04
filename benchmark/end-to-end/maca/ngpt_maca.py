#!/usr/bin/env python3
"""
nGPT (Normalized GPT) Model Benchmark for MetaX MACA GPU
Based on the original ngpt.py, adapted for MACA backend with 64-thread warps.

nGPT uses normalization after linear layers instead of before (like in LLaMA/GPT).
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

# Model configuration (nGPT style)
n_local_heads = 12
n_local_kv_heads = 12
head_dim = 128
intermediate_size = 4096
num_tokens = 4
num_kv_tokens = 4096
batch_size = 8

# MACA uses CUDA device through mcPytorch
device = "cuda"


def get_norm1():
    """Linear + RMS Norm fusion for QKV projection (nGPT style: norm after linear)"""
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim), dtype=yr.float16)
    D = graph.matmul(X, W)
    O = graph.rms_norm(D, normalized_shape=(n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim,))
    graph.mark_output(O)
    return graph.superoptimize(
        backend="maca",  # MACA backend with 64-thread warps
        previous_checkpoint=f"ngpt_norm1_maca_bs{batch_size}.json"
    )


def get_norm2():
    """
    Linear + L2Norm + Scale + L2Norm fusion for FFN (nGPT style).
    Computes: L2Norm(alpha * L2Norm(X @ W))
    
    Note: Using RMS norm as approximation for L2 norm here.
    """
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, intermediate_size * 2), dtype=yr.float16)
    alpha = graph.new_input(dims=(batch_size * num_tokens, intermediate_size * 2), dtype=yr.float16)
    
    # X @ W -> Norm -> Scale -> Norm
    D = graph.matmul(X, W)
    A = graph.rms_norm(D, normalized_shape=(intermediate_size * 2,))  # First norm
    B = graph.mul(A, alpha)  # Scale by learnable alpha
    O = graph.rms_norm(B, normalized_shape=(intermediate_size * 2,))  # Second norm
    graph.mark_output(O)
    
    return graph.superoptimize(
        backend="maca",
        previous_checkpoint=f"ngpt_norm2_maca_bs{batch_size}.json"
    )


def pytorch_attention(Xq, Kcache, Vcache):
    """PyTorch fallback attention when flashinfer not available"""
    scale = head_dim ** -0.5
    
    q = Xq.transpose(0, 1)
    k = Kcache.permute(1, 2, 0)
    v = Vcache.permute(1, 0, 2)
    
    scores = torch.matmul(q, k) * scale
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output.transpose(0, 1)


def yirage_ngpt(X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha, kernels):
    """Single nGPT layer forward pass with YiRage optimized kernels"""
    # QKV projection with post-linear normalization
    func = kernels[0]
    outputs = func(inputs=[X, Wqkv])
    Xqkv = outputs[0]
    
    # Split QKV
    Xq = Xqkv[:, : (n_local_heads * head_dim)]
    output_shape = Xq.shape
    Xkv = Xqkv[:, (n_local_heads * head_dim) :]
    Xk, Xv = Xkv.chunk(2, 1)
    
    # Reshape for attention
    Xq = Xq.view(Xq.shape[0], n_local_kv_heads, head_dim)
    Xk = Xk.view(Xk.shape[0], n_local_kv_heads, head_dim)
    Xv = Xv.view(Xv.shape[0], n_local_kv_heads, head_dim)
    
    # Attention
    if HAS_FLASHINFER:
        output = flashinfer.single_prefill_with_kv_cache(Xq, Kcache, Vcache, causal=True)
    else:
        output = pytorch_attention(Xq, Kcache, Vcache)
    
    # Output projection
    output = torch.matmul(output.reshape(output_shape), Wo)
    
    # FFN with double normalization (nGPT style)
    X = output
    func = kernels[1]
    outputs = func(inputs=[X, W13, alpha])
    X13 = outputs[0]
    X1, X3 = X13.chunk(2, -1)
    
    # Final projection
    output = torch.matmul(X1, W2)
    return output


def benchmark(kernels, X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha, warmup=16, repetitions=1000):
    """Run benchmark with proper CUDA timing"""
    # Warmup
    for _ in range(warmup):
        yirage_ngpt(X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha, kernels)
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    starter.record()
    for _ in range(repetitions):
        yirage_ngpt(X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha, kernels)
    ender.record()
    torch.cuda.synchronize()
    
    total_time = starter.elapsed_time(ender)
    mean_time = total_time / repetitions
    return mean_time


def main():
    parser = argparse.ArgumentParser(description="nGPT MACA Benchmark")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--warmup", type=int, default=16, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=1000, help="Benchmark repetitions")
    parser.add_argument("--skip-search", action="store_true", help="Skip optimization search")
    args = parser.parse_args()
    
    global batch_size
    batch_size = args.batch_size
    
    print("=" * 60)
    print("nGPT Model Benchmark - MACA Backend")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {batch_size}")
    print(f"Num tokens: {num_tokens}")
    print(f"nGPT style: Normalization after linear layers")
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
    alpha = torch.rand(batch_size * num_tokens, intermediate_size * 2, dtype=torch.float16, device=device)
    
    if args.skip_search:
        print("Skipping optimization search...")
        return
    
    # Run optimization search
    print("Searching for optimal kernel fusions...")
    print("(This may take several minutes on first run)")
    print()
    
    k1 = get_norm1()
    print("✓ Linear+Norm (QKV) optimized")
    
    k2 = get_norm2()
    print("✓ Linear+Norm+Scale+Norm (FFN) optimized")
    
    kernels = [k1, k2]
    
    if all(k is not None for k in kernels):
        print()
        print("Running benchmark...")
        mean_time = benchmark(kernels, X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha,
                             warmup=args.warmup, repetitions=args.repeat)
        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Mean latency: {mean_time:.4f} ms")
        print(f"Throughput: {1000 / mean_time:.2f} iter/s")
    else:
        print("Warning: Some kernels failed to optimize")


if __name__ == "__main__":
    main()

