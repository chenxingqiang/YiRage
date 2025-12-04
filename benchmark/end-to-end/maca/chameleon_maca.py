#!/usr/bin/env python3
"""
Chameleon Model Benchmark for MetaX MACA GPU
Based on the original chameleon.py, adapted for MACA backend with 64-thread warps.
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
n_local_kv_heads = 32
head_dim = 128
intermediate_size = 11008
num_tokens = 4
num_kv_tokens = 4096
batch_size = 8

# MACA uses CUDA device through mcPytorch
device = "cuda"


def get_rms_linear():
    """RMS Norm + Linear fusion for QKV projection"""
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim), dtype=yr.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    return graph.superoptimize(
        backend="maca",  # MACA backend with 64-thread warps
        config="mlp",
        previous_checkpoint=f"chameleon_rms_linear_maca_bs{batch_size}.json"
    )


def get_rms_linear2():
    """RMS Norm + Linear fusion for FFN projection"""
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, intermediate_size * 2), dtype=yr.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    return graph.superoptimize(
        backend="maca",
        config="mlp",
        previous_checkpoint=f"chameleon_rms_linear2_maca_bs{batch_size}.json"
    )


def get_chameleon_attention():
    """Attention computation graph"""
    graph = yr.new_kernel_graph()
    Q = graph.new_input(dims=(n_local_kv_heads, num_tokens, 128), dtype=yr.float16)
    K = graph.new_input(dims=(n_local_kv_heads, 128, num_kv_tokens), dtype=yr.float16)
    V = graph.new_input(dims=(n_local_kv_heads, num_kv_tokens, 128), dtype=yr.float16)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.mark_output(O)
    return graph.superoptimize(
        backend="maca",
        config="attention",
        previous_checkpoint=f"chameleon_attention_maca_bs{batch_size}.json"
    )


def pytorch_attention(Xq, Kcache, Vcache):
    """PyTorch fallback attention when flashinfer not available"""
    # Xq: [batch, n_heads, head_dim]
    # Kcache: [kv_len, n_kv_heads, head_dim]
    # Vcache: [kv_len, n_kv_heads, head_dim]
    
    # Simple scaled dot-product attention
    scale = head_dim ** -0.5
    # Reshape for attention
    q = Xq.transpose(0, 1)  # [n_heads, batch, head_dim]
    k = Kcache.permute(1, 2, 0)  # [n_kv_heads, head_dim, kv_len]
    v = Kcache.permute(1, 0, 2)  # [n_kv_heads, kv_len, head_dim]
    
    scores = torch.matmul(q, k) * scale
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output.transpose(0, 1)  # [batch, n_heads, head_dim]


def yirage_chameleon(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels):
    """Single Chameleon layer forward pass with YiRage optimized kernels"""
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
    
    # FFN with RMS norm fusion
    X = output
    func = kernels[1]
    outputs = func(inputs=[X, W13])
    X13 = outputs[0]
    X1, X3 = X13.chunk(2, -1)
    
    # Final projection
    output = torch.matmul(X1, W2)
    return output


def benchmark(kernels, X, Wqkv, Wo, W13, W2, Kcache, Vcache, warmup=16, repetitions=1000):
    """Run benchmark with proper CUDA timing"""
    # Warmup
    for _ in range(warmup):
        yirage_chameleon(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels)
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    starter.record()
    for _ in range(repetitions):
        yirage_chameleon(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels)
    ender.record()
    torch.cuda.synchronize()
    
    total_time = starter.elapsed_time(ender)
    mean_time = total_time / repetitions
    return mean_time


def main():
    parser = argparse.ArgumentParser(description="Chameleon MACA Benchmark")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--warmup", type=int, default=16, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=1000, help="Benchmark repetitions")
    parser.add_argument("--skip-search", action="store_true", help="Skip optimization search")
    args = parser.parse_args()
    
    global batch_size
    batch_size = args.batch_size
    
    print("=" * 60)
    print("Chameleon Model Benchmark - MACA Backend")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {batch_size}")
    print(f"Num tokens: {num_tokens}")
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
    
    if args.skip_search:
        print("Skipping optimization search...")
        return
    
    # Run optimization search
    print("Searching for optimal kernel fusions...")
    print("(This may take several minutes on first run)")
    print()
    
    k1 = get_rms_linear()
    print("✓ RMS+Linear (QKV) optimized")
    
    k2 = get_rms_linear2()
    print("✓ RMS+Linear (FFN) optimized")
    
    kernels = [k1, k2]
    
    if all(k is not None for k in kernels):
        print()
        print("Running benchmark...")
        mean_time = benchmark(kernels, X, Wqkv, Wo, W13, W2, Kcache, Vcache,
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

