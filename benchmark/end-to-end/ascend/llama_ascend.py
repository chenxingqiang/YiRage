#!/usr/bin/env python3
"""
LLaMA Model Benchmark for Huawei Ascend NPU
Adapted for Ascend backend with AI Core and Cube operations.

Requires:
- CANN toolkit installed
- torch_npu (PyTorch for Ascend)
"""

import yirage as yr
import torch
import argparse
import time

# Check for Ascend support
try:
    import torch_npu
    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False
    print("Warning: torch_npu not available, using CPU simulation")

# Model configuration (LLaMA 70B style)
n_local_heads = 32
n_local_kv_heads = 8  # GQA: 8 KV heads for 32 Q heads
head_dim = 128
intermediate_size = 14336
num_tokens = 1
num_kv_tokens = 4096
batch_size = 8

# Device selection
device = "npu" if HAS_NPU else "cpu"


def get_rms_linear():
    """RMS Norm + Linear fusion for QKV projection"""
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim), dtype=yr.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    return graph.superoptimize(
        backend="ascend",  # Ascend backend with AI Core optimization
        config="mlp",
        previous_checkpoint=f"llama_rms_linear_ascend_bs{batch_size}.json"
    )


def get_rms_linear2():
    """RMS Norm + Linear fusion for FFN gate/up projection"""
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, intermediate_size * 2), dtype=yr.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    return graph.superoptimize(
        backend="ascend",
        config="mlp",
        previous_checkpoint=f"llama_rms_linear2_ascend_bs{batch_size}.json"
    )


def simple_attention(Xq, Kcache, Vcache):
    """Simple attention implementation for Ascend"""
    scale = head_dim ** -0.5
    n_rep = n_local_heads // n_local_kv_heads
    
    # Reshape for attention
    q = Xq.transpose(0, 1)
    k = Kcache.permute(1, 2, 0)
    v = Vcache.permute(1, 0, 2)
    
    # Repeat KV for GQA
    k = k.repeat_interleave(n_rep, dim=0)
    v = v.repeat_interleave(n_rep, dim=0)
    
    scores = torch.matmul(q, k) * scale
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output.transpose(0, 1)


def yirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels):
    """Single LLaMA layer forward pass with YiRage optimized kernels"""
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
    output = simple_attention(Xq, Kcache, Vcache)
    
    # Output projection
    output = torch.matmul(output.reshape(output_shape), Wo)
    
    # FFN with RMS norm fusion (SwiGLU)
    X = output
    func = kernels[1]
    outputs = func(inputs=[X, W13])
    X13 = outputs[0]
    X1, X3 = X13.chunk(2, -1)
    
    # SwiGLU activation
    X1 = torch.nn.functional.silu(X1) * X3
    
    # Down projection
    output = torch.matmul(X1, W2)
    return output


def benchmark(kernels, X, Wqkv, Wo, W13, W2, Kcache, Vcache, warmup=16, repetitions=1000):
    """Run benchmark"""
    # Warmup
    for _ in range(warmup):
        yirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels)
    
    if HAS_NPU:
        torch.npu.synchronize()
        
        # NPU timing
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(repetitions):
            yirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels)
        end_event.record()
        torch.npu.synchronize()
        
        total_time = start_event.elapsed_time(end_event)
    else:
        # CPU timing
        start = time.perf_counter()
        for _ in range(repetitions):
            yirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels)
        total_time = (time.perf_counter() - start) * 1000
    
    return total_time / repetitions


def main():
    parser = argparse.ArgumentParser(description="LLaMA Ascend Benchmark")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--warmup", type=int, default=16, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=1000, help="Benchmark repetitions")
    parser.add_argument("--skip-search", action="store_true", help="Skip optimization search")
    args = parser.parse_args()
    
    global batch_size
    batch_size = args.batch_size
    
    print("=" * 60)
    print("LLaMA Model Benchmark - Ascend Backend")
    print("=" * 60)
    
    if HAS_NPU:
        print(f"Device: {torch.npu.get_device_name(0)}")
    else:
        print("Device: CPU (Ascend NPU not available)")
    
    print(f"Batch size: {batch_size}")
    print(f"Num tokens: {num_tokens}")
    print(f"GQA: {n_local_heads} Q heads, {n_local_kv_heads} KV heads")
    print()
    
    # Initialize tensors
    dtype = torch.float16
    X = torch.randn(batch_size * num_tokens, 4096, dtype=dtype, device=device)
    Wqkv = torch.randn(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim, dtype=dtype, device=device)
    Wo = torch.randn(n_local_heads * head_dim, 4096, dtype=dtype, device=device)
    W13 = torch.randn(4096, intermediate_size * 2, dtype=dtype, device=device)
    W2 = torch.rand(intermediate_size, 4096, dtype=dtype, device=device)
    Kcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=dtype, device=device)
    Vcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=dtype, device=device)
    
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

