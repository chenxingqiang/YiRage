#!/usr/bin/env python3
"""
Benchmark: YiRage vs PyTorch on MetaX C500
"""
import yirage
import torch
import time
import sys

def profile_pytorch(func, warmup=10, repeat=100):
    """Profile PyTorch function using CUDA events"""
    for _ in range(warmup):
        _ = func()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeat):
        _ = func()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeat

def benchmark():
    print("=" * 60)
    print("YiRage vs PyTorch Performance on MetaX C500")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"YiRage: {yirage.__version__}")
    print()
    
    # Small test for quick search
    M, K, N = 32, 32, 32
    print(f"Test: MatMul ({M}x{K} @ {K}x{N})")
    print("-" * 40)
    
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w = torch.randn(K, N, dtype=torch.float16, device="cuda")
    
    # PyTorch baseline
    pytorch_time = profile_pytorch(lambda: torch.matmul(x, w))
    print(f"PyTorch time: {pytorch_time:.4f} ms")
    
    # YiRage
    print("\nCreating YiRage graph...")
    graph = yirage.new_kernel_graph()
    X = graph.new_input(dims=(M, K), dtype=yirage.float16)
    W = graph.new_input(dims=(K, N), dtype=yirage.float16)
    Y = graph.matmul(X, W)
    graph.mark_output(Y)
    
    print("Running search (may take minutes)...")
    start = time.time()
    
    optimized = graph.superoptimize(
        backend="maca",
        config="mlp",
        verbose=False
    )
    
    search_time = time.time() - start
    print(f"Search time: {search_time:.1f}s")
    
    if optimized:
        print("\nProfiling optimized graph...")
        
        # Warmup
        for _ in range(10):
            _ = optimized(x, w)
        torch.cuda.synchronize()
        
        # Profile
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        for _ in range(100):
            _ = optimized(x, w)
        end_evt.record()
        torch.cuda.synchronize()
        
        yirage_time = start_evt.elapsed_time(end_evt) / 100
        speedup = pytorch_time / yirage_time
        
        print(f"YiRage time: {yirage_time:.4f} ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify correctness
        result = optimized(x, w)
        expected = torch.matmul(x, w)
        diff = (result - expected).abs().max().item()
        print(f"\nMax diff: {diff:.6f}")
        print("Correct!" if diff < 0.1 else "MISMATCH!")
    else:
        print("No optimized graph found")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    benchmark()

