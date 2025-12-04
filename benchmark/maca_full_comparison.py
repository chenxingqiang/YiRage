#!/usr/bin/env python3
"""
Complete YiRage vs PyTorch Comparison on MetaX C500
"""
import yirage
import torch
import time
import sys

def profile_pytorch(func, warmup=20, repeat=100):
    """Profile with CUDA events"""
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeat):
        func()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeat

def run_benchmark():
    print("=" * 65)
    print("  YiRage vs PyTorch Performance Comparison")
    print("  Device: MetaX C500 GPU")
    print("=" * 65)
    print(f"PyTorch: {torch.__version__}")
    print(f"YiRage: {yirage.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Simple matmul test (smaller for faster search)
    M, K, N = 16, 32, 32
    print(f"Test: MatMul Y = X @ W  ({M}x{K} @ {K}x{N}, FP16)")
    print("-" * 65)
    
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w = torch.randn(K, N, dtype=torch.float16, device="cuda")
    
    # PyTorch baseline
    pytorch_ms = profile_pytorch(lambda: torch.matmul(x, w))
    print(f"PyTorch time: {pytorch_ms:.4f} ms")
    
    # YiRage optimization
    print("\nYiRage search (please wait)...", flush=True)
    
    graph = yirage.new_kernel_graph()
    X = graph.new_input(dims=(M, K), dtype=yirage.float16)
    W = graph.new_input(dims=(K, N), dtype=yirage.float16)
    Y = graph.matmul(X, W)
    graph.mark_output(Y)
    
    start = time.time()
    optimized = graph.superoptimize(backend="maca", config="mlp", verbose=False)
    search_time = time.time() - start
    
    print(f"Search completed: {search_time:.1f}s")
    
    if optimized:
        # Profile optimized
        for _ in range(20):
            optimized(x, w)
        torch.cuda.synchronize()
        
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        for _ in range(100):
            optimized(x, w)
        end_evt.record()
        torch.cuda.synchronize()
        
        yirage_ms = start_evt.elapsed_time(end_evt) / 100
        speedup = pytorch_ms / yirage_ms
        
        print(f"YiRage time: {yirage_ms:.4f} ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify
        result = optimized(x, w)
        expected = torch.matmul(x, w)
        diff = (result - expected).abs().max().item()
        print(f"Max diff: {diff:.6f} ({'OK' if diff < 0.1 else 'MISMATCH'})")
    else:
        print("No optimized graph found")
    
    print()
    print("=" * 65)
    print("RESULT SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<20} {'Value':<20}")
    print("-" * 40)
    print(f"{'PyTorch':<20} {pytorch_ms:.4f} ms")
    if optimized:
        print(f"{'YiRage':<20} {yirage_ms:.4f} ms")
        print(f"{'Speedup':<20} {speedup:.2f}x")
    print(f"{'Search time':<20} {search_time:.1f}s")
    print()

if __name__ == "__main__":
    run_benchmark()

