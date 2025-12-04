#!/usr/bin/env python3
"""PyTorch baseline benchmark on MetaX C500"""
import torch

print("=" * 60)
print("PyTorch Baseline on MetaX C500 GPU")
print("=" * 60)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print()

def profile(func, name, warmup=20, repeat=100):
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
    
    ms = start.elapsed_time(end) / repeat
    print(f"  {name}: {ms:.4f} ms")
    return ms

# Matrix Multiplication tests
print("Benchmark: Matrix Multiplication (FP16)")
print("-" * 40)

sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]
results = []

for M, K, N in sizes:
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w = torch.randn(K, N, dtype=torch.float16, device="cuda")
    
    ms = profile(lambda x=x, w=w: torch.matmul(x, w), f"MatMul {M}x{K} @ {K}x{N}")
    results.append((f"{M}x{K}@{K}x{N}", ms))

print()
print("Benchmark: Fused Operations (FP16)")  
print("-" * 40)

# Fused: Y = ReLU(X @ W + B) - common in MLP
M, K, N = 64, 128, 128
x = torch.randn(M, K, dtype=torch.float16, device="cuda")
w = torch.randn(K, N, dtype=torch.float16, device="cuda")
b = torch.randn(M, N, dtype=torch.float16, device="cuda")

def mlp_unfused():
    xw = torch.matmul(x, w)
    xwb = xw + b
    return torch.relu(xwb)

unfused_ms = profile(mlp_unfused, "MLP (unfused): Y=ReLU(X@W+B)")

print()
print("=" * 60)
print("SUMMARY - PyTorch Baseline Timings")
print("=" * 60)
for name, ms in results:
    print(f"  {name}: {ms:.4f} ms")
print(f"  MLP unfused: {unfused_ms:.4f} ms")

print()
print("Note: YiRage kernel fusion eliminates intermediate memory")
print("      writes between matmul/add/relu, typically providing")
print("      1.2x-2x speedup for fused operations.")

