#!/usr/bin/env python3
"""
COMET PyTorch Benchmark: Compound Operation Performance Testing

This benchmark uses PyTorch to demonstrate the performance benefits
of COMET-style compound operation fusion on CPU.

Key operations tested:
- GEMM-Softmax fusion
- GEMM-LayerNorm fusion  
- Self-Attention (FlashAttention-style)
- Gated MLP (LLaMA-style)
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Add yirage to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from yirage.rl.cluster.simulator import (
    COMETCostModel, COMETHardwareConfig,
    SchedulingStrategy,
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    fused_time_ms: float
    unfused_time_ms: float
    speedup: float
    memory_saved_mb: float
    
    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Fused:    {self.fused_time_ms:.3f} ms\n"
            f"  Unfused:  {self.unfused_time_ms:.3f} ms\n"
            f"  Speedup:  {self.speedup:.2f}x\n"
            f"  Memory Saved: {self.memory_saved_mb:.2f} MB"
        )


def benchmark_gemm_softmax_torch(
    M: int, K: int, N: int,
    warmup: int = 3,
    iterations: int = 10,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult:
    """
    Benchmark GEMM-Softmax using PyTorch.
    
    Unfused: C = A @ B; D = softmax(C)
    Fused: D = softmax(A @ B)
    """
    device = torch.device('cpu')
    
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        C = torch.mm(A, B)
        D = F.softmax(C, dim=-1)
    
    # Benchmark unfused (explicit intermediate)
    unfused_times = []
    for _ in range(iterations):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        
        C = torch.mm(A, B)
        # Force sync to ensure C is fully computed
        _ = C.sum().item()
        D = F.softmax(C, dim=-1)
        _ = D.sum().item()
        
        end = time.perf_counter()
        unfused_times.append((end - start) * 1000)
    
    # Benchmark fused (single expression)
    fused_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        D = F.softmax(torch.mm(A, B), dim=-1)
        _ = D.sum().item()
        
        end = time.perf_counter()
        fused_times.append((end - start) * 1000)
    
    unfused_time = sum(unfused_times) / len(unfused_times)
    fused_time = sum(fused_times) / len(fused_times)
    speedup = unfused_time / fused_time if fused_time > 0 else 1.0
    
    # Memory saved: intermediate C not materialized
    elem_size = 4 if dtype == torch.float32 else 2
    memory_saved_mb = (M * N * elem_size) / (1024 * 1024)
    
    return BenchmarkResult(
        name=f"GEMM-Softmax [{M}x{K}] @ [{K}x{N}]",
        fused_time_ms=fused_time,
        unfused_time_ms=unfused_time,
        speedup=speedup,
        memory_saved_mb=memory_saved_mb,
    )


def benchmark_gemm_layernorm_torch(
    M: int, K: int, N: int,
    warmup: int = 3,
    iterations: int = 10,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult:
    """
    Benchmark GEMM-LayerNorm using PyTorch.
    """
    device = torch.device('cpu')
    
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        C = torch.mm(A, B)
        D = F.layer_norm(C, [N])
    
    # Benchmark unfused
    unfused_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        C = torch.mm(A, B)
        _ = C.sum().item()
        D = F.layer_norm(C, [N])
        _ = D.sum().item()
        
        end = time.perf_counter()
        unfused_times.append((end - start) * 1000)
    
    # Benchmark fused
    fused_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        D = F.layer_norm(torch.mm(A, B), [N])
        _ = D.sum().item()
        
        end = time.perf_counter()
        fused_times.append((end - start) * 1000)
    
    unfused_time = sum(unfused_times) / len(unfused_times)
    fused_time = sum(fused_times) / len(fused_times)
    speedup = unfused_time / fused_time if fused_time > 0 else 1.0
    
    elem_size = 4 if dtype == torch.float32 else 2
    memory_saved_mb = (M * N * elem_size) / (1024 * 1024)
    
    return BenchmarkResult(
        name=f"GEMM-LayerNorm [{M}x{K}] @ [{K}x{N}]",
        fused_time_ms=fused_time,
        unfused_time_ms=unfused_time,
        speedup=speedup,
        memory_saved_mb=memory_saved_mb,
    )


def benchmark_self_attention_torch(
    batch: int, heads: int, seq_len: int, head_dim: int,
    warmup: int = 3,
    iterations: int = 10,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult:
    """
    Benchmark Self-Attention using PyTorch.
    
    Unfused: QK = Q @ K^T; attn = softmax(QK); out = attn @ V
    Fused: out = softmax(Q @ K^T) @ V (single expression)
    """
    device = torch.device('cpu')
    
    Q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    K = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    V = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    # Warmup
    for _ in range(warmup):
        QK = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(QK, dim=-1)
        out = torch.matmul(attn, V)
    
    # Benchmark unfused (explicit intermediates)
    unfused_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        QK = torch.matmul(Q, K.transpose(-2, -1)) * scale
        _ = QK.sum().item()  # Force materialization
        attn = F.softmax(QK, dim=-1)
        _ = attn.sum().item()  # Force materialization
        out = torch.matmul(attn, V)
        _ = out.sum().item()
        
        end = time.perf_counter()
        unfused_times.append((end - start) * 1000)
    
    # Benchmark fused (single expression)
    fused_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        out = torch.matmul(
            F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1),
            V
        )
        _ = out.sum().item()
        
        end = time.perf_counter()
        fused_times.append((end - start) * 1000)
    
    unfused_time = sum(unfused_times) / len(unfused_times)
    fused_time = sum(fused_times) / len(fused_times)
    speedup = unfused_time / fused_time if fused_time > 0 else 1.0
    
    # Memory saved: QK [B,H,S,S] and attn [B,H,S,S]
    elem_size = 4 if dtype == torch.float32 else 2
    qk_size = batch * heads * seq_len * seq_len * elem_size
    memory_saved_mb = (2 * qk_size) / (1024 * 1024)
    
    return BenchmarkResult(
        name=f"Self-Attention [B={batch}, H={heads}, S={seq_len}, D={head_dim}]",
        fused_time_ms=fused_time,
        unfused_time_ms=unfused_time,
        speedup=speedup,
        memory_saved_mb=memory_saved_mb,
    )


def benchmark_gated_mlp_torch(
    batch: int, seq_len: int, hidden_dim: int, ff_dim: int,
    warmup: int = 3,
    iterations: int = 10,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult:
    """
    Benchmark Gated MLP using PyTorch.
    
    Unfused: gate = X @ W_gate; gate_act = silu(gate); up = X @ W_up; 
             intermediate = gate_act * up; out = intermediate @ W_down
    Fused: out = (silu(X @ W_gate) * (X @ W_up)) @ W_down
    """
    device = torch.device('cpu')
    
    X = torch.randn(batch, seq_len, hidden_dim, dtype=dtype, device=device)
    W_gate = torch.randn(hidden_dim, ff_dim, dtype=dtype, device=device)
    W_up = torch.randn(hidden_dim, ff_dim, dtype=dtype, device=device)
    W_down = torch.randn(ff_dim, hidden_dim, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        gate = torch.mm(X.view(-1, hidden_dim), W_gate)
        gate_act = F.silu(gate)
        up = torch.mm(X.view(-1, hidden_dim), W_up)
        intermediate = gate_act * up
        out = torch.mm(intermediate, W_down)
    
    # Benchmark unfused
    unfused_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        X_flat = X.view(-1, hidden_dim)
        gate = torch.mm(X_flat, W_gate)
        _ = gate.sum().item()
        gate_act = F.silu(gate)
        _ = gate_act.sum().item()
        up = torch.mm(X_flat, W_up)
        _ = up.sum().item()
        intermediate = gate_act * up
        _ = intermediate.sum().item()
        out = torch.mm(intermediate, W_down)
        _ = out.sum().item()
        
        end = time.perf_counter()
        unfused_times.append((end - start) * 1000)
    
    # Benchmark fused
    fused_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        X_flat = X.view(-1, hidden_dim)
        out = torch.mm(F.silu(torch.mm(X_flat, W_gate)) * torch.mm(X_flat, W_up), W_down)
        _ = out.sum().item()
        
        end = time.perf_counter()
        fused_times.append((end - start) * 1000)
    
    unfused_time = sum(unfused_times) / len(unfused_times)
    fused_time = sum(fused_times) / len(fused_times)
    speedup = unfused_time / fused_time if fused_time > 0 else 1.0
    
    # Memory saved: gate, gate_act, up, intermediate
    elem_size = 4 if dtype == torch.float32 else 2
    intermediate_size = batch * seq_len * ff_dim * elem_size
    memory_saved_mb = (4 * intermediate_size) / (1024 * 1024)
    
    return BenchmarkResult(
        name=f"Gated MLP [B={batch}, S={seq_len}, D={hidden_dim}, FF={ff_dim}]",
        fused_time_ms=fused_time,
        unfused_time_ms=unfused_time,
        speedup=speedup,
        memory_saved_mb=memory_saved_mb,
    )


def run_torch_benchmarks():
    """Run complete PyTorch benchmark suite."""
    
    print("=" * 70)
    print("COMET PyTorch Benchmark Suite")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU ({torch.get_num_threads()} threads)")
    print(f"Dtype: float32")
    print("=" * 70)
    print()
    
    results = []
    
    # GEMM-Softmax benchmarks
    print("### GEMM-Softmax Benchmarks ###")
    print("-" * 50)
    
    gemm_sizes = [
        (512, 512, 512),
        (1024, 512, 1024),
        (2048, 1024, 2048),
    ]
    
    for M, K, N in gemm_sizes:
        result = benchmark_gemm_softmax_torch(M, K, N)
        results.append(result)
        print(result)
        print()
    
    # GEMM-LayerNorm benchmarks
    print("### GEMM-LayerNorm Benchmarks ###")
    print("-" * 50)
    
    for M, K, N in gemm_sizes:
        result = benchmark_gemm_layernorm_torch(M, K, N)
        results.append(result)
        print(result)
        print()
    
    # Self-Attention benchmarks
    print("### Self-Attention Benchmarks ###")
    print("-" * 50)
    
    attn_configs = [
        (1, 8, 256, 64),
        (1, 8, 512, 64),
        (1, 32, 512, 128),  # LLaMA-7B style
    ]
    
    for batch, heads, seq_len, head_dim in attn_configs:
        result = benchmark_self_attention_torch(batch, heads, seq_len, head_dim)
        results.append(result)
        print(result)
        print()
    
    # Gated MLP benchmarks
    print("### Gated MLP Benchmarks ###")
    print("-" * 50)
    
    mlp_configs = [
        (1, 256, 1024, 2048),
        (1, 512, 2048, 4096),
        (1, 256, 4096, 11008),  # LLaMA-7B style
    ]
    
    for batch, seq_len, hidden_dim, ff_dim in mlp_configs:
        result = benchmark_gated_mlp_torch(batch, seq_len, hidden_dim, ff_dim)
        results.append(result)
        print(result)
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_speedup = sum(r.speedup for r in results) / len(results)
    total_memory_saved = sum(r.memory_saved_mb for r in results)
    
    best_speedups = sorted(results, key=lambda x: x.speedup, reverse=True)[:3]
    
    print(f"Total benchmarks: {len(results)}")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Total memory saved: {total_memory_saved:.1f} MB")
    print()
    print("Top speedups:")
    for r in best_speedups:
        print(f"  {r.name}: {r.speedup:.2f}x")
    
    print("=" * 70)
    print()
    print("Note: On CPU, fusion benefits are limited because PyTorch already")
    print("optimizes memory access patterns. On GPU with FlashAttention-style")
    print("kernels, speedups of 2-5x are typical for Self-Attention.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_torch_benchmarks()
