#!/usr/bin/env python3
"""
COMET Simulation: Compound Operation Performance Modeling

This script demonstrates COMET's cost model predictions for various
hardware configurations and compound operations.

Shows the expected speedup from operation fusion across:
- Different hardware types (CPU, GPU, TPU)
- Different memory bandwidths
- Different compute capabilities
- Different operation types
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from yirage.rl.cluster.simulator import (
    COMETCostModel, COMETHardwareConfig,
    SchedulingStrategy, MemoryLevel, CommunicationType,
)
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class HardwareProfile:
    """Hardware profile for simulation."""
    name: str
    config: COMETHardwareConfig


def get_hardware_profiles() -> List[HardwareProfile]:
    """Get predefined hardware profiles."""
    return [
        HardwareProfile(
            name="CPU (Intel Xeon)",
            config=COMETHardwareConfig(
                dram_bandwidth_gbps=50.0,
                global_buffer_bandwidth_gbps=200.0,
                local_buffer_bandwidth_gbps=500.0,
                num_compute_units=32,
                peak_tflops_fp16=1.0,
                peak_tflops_fp32=0.5,
            )
        ),
        HardwareProfile(
            name="GPU (NVIDIA A100)",
            config=COMETHardwareConfig(
                dram_bandwidth_gbps=2039.0,  # HBM2e
                global_buffer_bandwidth_gbps=8000.0,  # L2 cache
                local_buffer_bandwidth_gbps=20000.0,  # Shared memory
                num_compute_units=108,
                peak_tflops_fp16=312.0,
                peak_tflops_fp32=156.0,
            )
        ),
        HardwareProfile(
            name="GPU (NVIDIA H100)",
            config=COMETHardwareConfig(
                dram_bandwidth_gbps=3350.0,  # HBM3
                global_buffer_bandwidth_gbps=12000.0,
                local_buffer_bandwidth_gbps=30000.0,
                num_compute_units=132,
                peak_tflops_fp16=990.0,  # With sparsity
                peak_tflops_fp32=495.0,
            )
        ),
        HardwareProfile(
            name="TPU (Google v4)",
            config=COMETHardwareConfig(
                dram_bandwidth_gbps=1200.0,
                global_buffer_bandwidth_gbps=4000.0,
                local_buffer_bandwidth_gbps=10000.0,
                num_compute_units=4,  # 4 cores per chip
                peak_tflops_fp16=275.0,
                peak_tflops_fp32=137.5,
            )
        ),
        HardwareProfile(
            name="NPU (Huawei Ascend 910B)",
            config=COMETHardwareConfig(
                dram_bandwidth_gbps=1600.0,
                global_buffer_bandwidth_gbps=6000.0,
                local_buffer_bandwidth_gbps=15000.0,
                num_compute_units=64,  # AI cores
                peak_tflops_fp16=640.0,
                peak_tflops_fp32=320.0,
            )
        ),
    ]


def simulate_gemm_softmax(
    hw: HardwareProfile,
    M: int, K: int, N: int,
    dtype_bytes: int = 2,
) -> Dict:
    """Simulate GEMM-Softmax performance."""
    cost_model = COMETCostModel(hw_config=hw.config)
    
    # Fused execution
    fused_latency, fused_energy = cost_model.estimate_compound_operation(
        op_name="gemm_softmax",
        input_shapes=[(M, K), (K, N)],
        dtype_bytes=dtype_bytes,
        strategy=SchedulingStrategy.PIPELINED,
    )
    
    # Unfused execution (GEMM + Softmax separately)
    # GEMM: Read A, B, Write C
    # Softmax: Read C, Write D
    gemm_flops = 2 * M * K * N
    softmax_flops = 5 * M * N  # max, sub, exp, sum, div
    
    # Memory traffic for unfused
    unfused_mem_bytes = (
        M * K * dtype_bytes +  # Read A
        K * N * dtype_bytes +  # Read B
        M * N * dtype_bytes +  # Write C (GEMM output)
        M * N * dtype_bytes +  # Read C (Softmax input)
        M * N * dtype_bytes    # Write D (Softmax output)
    )
    
    # Memory traffic for fused (C stays in cache)
    fused_mem_bytes = (
        M * K * dtype_bytes +  # Read A
        K * N * dtype_bytes +  # Read B
        M * N * dtype_bytes    # Write D (final output)
    )
    
    # Unfused latency
    unfused_compute_ms = ((gemm_flops + softmax_flops) / 
                          (hw.config.peak_tflops_fp16 * 1e12)) * 1000
    unfused_mem_ms = (unfused_mem_bytes / 
                      (hw.config.dram_bandwidth_gbps * 1e9)) * 1000
    unfused_latency_ms = max(unfused_compute_ms, unfused_mem_ms)
    
    speedup = unfused_latency_ms / fused_latency.total_latency_ms
    
    return {
        "hardware": hw.name,
        "operation": f"GEMM-Softmax [{M}x{K}]@[{K}x{N}]",
        "fused_latency_ms": fused_latency.total_latency_ms,
        "unfused_latency_ms": unfused_latency_ms,
        "speedup": speedup,
        "memory_saved_mb": (unfused_mem_bytes - fused_mem_bytes) / 1024 / 1024,
        "compute_latency_ms": fused_latency.compute_latency_ms,
        "memory_latency_ms": fused_latency.total_memory_latency_ms,
    }


def simulate_self_attention(
    hw: HardwareProfile,
    batch: int, heads: int, seq_len: int, head_dim: int,
    dtype_bytes: int = 2,
) -> Dict:
    """Simulate Self-Attention performance."""
    cost_model = COMETCostModel(hw_config=hw.config)
    
    fused_latency, fused_energy = cost_model.estimate_compound_operation(
        op_name="self_attention",
        input_shapes=[
            (batch, heads, seq_len, head_dim),
            (batch, heads, seq_len, head_dim),
            (batch, heads, seq_len, head_dim),
        ],
        dtype_bytes=dtype_bytes,
        strategy=SchedulingStrategy.PIPELINED,
    )
    
    # Unfused attention memory traffic
    # QK^T: Read Q, K, Write QK (S*S per head)
    # Softmax: Read QK, Write Attn
    # Attn@V: Read Attn, V, Write Out
    qk_size = batch * heads * seq_len * seq_len * dtype_bytes
    qkv_size = batch * heads * seq_len * head_dim * dtype_bytes
    
    unfused_mem_bytes = (
        qkv_size +    # Read Q
        qkv_size +    # Read K
        qk_size +     # Write QK
        qk_size +     # Read QK for softmax
        qk_size +     # Write Attn
        qk_size +     # Read Attn
        qkv_size +    # Read V
        qkv_size      # Write Out
    )
    
    # Fused: QK stays in SRAM, Attn stays in SRAM
    fused_mem_bytes = (
        qkv_size * 3 +  # Read Q, K, V
        qkv_size        # Write Out
    )
    
    # Compute
    qk_flops = 2 * batch * heads * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch * heads * seq_len * seq_len
    av_flops = 2 * batch * heads * seq_len * head_dim * seq_len
    total_flops = qk_flops + softmax_flops + av_flops
    
    unfused_compute_ms = (total_flops / (hw.config.peak_tflops_fp16 * 1e12)) * 1000
    unfused_mem_ms = (unfused_mem_bytes / (hw.config.dram_bandwidth_gbps * 1e9)) * 1000
    unfused_latency_ms = max(unfused_compute_ms, unfused_mem_ms)
    
    speedup = unfused_latency_ms / fused_latency.total_latency_ms
    
    return {
        "hardware": hw.name,
        "operation": f"Self-Attn [B={batch}, H={heads}, S={seq_len}, D={head_dim}]",
        "fused_latency_ms": fused_latency.total_latency_ms,
        "unfused_latency_ms": unfused_latency_ms,
        "speedup": speedup,
        "memory_saved_mb": (unfused_mem_bytes - fused_mem_bytes) / 1024 / 1024,
    }


def run_simulation():
    """Run COMET simulation across hardware profiles."""
    
    print("=" * 80)
    print("COMET Performance Simulation")
    print("=" * 80)
    print("Simulating compound operation performance across different hardware")
    print("=" * 80)
    print()
    
    hardware_profiles = get_hardware_profiles()
    
    # Test configurations
    gemm_configs = [
        (1024, 512, 1024),
        (2048, 1024, 2048),
        (4096, 2048, 4096),
    ]
    
    attn_configs = [
        (1, 32, 1024, 128),   # B=1, H=32, S=1024, D=128 (LLaMA-7B)
        (1, 32, 2048, 128),   # B=1, H=32, S=2048, D=128
        (1, 32, 4096, 128),   # B=1, H=32, S=4096, D=128 (Long context)
    ]
    
    # GEMM-Softmax results
    print("### GEMM-Softmax Simulation ###")
    print("-" * 80)
    print(f"{'Hardware':<25} | {'Size':<25} | {'Speedup':>8} | {'Mem Saved':>10}")
    print("-" * 80)
    
    gemm_results = []
    for hw in hardware_profiles:
        for M, K, N in gemm_configs:
            result = simulate_gemm_softmax(hw, M, K, N)
            gemm_results.append(result)
            print(f"{result['hardware']:<25} | {f'[{M}x{K}]@[{K}x{N}]':<25} | "
                  f"{result['speedup']:>7.2f}x | {result['memory_saved_mb']:>8.1f} MB")
    
    print()
    
    # Self-Attention results
    print("### Self-Attention Simulation ###")
    print("-" * 80)
    print(f"{'Hardware':<25} | {'Config':<25} | {'Speedup':>8} | {'Mem Saved':>10}")
    print("-" * 80)
    
    attn_results = []
    for hw in hardware_profiles:
        for B, H, S, D in attn_configs:
            result = simulate_self_attention(hw, B, H, S, D)
            attn_results.append(result)
            print(f"{result['hardware']:<25} | {f'B={B},H={H},S={S},D={D}':<25} | "
                  f"{result['speedup']:>7.2f}x | {result['memory_saved_mb']:>8.1f} MB")
    
    print()
    
    # Summary by hardware
    print("=" * 80)
    print("SUMMARY BY HARDWARE")
    print("=" * 80)
    
    for hw in hardware_profiles:
        hw_gemm = [r for r in gemm_results if r['hardware'] == hw.name]
        hw_attn = [r for r in attn_results if r['hardware'] == hw.name]
        
        avg_gemm_speedup = np.mean([r['speedup'] for r in hw_gemm])
        avg_attn_speedup = np.mean([r['speedup'] for r in hw_attn])
        total_mem_saved = sum(r['memory_saved_mb'] for r in hw_gemm + hw_attn)
        
        print(f"\n{hw.name}:")
        print(f"  GEMM-Softmax avg speedup: {avg_gemm_speedup:.2f}x")
        print(f"  Self-Attention avg speedup: {avg_attn_speedup:.2f}x")
        print(f"  Total memory traffic saved: {total_mem_saved:.1f} MB")
    
    print()
    print("=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. GPUs benefit most from fusion due to high compute-to-memory ratio")
    print("2. Self-Attention shows larger speedups due to O(S^2) intermediate tensors")
    print("3. Longer sequences (S) amplify the benefits of COMET fusion")
    print("4. Memory bandwidth is often the bottleneck, not compute")
    print("=" * 80)


def run_distributed_comparison():
    """Compare local vs distributed execution."""
    
    print("\n" + "=" * 80)
    print("COMET Distributed Execution Analysis")
    print("=" * 80)
    print("Comparing local vs distributed compound operation execution")
    print("=" * 80 + "\n")
    
    hw = HardwareProfile(
        name="GPU (8x A100 cluster)",
        config=COMETHardwareConfig(
            dram_bandwidth_gbps=2039.0,
            global_buffer_bandwidth_gbps=8000.0,
            local_buffer_bandwidth_gbps=20000.0,
            num_compute_units=108 * 8,  # 8 GPUs
            peak_tflops_fp16=312.0 * 8,
            peak_tflops_fp32=156.0 * 8,
            noc_bandwidth_gbps=600.0,  # NVLink between GPUs
        )
    )
    
    cost_model = COMETCostModel(hw_config=hw.config)
    
    # Large model configurations
    configs = [
        ("LLaMA-7B Attention", "self_attention", 
         [(1, 32, 4096, 128), (1, 32, 4096, 128), (1, 32, 4096, 128)]),
        ("LLaMA-70B MLP", "gemm_softmax",
         [(8192, 8192), (8192, 28672)]),
    ]
    
    for name, op_type, shapes in configs:
        print(f"\n{name}")
        print("-" * 50)
        
        # Compare different device counts
        for num_devices in [1, 2, 4, 8]:
            results = cost_model.compare_distributed_variants(
                op_name=op_type,
                input_shapes=shapes,
                num_devices=num_devices,
            )
            
            print(f"  {num_devices} GPU(s): "
                  f"Latency={results['distributed']['latency_ms']:.2f}ms, "
                  f"vs 1GPU speedup={results.get('speedup', 1.0):.2f}x")
    
    print()
    print("=" * 80)
    print("Note: Distributed execution adds collective communication overhead")
    print("COMET models this explicitly to find optimal parallelism strategy")
    print("=" * 80)


if __name__ == "__main__":
    run_simulation()
    run_distributed_comparison()
