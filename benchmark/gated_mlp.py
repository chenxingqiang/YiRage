import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
os.environ['MACA_PATH'] = '/opt/maca'
os.environ['LD_LIBRARY_PATH'] = '/opt/maca/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import yirage as yr
import numpy as np
import torch
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--file', type=str, default='gated_mlp.json')
    parser.add_argument('--backend', type=str, default='cuda', choices=['cuda', 'mps', 'cpu', 'maca'])
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--profile', type=int, default=1000)
    parser.add_argument('--benchmark_iters', type=int, default=100)
    parser.add_argument('--save_codes', type=bool, default=False)
    parser.add_argument('--fast', action='store_true', help='Fast test mode: skip optimization, use cached graph')

    args = parser.parse_args()
    batch_size = args.batch
    print(f"Batch size: {batch_size}")
    backend = args.backend
    
    # Fast mode: reduce iterations for quick testing
    if args.fast:
        warmup_iters = 1
        profile_iters = 10
        benchmark_iters = 10
        print("🚀 Fast test mode: warmup=1, profile=10, benchmark=10")
    else:
        warmup_iters = args.warmup
        profile_iters = args.profile
        benchmark_iters = args.benchmark_iters
    
    save_codes = args.save_codes
    
    # Backend-specific file path
    filename = f'benchmark/saved_mugraphs/{backend}/{args.file}'
    
    # Select device
    # MACA uses CUDA interface via mcPytorch
    if backend == 'maca' and torch.cuda.is_available():
        device = 'cuda:0'  # MACA maps to CUDA device
    elif backend == 'cuda' and torch.cuda.is_available():
        device = 'cuda:0'
    elif backend == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Backend: {backend}, Device: {device}")

    # Create graph (match PyTorch baseline: X[batch, 4096] @ W[4096, 4096])
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size, 4096), dtype=yr.float16)
    W1 = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
    W2 = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
    O1 = graph.matmul(X, W1)
    O2 = graph.matmul(X, W2)
    O1 = graph.silu(O1)
    O = graph.mul(O1, O2)
    graph.mark_output(O)
    
    # Optimize (or load cached if in fast mode)
    if args.fast:
        import os
        # Try to load cached graph
        if os.path.exists(filename):
            print(f"⚡ Loading cached graph from {filename}")
            optimized_graph = graph.superoptimize(
                config="mlp",
                backend=backend,
                save_codes=False,
                warmup_iters=1,
                profile_iters=1,
                use_cached_graphs=True
            )
        else:
            print(f"⚡ Fast optimization (limited search)")
            optimized_graph = graph.superoptimize(
                config="mlp",
                backend=backend,
                save_codes=save_codes,
                warmup_iters=1,
                profile_iters=10
            )
    else:
        optimized_graph = graph.superoptimize(
            config="mlp",
            backend=backend,
            save_codes=save_codes,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters
        )

    # Create input tensors (match PyTorch baseline dimensions)
    input_tensors = [
        torch.randn(batch_size, 4096, dtype=torch.float16, device=device),
        torch.randn(4096, 4096, dtype=torch.float16, device=device),
        torch.randn(4096, 4096, dtype=torch.float16, device=device)
    ]

    # Warmup
    for _ in range(16):
        optimized_graph(inputs=input_tensors)

    # Benchmark with backend-specific timing
    if device == 'cuda:0':
        # CUDA backend: use CUDA events for precise timing
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(benchmark_iters):
            optimized_graph(inputs=input_tensors)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        mean_syn = curr_time / benchmark_iters
    elif device == 'mps':
        # MPS backend: use Python timing with MPS synchronization
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        start_time = time.perf_counter()
        for _ in range(benchmark_iters):
            optimized_graph(inputs=input_tensors)
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        end_time = time.perf_counter()
        mean_syn = (end_time - start_time) / benchmark_iters * 1000
    else:
        # CPU backend: use Python timing
        start_time = time.perf_counter()
        for _ in range(benchmark_iters):
            optimized_graph(inputs=input_tensors)
        end_time = time.perf_counter()
        mean_syn = (end_time - start_time) / benchmark_iters * 1000

    print(f"Best muGraph run time (ms): {mean_syn}")
