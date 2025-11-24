import yirage as yr
import numpy as np
import torch
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--file', type=str, default='moe.json')
    parser.add_argument('--backend', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--profile', type=int, default=1000)
    parser.add_argument('--benchmark_iters', type=int, default=100)
    parser.add_argument('--save_codes', type=bool, default=False)
    
    args = parser.parse_args()
    batch_size = args.bs
    backend = args.backend
    warmup_iters = args.warmup
    profile_iters = args.profile
    benchmark_iters = args.benchmark_iters
    save_codes = args.save_codes
    
    filename = f'benchmark/saved_mugraphs/{backend}/{args.file}'
    
    # Select device
    if backend == 'cuda' and torch.cuda.is_available():
        device = 'cuda:0'
    elif backend == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Backend: {backend}, Device: {device}")

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(8, 8, 4096), dtype=yr.float16)
    A = graph.new_input(dims=(8, 4096, 4096), dtype=yr.float16)
    B = graph.new_input(dims=(8, 4096, 4096), dtype=yr.float16)
    D = graph.matmul(X, A)
    E = graph.exp(D)
    O = graph.matmul(E, B)
    graph.mark_output(O)
    
    optimized_graph = graph.superoptimize(
        backend=backend,
        save_codes=save_codes,
        warmup_iters=warmup_iters,
        profile_iters=profile_iters
    )

    input_tensors = [
        torch.randn(8, 8, 4096, dtype=torch.float16, device=device),
        torch.randn(8, 4096, 4096, dtype=torch.float16, device=device),
        torch.randn(8, 4096, 4096, dtype=torch.float16, device=device)
    ]

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
