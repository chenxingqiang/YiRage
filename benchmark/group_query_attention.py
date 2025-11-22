import yirage as yr
import numpy as np
import torch
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--file', type=str, default='group_query_attention.json')
    parser.add_argument('--backend', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--profile', type=int, default=1000)
    parser.add_argument('--save_codes', type=bool, default=False)

    args = parser.parse_args()
    batch_size = args.bs
    backend = args.backend
    warmup_iters = args.warmup
    profile_iters = args.profile
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
    Q = graph.new_input(dims=(2 * batch_size, 256, 64), dtype=yr.float16)
    K = graph.new_input(dims=(2 * batch_size, 64, 4096), dtype=yr.float16)
    V = graph.new_input(dims=(2 * batch_size, 4096, 64), dtype=yr.float16)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.mark_output(O)
    
    optimized_graph = graph.superoptimize(
        config="attention",
        backend=backend,
        save_codes=save_codes,
        warmup_iters=warmup_iters,
        profile_iters=profile_iters
    )

    input_tensors = [
        torch.randn(2 * batch_size, 256, 64, dtype=torch.float16, device=device),
        torch.randn(2 * batch_size, 64, 4096, dtype=torch.float16, device=device),
        torch.randn(2 * batch_size, 4096, 64, dtype=torch.float16, device=device)
    ]

    for _ in range(16):
        optimized_graph(inputs=input_tensors)

    # Benchmark with backend-specific timing
    if device == 'cuda:0':
        # CUDA backend: use CUDA events for precise timing
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(1000):
            optimized_graph(inputs=input_tensors)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        mean_syn = curr_time / 1000
    elif device == 'mps':
        # MPS backend: use Python timing with MPS synchronization
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        start_time = time.perf_counter()
        for _ in range(1000):
            optimized_graph(inputs=input_tensors)
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        end_time = time.perf_counter()
        mean_syn = (end_time - start_time) / 1000 * 1000  # Convert to ms
    else:
        # CPU backend: use Python timing
        start_time = time.perf_counter()
        for _ in range(1000):
            optimized_graph(inputs=input_tensors)
        end_time = time.perf_counter()
        mean_syn = (end_time - start_time) / 1000 * 1000  # Convert to ms

    print(f"Best muGraph run time (ms): {mean_syn}")
