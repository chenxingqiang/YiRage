import argparse
import sys

import yirage as yr
import torch

_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def __sync():
    if _DEVICE == "cuda":
        torch.cuda.synchronize()
    elif _DEVICE == "mps":
        torch.mps.synchronize()


def __bench_ms(fn, warmup=16, reps=1000):
    import time

    for _ in range(warmup):
        fn()
    __sync()
    if _DEVICE == "cuda":
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        __sync()
        return s.elapsed_time(e) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    __sync()
    return (time.perf_counter() - t0) / reps * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small shapes and few bench reps for CPU CI smoke",
    )
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    quick = args.quick or _DEVICE == "cpu"
    if quick:
        m, k, n, rank = 1, 128, 256, 8
        warmup, reps = 2, 5
    else:
        m, k, n, rank = 1, 4096, 4096, 16
        warmup, reps = 16, 1000

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(m, k), dtype=yr.float16)
    W = graph.new_input(dims=(k, n), dtype=yr.float16)
    A = graph.new_input(dims=(k, rank), dtype=yr.float16)
    B = graph.new_input(dims=(rank, n), dtype=yr.float16)
    D = graph.matmul(X, A)
    E = graph.matmul(D, B)
    C = graph.matmul(X, W)
    O = graph.add(C, E)
    graph.mark_output(O)

    if _DEVICE != "cuda":
        graph.backend = "cpu"

    input_tensors = [
        torch.randn(m, k, dtype=torch.float16, device=_DEVICE),
        torch.randn(k, n, dtype=torch.float16, device=_DEVICE),
        torch.randn(k, rank, dtype=torch.float16, device=_DEVICE),
        torch.randn(rank, n, dtype=torch.float16, device=_DEVICE),
    ]

    mean_syn = __bench_ms(
        lambda: graph(inputs=input_tensors),
        warmup=warmup,
        reps=reps,
    )
    print(f"reference_muGraph run time (ms): {mean_syn:.6g}")
    if not quick and _DEVICE == "cuda":
        from yirage import visualizer

        graph.visualize("lora")
    sys.exit(0)
