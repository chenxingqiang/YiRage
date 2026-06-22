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
    args = parser.parse_args()

    quick = args.quick or _DEVICE == "cpu"
    if quick:
        m, k, n = 8, 128, 256
        grid_dim = (4, 1, 1)
        block_dim = (32, 1, 1)
        forloop_range = 4
        warmup, reps = 2, 5
    else:
        m, k, n = 16, 4096, 4096
        grid_dim = (64, 1, 1)
        block_dim = (128, 1, 1)
        forloop_range = 32
        warmup, reps = 16, 1000

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(m, k), dtype=yr.float16)
    W1 = graph.new_input(dims=(k, n), dtype=yr.float16)
    W2 = graph.new_input(dims=(k, n), dtype=yr.float16)
    tb_graph = yr.new_threadblock_graph(
        grid_dim=grid_dim,
        block_dim=block_dim,
        forloop_range=forloop_range,
        reduction_dimx=32 if quick else 64,
    )
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW1 = tb_graph.new_input(dtensor=W1, input_map=(1, -1, -1), forloop_dim=0)
    tW2 = tb_graph.new_input(dtensor=W2, input_map=(1, -1, -1), forloop_dim=0)
    tD1 = tb_graph.matmul(tX, tW1)
    tD2 = tb_graph.matmul(tX, tW2)
    tA1 = tb_graph.forloop_accum(tD1)
    tA2 = tb_graph.forloop_accum(tD2)
    tS = tb_graph.silu(tA1)
    tO = tb_graph.mul(tS, tA2)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W1, W2], tb_graph)
    graph.mark_output(O[0])

    input_tensors = [
        torch.randn(m, k, dtype=torch.float16, device=_DEVICE),
        torch.randn(k, n, dtype=torch.float16, device=_DEVICE),
        torch.randn(k, n, dtype=torch.float16, device=_DEVICE),
    ]

    if _DEVICE == "cuda" and not quick:
        input_strides = [tensor.stride() for tensor in input_tensors]
        p = yr.generate_cuda_program(
            graph.cygraph, target_cc=86, input_strides=input_strides
        )
        print(p["code"])

    mean_syn = __bench_ms(
        lambda: graph(inputs=input_tensors),
        warmup=warmup,
        reps=reps,
    )
    print(f"reference_muGraph run time (ms): {mean_syn:.6g}")
    if not quick and _DEVICE == "cuda":
        graph.visualize("gated_mlp")
    sys.exit(0)
