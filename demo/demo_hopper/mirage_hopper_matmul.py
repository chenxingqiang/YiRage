import yirage as yr
import argparse
import os
import torch
_DEVICE = ("cuda" if __import__("torch").cuda.is_available() 
           else "mps" if __import__("torch").backends.mps.is_available() 
           else "cpu")
def __sync():
    d = _DEVICE
    t = __import__("torch")
    if d == "cuda": t.cuda.synchronize()
    elif d == "mps": t.mps.synchronize()
def __bench_ms(fn, warmup=16, reps=1000):
    import time
    for _ in range(warmup): fn()
    __sync()
    if _DEVICE == "cuda":
        t = __import__("torch")
        s, e = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps): fn()
        e.record()
        __sync()
        return s.elapsed_time(e) / reps
    else:
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        __sync()
        return (time.perf_counter() - t0) / reps * 1000


datatype = {
    "e4m3": (yr.float8_e4m3, torch.float8_e4m3fn),
    "e5m2": (yr.float8_e5m2, torch.float8_e5m2),
}

config = {
    "M": 16,
    "N": 4096,
    "K": 4096,
}


def matmul_fp8(M, N, K):
    kn_graph = yr.new_kernel_graph()
    X = kn_graph.new_input(dims=(M, K), dtype=yr.float8_e4m3)
    W = kn_graph.new_input(dims=(K, N), dtype=yr.float8_e4m3)

    # launch 64x1x1 blocks, each running a warp group (128 threads)
    tb_graph = yr.new_threadblock_graph(
        grid_dim=(64, 1, 1), block_dim=(128, 1, 1), forloop_range=64, reduction_dimx=64
    )
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tO = tb_graph.forloop_accum(tM)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))

    O = kn_graph.customized([X, W], tb_graph)

    kn_graph.mark_output(O)
    return kn_graph


if __name__ == "__main__":
    M, N, K = config["M"], config["N"], config["K"]

    mm = matmul_fp8(M, N, K)

    # real inputs
    input_tensors = [
        torch.randn(M, K, dtype=torch.float8_e4m3fn, device=_DEVICE),
        torch.randn(K, N, dtype=torch.float8_e4m3fn, device=_DEVICE),
    ]

    # debug: view transpiled CUDA code
    # input_strides = [tensor.stride() for tensor in input_tensors]
    # p = yr.generate_cuda_program(mm.cygraph, target_cc=86, input_strides=input_strides)

    # run kernel graph
    output = mm(inputs=input_tensors)[0]

    # print(output.shape)
    # print(output.stride(0), output.stride(1))
