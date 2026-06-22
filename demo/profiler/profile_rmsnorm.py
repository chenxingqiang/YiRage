import torch
import yirage as yr
import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiling", action="store_true", help="Enable yirage profiling mode")
    args = parser.parse_args()

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(64, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
    tb_graph = yr.new_threadblock_graph(
        grid_dim=(64, 1, 1), block_dim=(384, 1, 1), forloop_range=64, reduction_dimx=64
    )
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccX = tb_graph.forloop_accum(tX, "rms")
    tAccM = tb_graph.forloop_accum(tM)
    tO = tb_graph.div(tAccM, tAccX)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])

    input_tensors = [
        torch.randn(64, 4096, dtype=torch.float16, device=_DEVICE),
        torch.randn(4096, 4096, dtype=torch.float16, device=_DEVICE),
    ]

    outputs = graph(
        inputs=input_tensors, num_warp_groups=3, pipeline_stages=4, profiling=args.profiling
    )
