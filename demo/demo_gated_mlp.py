import yirage as yr
import numpy as np
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

if __name__ == "__main__":
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
    W1 = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
    W2 = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
    O1 = graph.matmul(X, W1)
    O2 = graph.matmul(X, W2)
    O1 = graph.silu(O1)
    O = graph.mul(O1, O2)
    graph.mark_output(O)
    if _DEVICE == "cuda":
        optimized_graph = graph.superoptimize(config="mlp")
    else:
        optimized_graph = graph

    input_tensors = [
        torch.randn(8, 4096, dtype=torch.float16, device=_DEVICE),
        torch.randn(4096, 4096, dtype=torch.float16, device=_DEVICE),
        torch.randn(4096, 4096, dtype=torch.float16, device=_DEVICE),
    ]

    mean_syn = __bench_ms(lambda: optimized_graph(inputs=input_tensors))
    print("Best muGraph run time (ms): ", mean_syn)
