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
    Q = graph.new_input(dims=(2, 256, 64), dtype=yr.float16)
    K = graph.new_input(dims=(2, 64, 4096), dtype=yr.float16)
    V = graph.new_input(dims=(2, 4096, 64), dtype=yr.float16)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.mark_output(O)
    if _DEVICE == "cuda":
        optimized_graph = graph.superoptimize(config="attention")
    else:
        optimized_graph = graph

    input_tensors = [
        torch.randn(2, 256, 64, dtype=torch.float16, device=_DEVICE),
        torch.randn(2, 64, 4096, dtype=torch.float16, device=_DEVICE),
        torch.randn(2, 4096, 64, dtype=torch.float16, device=_DEVICE),
    ]

    mean_syn = __bench_ms(lambda: optimized_graph(inputs=input_tensors))
    print("Best muGraph run time (ms): ", mean_syn)
