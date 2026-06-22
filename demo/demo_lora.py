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
    X = graph.new_input(dims=(1, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
    A = graph.new_input(dims=(4096, 16), dtype=yr.float16)
    B = graph.new_input(dims=(16, 4096), dtype=yr.float16)
    D = graph.matmul(X, A)
    E = graph.matmul(D, B)
    C = graph.matmul(X, W)
    O = graph.add(C, E)
    graph.mark_output(O)
    if _DEVICE == "cuda":
        optimized_graph = graph.superoptimize(config="lora")
    else:
        graph.backend = "cpu"
        optimized_graph = graph

    input_tensors = [
        torch.randn(1, 4096, dtype=torch.float16, device=_DEVICE),
        torch.randn(4096, 4096, dtype=torch.float16, device=_DEVICE),
        torch.randn(4096, 16, dtype=torch.float16, device=_DEVICE),
        torch.randn(16, 4096, dtype=torch.float16, device=_DEVICE),
    ]

    mean_syn = __bench_ms(lambda: optimized_graph(inputs=input_tensors), warmup=4, reps=20)
    print(f"LoRA muGraph run time (ms): {mean_syn:.6g}")
