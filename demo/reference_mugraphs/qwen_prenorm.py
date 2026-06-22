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

torch.set_printoptions(sci_mode=False)


def torch_qwen_prenorm(X, G, W):
    variance = X.pow(2).mean(-1, keepdim=True)
    X = X * torch.rsqrt(variance)
    X = torch.mul(X, G)
    O = torch.matmul(X, W)

    return O


if __name__ == "__main__":

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(1, 2048), dtype=yr.bfloat16)
    G = graph.new_input(dims=(1, 2048), dtype=yr.bfloat16)
    W = graph.new_input(dims=(2048, 2560), strides=(1, 2048), dtype=yr.bfloat16)
    D = graph.rms_norm(X, normalized_shape=(2048,))
    D = graph.mul(D, G)
    O = graph.matmul(D, W)
    graph.mark_output(O)
    if _DEVICE == "cuda":
        opt_kernel = graph.superoptimize(config="mlp")
    else:
        opt_kernel = graph

    input_tensors = [
        torch.randn(1, 1, 2048, dtype=torch.bfloat16, device=_DEVICE),
        torch.randn(2048, dtype=torch.bfloat16, device=_DEVICE),
        torch.randn(2048, 2560, dtype=torch.bfloat16, device=_DEVICE),
    ]

    input_tensors[2] = torch.as_strided(input_tensors[2], (2048, 2560), (1, 2048))
    outputs = opt_kernel(inputs=input_tensors)
    print(outputs[0])
    print(torch_qwen_prenorm(input_tensors[0], input_tensors[1], input_tensors[2]))
