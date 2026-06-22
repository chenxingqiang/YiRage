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


def torch_qwen_mlp(X, Y, W):
    silu = torch.nn.SiLU()
    O = torch.matmul(silu(X) * Y, W)

    return O


if __name__ == "__main__":
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(1, 18944), dtype=yr.bfloat16)
    Y = graph.new_input(dims=(1, 18944), dtype=yr.bfloat16)
    W = graph.new_input(dims=(18944, 3584), dtype=yr.bfloat16)
    D = graph.mul(graph.silu(X), Y)
    O = graph.matmul(D, W)
    graph.mark_output(O)
    if _DEVICE != "cuda":
        import sys
        sys.stderr.write(f"This demo requires CUDA. Detected {_DEVICE}. Exiting.\n")
        sys.exit(0)
    optimized_graph = graph.superoptimize(config="mlp")

    input_tensors = [
        torch.randn(1, 1, 18944, dtype=torch.bfloat16, device=_DEVICE),
        torch.randn(1, 1, 18944, dtype=torch.bfloat16, device=_DEVICE),
        torch.randn(18944, 3584, dtype=torch.bfloat16, device=_DEVICE),
    ]

    input_strides = []
    dtensors = optimized_graph.cygraph.get_input_dtensors()
    assert len(dtensors) == len(
        input_tensors
    ), "Given number of inputs do not match the uGraph's inputs"
    for i in range(len(dtensors)):
        dims, strides = optimized_graph.cygraph.get_input_dtensor_shape_and_stride(dtensors[i])
        input_strides.append(strides)

    # input_strides = [tensor.stride() for tensor in input_tensors]
    p = yr.generate_cuda_program(optimized_graph.cygraph, target_cc=86, input_strides=input_strides)
    print(p["code"])

    outputs = optimized_graph(inputs=input_tensors)
    print(outputs[0])
    print(torch_qwen_mlp(input_tensors[0], input_tensors[1], input_tensors[2]))
