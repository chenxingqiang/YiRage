"""Qwen MLP reference muGraph — device-agnostic (CUDA / MPS / CPU).

Uses ``superoptimize`` on CUDA for full auto-tuning; on MPS uses the
non-optimized graph (superoptimize Ray workers hit a from_json
deserialization assert on Apple Silicon).  For MPS-optimized kernels
see ``demo/mps/`` which uses ``yr.search()`` with explicit block/grid dims.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import yirage as yr
from demo._device_utils import DEVICE, sync, bench_ms, get_dtype, get_yirage_dtype, print_device_info

torch.set_printoptions(sci_mode=False)


def torch_qwen_mlp(X, G, W):
    variance = X.pow(2).mean(-1, keepdim=True)
    X = X * torch.rsqrt(variance)
    X = torch.mul(X, G)
    O = torch.matmul(X, W)
    return O


if __name__ == "__main__":
    print_device_info()

    yr_dtype = get_yirage_dtype()
    torch_dtype = get_dtype()
    backend = DEVICE

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(1, 3584), dtype=yr_dtype)
    G = graph.new_input(dims=(1, 3584), dtype=yr_dtype)
    W = graph.new_input(dims=(3584, 2 * 18944), strides=(1, 3584), dtype=yr_dtype)
    D = graph.rms_norm(X, normalized_shape=(3584,))
    D = graph.mul(D, G)
    O = graph.matmul(D, W)
    graph.mark_output(O)

    if backend == "cuda":
        optimized_graph = graph.superoptimize(config="mlp")
    else:
        # superoptimize Ray workers abort on MPS (from_json deserialization).
        # Use the non-optimized graph directly; for MPS-optimized kernels
        # see demo/mps/ which uses yr.search() with explicit search dims.
        optimized_graph = graph

    input_tensors = [
        torch.randn(1, 1, 3584, dtype=torch_dtype, device=DEVICE),
        torch.randn(3584, dtype=torch_dtype, device=DEVICE),
        torch.randn(3584, 2 * 18944, dtype=torch_dtype, device=DEVICE),
    ]
    input_tensors[2] = torch.as_strided(input_tensors[2], (3584, 37888), (1, 3584))

    if backend == "cuda":
        input_strides = []
        dtensors = optimized_graph.cygraph.get_input_dtensors()
        assert len(dtensors) == len(input_tensors), (
            "Given number of inputs do not match the uGraph's inputs"
        )
        for i in range(len(dtensors)):
            input_strides.append(optimized_graph.cygraph.get_input_dtensor_layout(dtensors[i]))
        p = yr.generate_cuda_program(optimized_graph.cygraph, target_cc=86, input_strides=input_strides)
        print(p["code"])

    outputs = optimized_graph(inputs=input_tensors)

    print(f"Shape1: {input_tensors[0].shape}  Stride1: {input_tensors[0].stride()}  Dtype1: {input_tensors[0].dtype}")
    print(f"Shape2: {input_tensors[1].shape}  Stride2: {input_tensors[1].stride()}  Dtype2: {input_tensors[1].dtype}")
    print(f"Shape3: {input_tensors[2].shape}  Stride3: {input_tensors[2].stride()}  Dtype3: {input_tensors[2].dtype}")
    print("input", input_tensors[0])
    print("YiRage output:", outputs[0])
    print("PyTorch ref:  ", torch_qwen_mlp(input_tensors[0], input_tensors[1], input_tensors[2]))
