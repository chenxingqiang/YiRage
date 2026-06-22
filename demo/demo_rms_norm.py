#!/usr/bin/env python3
"""RMSNorm + MatMul demo using YiRage (CPU / CUDA / MPS)."""

import argparse
import sys

from demo._device_utils import (
    DEVICE,
    backend_for_device,
    bench_ms,
    configure_device,
    ensure_native_ld_library_path,
    ensure_repo_on_path,
    get_yirage_dtype,
    print_device_info,
)

ensure_repo_on_path()
ensure_native_ld_library_path()

import torch
import yirage as yr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use large LLM-shaped dims (slow on CPU)",
    )
    args = parser.parse_args()

    configure_device("auto")
    print_device_info()

    hidden = 7168 if args.full else 256
    out_dim = 16384 if args.full else 512

    yirage_dtype = get_yirage_dtype()
    torch_dtype = yr.convert_dtype_to_torch_type(yirage_dtype)

    graph = yr.new_kernel_graph()
    graph.backend = backend_for_device(DEVICE)
    X = graph.new_input(dims=(1, hidden), dtype=yirage_dtype)
    W = graph.new_input(dims=(hidden, out_dim), dtype=yirage_dtype)
    D = graph.rms_norm(X, normalized_shape=(hidden,))
    O = graph.matmul(D, W)
    graph.mark_output(O)

    if DEVICE.startswith("cuda"):
        optimized_graph = graph.superoptimize(config="mlp")
    else:
        optimized_graph = graph

    input_tensors = [
        torch.randn(1, hidden, dtype=torch_dtype, device=DEVICE),
        torch.randn(hidden, out_dim, dtype=torch_dtype, device=DEVICE),
    ]

    outputs = optimized_graph(inputs=input_tensors)
    _ = outputs[0]

    mean_syn = bench_ms(lambda: optimized_graph(inputs=input_tensors))
    print("Best muGraph run time (ms): ", mean_syn)
    sys.exit(0)
