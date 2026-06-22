#!/usr/bin/env python3
"""Minimal YiRage JIT demo: build a small graph and verify add/exp on CPU/CUDA/MPS."""

from __future__ import annotations

import argparse
import sys

from demo._device_utils import ensure_native_ld_library_path, ensure_repo_on_path

ensure_repo_on_path()
ensure_native_ld_library_path()

import torch
import yirage as yr


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested, but CUDA is not available.")
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise SystemExit("MPS device requested, but MPS is not available.")
    return device


def _backend_for_device(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


def _dtypes_for_backend(backend: str) -> tuple:
    """Return (yirage_dtype, torch_dtype) for the execution backend."""
    if backend == "cuda":
        return yr.float16, torch.float16
    return yr.float32, torch.float32


def _check_correctness(outputs, input_tensors, atol: float = 1e-3) -> tuple[bool, bool]:
    mid0_correct = input_tensors[0] + input_tensors[1]
    output0_correct = input_tensors[0] + mid0_correct
    output1_correct = torch.exp(mid0_correct)

    def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
        return bool(
            torch.allclose(
                a.detach().float().cpu(),
                b.detach().float().cpu(),
                atol=atol,
                rtol=1e-4,
            )
        )

    return _close(outputs[0], output0_correct), _close(outputs[1], output1_correct)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, mps, or cuda:0",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print correctness lines (for CI subprocess tests)",
    )
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    backend = _backend_for_device(device)
    graph_dtype, torch_dtype = _dtypes_for_backend(backend)

    graph = yr.new_kernel_graph()
    graph.backend = backend
    input0 = graph.new_input(dims=(12, 256), dtype=graph_dtype)
    input1 = graph.new_input(dims=(12, 256), dtype=graph_dtype)
    mid0 = graph.add(input0, input1)
    output0 = graph.add(input0, mid0)
    output1 = graph.exp(mid0)
    graph.mark_output(output0)
    graph.mark_output(output1)

    input_tensors = [
        torch.randn(12, 256, dtype=torch_dtype, device=device),
        torch.randn(12, 256, dtype=torch_dtype, device=device),
    ]

    outputs = graph(inputs=input_tensors)
    ok0, ok1 = _check_correctness(outputs, input_tensors)

    if not args.quiet:
        print("yirage output[0]:", outputs[0], sep="\n")
        print()
        print("correct output[0]:", input_tensors[0] + input_tensors[1] + input_tensors[0], sep="\n")

    print("Correctness of output[0]:", ok0)
    print("Correctness of output[1]:", ok1)

    return 0 if (ok0 and ok1) else 1


if __name__ == "__main__":
    sys.exit(main())
