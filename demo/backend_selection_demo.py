#!/usr/bin/env python3
"""
YiRage Multi-Backend Selection Demo

This demo shows how to:
1. Query available backends
2. Get backend information
3. Select a specific backend for computation
4. Use fallback backends if primary is unavailable
"""

from __future__ import annotations

import sys

from demo._device_utils import ensure_native_ld_library_path, ensure_repo_on_path

ensure_repo_on_path()
ensure_native_ld_library_path()

try:
    import torch
    import yirage as yr
except ImportError as exc:
    print(f"Error: YiRage not available: {exc}", file=sys.stderr)
    print("Install with: YIRAGE_BACKEND=cpu pip install -e . --no-build-isolation", file=sys.stderr)
    sys.exit(1)


def _run_sample_graph(backend: str) -> None:
    """Build and execute a tiny add graph on *backend*."""
    graph = yr.new_kernel_graph()
    graph.backend = backend
    ydtype = yr.float16 if backend in ("cuda", "mps") else yr.float32
    tdtype = torch.float16 if backend in ("cuda", "mps") else torch.float32
    device = "cpu"
    if backend == "cuda" and torch.cuda.is_available():
        device = "cuda:0"
    elif backend == "mps" and torch.backends.mps.is_available():
        device = "mps"

    a = graph.new_input(dims=(4, 8), dtype=ydtype)
    b = graph.new_input(dims=(4, 8), dtype=ydtype)
    out = graph.add(a, b)
    graph.mark_output(out)

    t_a = torch.ones(4, 8, dtype=tdtype, device=device)
    t_b = torch.ones(4, 8, dtype=tdtype, device=device) * 2
    results = graph(inputs=[t_a, t_b])
    expected = t_a + t_b
    ok = torch.allclose(
        results[0].detach().float().cpu(),
        expected.detach().float().cpu(),
        atol=1e-2,
    )
    print(f"    ✓ Graph executed on {backend} (correct={ok})")


def main() -> int:
    print("=" * 60)
    print("YiRage Multi-Backend Selection Demo")
    print("=" * 60)

    print("\n[1] Available Backends:")
    print("-" * 40)
    backends = yr.get_available_backends()
    if not backends:
        print("  No backends available!")
        return 1

    for backend in backends:
        print(f"  - {backend}")

    print("\n[2] Backend Details:")
    print("-" * 40)
    for backend in backends:
        print(f"\n  {backend.upper()}:")
        info = yr.get_backend_info(backend)
        for key, value in info.items():
            print(f"    {key}: {value}")

    print("\n[3] Default Backend:")
    print("-" * 40)
    default = yr.get_default_backend()
    print(f"  {default}")

    print("\n[4] Backend Availability Check:")
    print("-" * 40)
    test_backends = ["cuda", "cpu", "mps", "nki", "triton"]
    for backend in test_backends:
        available = yr.is_backend_available(backend)
        status = "✓" if available else "✗"
        print(f"  {status} {backend}: {'Available' if available else 'Not Available'}")

    print("\n[5] Setting Default Backend:")
    print("-" * 40)
    if "cuda" in backends:
        success = yr.set_default_backend("cuda")
        print(f"  Set CUDA as default: {success}")
    elif "cpu" in backends:
        success = yr.set_default_backend("cpu")
        print(f"  Set CPU as default: {success}")

    print("\n[6] Creating Graphs with Different Backends:")
    print("-" * 40)
    for backend in backends[:3]:
        print(f"\n  Testing with {backend} backend:")
        try:
            _run_sample_graph(backend)
        except Exception as e:
            print(f"    ✗ Error with {backend}: {e}")

    print("\n[7] Detailed Backend Listing:")
    print("-" * 40)
    yr.list_backends(verbose=True)

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    return 0


def example_with_fallback() -> None:
    print("\n" + "=" * 60)
    print("Fallback Backend Example")
    print("=" * 60)

    preferred_backends = ["cuda", "mps", "cpu"]
    selected_backend = None

    print("\nTrying backends in order:")
    for backend in preferred_backends:
        if yr.is_backend_available(backend):
            selected_backend = backend
            print(f"  ✓ Selected: {backend}")
            break
        print(f"  ✗ Not available: {backend}")

    if not selected_backend:
        print("\n  Error: No suitable backend found!")
        return

    print(f"\nUsing backend: {selected_backend}")


if __name__ == "__main__":
    code = main()
    example_with_fallback()
    sys.exit(code)
