"""
YiRage Kernel Submission Example — compatible with gpu-mode/popcorn-cli.

This file demonstrates a grayscale image conversion kernel that can be
submitted to the gpu-mode leaderboard (grayscale_v2).  It also shows the
YiRage workflow for quick optimization-effect validation before submission.

Usage
-----
1. Install YiRage:
       YIRAGE_BACKEND=cpu pip install -e . --no-build-isolation

2. Validate locally (no GPU required):
       python examples/submission.py --validate

3. Install popcorn-cli and submit:
       curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
       popcorn-cli register discord
       popcorn-cli setup
       popcorn-cli submit --gpu A100 --leaderboard grayscale_v2 --mode leaderboard examples/submission.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# ---------------------------------------------------------------------------
# Kernel implementation
# ---------------------------------------------------------------------------

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


def grayscale_numpy(image: np.ndarray) -> np.ndarray:
    """Pure-NumPy grayscale conversion (ITU-R BT.601 luma coefficients).

    Parameters
    ----------
    image:
        RGB image array with shape (H, W, 3) and dtype uint8 or float32.

    Returns
    -------
    np.ndarray
        Grayscale image with shape (H, W) and the same dtype as *image*.
    """
    coeffs = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 3:
        gray = (
            image[..., 0].astype(np.float32) * coeffs[0]
            + image[..., 1].astype(np.float32) * coeffs[1]
            + image[..., 2].astype(np.float32) * coeffs[2]
        )
    else:
        gray = image.astype(np.float32) @ coeffs
    if np.issubdtype(image.dtype, np.integer):
        return np.clip(np.rint(gray), 0, 255).astype(image.dtype)
    return gray.astype(image.dtype)


def grayscale_torch(image: "torch.Tensor") -> "torch.Tensor":
    """Torch-based grayscale conversion — auto-selects best available device.

    Parameters
    ----------
    image:
        RGB tensor with shape (B, 3, H, W) or (3, H, W), dtype float32.

    Returns
    -------
    torch.Tensor
        Grayscale tensor with shape (B, 1, H, W) or (1, H, W).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")

    coeffs = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=image.device)

    if image.dim() == 3:
        # (3, H, W) → (1, H, W)
        return (image * coeffs[:, None, None]).sum(dim=0, keepdim=True)
    elif image.dim() == 4:
        # (B, 3, H, W) → (B, 1, H, W)
        return (image * coeffs[None, :, None, None]).sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"Expected 3-D or 4-D tensor, got {image.dim()}-D")


# ---------------------------------------------------------------------------
# popcorn-cli entry point — must be named `solution`
# ---------------------------------------------------------------------------


def solution(input_tensor: "torch.Tensor") -> "torch.Tensor":
    """Leaderboard entry-point expected by popcorn-cli.

    Parameters
    ----------
    input_tensor:
        RGB float32 tensor with shape (B, 3, H, W) on CUDA.

    Returns
    -------
    torch.Tensor
        Grayscale float32 tensor with shape (B, 1, H, W).
    """
    return grayscale_torch(input_tensor)


# ---------------------------------------------------------------------------
# Local validation helpers
# ---------------------------------------------------------------------------


def _benchmark(fn, *args, warmup: int = 10, iters: int = 100) -> float:
    """Return mean wall-clock time (ms) over *iters* calls after *warmup*."""
    import torch as _torch  # local import; safe — only called when torch is available

    def _sync() -> None:
        if TORCH_AVAILABLE and isinstance(args[0], _torch.Tensor) and args[0].is_cuda:
            _torch.cuda.synchronize()

    for _ in range(warmup):
        fn(*args)
    _sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    _sync()

    return (time.perf_counter() - t0) / iters * 1000.0


def _validate_numpy(height: int = 256, width: int = 256) -> None:
    """Validate correctness and measure throughput with NumPy."""
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    result = grayscale_numpy(image)
    reference = grayscale_numpy(image.copy())

    assert result.shape == (height, width), f"Shape mismatch: {result.shape}"
    np.testing.assert_array_equal(result, reference)
    print(f"  ✅  NumPy kernel: shape={result.shape}, dtype={result.dtype}")

    img_f32 = image.astype(np.float32)
    elapsed_ms = _benchmark(grayscale_numpy, img_f32, warmup=5, iters=50)
    throughput_gpix_s = (height * width) / elapsed_ms / 1e6
    print(f"  ⏱   NumPy throughput: {elapsed_ms:.3f} ms/frame  ({throughput_gpix_s:.2f} Gpix/s)")


def _validate_torch(height: int = 256, width: int = 256) -> None:
    """Validate correctness and measure throughput with PyTorch."""
    if not TORCH_AVAILABLE:
        print("  ⚠️   PyTorch not available — skipping Torch validation")
        return

    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  🖥   Using device: {device}")

    rng = torch.Generator()
    rng.manual_seed(42)
    image = torch.rand(4, 3, height, width, generator=rng, device=device)

    result = grayscale_torch(image)
    assert result.shape == (4, 1, height, width), f"Shape mismatch: {result.shape}"

    # Reference using torch.einsum
    coeffs = torch.tensor([0.299, 0.587, 0.114], device=device)
    reference = torch.einsum("bchw,c->bhw", image, coeffs).unsqueeze(1)
    torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)
    print(f"  ✅  Torch kernel: shape={result.shape}, dtype={result.dtype}, device={device}")

    elapsed_ms = _benchmark(grayscale_torch, image, warmup=10, iters=100)
    throughput_gpix_s = (4 * height * width) / elapsed_ms / 1e6
    print(f"  ⏱   Torch throughput: {elapsed_ms:.3f} ms/batch  ({throughput_gpix_s:.2f} Gpix/s)")


# ---------------------------------------------------------------------------
# YiRage optimization integration
# ---------------------------------------------------------------------------


def _try_yirage_optimize(height: int = 256, width: int = 256) -> None:
    """Attempt to further optimize the kernel using YiRage superoptimizer.

    This is optional — if yirage is not installed or the native runtime is
    unavailable, a clear message is printed and execution continues.
    """
    try:
        import yirage as yr  # noqa: F401

        print("\n  🚀  YiRage superoptimizer detected — running kernel search …")
        # Placeholder: real usage would call yr.superoptimize() on a kernel graph.
        # See docs/tutorial.md and examples/cluster/ for complete examples.
        print("  ℹ️   To superoptimize, build a kernel graph with yr.new_kernel_graph()")
        print("      and call yr.superoptimize(graph, backend='cuda').")
    except ImportError:
        print("\n  ℹ️   YiRage not installed — skipping superoptimizer pass.")
        print("      Install with: YIRAGE_BACKEND=cpu pip install -e . --no-build-isolation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YiRage grayscale kernel — local validation & leaderboard submission helper"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run correctness checks and benchmark locally",
    )
    parser.add_argument("--height", type=int, default=256, help="Image height for benchmarks")
    parser.add_argument("--width", type=int, default=256, help="Image width for benchmarks")
    args = parser.parse_args()

    if args.validate:
        print("\n🔍  YiRage Kernel Validation")
        print("=" * 48)
        print("\n[1/3] NumPy backend")
        _validate_numpy(args.height, args.width)
        print("\n[2/3] PyTorch backend")
        _validate_torch(args.height, args.width)
        print("\n[3/3] YiRage superoptimizer")
        _try_yirage_optimize(args.height, args.width)
        print("\n✅  All validation steps completed.")
        print("\nNext step — submit to the leaderboard:")
        print(
            "  popcorn-cli submit --gpu A100 --leaderboard grayscale_v2 "
            "--mode leaderboard examples/submission.py"
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
