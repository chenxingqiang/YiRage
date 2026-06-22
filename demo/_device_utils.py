"""Device-agnostic utilities for demos and benchmarks.

Import this module to write demos that run on CUDA, MPS (Apple Silicon),
and CPU without platform-specific branches.

Usage::

    from demo._device_utils import configure_device, DEVICE, sync, bench_ms, get_dtype

    configure_device("auto")  # or pass via --device in argparse
    dtype = get_dtype()
    t = torch.randn(..., device=DEVICE, dtype=dtype)
    elapsed = bench_ms(lambda: model(t))
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Updated by configure_device(); default is auto-detected once at import.
DEVICE: str = "cpu"
TORCH_DEVICE: torch.device = torch.device("cpu")


def ensure_repo_on_path() -> None:
    """Allow ``import yirage`` when running ``python demo/...`` from any cwd."""
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    python_pkg = str(_REPO_ROOT / "python")
    if python_pkg not in sys.path:
        sys.path.insert(0, python_pkg)


def ensure_native_ld_library_path() -> None:
    """Prepend YiRage Rust helper libraries when running demos as scripts."""
    build = _REPO_ROOT / "build"
    parts = [
        build / "abstract_subexpr" / "release",
        build / "formal_verifier" / "release",
    ]
    extra = os.pathsep.join(str(p) for p in parts if p.exists())
    if not extra:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if extra not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = f"{extra}{os.pathsep}{current}" if current else extra


def mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def cuda_available() -> bool:
    return torch.cuda.is_available()


def resolve_device(requested: str) -> str:
    """Resolve execution device string: auto | cpu | mps | cuda | cuda:0."""
    req = (requested or "auto").lower()
    if req == "auto":
        if cuda_available():
            return "cuda:0"
        if mps_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda"):
        if not cuda_available():
            raise RuntimeError("CUDA requested but not available.")
        return req if ":" in req else "cuda:0"
    if req == "mps":
        if not mps_available():
            raise RuntimeError("MPS requested but not available (Apple Silicon required).")
        return "mps"
    if req == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device {requested!r}; use auto, cpu, mps, or cuda")


def backend_for_device(device: str) -> str:
    """Map torch device string to YiRage backend name."""
    if device.startswith("cuda"):
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


def configure_device(requested: str = "auto") -> str:
    """Set module-level DEVICE / TORCH_DEVICE from *requested*."""
    global DEVICE, TORCH_DEVICE
    DEVICE = resolve_device(requested)
    TORCH_DEVICE = torch.device(DEVICE)
    return DEVICE


def require_mps(message: str | None = None) -> None:
    """Exit with code 1 if MPS is not available."""
    if mps_available():
        return
    msg = message or "This demo requires Apple Silicon MPS."
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def get_device() -> str:
    """Legacy helper: auto-detect without explicit configure."""
    return resolve_device("auto")


def get_dtype(use_bfloat16: bool = True):
    """Return a dtype suitable for the current device."""
    if use_bfloat16 and DEVICE == "mps":
        return torch.float16
    if use_bfloat16 and DEVICE.startswith("cuda"):
        return torch.bfloat16
    if use_bfloat16:
        return torch.bfloat16
    return torch.float16


def get_yirage_backend() -> str:
    return backend_for_device(DEVICE)


def get_yirage_dtype():
    """Return the YiRage dtype equivalent for the current device."""
    import yirage as yr

    if DEVICE == "mps" or DEVICE.startswith("cuda"):
        return yr.float16
    return yr.bfloat16


def sync():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def bench_ms(fn, warmup=16, reps=1000):
    """Run *fn* warmup+reps times and return mean elapsed milliseconds."""
    for _ in range(warmup):
        fn()
    sync()
    if DEVICE.startswith("cuda"):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(reps):
            fn()
        ender.record()
        sync()
        return starter.elapsed_time(ender) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t0) / reps * 1000


def print_device_info():
    """Print detected device information for the demo header."""
    print(f"Device: {DEVICE}")
    if DEVICE.startswith("cuda"):
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    elif DEVICE == "mps":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            print(f"  Chip: {brand}")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    print(f"  Dtype: {get_dtype()}")
    print(f"  YiRage backend: {get_yirage_backend()}")
    print()


# Auto-detect once on import for backwards compatibility with older demos.
configure_device("auto")
