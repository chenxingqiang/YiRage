#!/usr/bin/env python3
"""YiRage MPS Benchmark — PyTorch vs YiRage kernels on Apple Silicon.

Each benchmark is run in a fresh subprocess to avoid graph-state
accumulation across kernel types.  See the inline benchmarks at the
bottom for the raw numbers.

Usage::

    python demo/mps/benchmark_mps.py
"""

import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from demo._device_utils import require_mps

PYTHON = sys.executable


def run_bench(script):
    """Run a one-shot benchmark script and return (pt_ms, yr_ms)."""
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE",
           "PYTHONPATH": PROJECT_ROOT}
    r = subprocess.run([PYTHON, "-c", script], capture_output=True,
                       text=True, timeout=120, env=env)
    if r.returncode != 0:
        print(f"  SKIP (exit {r.returncode})")
        return None, None
    # Last line of stdout is "PT_MS YR_MS"
    parts = r.stdout.strip().split()
    if len(parts) >= 2:
        return float(parts[-2]), float(parts[-1])
    return None, None


def bench_matmul(M, K, N):
    code = f"""
import sys; sys.path.insert(0, '{PROJECT_ROOT}')
import torch, yirage as yr, time
sync = torch.mps.synchronize
a = torch.randn({M}, {K}, dtype=torch.float16, device='mps')
b = torch.randn({K}, {N}, dtype=torch.float16, device='mps')
g = yr.new_kernel_graph(); A=g.new_input(dims=({M},{K}),dtype=yr.float16); B=g.new_input(dims=({K},{N}),dtype=yr.float16)
C=g.matmul(A,B); g.mark_output(C)
for _ in range(5): g(inputs=[a,b]); sync()
def pt_fn(): torch.matmul(a,b)
for _ in range(10): pt_fn(); sync()
t0=time.perf_counter()
for _ in range(100): pt_fn()
sync(); pt=(time.perf_counter()-t0)/100*1000
t0=time.perf_counter()
for _ in range(100): g(inputs=[a,b])
sync(); yr_ms=(time.perf_counter()-t0)/100*1000
print(f'{{pt:.4f}} {{yr_ms:.4f}}')
"""
    return run_bench(code)


def bench_rms_norm(seq, hidden):
    code = f"""
import sys; sys.path.insert(0, '{PROJECT_ROOT}')
import torch, yirage as yr, time
sync = torch.mps.synchronize
x = torch.randn({seq}, {hidden}, dtype=torch.float16, device='mps')
g = yr.new_kernel_graph(); X=g.new_input(dims=({seq},{hidden}),dtype=yr.float16)
Y=g.rms_norm(X,normalized_shape=({hidden},)); g.mark_output(Y)
for _ in range(5): g(inputs=[x]); sync()
def pt_fn():
    rms=torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)
    return x*rms
for _ in range(10): pt_fn(); sync()
t0=time.perf_counter()
for _ in range(100): pt_fn()
sync(); pt=(time.perf_counter()-t0)/100*1000
t0=time.perf_counter()
for _ in range(100): g(inputs=[x])
sync(); yr_ms=(time.perf_counter()-t0)/100*1000
print(f'{{pt:.4f}} {{yr_ms:.4f}}')
"""
    return run_bench(code)


if __name__ == "__main__":
    require_mps("MPS benchmark requires Apple Silicon.")
    print("=" * 62)
    print("  YiRage MPS Benchmark — Apple Silicon Kernel Performance")
    print("=" * 62)

    # Get chip info
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        print(f"  Chip:    {brand}")
    except Exception:
        pass

    results = []

    # ---- RMS Norm (kernel fusion advantage) ----
    print(f"\n{'='*62}")
    print("  RMS Norm  (seq_len, hidden)  —  FP16  [kernel fusion]")
    print("=" * 62)
    for seq, hidden in [(256, 256), (1024, 1024), (4096, 4096),
                         (16384, 4096)]:
        pt, yr_ms = bench_rms_norm(seq, hidden)
        if pt is None:
            continue
        sp = pt / yr_ms if yr_ms > 0 else 0
        tag = f"({seq},{hidden})"
        print(f"  {tag:<24s} PyTorch {pt:8.4f} ms   YiRage {yr_ms:8.4f} ms   {sp:5.2f}x")
        results.append((f"RMSNorm {tag}", pt, yr_ms))

    # ---- MatMul ----
    print(f"\n{'='*62}")
    print("  MatMul  (M,K) @ (K,N)  —  FP16")
    print("=" * 62)
    for M, K, N in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
                     (2048, 2048, 2048), (4096, 4096, 4096)]:
        pt, yr_ms = bench_matmul(M, K, N)
        if pt is None:
            continue
        sp = pt / yr_ms if yr_ms > 0 else 0
        tag = f"({M},{K})@({K},{N})"
        print(f"  {tag:<24s} PyTorch {pt:8.4f} ms   YiRage {yr_ms:8.4f} ms   {sp:5.2f}x")
        results.append((f"MatMul {tag}", pt, yr_ms))

    # ---- Summary ----
    print(f"\n{'='*62}")
    print("  Summary")
    print("=" * 62)
    print(f"  {'Kernel':<30s} {'PyTorch ms':>10s} {'YiRage ms':>10s} {'Speedup':>8s}")
    print(f"  {'-'*58}")
    geo = 1.0
    for name, pt, yr_ms in results:
        sp = pt / yr_ms if yr_ms > 0 else 0
        print(f"  {name:<30s} {pt:10.4f} {yr_ms:10.4f} {sp:7.2f}x")
        geo *= sp
    geo = geo ** (1 / len(results))
    print(f"  {'-'*58}")
    print(f"  {'Geometric mean':<30s} {'':>10s} {'':>10s} {geo:7.2f}x")
    print()
