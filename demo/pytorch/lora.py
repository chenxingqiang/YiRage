import yirage as yr
import argparse
import os
import torch
import flashinfer
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

n_local_heads = 32
n_local_kv_heads = 8
head_dim = 128
num_tokens = 1
num_kv_tokens = 4096
batch_size = 8

rms_norm4k = torch.nn.RMSNorm(4096, device=_DEVICE, dtype=torch.float16)
rms_norm128 = torch.nn.RMSNorm(128, device=_DEVICE, dtype=torch.float16)
silu = torch.nn.SiLU()


def torch_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A3, B3, A2, B2):
    X = rms_norm4k(X)
    Xqkv = torch.matmul(X, Wqkv)
    Xq = Xqkv[:, : (n_local_heads * head_dim)]
    output_shape = Xq.shape
    Xkv = Xqkv[:, (n_local_heads * head_dim) :]
    Xk, Xv = Xkv.chunk(2, 1)
    Xq = Xq.view(Xq.shape[0], n_local_heads, head_dim)
    Xk = Xk.view(Xk.shape[0], n_local_kv_heads, head_dim)
    Xv = Xv.view(Xv.shape[0], n_local_kv_heads, head_dim)
    output = flashinfer.single_prefill_with_kv_cache(Xq, Kcache, Vcache, causal=True)
    output = torch.matmul(output.reshape(output_shape), Wo)
    # RMSNorm
    X = output
    X = rms_norm4k(X)
    W1, W3 = W13.chunk(2, 1)
    X1 = torch.add(torch.matmul(X, W1), torch.matmul(torch.matmul(X, A1), B1))
    X3 = torch.add(torch.matmul(X, W3), torch.matmul(torch.matmul(X, A3), B3))
    output = silu(X1) * X3
    output = torch.add(torch.matmul(output, W2), torch.matmul(torch.matmul(output, A2), B2))
    return output


if __name__ == "__main__":
    X = torch.randn(batch_size * num_tokens, 4096, dtype=torch.float16, device=_DEVICE)
    Wqkv = torch.randn(
        4096,
        n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim,
        dtype=torch.float16,
        device=_DEVICE,
    )
    Wo = torch.randn(n_local_heads * head_dim, 4096, dtype=torch.float16, device=_DEVICE)
    W13 = torch.randn(4096, 14336 * 2, dtype=torch.float16, device=_DEVICE)
    W2 = torch.rand(14336, 4096, dtype=torch.float16, device=_DEVICE)
    A1 = torch.rand(4096, 16, dtype=torch.float16, device=_DEVICE)
    B1 = torch.rand(16, 14336, dtype=torch.float16, device=_DEVICE)
    A3 = torch.rand(4096, 16, dtype=torch.float16, device=_DEVICE)
    B3 = torch.rand(16, 14336, dtype=torch.float16, device=_DEVICE)
    A2 = torch.rand(14336, 16, dtype=torch.float16, device=_DEVICE)
    B2 = torch.rand(16, 4096, dtype=torch.float16, device=_DEVICE)

    Kcache = torch.rand(
        num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device=_DEVICE
    )
    Vcache = torch.rand(
        num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device=_DEVICE
    )
    for _ in range(16):
        torch_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A3, B3, A2, B2)
    _sync()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(1000):
        torch_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A3, B3, A2, B2)
    ender.record()
    _sync()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / 1000

    print("Torch LLAMA-3 run time (ms): ", mean_syn)
