import yirage as yr
import torch
import flashinfer

n_local_heads = 12
n_local_kv_heads = 12
head_dim = 128
intermediate_size = 4096
num_tokens = 4
num_kv_tokens = 4096
batch_size = 8

def get_norm1():
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim), dtype=yr.float16)
    D = graph.matmul(X, W)
    O = graph.rms_norm(D, normalized_shape=(n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim,))
    graph.mark_output(O)
    return graph.superoptimize(previous_checkpoint="ngpt_norm1_bs{batch_size}.json")

def get_norm2():
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, intermediate_size * 2), dtype=yr.float16)
    alpha = graph.new_input(dims=(batch_size * num_tokens, intermediate_size * 2), dtype=yr.float16)
    D = graph.matmul(X, W)
    A = graph.rms_norm(D, normalized_shape=(intermediate_size * 2,)) # TODO: replace with standard L2 norm
    B = graph.mul(A, alpha)
    O = graph.rms_norm(B, normalized_shape=(intermediate_size * 2,)) # TODO: replace with standard L2 norm
    graph.mark_output(O)
    return graph.superoptimize(previous_checkpoint="ngpt_norm2_bs{batch_size}.json")

def yirage_ngpt(X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha, kernels):
    func = kernels[0]
    outputs = func(inputs=[X, Wqkv])
    Xqkv = outputs[0]
    Xq = Xqkv[:, : (n_local_heads * head_dim)]
    output_shape = Xq.shape
    Xkv = Xqkv[:, (n_local_heads * head_dim) :]
    Xk, Xv = Xkv.chunk(2, 1)
    Xq = Xq.view(Xq.shape[0], n_local_kv_heads, head_dim)
    Xk = Xk.view(Xk.shape[0], n_local_kv_heads, head_dim)
    Xv = Xv.view(Xv.shape[0], n_local_kv_heads, head_dim)
    # func = kernels[2]
    # outputs = func(inputs=[Xq, Kcache, Vcache])
    # output = outputs[0]
    #Xq = rms_norm2(Xq)
    #Xk = rms_norm2(Xk)
    output = flashinfer.single_prefill_with_kv_cache(Xq, Kcache, Vcache, causal=True)
    output = torch.matmul(output.reshape(output_shape), Wo)
    # Norm
    X = output
    func = kernels[1]
    outputs = func(inputs=[X, W13, alpha])
    X13 = outputs[0]
    X1, X3 = X13.chunk(2, -1)
    output = torch.matmul(X1, W2)
    return output

if __name__ == "__main__":
    X = torch.randn(batch_size * num_tokens, 4096, dtype=torch.float16, device=device)
    Wqkv = torch.randn(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim, dtype=torch.float16, device=device)
    Wo = torch.randn(n_local_heads * head_dim, 4096, dtype=torch.float16, device=device)
    W13 = torch.randn(4096, intermediate_size * 2, dtype=torch.float16, device=device)
    W2 = torch.rand(intermediate_size, 4096, dtype=torch.float16, device=device)
    Kcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device=device)
    Vcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device=device)
    alpha = torch.rand(batch_size * num_tokens, intermediate_size * 2, dtype=torch.float16, device=device)

    k1 = get_norm1()
    k2 = get_norm2()
    # k3 = get_chameleon_attention()
    # kernels = [k1, k2, k3]
    kernels = [k1, k2]

    for _ in range(16):
        yirage_ngpt(X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha, kernels)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elif device == 'mps' and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()

    if device.startswith('cuda'):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    if device.startswith('cuda'):
        starter.record()
    for rep in range(repetitions):
        yirage_ngpt(X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha, kernels)

    ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
    else:
        start_time = time.perf_counter()
        # Execute operations (copy from above)
        # ...
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elif device == 'mps' and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / 1000
    print(mean_syn)
