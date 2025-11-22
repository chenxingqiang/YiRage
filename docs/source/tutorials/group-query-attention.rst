:tocdepth: 1
*************************************
Superoptimizing Group-Query Attention
*************************************

Introduction
============

The follow code snippet shows how to use YiRage to automatically generate highly-optimized CUDA programs for group-query attention (GQA) in LLAMA-3-70B. We assume the model is served in half precision and is tensor model parallelized across 4 GPUs to fit in GPU device memory. Therefore, the GQA operator computes attention across 8 query heads and 2 key-value heads.

First, we define the computation graph for GQA, which takes three input tensors `Q`, `K`, and `V`, and produces a single output tensor `O` that contains the attention result:

.. code-block:: Python

    import yirage as yr
    graph = yr.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=yr.float16)
    K = graph.new_input(dims=(2, 64, 4096), dtype=yr.float16)
    V = graph.new_input(dims=(2, 4096, 64), dtype=yr.float16)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    optimized_graph = graph.superoptimize(config="attention")

Second, we will use `yr.superoptimize` to superoptimize GQA. YiRage will automatically search the space of potential mugraphs that are functionally equivalent to the input graph to discover highly-optimized CUDA programs. MuGraphs are a new multi-level graph representation in YiRage that specifies computation at the kernel, thread block, and thread levels of the GPU compute hierarchy. An introduction to uGraph is available [here](https://yirage-project.readthedocs.io/en/latest/mugraph.html). YiRage can automatically find uGraphs that represent today's expert-designed GPU optimizations such as FlashAttention, FlashDecoding, and FlashInfer. In addition, YiRage also discovers other uGraphs that outperform these expert-designed implementations for certain cases.

The `superoptimize` function returns the best uGraph discovered by YiRage. The object `optimized_graph` can directly run as a function, and doing so will let YiRage transpile the uGraph into CUDA code, compile the code for execution, and launch the compiled kernel. This allows users to directly run YiRage-generated kernels in their Python programs.

.. code-block:: Python

    import torch
    input_tensors = [
        torch.randn(64, 1, 128, dtype=torch.float16, device='cuda:0'),
        torch.randn(64, 128, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(64, 4096, 128, dtype=torch.float16, device='cuda:0')
    ]
    # Launch the YiRage-generated kernel to perform attention
    output = optimized_graph(input_tensors)
