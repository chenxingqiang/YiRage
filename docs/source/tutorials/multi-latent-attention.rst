:tocdepth: 1
**************************************
Superoptimizing Multi-Latent Attention
**************************************

Introduction
============

.. code-block:: Python
    import torch
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
    graph.mark_output(O)
    optimized_graph = yr.superoptimize(graph)


