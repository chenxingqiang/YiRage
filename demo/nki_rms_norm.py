import yirage as yr
import numpy as np
import torch

if __name__ == "__main__":
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, 6144), dtype=yr.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="mlp", backend="nki")
    for g in optimized_graph:
        print("Next MuGraph")
        print(yr.generate_nki_program(g.cygraph, target_cc=10)["code"])
