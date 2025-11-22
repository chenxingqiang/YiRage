import torch
import yirage as yr
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiling", action="store_true", help="Enable yirage profiling mode"
    )
    args = parser.parse_args()

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(256, 4096), dtype=yr.float16)
    W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
    tb_graph = yr.new_threadblock_graph(
        grid_dim=(4, 16, 1), block_dim=(256, 1, 1), forloop_range=64, reduction_dimx=64
    )
    tX = tb_graph.new_input(dtensor=X, input_map=(0, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(-1, 1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tO = tb_graph.forloop_accum(tM)
    tb_graph.new_output(stensor=tO, output_map=(0, 1, -1))
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])

    input_tensors = [
        torch.ones(256, 4096, dtype=torch.float16, device="cuda:0"),
        torch.ones(4096, 4096, dtype=torch.float16, device="cuda:0"),
    ]

    outputs = graph(
        inputs=input_tensors,
        num_warp_groups=2,
        pipeline_stages=4,
        profiling=args.profiling,
        file_id=3,
        target_cc=100,
    )
