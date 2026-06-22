from ..core import *


def _validate_threadblock_matmul(A: STensor, B: STensor) -> None:
    """Match ``yirage::threadblock::Graph::create_matmul_op`` rules."""
    if A.num_dims is None or B.num_dims is None:
        raise ValueError("matmul: invalid STensor (missing dimensions)")
    if A.num_dims != B.num_dims:
        raise ValueError(
            f"matmul: rank mismatch (A.num_dims={A.num_dims}, B.num_dims={B.num_dims})"
        )
    nd = A.num_dims
    if nd < 2:
        raise ValueError("matmul: tensors need at least 2 dimensions")
    if A.dim(nd - 1) != B.dim(nd - 2):
        raise ValueError(
            "matmul: inner dimensions do not match: "
            f"A[..., -1]={A.dim(nd - 1)} vs B[..., -2]={B.dim(nd - 2)}"
        )
    for i in range(nd - 2):
        if A.dim(i) != 1 or B.dim(i) != 1:
            raise ValueError(
                "threadblock matmul requires leading dimensions to be 1; "
                f"got A.dim({i})={A.dim(i)}, B.dim({i})={B.dim(i)}"
            )


class TBGraph:
    def __init__(self, graph):
        self.cygraph = graph

    def new_input(
        self,
        dtensor: DTensor,
        input_map: tuple,
        forloop_dim: int,
        store_in_dmem: bool = False,
    ):
        return self.cygraph.new_input(dtensor, input_map, forloop_dim, store_in_dmem)

    def new_output(self, stensor: STensor, output_map: tuple, forloop_dim: int = -1):
        return self.cygraph.new_output(stensor, output_map, forloop_dim)

    def matmul(self, A: STensor, B: STensor):
        _validate_threadblock_matmul(A, B)
        return self.cygraph.matmul(A, B)

    def exp(self, A: STensor):
        return self.cygraph.exp(A)

    def silu(self, A: STensor):
        return self.cygraph.silu(A)

    def gelu(self, A: STensor):
        return self.cygraph.gelu(A)

    def relu(self, A: STensor):
        return self.cygraph.relu(A)

    def sigmoid(self, A: STensor):
        return self.cygraph.sigmoid(A)

    def log(self, A: STensor):
        return self.cygraph.log(A)

    def clamp(self, A: STensor, min_val: float, max_val: float):
        return self.cygraph.clamp(A, min_val, max_val)

    def square(self, A: STensor):
        return self.cygraph.square(A)

    def sqrt(self, A: STensor):
        return self.cygraph.sqrt(A)

    def mul_scalar(self, A: STensor, scalar: float):
        return self.cygraph.mul_scalar(A, scalar)

    def add(self, A: STensor, B: STensor):
        return self.cygraph.add(A, B)

    def mul(self, A: STensor, B: STensor):
        return self.cygraph.mul(A, B)

    def div(self, A: STensor, B: STensor):
        return self.cygraph.div(A, B)

    def sub(self, A: STensor, B: STensor):
        return self.cygraph.sub(A, B)

    def pow(self, A: STensor, B: STensor):
        return self.cygraph.pow(A, B)

    def reduction(self, A: STensor, dim: int):
        return self.cygraph.reduction(A, dim)

    def reduction_to_dimx(self, A: STensor, dim: int):
        return self.cygraph.reduction_to_dimx(A, dim)

    def reduction_max(self, A: STensor, dim: int):
        return self.cygraph.reduction_max(A, dim)

    def rms_norm(self, A: STensor):
        return self.cygraph.rms_norm(A)

    def concat(self, A: STensor, B: STensor, dim: int):
        return self.cygraph.concat(A, B, dim)

    def split(self, A: STensor, split_size: int, dim: int):
        return self.cygraph.split(A, split_size, dim)

    def chunk(self, A: STensor, chunk_size: int, dim: int):
        return self.cygraph.chunk(A, chunk_size, dim)

    def forloop_accum(self, A: STensor, acc: str = None):
        return self.cygraph.forloop_accum(A, acc)

    def forloop_accum_rescale(self, A: STensor, B: STensor, acc: str = None):
        return self.cygraph.forloop_accum_rescale(A, B, acc)

    def forloop_accum_max(self, A: STensor):
        return self.cygraph.forloop_accum_max(A)
