module {
  func.func @mugraph(%arg0: tensor<2048x4096xf16>, %arg1: tensor<4096x11008xf16>, %arg2: tensor<4096x11008xf16>, %arg3: tensor<11008x4096xf16>) -> (tensor<2048x4096xf16>) {
    %0 = yirage.matmul %arg0, %arg1 : tensor<2048x4096xf16>, tensor<4096x11008xf16> -> tensor<2048x11008xf16>
    %1 = yirage.matmul %arg0, %arg2 : tensor<2048x4096xf16>, tensor<4096x11008xf16> -> tensor<2048x11008xf16>
    %2 = yirage.silu %0 : tensor<2048x11008xf16>
    %3 = arith.mulf %2, %1 : tensor<2048x11008xf16>
    %4 = yirage.matmul %3, %arg3 : tensor<2048x11008xf16>, tensor<11008x4096xf16> -> tensor<2048x4096xf16>
    return %4 : tensor<2048x4096xf16>
  }
}
