module {
  func.func @forward(%arg0: tensor<32x32xf32> {txe.name = "input_0"}, %arg1: tensor<32xf32> {txe.name = "input_1"}) -> (tensor<32xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.expand_shape %arg1 [[0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
    %1 = tensor.empty() : tensor<32x32xf32>
    %2 = linalg.transpose ins(%arg0 : tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) permutation = [1, 0]
    %3 = tensor.empty() : tensor<1x32xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %5 = linalg.matmul ins(%0, %2 : tensor<1x32xf32>, tensor<32x32xf32>) outs(%4 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %6 = tensor.collapse_shape %5 [[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
    return %6 : tensor<32xf32>
  }
}
