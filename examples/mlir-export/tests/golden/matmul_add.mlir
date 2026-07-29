module {
  func.func @forward(%arg0: tensor<32x32xf32> {txe.name = "input_0"}, %arg1: tensor<32x32xf32> {txe.name = "input_1"}, %arg2: tensor<32x32xf32> {txe.name = "input_2"}) -> (tensor<32x32xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x32xf32>
    %1 = linalg.transpose ins(%arg0 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) permutation = [1, 0]
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = linalg.matmul ins(%arg1, %1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = tensor.empty() : tensor<32x32xf32>
    %6 = linalg.add ins(%4, %arg2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %6 : tensor<32x32xf32>
  }
}
