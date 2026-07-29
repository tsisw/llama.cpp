module {
  func.func @forward(%arg0: tensor<2x32x32xf32> {txe.name = "input_0"}, %arg1: tensor<4x32x32xf32> {txe.name = "input_1"}) -> (tensor<4x32x32xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<4x32x32xf32>
    %1 = tensor.extract_slice %arg0[0, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xf32> to tensor<1x32x32xf32>
    %2 = tensor.insert_slice %1 into %0[0, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<1x32x32xf32> into tensor<4x32x32xf32>
    %3 = tensor.insert_slice %1 into %2[1, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<1x32x32xf32> into tensor<4x32x32xf32>
    %4 = tensor.extract_slice %arg0[1, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xf32> to tensor<1x32x32xf32>
    %5 = tensor.insert_slice %4 into %3[2, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<1x32x32xf32> into tensor<4x32x32xf32>
    %6 = tensor.insert_slice %4 into %5[3, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<1x32x32xf32> into tensor<4x32x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %7 = tensor.empty() : tensor<4x32x32xf32>
    %8 = linalg.transpose ins(%6 : tensor<4x32x32xf32>) outs(%7 : tensor<4x32x32xf32>) permutation = [0, 2, 1]
    %9 = tensor.empty() : tensor<4x32x32xf32>
    %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    %11 = linalg.batch_matmul ins(%arg1, %8 : tensor<4x32x32xf32>, tensor<4x32x32xf32>) outs(%10 : tensor<4x32x32xf32>) -> tensor<4x32x32xf32>
    return %11 : tensor<4x32x32xf32>
  }
}
