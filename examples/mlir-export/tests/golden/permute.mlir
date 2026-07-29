module {
  func.func @forward(%arg0: tensor<2x4x8xf32> {txe.name = "input_0"}) -> (tensor<2x8x4xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<2x8x4xf32>
    %1 = linalg.transpose ins(%arg0 : tensor<2x4x8xf32>) outs(%0 : tensor<2x8x4xf32>) permutation = [0, 2, 1]
    return %1 : tensor<2x8x4xf32>
  }
}
