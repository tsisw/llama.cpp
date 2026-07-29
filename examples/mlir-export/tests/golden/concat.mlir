module {
  func.func @forward(%arg0: tensor<2x8x32xf32> {txe.name = "input_0"}, %arg1: tensor<2x4x32xf32> {txe.name = "input_1"}) -> (tensor<2x12x32xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<2x12x32xf32>
    %1 = tensor.insert_slice %arg0 into %0[0, 0, 0] [2, 8, 32] [1, 1, 1] : tensor<2x8x32xf32> into tensor<2x12x32xf32>
    %2 = tensor.insert_slice %arg1 into %1[0, 8, 0] [2, 4, 32] [1, 1, 1] : tensor<2x4x32xf32> into tensor<2x12x32xf32>
    return %2 : tensor<2x12x32xf32>
  }
}
