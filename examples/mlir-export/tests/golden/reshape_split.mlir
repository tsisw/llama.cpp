module {
  func.func @forward(%arg0: tensor<4x64xf32> {txe.name = "input_0"}) -> (tensor<4x4x16xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [4, 4, 16] : tensor<4x64xf32> into tensor<4x4x16xf32>
    return %0 : tensor<4x4x16xf32>
  }
}
