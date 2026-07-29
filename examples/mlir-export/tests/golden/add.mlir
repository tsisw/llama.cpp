module {
  func.func @forward(%arg0: tensor<128xf32> {txe.name = "input_0"}, %arg1: tensor<128xf32> {txe.name = "input_1"}) -> (tensor<128xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }
}
