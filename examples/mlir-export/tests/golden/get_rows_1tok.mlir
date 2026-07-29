module {
  func.func @forward(%arg0: tensor<16x32xf32> {txe.name = "input_0"}, %arg1: tensor<1xi32> {txe.name = "input_1"}) -> (tensor<32xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = arith.constant 0 : index
    %1 = tensor.extract %arg1[%0] : tensor<1xi32>
    %2 = arith.index_cast %1 : i32 to index
    %3 = tensor.extract_slice %arg0[%2, 0] [1, 32] [1, 1] : tensor<16x32xf32> to tensor<32xf32>
    return %3 : tensor<32xf32>
  }
}
