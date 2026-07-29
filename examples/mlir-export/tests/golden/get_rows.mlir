module {
  func.func @forward(%arg0: tensor<16x32xf32> {txe.name = "input_0"}, %arg1: tensor<4xi32> {txe.name = "input_1"}) -> (tensor<4x32xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<4x32xf32>
    %1 = arith.constant 0 : index
    %2 = tensor.extract %arg1[%1] : tensor<4xi32>
    %3 = arith.index_cast %2 : i32 to index
    %4 = tensor.extract_slice %arg0[%3, 0] [1, 32] [1, 1] : tensor<16x32xf32> to tensor<1x32xf32>
    %5 = tensor.insert_slice %4 into %0[0, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
    %6 = arith.constant 1 : index
    %7 = tensor.extract %arg1[%6] : tensor<4xi32>
    %8 = arith.index_cast %7 : i32 to index
    %9 = tensor.extract_slice %arg0[%8, 0] [1, 32] [1, 1] : tensor<16x32xf32> to tensor<1x32xf32>
    %10 = tensor.insert_slice %9 into %5[1, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
    %11 = arith.constant 2 : index
    %12 = tensor.extract %arg1[%11] : tensor<4xi32>
    %13 = arith.index_cast %12 : i32 to index
    %14 = tensor.extract_slice %arg0[%13, 0] [1, 32] [1, 1] : tensor<16x32xf32> to tensor<1x32xf32>
    %15 = tensor.insert_slice %14 into %10[2, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
    %16 = arith.constant 3 : index
    %17 = tensor.extract %arg1[%16] : tensor<4xi32>
    %18 = arith.index_cast %17 : i32 to index
    %19 = tensor.extract_slice %arg0[%18, 0] [1, 32] [1, 1] : tensor<16x32xf32> to tensor<1x32xf32>
    %20 = tensor.insert_slice %19 into %15[3, 0] [1, 32] [1, 1] : tensor<1x32xf32> into tensor<4x32xf32>
    return %20 : tensor<4x32xf32>
  }
}
