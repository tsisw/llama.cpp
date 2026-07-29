module {
  func.func @forward(%arg0: tensor<4x16xf32> {txe.name = "input_0"}, %arg1: tensor<1xi32> {txe.name = "input_1"}) -> (tensor<4x16xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = arith.constant dense<[1.0, 0.316227764, 0.100000001, 0.0316227749, 0.00999999978, 0.00316227763, 0.00100000005, 0.000316227757]> : tensor<8xf32>
    %1 = arith.constant 0 : index
    %2 = tensor.extract %arg1[%1] : tensor<1xi32>
    %3 = arith.sitofp %2 : i32 to f32
    %4 = tensor.empty() : tensor<8xf32>
    %5 = linalg.fill ins(%3 : f32) outs(%4 : tensor<8xf32>) -> tensor<8xf32>
    %6 = tensor.empty() : tensor<8xf32>
    %7 = linalg.mul ins(%0, %5 : tensor<8xf32>, tensor<8xf32>) outs(%6 : tensor<8xf32>) -> tensor<8xf32>
    %8 = tensor.empty() : tensor<8xf32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7 : tensor<8xf32>) outs(%8 : tensor<8xf32>) {
    ^bb0(%t: f32, %out: f32):
      %c = math.cos %t : f32
      linalg.yield %c : f32
    } -> tensor<8xf32>
    %10 = tensor.empty() : tensor<8xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7 : tensor<8xf32>) outs(%10 : tensor<8xf32>) {
    ^bb0(%t: f32, %out: f32):
      %s = math.sin %t : f32
      linalg.yield %s : f32
    } -> tensor<8xf32>
    %12 = tensor.extract_slice %arg0[0, 0] [4, 8] [1, 2] : tensor<4x16xf32> to tensor<4x8xf32>
    %13 = tensor.extract_slice %arg0[0, 1] [4, 8] [1, 2] : tensor<4x16xf32> to tensor<4x8xf32>
    %14 = tensor.empty() : tensor<4x8xf32>
    %15 = tensor.empty() : tensor<4x8xf32>
    %16, %17 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d1)>, affine_map<(d0,d1) -> (d1)>, affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %13, %9, %11 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>) outs(%14, %15 : tensor<4x8xf32>, tensor<4x8xf32>) {
    ^bb0(%xe: f32, %xo: f32, %c: f32, %s: f32, %oe: f32, %oo: f32):
      %e1 = arith.mulf %xe, %c : f32
      %e2 = arith.mulf %xo, %s : f32
      %new_e = arith.subf %e1, %e2 : f32
      %o1 = arith.mulf %xe, %s : f32
      %o2 = arith.mulf %xo, %c : f32
      %new_o = arith.addf %o1, %o2 : f32
      linalg.yield %new_e, %new_o : f32, f32
    } -> (tensor<4x8xf32>, tensor<4x8xf32>)
    %18 = tensor.empty() : tensor<4x16xf32>
    %19 = tensor.insert_slice %16 into %18[0, 0] [4, 8] [1, 2] : tensor<4x8xf32> into tensor<4x16xf32>
    %20 = tensor.insert_slice %17 into %19[0, 1] [4, 8] [1, 2] : tensor<4x8xf32> into tensor<4x16xf32>
    return %20 : tensor<4x16xf32>
  }
}
