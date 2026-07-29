module {
  func.func @forward(%arg0: tensor<3x4x16xf32> {txe.name = "input_0"}, %arg1: tensor<3xi32> {txe.name = "input_1"}) -> (tensor<3x4x16xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = arith.constant dense<[1.0, 0.316227764, 0.100000001, 0.0316227749, 0.00999999978, 0.00316227763, 0.00100000005, 0.000316227757]> : tensor<8xf32>
    %1 = tensor.empty() : tensor<3xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg1 : tensor<3xi32>) outs(%1 : tensor<3xf32>) {
    ^bb0(%in: i32, %out: f32):
      %f = arith.sitofp %in : i32 to f32
      linalg.yield %f : f32
    } -> tensor<3xf32>
    %3 = tensor.empty() : tensor<3x8xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0)>, affine_map<(d0,d1) -> (d1)>, affine_map<(d0,d1) -> (d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %0 : tensor<3xf32>, tensor<8xf32>) outs(%3 : tensor<3x8xf32>) {
    ^bb0(%p: f32, %fr: f32, %out: f32):
      %th = arith.mulf %p, %fr : f32
      linalg.yield %th : f32
    } -> tensor<3x8xf32>
    %5 = tensor.empty() : tensor<3x8xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<3x8xf32>) outs(%5 : tensor<3x8xf32>) {
    ^bb0(%t: f32, %out: f32):
      %c = math.cos %t : f32
      linalg.yield %c : f32
    } -> tensor<3x8xf32>
    %7 = tensor.empty() : tensor<3x8xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<3x8xf32>) outs(%7 : tensor<3x8xf32>) {
    ^bb0(%t: f32, %out: f32):
      %s = math.sin %t : f32
      linalg.yield %s : f32
    } -> tensor<3x8xf32>
    %9 = tensor.extract_slice %arg0[0, 0, 0] [3, 4, 8] [1, 1, 2] : tensor<3x4x16xf32> to tensor<3x4x8xf32>
    %10 = tensor.extract_slice %arg0[0, 0, 1] [3, 4, 8] [1, 1, 2] : tensor<3x4x16xf32> to tensor<3x4x8xf32>
    %11 = tensor.empty() : tensor<3x4x8xf32>
    %12 = tensor.empty() : tensor<3x4x8xf32>
    %13, %14 = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2) -> (d0,d1,d2)>, affine_map<(d0,d1,d2) -> (d0,d1,d2)>, affine_map<(d0,d1,d2) -> (d0,d2)>, affine_map<(d0,d1,d2) -> (d0,d2)>, affine_map<(d0,d1,d2) -> (d0,d1,d2)>, affine_map<(d0,d1,d2) -> (d0,d1,d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %10, %6, %8 : tensor<3x4x8xf32>, tensor<3x4x8xf32>, tensor<3x8xf32>, tensor<3x8xf32>) outs(%11, %12 : tensor<3x4x8xf32>, tensor<3x4x8xf32>) {
    ^bb0(%xe: f32, %xo: f32, %c: f32, %s: f32, %oe: f32, %oo: f32):
      %e1 = arith.mulf %xe, %c : f32
      %e2 = arith.mulf %xo, %s : f32
      %new_e = arith.subf %e1, %e2 : f32
      %o1 = arith.mulf %xe, %s : f32
      %o2 = arith.mulf %xo, %c : f32
      %new_o = arith.addf %o1, %o2 : f32
      linalg.yield %new_e, %new_o : f32, f32
    } -> (tensor<3x4x8xf32>, tensor<3x4x8xf32>)
    %15 = tensor.empty() : tensor<3x4x16xf32>
    %16 = tensor.insert_slice %13 into %15[0, 0, 0] [3, 4, 8] [1, 1, 2] : tensor<3x4x8xf32> into tensor<3x4x16xf32>
    %17 = tensor.insert_slice %14 into %16[0, 0, 1] [3, 4, 8] [1, 1, 2] : tensor<3x4x8xf32> into tensor<3x4x16xf32>
    return %17 : tensor<3x4x16xf32>
  }
}
