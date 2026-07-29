module {
  func.func @forward(%arg0: tensor<8x64xf32> {txe.name = "input_0"}) -> (tensor<8x64xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.constant 0.015625 : f32
    %1 = arith.constant 9.99999975e-06 : f32
    %2 = tensor.empty() : tensor<8xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<8xf32>) -> tensor<8xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<8x64xf32>) outs(%3 : tensor<8xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sq = arith.mulf %in, %in : f32
      %newacc = arith.addf %acc, %sq : f32
      linalg.yield %newacc : f32
    } -> tensor<8xf32>
    %5 = tensor.empty() : tensor<8xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4 : tensor<8xf32>) outs(%5 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %mean = arith.mulf %in, %0 : f32
      %meaneps = arith.addf %mean, %1 : f32
      %rs = math.rsqrt %meaneps : f32
      linalg.yield %rs : f32
    } -> tensor<8xf32>
    %7 = tensor.empty() : tensor<8x64xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0)>, affine_map<(d0,d1) -> (d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %6 : tensor<8x64xf32>, tensor<8xf32>) outs(%7 : tensor<8x64xf32>) {
    ^bb0(%in: f32, %sc: f32, %out: f32):
      %m = arith.mulf %in, %sc : f32
      linalg.yield %m : f32
    } -> tensor<8x64xf32>
    return %8 : tensor<8x64xf32>
  }
}
