module {
  func.func @forward(%arg0: tensor<8x64xf32> {txe.name = "input_0"}) -> (tensor<8x64xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.constant 0xFF800000 : f32
    %1 = tensor.empty() : tensor<8xf32>
    %2 = linalg.fill ins(%0 : f32) outs(%1 : tensor<8xf32>) -> tensor<8xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<8x64xf32>) outs(%2 : tensor<8xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %m = arith.maximumf %in, %acc : f32
      linalg.yield %m : f32
    } -> tensor<8xf32>
    %4 = tensor.empty() : tensor<8x64xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0)>, affine_map<(d0,d1) -> (d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %3 : tensor<8x64xf32>, tensor<8xf32>) outs(%4 : tensor<8x64xf32>) {
    ^bb0(%in: f32, %mx: f32, %out: f32):
      %sub = arith.subf %in, %mx : f32
      %e = math.exp %sub : f32
      linalg.yield %e : f32
    } -> tensor<8x64xf32>
    %6 = tensor.empty() : tensor<8xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<8xf32>) -> tensor<8xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<8x64xf32>) outs(%7 : tensor<8xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %a = arith.addf %in, %acc : f32
      linalg.yield %a : f32
    } -> tensor<8xf32>
    %9 = tensor.empty() : tensor<8x64xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0)>, affine_map<(d0,d1) -> (d0,d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %8 : tensor<8x64xf32>, tensor<8xf32>) outs(%9 : tensor<8x64xf32>) {
    ^bb0(%in: f32, %sm: f32, %out: f32):
      %d = arith.divf %in, %sm : f32
      linalg.yield %d : f32
    } -> tensor<8x64xf32>
    return %10 : tensor<8x64xf32>
  }
}
