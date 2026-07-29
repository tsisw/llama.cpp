module {
  func.func @forward(%arg0: tensor<128xf32> {txe.name = "input_0"}) -> (tensor<128xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
    %0 = arith.constant 0.5 : f32
    %1 = tensor.empty() : tensor<128xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : tensor<128xf32>) outs(%1 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %r = arith.mulf %in, %0 : f32
      linalg.yield %r : f32
    } -> tensor<128xf32>
    return %2 : tensor<128xf32>
  }
}
