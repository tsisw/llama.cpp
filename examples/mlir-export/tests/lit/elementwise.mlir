// RUN: %tsi-ggml-opt --convert-ggml-to-linalg %s | FileCheck %s

// Equal shapes use the named linalg op, which infers its indexing maps from the operand shapes.
// CHECK-LABEL: func.func @add_same
// CHECK: %[[E:.*]] = tensor.empty() : tensor<128xf32>
// CHECK: linalg.add ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) outs(%[[E]] : tensor<128xf32>)
func.func @add_same(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
  %0 = ggml.add %a, %b : tensor<128xf32>, tensor<128xf32> -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// The rms_norm(x)*weight shape: a rank-1 operand broadcast over the innermost dim needs an explicit
// generic, since the named op cannot express the differing maps.
// CHECK-LABEL: func.func @mul_broadcast
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK: arith.mulf
func.func @mul_broadcast(%a: tensor<8x64xf32>, %w: tensor<64xf32>) -> tensor<8x64xf32> {
  %0 = ggml.mul %a, %w : tensor<8x64xf32>, tensor<64xf32> -> tensor<8x64xf32>
  return %0 : tensor<8x64xf32>
}

// The scale factor is a compile-time constant, not an operand.
// CHECK-LABEL: func.func @scale
// CHECK: %[[C:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: arith.mulf %{{.*}}, %[[C]]
func.func @scale(%a: tensor<128xf32>) -> tensor<128xf32> {
  %0 = ggml.scale %a {scale = 0.5 : f32} : tensor<128xf32> -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// SiLU(x) = x / (1 + exp(-x)), matching ggml_silu_f32 exactly.
// CHECK-LABEL: func.func @silu
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
func.func @silu(%a: tensor<128xf32>) -> tensor<128xf32> {
  %0 = ggml.silu %a : tensor<128xf32> -> tensor<128xf32>
  return %0 : tensor<128xf32>
}
