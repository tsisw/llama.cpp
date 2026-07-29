// RUN: %tsi-ggml-opt --convert-ggml-to-linalg %s | FileCheck %s

// Three stages: sum-of-squares reduction over the innermost dim, a pointwise mean+eps+rsqrt on the
// reduced vector, then a broadcast multiply. 1/cols is folded into a constant at lowering time.
// CHECK-LABEL: func.func @rms_norm
// CHECK: arith.constant 1.562500e-02 : f32
// CHECK: arith.constant 9.99999974E-6 : f32
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK: math.rsqrt
func.func @rms_norm(%a: tensor<8x64xf32>) -> tensor<8x64xf32> {
  %0 = ggml.rms_norm %a {eps = 9.99999974E-6 : f32} : tensor<8x64xf32> -> tensor<8x64xf32>
  return %0 : tensor<8x64xf32>
}

// Row max seeded with -inf, exp of the shifted values, row sum, divide. With scale 1 and no mask the
// scaling stage is skipped entirely rather than emitting a multiply by one.
// CHECK-LABEL: func.func @soft_max
// CHECK: arith.constant 0xFF800000 : f32
// CHECK: arith.maximumf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK-NOT: arith.mulf
func.func @soft_max(%a: tensor<8x64xf32>) -> tensor<8x64xf32> {
  %0 = ggml.soft_max %a : tensor<8x64xf32> -> tensor<8x64xf32>
  return %0 : tensor<8x64xf32>
}

// A mask is added after scaling, broadcast over the head dim.
// CHECK-LABEL: func.func @soft_max_masked
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.maximumf
func.func @soft_max_masked(%a: tensor<4x8x16xf32>, %m: tensor<8x16xf32>) -> tensor<4x8x16xf32> {
  %0 = ggml.soft_max %a, %m : tensor<8x16xf32> {scale = 0.125 : f32} : tensor<4x8x16xf32> -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}
