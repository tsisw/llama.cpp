// RUN: %tsi-ggml-opt --promote-ggml-to-f32 %s | FileCheck %s
// RUN: %tsi-ggml-opt --promote-ggml-to-f32 --convert-ggml-to-linalg %s | FileCheck %s --check-prefix=LOWERED

// f32 accumulation is the whole point: half-precision operands are widened, the op runs at f32, and
// the result is narrowed back so consumers still see the type the graph declared.
// CHECK-LABEL: func.func @add_f16
// CHECK: arith.extf
// CHECK: arith.extf
// CHECK: ggml.add {{.*}} : tensor<128xf32>, tensor<128xf32> -> tensor<128xf32>
// CHECK: arith.truncf
func.func @add_f16(%a: tensor<128xf16>, %b: tensor<128xf16>) -> tensor<128xf16> {
  %0 = ggml.add %a, %b : tensor<128xf16>, tensor<128xf16> -> tensor<128xf16>
  return %0 : tensor<128xf16>
}

// bf16 takes the same path. It matters separately from f16 because bf16 has f32's exponent range but
// 8 mantissa bits, so it is the type where a narrow accumulator hurts most.
// CHECK-LABEL: func.func @rms_norm_bf16
// CHECK: arith.extf
// CHECK: ggml.rms_norm {{.*}} : tensor<4x64xf32> -> tensor<4x64xf32>
// CHECK: arith.truncf
func.func @rms_norm_bf16(%x: tensor<4x64xbf16>) -> tensor<4x64xbf16> {
  %0 = ggml.rms_norm %x {eps = 1.000000e-05 : f32} : tensor<4x64xbf16> -> tensor<4x64xbf16>
  return %0 : tensor<4x64xbf16>
}

// The case llama actually produces: an f16 weight against an f32 activation, f32 out. Only the f16
// operand is extended and there is nothing to narrow, so no truncf appears.
// CHECK-LABEL: func.func @matmul_f16_weight
// CHECK: arith.extf
// CHECK: ggml.mul_mat {{.*}} : tensor<32x64xf32>, tensor<8x64xf32> -> tensor<8x32xf32>
// CHECK-NOT: arith.truncf
func.func @matmul_f16_weight(%w: tensor<32x64xf16>, %x: tensor<8x64xf32>) -> tensor<8x32xf32> {
  %0 = ggml.mul_mat %w, %x : tensor<32x64xf16>, tensor<8x64xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// An all-f32 graph must come out untouched: no casts, and the op still standing for the lowering.
// CHECK-LABEL: func.func @add_f32_untouched
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.truncf
// CHECK: ggml.add
func.func @add_f32_untouched(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
  %0 = ggml.add %a, %b : tensor<128xf32>, tensor<128xf32> -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// Promotion then lowering leaves no ggml op behind, which is what proves the two passes compose:
// the patterns only ever see f32 and none of them needed changing.
// LOWERED-LABEL: func.func @add_f16
// LOWERED-NOT: ggml.
// LOWERED: linalg.add
