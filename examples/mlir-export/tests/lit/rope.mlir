// RUN: %tsi-ggml-opt --convert-ggml-to-linalg %s | FileCheck %s

// rank-2 x: ggml ne[2] is implicitly 1, so there is one shared position. It is broadcast with
// fill + linalg.mul rather than captured as a scalar in a generic body or splatted: the scalar
// capture fails to legalize in the txe-to-LLVM stage and tensor.splat fails to bufferize.
// The frequency table is a compile-time dense constant.
// CHECK-LABEL: func.func @rope_2d
// CHECK: arith.constant dense<{{.*}}> : tensor<8xf32>
// CHECK: tensor.extract
// CHECK: arith.sitofp
// CHECK: linalg.fill
// CHECK: linalg.mul
// CHECK: math.cos
// CHECK: math.sin
// Deinterleave the even/odd pairs with a stride-2 slice, rotate, then reinterleave.
// CHECK: tensor.extract_slice %arg0[0, 0] [4, 8] [1, 2]
// CHECK: tensor.extract_slice %arg0[0, 1] [4, 8] [1, 2]
// CHECK: tensor.insert_slice %{{.*}}[0, 0] [4, 8] [1, 2]
// CHECK: tensor.insert_slice %{{.*}}[0, 1] [4, 8] [1, 2]
func.func @rope_2d(%x: tensor<4x16xf32>, %p: tensor<1xi32>) -> tensor<4x16xf32> {
  %0 = ggml.rope %x, %p {n_dims = 16 : i32, mode = 0 : i32} : tensor<4x16xf32>, tensor<1xi32> -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// rank-3 x: one position per token, so the whole position vector is converted elementwise (staying a
// real tensor operand) and theta is an outer product over (token, pair). cos/sin then broadcast over
// the head dim, which carries no position dependence.
// CHECK-LABEL: func.func @rope_3d
// CHECK: linalg.generic
// CHECK: arith.sitofp
// CHECK: math.cos
// CHECK: math.sin
// CHECK: tensor.extract_slice %arg0[0, 0, 0] [3, 4, 8] [1, 1, 2]
// CHECK: tensor.extract_slice %arg0[0, 0, 1] [3, 4, 8] [1, 1, 2]
func.func @rope_3d(%x: tensor<3x4x16xf32>, %p: tensor<3xi32>) -> tensor<3x4x16xf32> {
  %0 = ggml.rope %x, %p {n_dims = 16 : i32, mode = 0 : i32} : tensor<3x4x16xf32>, tensor<3xi32> -> tensor<3x4x16xf32>
  return %0 : tensor<3x4x16xf32>
}
