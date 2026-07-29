// RUN: %tsi-ggml-opt --convert-ggml-to-linalg %s | FileCheck %s

// A permute that reorders two non-1 dims moves data, so it becomes a real transpose.
// CHECK-LABEL: func.func @permute
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [0, 2, 1]
func.func @permute(%a: tensor<2x4x8xf32>) -> tensor<2x8x4xf32> {
  %0 = ggml.permute %a {axes = array<i32: 1, 0, 2, 3>} : tensor<2x4x8xf32> -> tensor<2x8x4xf32>
  return %0 : tensor<2x8x4xf32>
}

// A permute that only reshuffles size-1 dims moves nothing, so it collapses instead of transposing.
// CHECK-LABEL: func.func @permute_size1
// CHECK-NOT: linalg.transpose
// CHECK: tensor.collapse_shape
func.func @permute_size1(%a: tensor<4x1x16xf32>) -> tensor<4x16xf32> {
  %0 = ggml.permute %a {axes = array<i32: 0, 2, 1, 3>} : tensor<4x1x16xf32> -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// A cont with no shape change is a pure passthrough: nothing at all is emitted for it.
// CHECK-LABEL: func.func @cont_noop
// CHECK-NEXT: return %arg0
func.func @cont_noop(%a: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = ggml.cont %a : tensor<8x16xf32> -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// 2D -> 3D head split.
// CHECK-LABEL: func.func @reshape_split
// CHECK: tensor.expand_shape %arg0 {{\[}}[0], [1, 2]]
func.func @reshape_split(%a: tensor<4x64xf32>) -> tensor<4x4x16xf32> {
  %0 = ggml.reshape %a : tensor<4x64xf32> -> tensor<4x4x16xf32>
  return %0 : tensor<4x4x16xf32>
}

// Built from empty + insert_slice, because the TSI pipeline does not bufferize tensor.concat. The
// second operand starts at the first's extent along the concat dim (ggml dim 1 -> MLIR dim 1 here).
// CHECK-LABEL: func.func @concat
// CHECK-NOT: tensor.concat
// CHECK: tensor.insert_slice %arg0 into %{{.*}}[0, 0, 0] [2, 8, 32] [1, 1, 1]
// CHECK: tensor.insert_slice %arg1 into %{{.*}}[0, 8, 0] [2, 4, 32] [1, 1, 1]
func.func @concat(%a: tensor<2x8x32xf32>, %b: tensor<2x4x32xf32>) -> tensor<2x12x32xf32> {
  %0 = ggml.concat %a, %b {dim = 1 : i32} : tensor<2x8x32xf32>, tensor<2x4x32xf32> -> tensor<2x12x32xf32>
  return %0 : tensor<2x12x32xf32>
}

// Token ids are runtime data, so each row needs a dynamic-offset slice: extract the id, cast it to
// an index, slice that row. Unrolled once per token since the count is compile-time known.
// CHECK-LABEL: func.func @get_rows
// CHECK-COUNT-4: arith.index_cast
// CHECK-LABEL: func.func @get_rows_1tok
// CHECK: arith.index_cast
// CHECK: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
func.func @get_rows(%t: tensor<16x32xf32>, %i: tensor<4xi32>) -> tensor<4x32xf32> {
  %0 = ggml.get_rows %t, %i : tensor<16x32xf32>, tensor<4xi32> -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
func.func @get_rows_1tok(%t: tensor<16x32xf32>, %i: tensor<1xi32>) -> tensor<32xf32> {
  %0 = ggml.get_rows %t, %i : tensor<16x32xf32>, tensor<1xi32> -> tensor<32xf32>
  return %0 : tensor<32xf32>
}
