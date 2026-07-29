// RUN: %tsi-ggml-opt --convert-ggml-to-linalg %s | FileCheck %s

// ggml_mul_mat computes A*B^T with a transposed result layout, so the lowering transposes A (not B)
// and emits matmul(B, transpose(A)). The accumulator must be zero-filled before the matmul.
// CHECK-LABEL: func.func @matmul_2d
// CHECK: linalg.transpose ins(%arg0 : tensor<32x32xf32>) outs(%{{.*}} : tensor<32x32xf32>) permutation = [1, 0]
// CHECK: linalg.fill
// CHECK: linalg.matmul ins(%arg1, %{{.*}}
func.func @matmul_2d(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = ggml.mul_mat %a, %b : tensor<32x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// n_tokens=1 decode: a rank-1 rhs is expanded to [1,k], run through the same path, then collapsed.
// CHECK-LABEL: func.func @matmul_vec
// CHECK: tensor.expand_shape
// CHECK: linalg.matmul
// CHECK: tensor.collapse_shape
func.func @matmul_vec(%a: tensor<32x32xf32>, %b: tensor<32xf32>) -> tensor<32xf32> {
  %0 = ggml.mul_mat %a, %b : tensor<32x32xf32>, tensor<32xf32> -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// Equal head counts: the batch dim passes through the transpose and matmul unchanged.
// CHECK-LABEL: func.func @matmul_3d
// CHECK: linalg.transpose ins(%arg0 : tensor<2x32x32xf32>) outs(%{{.*}}) permutation = [0, 2, 1]
// CHECK: linalg.batch_matmul
func.func @matmul_3d(%a: tensor<2x32x32xf32>, %b: tensor<2x32x32xf32>) -> tensor<2x32x32xf32> {
  %0 = ggml.mul_mat %a, %b : tensor<2x32x32xf32>, tensor<2x32x32xf32> -> tensor<2x32x32xf32>
  return %0 : tensor<2x32x32xf32>
}

// GQA: rhs has 2x the heads, so lhs's heads are repeated first with slices (not a floordiv indexing
// map, which is valid MLIR but untested through the TSI pipeline's tile/vectorize stages). Four
// inserts for four destination heads, then the identical batched core.
// CHECK-LABEL: func.func @matmul_gqa
// CHECK-COUNT-4: tensor.insert_slice
// CHECK: linalg.batch_matmul
func.func @matmul_gqa(%a: tensor<2x32x32xf32>, %b: tensor<4x32x32xf32>) -> tensor<4x32x32xf32> {
  %0 = ggml.mul_mat %a, %b : tensor<2x32x32xf32>, tensor<4x32x32xf32> -> tensor<4x32x32xf32>
  return %0 : tensor<4x32x32xf32>
}
