// Negative tests. These matter as much as the positive ones: they pin down WHICH layer rejects what.
// A ggml invariant must fail in the dialect verifier, and a limit of our lowering must fail in the
// conversion. If those ever swap places the separation has broken.
//
// RUN: not %tsi-ggml-opt --convert-ggml-to-linalg --split-input-file %s 2>&1 | FileCheck %s

// --- ggml's own invariants: dialect verifiers ------------------------------------------------

// CHECK: reduction dim mismatch: lhs innermost 4 vs rhs innermost 7
func.func @mul_mat_bad_k(%a: tensor<8x4xf32>, %b: tensor<8x7xf32>) -> tensor<8x8xf32> {
  %0 = ggml.mul_mat %a, %b : tensor<8x4xf32>, tensor<8x7xf32> -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK: rhs batch dim 5 must be a multiple of lhs batch dim 2
func.func @mul_mat_bad_gqa(%a: tensor<2x8x4xf32>, %b: tensor<5x8x4xf32>) -> tensor<5x8x8xf32> {
  %0 = ggml.mul_mat %a, %b : tensor<2x8x4xf32>, tensor<5x8x4xf32> -> tensor<5x8x8xf32>
  return %0 : tensor<5x8x8xf32>
}

// -----

// CHECK: positions has 2 entries but input needs 3
func.func @rope_bad_pos(%x: tensor<3x4x16xf32>, %p: tensor<2xi32>) -> tensor<3x4x16xf32> {
  %0 = ggml.rope %x, %p {n_dims = 16 : i32, mode = 0 : i32} : tensor<3x4x16xf32>, tensor<2xi32> -> tensor<3x4x16xf32>
  return %0 : tensor<3x4x16xf32>
}

// -----

// CHECK: element count must be preserved: 64 vs 32
func.func @reshape_bad_count(%a: tensor<4x16xf32>) -> tensor<2x16xf32> {
  %0 = ggml.reshape %a : tensor<4x16xf32> -> tensor<2x16xf32>
  return %0 : tensor<2x16xf32>
}

// -----

// CHECK: axes must be a permutation of 0..3
func.func @permute_bad_axes(%a: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = ggml.permute %a {axes = array<i32: 1, 1, 2, 3>} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}

// --- our lowering's limits: conversion failures ----------------------------------------------

// -----

// ggml_scale_bias's non-zero bias form is expressible in ggml, so it imports cleanly and is only
// rejected by the lowering.
// CHECK: failed to legalize operation 'ggml.scale'
func.func @scale_with_bias(%a: tensor<8xf32>) -> tensor<8xf32> {
  %0 = ggml.scale %a {scale = 2.0 : f32, bias = 1.0 : f32} : tensor<8xf32> -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// ALiBi (max_bias != 0).
// CHECK: failed to legalize operation 'ggml.soft_max'
func.func @soft_max_alibi(%a: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = ggml.soft_max %a {max_bias = 8.0 : f32} : tensor<8x16xf32> -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// NEOX rope mode.
// CHECK: failed to legalize operation 'ggml.rope'
func.func @rope_neox(%x: tensor<4x16xf32>, %p: tensor<1xi32>) -> tensor<4x16xf32> {
  %0 = ggml.rope %x, %p {n_dims = 16 : i32, mode = 2 : i32} : tensor<4x16xf32>, tensor<1xi32> -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

// Partial rotation (n_dims < head_dim).
// CHECK: failed to legalize operation 'ggml.rope'
func.func @rope_partial(%x: tensor<4x16xf32>, %p: tensor<1xi32>) -> tensor<4x16xf32> {
  %0 = ggml.rope %x, %p {n_dims = 8 : i32, mode = 0 : i32} : tensor<4x16xf32>, tensor<1xi32> -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
