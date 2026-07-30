//===- GgmlOps.td - the ggml dialect --------------------------*- tablegen -*-===//
//
// An MLIR dialect mirroring the subset of ggml ops the TSI exporter handles. A ggml_cgraph is
// imported 1:1 into these ops, then a single conversion pass lowers them to linalg.
//
// SHAPE CONVENTION: operands and results carry MLIR-ordered tensor types, that is ggml's `ne`
// reversed over ggml_n_dims, because ggml's ne[0] is the innermost/fastest dim and MLIR's last dim
// is. The reversal happens once, in the importer. Everything downstream sees one convention.
//
// Attributes that NAME a dimension (permute axes, concat dim) are the opposite: they stay in ggml's
// own dim space, verbatim from op_params, because they are raw ggml data. Translating them into MLIR
// dim indices is part of the lowering, which is where that logic belongs.
//
// Verifiers here encode ggml's OWN invariants, the things true of any valid ggml graph. Limits that
// belong to our lowering rather than to ggml (rank <= 3, ROPE mode must be NORMAL, no YaRN scaling)
// are deliberately NOT verifiers: they are match failures in the conversion patterns, so the dialect
// stays a faithful representation and only the lowering admits it cannot handle something.
//
//===----------------------------------------------------------------------===//

#ifndef TSI_GGML_OPS
#define TSI_GGML_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// SameOperandsAndResultType lives here, not in OpBase.td (which only has the ElementType variant).
include "mlir/Interfaces/InferTypeOpInterface.td"

def Ggml_Dialect : Dialect {
  let name = "ggml";
  let summary = "ggml compute-graph ops, as imported from a ggml_cgraph";
  let cppNamespace = "::tsi::ggml";
}

class Ggml_Op<string mnemonic, list<Trait> traits = []> : Op<Ggml_Dialect, mnemonic, traits>;

// Every tensor in an imported graph is statically shaped, with one of ggml's element types that we
// map directly: f32, f16 and bf16 for data, i32 for indices and positions.
//
// The dialect admits f16/bf16 so the imported IR stays a faithful record of the graph. Arithmetic is
// a different question: promote-ggml-to-f32 rewrites every op to f32 before lowering, so no pattern
// ever sees a half-precision operand and every reduction accumulates in f32.
def Ggml_Tensor : AnyTypeOf<[
  TensorOf<[F32, F16, BF16, I32]>
], "a statically shaped f32, f16, bf16 or i32 tensor">;

//===----------------------------------------------------------------------===//
// elementwise / unary
//===----------------------------------------------------------------------===//

// ggml_add / ggml_mul. Either identical shapes, or ggml's repeat-broadcast of `rhs` over `lhs`.
// The exact broadcast shapes the lowering supports are its own business, not verified here.
class Ggml_BinaryOp<string mnemonic> : Ggml_Op<mnemonic, [Pure, SameOperandsAndResultElementType]> {
  let arguments = (ins Ggml_Tensor:$lhs, Ggml_Tensor:$rhs);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def Ggml_AddOp : Ggml_BinaryOp<"add"> {
  let summary = "ggml_add: elementwise sum, with ggml repeat-broadcast of rhs";
}

def Ggml_MulOp : Ggml_BinaryOp<"mul"> {
  let summary = "ggml_mul: elementwise product, with ggml repeat-broadcast of rhs";
}

// ggml_scale. `bias` is op_params[1]; ggml_scale itself always sets it to 0, ggml_scale_bias does
// not, so it is carried faithfully and the lowering rejects a non-zero value.
def Ggml_ScaleOp : Ggml_Op<"scale", [Pure, SameOperandsAndResultType]> {
  let summary = "ggml_scale: multiply by a compile-time scalar";
  let arguments = (ins Ggml_Tensor:$input, F32Attr:$scale, DefaultValuedAttr<F32Attr, "0.0">:$bias);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// GGML_OP_UNARY / GGML_UNARY_OP_SILU. SiLU(x) = x / (1 + exp(-x)).
def Ggml_SiluOp : Ggml_Op<"silu", [Pure, SameOperandsAndResultType]> {
  let summary = "ggml_silu";
  let arguments = (ins Ggml_Tensor:$input);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// norms
//===----------------------------------------------------------------------===//

// ggml_rms_norm: x / sqrt(mean(x^2 over ne[0]) + eps), per row.
def Ggml_RmsNormOp : Ggml_Op<"rms_norm", [Pure, SameOperandsAndResultType]> {
  let summary = "ggml_rms_norm: RMS normalization over the innermost dim";
  let arguments = (ins Ggml_Tensor:$input, F32Attr:$eps);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

// ggml_soft_max_ext: softmax(x*scale + mask) over ne[0]. `mask` is optional; `max_bias` drives
// ALiBi and is carried so the lowering can reject a non-zero value.
def Ggml_SoftMaxOp : Ggml_Op<"soft_max", [Pure]> {
  let summary = "ggml_soft_max_ext: softmax over the innermost dim, with optional additive mask";
  let arguments = (ins Ggml_Tensor:$input, Optional<Ggml_Tensor>:$mask,
                       DefaultValuedAttr<F32Attr, "1.0">:$scale,
                       DefaultValuedAttr<F32Attr, "0.0">:$max_bias);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat = "$input (`,` $mask^ `:` type($mask))? attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// matmul
//===----------------------------------------------------------------------===//

// ggml_mul_mat(a, b). ggml's own precondition is a->ne[0] == b->ne[0] and, for the batch dims,
// b->ne[2] % a->ne[2] == 0 (the GQA grouping rule). In MLIR order that reads as: the LAST dim of
// both operands matches (the K dim), and the leading batch dim of rhs is a multiple of lhs's.
// Verified here because it is ggml's invariant, not our restriction.
def Ggml_MulMatOp : Ggml_Op<"mul_mat", [Pure]> {
  let summary = "ggml_mul_mat: A * B^T with ggml's transposed result layout";
  let arguments = (ins Ggml_Tensor:$lhs, Ggml_Tensor:$rhs);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// rope
//===----------------------------------------------------------------------===//

// ggml_rope. All frequency parameters are carried verbatim from op_params so the lowering can
// reject the combinations it does not implement (YaRN scaling, partial rotation, NEOX mode).
// `positions` is an i32 tensor with one entry per ggml ne[2] of the input.
def Ggml_RopeOp : Ggml_Op<"rope", [Pure]> {
  let summary = "ggml_rope: rotary position embedding";
  let arguments = (ins Ggml_Tensor:$input, Ggml_Tensor:$positions,
                       I32Attr:$n_dims, I32Attr:$mode,
                       DefaultValuedAttr<F32Attr, "10000.0">:$freq_base,
                       DefaultValuedAttr<F32Attr, "1.0">:$freq_scale,
                       DefaultValuedAttr<F32Attr, "0.0">:$ext_factor,
                       DefaultValuedAttr<F32Attr, "1.0">:$attn_factor);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat =
      "$input `,` $positions attr-dict `:` type($input) `,` type($positions) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// shape
//===----------------------------------------------------------------------===//

// ggml_permute. `axes` is op_params[0..3] verbatim, in GGML dim space: ggml sets
// result.ne[axes[i]] = src.ne[i]. Always four entries, even when the tensor has lower rank.
def Ggml_PermuteOp : Ggml_Op<"permute", [Pure]> {
  let summary = "ggml_permute: a strided view with reordered dims";
  let arguments = (ins Ggml_Tensor:$input, DenseI32ArrayAttr:$axes);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

// ggml_reshape and ggml_cont are kept distinct so the imported IR is faithful to the graph, even
// though both lower through the same pattern: the exporter materializes every value eagerly, so by
// the time a CONT is reached its source is already what ggml's own cont would produce, leaving only
// the shape change (if any) to emit. The new shape is carried by the result type.
class Ggml_ReshapeLikeOp<string mnemonic> : Ggml_Op<mnemonic, [Pure, SameOperandsAndResultElementType]> {
  let arguments = (ins Ggml_Tensor:$input);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def Ggml_ReshapeOp : Ggml_ReshapeLikeOp<"reshape"> {
  let summary = "ggml_reshape: reinterpret the same elements under a new shape";
}

def Ggml_ContOp : Ggml_ReshapeLikeOp<"cont"> {
  let summary = "ggml_cont: materialize a possibly non-contiguous view, optionally reshaping";
}

// ggml_concat(a, b, dim). `dim` is in GGML dim space, from op_params[0].
def Ggml_ConcatOp : Ggml_Op<"concat", [Pure, SameOperandsAndResultElementType]> {
  let summary = "ggml_concat: join two tensors along a ggml dim";
  let arguments = (ins Ggml_Tensor:$lhs, Ggml_Tensor:$rhs, I32Attr:$dim);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

// ggml_get_rows(table, ids): gather rows of an embedding table by i32 indices.
def Ggml_GetRowsOp : Ggml_Op<"get_rows", [Pure]> {
  let summary = "ggml_get_rows: gather rows of an embedding table by index";
  let arguments = (ins Ggml_Tensor:$table, TensorOf<[I32]>:$ids);
  let results   = (outs Ggml_Tensor:$result);
  let assemblyFormat =
      "$table `,` $ids attr-dict `:` type($table) `,` type($ids) `->` type($result)";
  let hasVerifier = 1;
}

#endif  // TSI_GGML_OPS
