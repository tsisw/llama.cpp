// Elementwise and unary ops: ADD, MUL, SCALE, and the SILU unary.
#include "Builder.h"

#include <cstring>

using namespace mlir;

namespace tsi::mlir_export {

// Two shapes supported: identical shape at any rank (linalg.add/linalg.mul are named ops that
// infer identity indexing maps from the operand shapes, so no explicit affine maps are needed), or
// a's innermost dim broadcast against a 1D b matching that dim - the real rms_norm(x)*weight
// pattern, matching ggml's ggml_can_repeat(b,a) rule when b's non-innermost dims are all 1.
Value GraphBuilder::emitElementwiseBinop(const ggml_tensor * node, const char * which, const char * ggmlName) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];

    const int n_dims     = ggml_n_dims(a);
    bool      same_shape = ggml_n_dims(b) == n_dims;
    for (int i = 0; same_shape && i < n_dims; i++) {
        same_shape = a->ne[i] == b->ne[i];
    }

    Value            aVal = valueOf(a);
    Value            bVal = valueOf(b);
    RankedTensorType ty   = tensorType(node);

    const bool isAdd = std::strcmp(which, "add") == 0;

    if (same_shape) {
        Value init = empty(ty);
        if (isAdd) {
            return linalg::AddOp::create(b_, loc_, TypeRange{ty}, ValueRange{aVal, bVal}, ValueRange{init})
                .getResult(0);
        }
        return linalg::MulOp::create(b_, loc_, TypeRange{ty}, ValueRange{aVal, bVal}, ValueRange{init}).getResult(0);
    }

    if (n_dims >= 2 && ggml_n_dims(b) == 1 && b->ne[0] == a->ne[0]) {
        AffineMap fullMap = mapFull(n_dims);
        AffineMap lastMap = mapSelect(n_dims, {n_dims - 1});
        Value     init    = empty(ty);
        return generic(ty, ValueRange{aVal, bVal}, init, {fullMap, lastMap, fullMap}, itersAllParallel(n_dims),
                       [&](OpBuilder & nb, Location nloc, ValueRange args) -> Value {
                           if (isAdd) {
                               return arith::AddFOp::create(nb, nloc, args[0], args[1]);
                           }
                           return arith::MulFOp::create(nb, nloc, args[0], args[1]);
                       });
    }

    unsupported("%s only supports equal-shape tensors, or a's innermost-dim broadcast against a matching 1D "
                "tensor, for now",
                ggmlName);
}

Value GraphBuilder::emitAdd(const ggml_tensor * node) {
    return emitElementwiseBinop(node, "add", "ADD");
}

Value GraphBuilder::emitMul(const ggml_tensor * node) {
    return emitElementwiseBinop(node, "mul", "MUL");
}

// out = in * s, s a compile-time constant from op_params[0]. Only plain ggml_scale: the bias at
// op_params[1] must be zero, since ggml_scale_bias's non-zero-bias form is not handled.
Value GraphBuilder::emitScale(const ggml_tensor * node) {
    const ggml_tensor * x = node->src[0];

    float s, bias;
    std::memcpy(&s, node->op_params, sizeof(float));
    std::memcpy(&bias, (const char *) node->op_params + sizeof(float), sizeof(float));
    if (bias != 0.0f) {
        unsupported("SCALE only supports zero bias for now");
    }

    Value            xVal   = valueOf(x);
    RankedTensorType ty     = tensorType(node);
    const int        n_dims = ggml_n_dims(node);
    AffineMap        idMap  = mapFull(n_dims);

    Value sVal = constF32(s);
    Value init = empty(ty);
    return generic(ty, ValueRange{xVal}, init, {idMap, idMap}, itersAllParallel(n_dims),
                   [&](OpBuilder & nb, Location nloc, ValueRange args) -> Value {
                       return arith::MulFOp::create(nb, nloc, args[0], sVal);
                   });
}

// SiLU(x) = x / (1 + exp(-x)), matching ggml_silu_f32 in src/ggml-cpu/vec.h exactly.
Value GraphBuilder::emitSilu(const ggml_tensor * node) {
    const ggml_tensor * x = node->src[0];

    Value            xVal   = valueOf(x);
    RankedTensorType ty     = tensorType(node);
    const int        n_dims = ggml_n_dims(node);
    AffineMap        idMap  = mapFull(n_dims);

    Value one  = constF32(1.0f);
    Value init = empty(ty);
    return generic(ty, ValueRange{xVal}, init, {idMap, idMap}, itersAllParallel(n_dims),
                   [&](OpBuilder & nb, Location nloc, ValueRange args) -> Value {
                       Value neg   = arith::NegFOp::create(nb, nloc, args[0]);
                       Value e     = math::ExpOp::create(nb, nloc, neg);
                       Value denom = arith::AddFOp::create(nb, nloc, one, e);
                       return arith::DivFOp::create(nb, nloc, args[0], denom);
                   });
}

}  // namespace tsi::mlir_export
