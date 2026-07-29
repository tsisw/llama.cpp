// ggml.add, ggml.mul, ggml.scale, ggml.silu -> linalg.
#include "Builder.h"
#include "GgmlDialect.h"
#include "GgmlToLinalg.h"
#include "PatternSupport.h"

using namespace mlir;

namespace tsi::mlir_export {

namespace {

// Shared by add and mul. Two shapes are handled: identical shapes at any rank (linalg.add and
// linalg.mul are named ops that infer identity indexing maps from the operand shapes, so no explicit
// affine maps are needed), or lhs's innermost dim broadcast against a rank-1 rhs matching that dim -
// the real rms_norm(x)*weight pattern, matching ggml's ggml_can_repeat rule when rhs's outer dims
// are all 1. Other broadcast shapes are a match failure, which is our limitation and so belongs
// here rather than in the dialect verifier.
template <typename GgmlOp, typename LinalgOp, bool IsAdd>
struct BinaryLowering : public OpConversionPattern<GgmlOp> {
    using OpConversionPattern<GgmlOp>::OpConversionPattern;
    using Adaptor = typename GgmlOp::Adaptor;

    LogicalResult matchAndRewrite(GgmlOp op, Adaptor adaptor, ConversionPatternRewriter & rw) const override {
        auto lhsTy = llvm::cast<RankedTensorType>(adaptor.getLhs().getType());
        auto rhsTy = llvm::cast<RankedTensorType>(adaptor.getRhs().getType());
        auto ty    = llvm::cast<RankedTensorType>(op.getResult().getType());
        Location loc = op.getLoc();

        if (lhsTy.getShape() == rhsTy.getShape()) {
            Value init = ps::empty(rw, loc, ty);
            auto  out  = LinalgOp::create(rw, loc, TypeRange{ty}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                                          ValueRange{init});
            rw.replaceOp(op, out.getResult(0));
            return success();
        }

        const int rank = lhsTy.getRank();
        if (rank >= 2 && rhsTy.getRank() == 1 && rhsTy.getShape()[0] == lhsTy.getShape()[rank - 1]) {
            AffineMap full = ps::mapFull(rw.getContext(), rank);
            AffineMap last = ps::mapSelect(rw.getContext(), rank, {rank - 1});
            Value     init = ps::empty(rw, loc, ty);
            Value     res  = ps::generic(
                rw, loc, ty, ValueRange{adaptor.getLhs(), adaptor.getRhs()}, init, {full, last, full},
                ps::itersAllParallel(rank), [&](OpBuilder & nb, Location nloc, ValueRange args) -> Value {
                    if (IsAdd) {
                        return arith::AddFOp::create(nb, nloc, args[0], args[1]);
                    }
                    return arith::MulFOp::create(nb, nloc, args[0], args[1]);
                });
            rw.replaceOp(op, res);
            return success();
        }

        return rw.notifyMatchFailure(op, "only equal shapes, or an innermost-dim broadcast against a matching "
                                         "rank-1 operand, are supported");
    }
};

using AddLowering = BinaryLowering<ggml::AddOp, linalg::AddOp, true>;
using MulLowering = BinaryLowering<ggml::MulOp, linalg::MulOp, false>;

// out = in * scale. ggml_scale always sets bias to 0; ggml_scale_bias does not, and that form is
// not implemented.
struct ScaleLowering : public OpConversionPattern<ggml::ScaleOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ggml::ScaleOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter & rw) const override {
        if (op.getBias().convertToFloat() != 0.0f) {
            return rw.notifyMatchFailure(op, "only zero bias is supported");
        }
        auto     ty   = llvm::cast<RankedTensorType>(op.getResult().getType());
        Location loc  = op.getLoc();
        const int rank = ty.getRank();
        AffineMap id  = ps::mapFull(rw.getContext(), rank);

        Value s    = ps::constF32(rw, loc, op.getScale().convertToFloat());
        Value init = ps::empty(rw, loc, ty);
        Value res  = ps::generic(rw, loc, ty, ValueRange{adaptor.getInput()}, init, {id, id},
                                 ps::itersAllParallel(rank),
                                 [&](OpBuilder & nb, Location nloc, ValueRange args) -> Value {
                                     return arith::MulFOp::create(nb, nloc, args[0], s);
                                 });
        rw.replaceOp(op, res);
        return success();
    }
};

// SiLU(x) = x / (1 + exp(-x)), matching ggml_silu_f32 in src/ggml-cpu/vec.h exactly.
struct SiluLowering : public OpConversionPattern<ggml::SiluOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ggml::SiluOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter & rw) const override {
        auto      ty   = llvm::cast<RankedTensorType>(op.getResult().getType());
        Location  loc  = op.getLoc();
        const int rank = ty.getRank();
        AffineMap id   = ps::mapFull(rw.getContext(), rank);

        Value one  = ps::constF32(rw, loc, 1.0f);
        Value init = ps::empty(rw, loc, ty);
        Value res  = ps::generic(rw, loc, ty, ValueRange{adaptor.getInput()}, init, {id, id},
                                 ps::itersAllParallel(rank),
                                 [&](OpBuilder & nb, Location nloc, ValueRange args) -> Value {
                                     Value neg   = arith::NegFOp::create(nb, nloc, args[0]);
                                     Value e     = math::ExpOp::create(nb, nloc, neg);
                                     Value denom = arith::AddFOp::create(nb, nloc, one, e);
                                     return arith::DivFOp::create(nb, nloc, args[0], denom);
                                 });
        rw.replaceOp(op, res);
        return success();
    }
};

}  // namespace

void populateElementwisePatterns(RewritePatternSet & patterns) {
    patterns.add<AddLowering, MulLowering, ScaleLowering, SiluLowering>(patterns.getContext());
}

}  // namespace tsi::mlir_export
