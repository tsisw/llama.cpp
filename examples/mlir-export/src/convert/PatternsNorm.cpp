// ggml.rms_norm and ggml.soft_max -> linalg.
//
// Both reduce over the innermost dim (ggml's ne[0]), treating every outer dim as an independent
// batch dim, and both are decomposed into a chain of linalg.generic ops: one reduction, a pointwise
// stage on the reduced vector, then a broadcast back over the full shape.
#include "GgmlDialect.h"
#include "GgmlToLinalg.h"
#include "PatternSupport.h"

#include <limits>

using namespace mlir;

namespace tsi::mlir_export {

namespace {

// ggml_rms_norm(x, eps) = x / sqrt(mean(x^2 over ne[0]) + eps), per row. Three stages:
// sum-of-squares per row, a pointwise mean+eps+rsqrt giving a per-row scale, then a
// broadcast-multiply. A rank-1 input reduces to a rank-0 scalar, which is valid.
struct RmsNormLowering : public OpConversionPattern<ggml::RmsNormOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ggml::RmsNormOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter & rw) const override {
        Value         x    = adaptor.getInput();
        auto          xTy  = llvm::cast<RankedTensorType>(x.getType());
        auto          outTy = llvm::cast<RankedTensorType>(op.getResult().getType());
        Location      loc  = op.getLoc();
        MLIRContext * ctx  = rw.getContext();

        const int     rank = xTy.getRank();
        const int64_t cols = xTy.getShape().back();
        auto          vecTy = ps::reducedType(xTy);

        AffineMap full  = ps::mapFull(ctx, rank);
        AffineMap drop  = ps::mapDropLast(ctx, rank);
        AffineMap vecId = ps::mapFull(ctx, rank - 1);

        Value zero = ps::constF32(rw, loc, 0.0f);
        Value ninv = ps::constF32(rw, loc, 1.0f / (float) cols);
        Value eps  = ps::constF32(rw, loc, op.getEps().convertToFloat());

        // 1. sum of squares per row, reducing the innermost dim.
        Value sumFilled = ps::fillWith(rw, loc, zero, vecTy);
        Value sumsq     = ps::generic(rw, loc, vecTy, ValueRange{x}, sumFilled, {full, drop},
                                      ps::itersReduceLast(rank),
                                      [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                          Value sq = arith::MulFOp::create(nb, nloc, a[0], a[0]);
                                          return arith::AddFOp::create(nb, nloc, a[1], sq);
                                      });

        // 2. per-row scale = rsqrt(mean + eps).
        Value scaleInit = ps::empty(rw, loc, vecTy);
        Value scale     = ps::generic(rw, loc, vecTy, ValueRange{sumsq}, scaleInit, {vecId, vecId},
                                      ps::itersAllParallel(rank - 1),
                                      [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                          Value mean    = arith::MulFOp::create(nb, nloc, a[0], ninv);
                                          Value meaneps = arith::AddFOp::create(nb, nloc, mean, eps);
                                          return math::RsqrtOp::create(nb, nloc, meaneps);
                                      });

        // 3. broadcast-multiply: out[..., c] = x[..., c] * scale[...].
        Value outInit = ps::empty(rw, loc, outTy);
        Value res     = ps::generic(rw, loc, outTy, ValueRange{x, scale}, outInit, {full, drop, full},
                                    ps::itersAllParallel(rank),
                                    [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                        return arith::MulFOp::create(nb, nloc, a[0], a[1]);
                                    });

        rw.replaceOp(op, res);
        return success();
    }
};

// ggml_soft_max_ext(x, mask, scale, max_bias) = softmax(x*scale + mask) over ne[0].
//
// An optional stage 0 applies the scale and mask, then the same four-stage reduction runs: row max,
// broadcast-subtract and exp, row sum, broadcast-divide. When scale is 1 and there is no mask,
// stage 0 is skipped entirely rather than emitting a multiply by 1.
struct SoftMaxLowering : public OpConversionPattern<ggml::SoftMaxOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ggml::SoftMaxOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter & rw) const override {
        const float scale    = op.getScale().convertToFloat();
        const float max_bias = op.getMaxBias().convertToFloat();
        if (max_bias != 0.0f) {
            return rw.notifyMatchFailure(op, "only max_bias=0 (no ALiBi) is supported");
        }

        Value         x    = adaptor.getInput();
        Value         mask = adaptor.getMask();
        auto          xTy  = llvm::cast<RankedTensorType>(x.getType());
        auto          outTy = llvm::cast<RankedTensorType>(op.getResult().getType());
        Location      loc  = op.getLoc();
        MLIRContext * ctx  = rw.getContext();
        const int     rank = xTy.getRank();

        if (rank < 2) {
            return rw.notifyMatchFailure(op, "requires at least 2 dims");
        }

        // A mask is only handled for rank-3 x: a rank-2 [n_q, n_kv] mask broadcast identically over
        // the head dim, or a rank-1 [n_kv] mask when n_q is 1 (the decode shape), broadcast over
        // both the head and query dims.
        int maskRank = 0;
        if (mask) {
            auto mTy = llvm::cast<RankedTensorType>(mask.getType());
            maskRank = mTy.getRank();
            const bool ok = rank == 3 && ((maskRank == 2 && mTy.getShape()[0] == xTy.getShape()[1]) ||
                                          (maskRank == 1 && xTy.getShape()[1] == 1));
            if (!ok) {
                return rw.notifyMatchFailure(op, "a mask is only supported for rank-3 x with a matching rank-2 "
                                                 "mask, or a rank-1 [n_kv] mask when n_q is 1");
            }
        }

        auto      vecTy = ps::reducedType(xTy);
        AffineMap full  = ps::mapFull(ctx, rank);
        AffineMap drop  = ps::mapDropLast(ctx, rank);

        Value zero = ps::constF32(rw, loc, 0.0f);

        // stage 0: combined = x*scale [+ mask].
        Value combined = x;
        if (mask || scale != 1.0f) {
            Value scaleVal = ps::constF32(rw, loc, scale);
            Value combInit = ps::empty(rw, loc, xTy);
            if (mask) {
                AffineMap maskMap = maskRank == 1 ? ps::mapSelect(ctx, 3, {2}) : ps::mapSelect(ctx, 3, {1, 2});
                combined = ps::generic(rw, loc, xTy, ValueRange{x, mask}, combInit, {full, maskMap, full},
                                       ps::itersAllParallel(rank),
                                       [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                           Value scaled = arith::MulFOp::create(nb, nloc, a[0], scaleVal);
                                           return arith::AddFOp::create(nb, nloc, scaled, a[1]);
                                       });
            } else {
                combined = ps::generic(rw, loc, xTy, ValueRange{x}, combInit, {full, full},
                                       ps::itersAllParallel(rank),
                                       [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                           return arith::MulFOp::create(nb, nloc, a[0], scaleVal);
                                       });
            }
        }

        // 1. row max. Seeded with -inf, which MLIR prints as the 0xFF800000 bit pattern.
        Value neginf   = ps::constF32(rw, loc, -std::numeric_limits<float>::infinity());
        Value maxFilled = ps::fillWith(rw, loc, neginf, vecTy);
        Value rowmax   = ps::generic(rw, loc, vecTy, ValueRange{combined}, maxFilled, {full, drop},
                                     ps::itersReduceLast(rank),
                                     [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                         return arith::MaximumFOp::create(nb, nloc, a[0], a[1]);
                                     });

        // 2. exp(x - rowmax), broadcasting rowmax over the innermost dim.
        Value expInit = ps::empty(rw, loc, xTy);
        Value expx    = ps::generic(rw, loc, xTy, ValueRange{combined, rowmax}, expInit, {full, drop, full},
                                    ps::itersAllParallel(rank),
                                    [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                        Value sub = arith::SubFOp::create(nb, nloc, a[0], a[1]);
                                        return math::ExpOp::create(nb, nloc, sub);
                                    });

        // 3. row sum.
        Value sumFilled = ps::fillWith(rw, loc, zero, vecTy);
        Value rowsum    = ps::generic(rw, loc, vecTy, ValueRange{expx}, sumFilled, {full, drop},
                                      ps::itersReduceLast(rank),
                                      [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                          return arith::AddFOp::create(nb, nloc, a[0], a[1]);
                                      });

        // 4. divide by the row sum, broadcasting over the innermost dim.
        Value outInit = ps::empty(rw, loc, outTy);
        Value res     = ps::generic(rw, loc, outTy, ValueRange{expx, rowsum}, outInit, {full, drop, full},
                                    ps::itersAllParallel(rank),
                                    [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                        return arith::DivFOp::create(nb, nloc, a[0], a[1]);
                                    });

        rw.replaceOp(op, res);
        return success();
    }
};

}  // namespace

void populateNormPatterns(RewritePatternSet & patterns) {
    patterns.add<RmsNormLowering, SoftMaxLowering>(patterns.getContext());
}

}  // namespace tsi::mlir_export
