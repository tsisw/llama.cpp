// ggml.rope -> linalg.
//
// GGML_ROPE_TYPE_NORMAL rotates interleaved pairs (2k, 2k+1) of each row by
// theta_k = pos * freq_base^(-2k/n_dims), matching LLAMA_ROPE_TYPE_NORM. (That only agrees with the
// HF Llama model's rotate_half because convert_hf_to_gguf.py permutes the Q/K weights so this
// simpler interleaved-pairs rotation can be used.)
//
// Note what is NOT checked here: the position count. ggml requires pos->ne[0] == x->ne[2], and that
// is now a RopeOp dialect verifier, so by the time this pattern runs it holds. Only limits that are
// ours (mode, partial rotation, YaRN scaling, rank) are match failures.
#include "GgmlDialect.h"
#include "GgmlToLinalg.h"
#include "PatternSupport.h"
#include "tsi/export/Exporter.h"

#include <cmath>

using namespace mlir;

namespace tsi::mlir_export {

namespace {

SmallVector<OpFoldResult> idxAttrs(OpBuilder & b, ArrayRef<int64_t> vals) {
    SmallVector<OpFoldResult> r;
    for (int64_t v : vals) {
        r.push_back(b.getIndexAttr(v));
    }
    return r;
}

// freq_k = freq_base^(-2k/n_dims) for k in [0, half), a compile-time constant vector.
Value freqConstant(OpBuilder & b, Location loc, int64_t half, int32_t rotDims, float freqBase,
                   RankedTensorType freqTy) {
    SmallVector<float> freqs;
    freqs.reserve(half);
    for (int64_t k = 0; k < half; k++) {
        freqs.push_back(powf(freqBase, -2.0f * (float) k / (float) rotDims));
    }
    return ps::denseF32(b, loc, freqs, freqTy);
}

// The shared rotation: out_even = xe*cos - xo*sin, out_odd = xe*sin + xo*cos, as one two-result
// linalg.generic. `maps` supplies the per-operand indexing so the same body serves both ranks.
std::pair<Value, Value> rotate(OpBuilder & b, Location loc, RankedTensorType pairTy, Value xEven, Value xOdd,
                               Value cosT, Value sinT, ArrayRef<AffineMap> maps,
                               ArrayRef<utils::IteratorType> iters) {
    Value evenInit = ps::empty(b, loc, pairTy);
    Value oddInit  = ps::empty(b, loc, pairTy);
    auto  op       = linalg::GenericOp::create(
        b, loc, TypeRange{pairTy, pairTy}, ValueRange{xEven, xOdd, cosT, sinT}, ValueRange{evenInit, oddInit},
        maps, iters, [&](OpBuilder & nb, Location nloc, ValueRange a) {
            Value e1 = arith::MulFOp::create(nb, nloc, a[0], a[2]);
            Value e2 = arith::MulFOp::create(nb, nloc, a[1], a[3]);
            Value ne = arith::SubFOp::create(nb, nloc, e1, e2);
            Value o1 = arith::MulFOp::create(nb, nloc, a[0], a[3]);
            Value o2 = arith::MulFOp::create(nb, nloc, a[1], a[2]);
            Value no = arith::AddFOp::create(nb, nloc, o1, o2);
            linalg::YieldOp::create(nb, nloc, ValueRange{ne, no});
        });
    return {op.getResult(0), op.getResult(1)};
}

// Elementwise unary over a tensor, used for cos and sin.
template <typename MathOp>
Value unary(OpBuilder & b, Location loc, Value in, RankedTensorType ty, int rank) {
    AffineMap id   = ps::mapFull(b.getContext(), rank);
    Value     init = ps::empty(b, loc, ty);
    return ps::generic(b, loc, ty, ValueRange{in}, init, {id, id}, ps::itersAllParallel(rank),
                       [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                           return MathOp::create(nb, nloc, a[0]);
                       });
}

struct RopeLowering : public OpConversionPattern<ggml::RopeOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ggml::RopeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter & rw) const override {
        if (op.getMode() != GGML_ROPE_TYPE_NORMAL) {
            return rw.notifyMatchFailure(op, "only GGML_ROPE_TYPE_NORMAL is supported");
        }
        if (op.getFreqScale().convertToFloat() != 1.0f || op.getExtFactor().convertToFloat() != 0.0f ||
            op.getAttnFactor().convertToFloat() != 1.0f) {
            return rw.notifyMatchFailure(op, "only freq_scale=1, ext_factor=0, attn_factor=1 are supported");
        }

        Value x   = adaptor.getInput();
        auto  xTy = llvm::cast<RankedTensorType>(x.getType());

        const int32_t rotDims = op.getNDims();
        if (rotDims != xTy.getShape().back() || rotDims % 2 != 0) {
            return rw.notifyMatchFailure(op, "only full-row rotation (n_dims == ne[0], even) is supported");
        }

        const int rank = xTy.getRank();
        if (rank == 2) {
            return lowerRank2(op, adaptor, rw, xTy, rotDims);
        }
        if (rank == 3) {
            return lowerRank3(op, adaptor, rw, xTy, rotDims);
        }
        return rw.notifyMatchFailure(op, "only rank-2 or rank-3 input is supported");
    }

  private:
    // rank-2 x = (head_dim, n_head) in ggml order -> MLIR (rows, D). ggml's ne[2] is implicitly 1,
    // so there is exactly one position, shared by every row.
    LogicalResult lowerRank2(ggml::RopeOp op, OpAdaptor adaptor, ConversionPatternRewriter & rw,
                             RankedTensorType xTy, int32_t rotDims) const {
        Location      loc  = op.getLoc();
        MLIRContext * ctx  = rw.getContext();
        auto          outTy = llvm::cast<RankedTensorType>(op.getResult().getType());
        Type          elem = xTy.getElementType();

        const int64_t rows = xTy.getShape()[0];
        const int64_t half = rotDims / 2;
        auto          halfTy = ps::typeOf({half}, elem);
        auto          pairTy = ps::typeOf({rows, half}, elem);

        Value freq = freqConstant(rw, loc, half, rotDims, op.getFreqBase().convertToFloat(), halfTy);

        // theta = freq * pos, with pos broadcast via linalg.fill + linalg.mul rather than captured
        // as a scalar inside a generic body or splatted: the scalar-capture form fails to legalize
        // in the txe-to-LLVM stage and tensor.splat fails to bufferize, while fill and mul are
        // already proven through the full pipeline.
        Value c0     = ps::constIndex(rw, loc, 0);
        Value posI32 = tensor::ExtractOp::create(rw, loc, adaptor.getPositions(), ValueRange{c0});
        Value posF32 = arith::SIToFPOp::create(rw, loc, rw.getF32Type(), posI32);
        Value posB   = ps::fillWith(rw, loc, posF32, halfTy);
        Value thInit = ps::empty(rw, loc, halfTy);
        Value theta  = linalg::MulOp::create(rw, loc, TypeRange{halfTy}, ValueRange{freq, posB},
                                             ValueRange{thInit}).getResult(0);

        Value cosT = unary<math::CosOp>(rw, loc, theta, halfTy, 1);
        Value sinT = unary<math::SinOp>(rw, loc, theta, halfTy, 1);

        SmallVector<OpFoldResult> sizes   = idxAttrs(rw, {rows, half});
        SmallVector<OpFoldResult> strides = idxAttrs(rw, {1, 2});
        Value xEven = tensor::ExtractSliceOp::create(rw, loc, pairTy, adaptor.getInput(), idxAttrs(rw, {0, 0}),
                                                    sizes, strides);
        Value xOdd  = tensor::ExtractSliceOp::create(rw, loc, pairTy, adaptor.getInput(), idxAttrs(rw, {0, 1}),
                                                    sizes, strides);

        AffineMap full = ps::mapFull(ctx, 2);
        AffineMap bc   = ps::mapSelect(ctx, 2, {1});   // cos/sin depend only on the head_dim pair index
        auto [outEven, outOdd] =
            rotate(rw, loc, pairTy, xEven, xOdd, cosT, sinT, {full, full, bc, bc, full, full},
                   ps::itersAllParallel(2));

        Value out = ps::empty(rw, loc, outTy);
        out = tensor::InsertSliceOp::create(rw, loc, outEven, out, idxAttrs(rw, {0, 0}), sizes, strides);
        out = tensor::InsertSliceOp::create(rw, loc, outOdd, out, idxAttrs(rw, {0, 1}), sizes, strides);
        rw.replaceOp(op, out);
        return success();
    }

    // rank-3 x = (head_dim, n_head, n_tokens) in ggml order -> MLIR (T, H, D). Position varies along
    // T (the outermost MLIR dim); H is a pure broadcast dim with no position dependence.
    LogicalResult lowerRank3(ggml::RopeOp op, OpAdaptor adaptor, ConversionPatternRewriter & rw,
                             RankedTensorType xTy, int32_t rotDims) const {
        Location      loc   = op.getLoc();
        MLIRContext * ctx   = rw.getContext();
        auto          outTy = llvm::cast<RankedTensorType>(op.getResult().getType());
        Type          elem  = xTy.getElementType();

        const int64_t T    = xTy.getShape()[0];
        const int64_t H    = xTy.getShape()[1];
        const int64_t half = rotDims / 2;

        auto posF32Ty = ps::typeOf({T}, elem);
        auto freqTy   = ps::typeOf({half}, elem);
        auto thetaTy  = ps::typeOf({T, half}, elem);
        auto pairTy   = ps::typeOf({T, H, half}, elem);

        Value freq = freqConstant(rw, loc, half, rotDims, op.getFreqBase().convertToFloat(), freqTy);

        // pos_f32[t] = sitofp(pos[t]), elementwise over the whole vector so it stays a real tensor
        // operand rather than a captured scalar (see the rank-2 note on why that matters).
        AffineMap id1     = ps::mapFull(ctx, 1);
        Value     pfInit  = ps::empty(rw, loc, posF32Ty);
        Value     posF32  = ps::generic(rw, loc, posF32Ty, ValueRange{adaptor.getPositions()}, pfInit, {id1, id1},
                                        ps::itersAllParallel(1),
                                        [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                            return arith::SIToFPOp::create(nb, nloc, nb.getF32Type(), a[0]);
                                        });

        // theta[t,k] = pos_f32[t] * freq[k], an outer product expressed by per-operand affine maps.
        Value     thInit = ps::empty(rw, loc, thetaTy);
        AffineMap mT     = ps::mapSelect(ctx, 2, {0});
        AffineMap mK     = ps::mapSelect(ctx, 2, {1});
        AffineMap full2  = ps::mapFull(ctx, 2);
        Value     theta  = ps::generic(rw, loc, thetaTy, ValueRange{posF32, freq}, thInit, {mT, mK, full2},
                                       ps::itersAllParallel(2),
                                       [&](OpBuilder & nb, Location nloc, ValueRange a) -> Value {
                                           return arith::MulFOp::create(nb, nloc, a[0], a[1]);
                                       });

        Value cosT = unary<math::CosOp>(rw, loc, theta, thetaTy, 2);
        Value sinT = unary<math::SinOp>(rw, loc, theta, thetaTy, 2);

        SmallVector<OpFoldResult> sizes   = idxAttrs(rw, {T, H, half});
        SmallVector<OpFoldResult> strides = idxAttrs(rw, {1, 1, 2});
        Value xEven = tensor::ExtractSliceOp::create(rw, loc, pairTy, adaptor.getInput(), idxAttrs(rw, {0, 0, 0}),
                                                    sizes, strides);
        Value xOdd  = tensor::ExtractSliceOp::create(rw, loc, pairTy, adaptor.getInput(), idxAttrs(rw, {0, 0, 1}),
                                                    sizes, strides);

        AffineMap full3 = ps::mapFull(ctx, 3);
        AffineMap bc    = ps::mapSelect(ctx, 3, {0, 2});   // cos/sin broadcast over the head dim
        auto [outEven, outOdd] =
            rotate(rw, loc, pairTy, xEven, xOdd, cosT, sinT, {full3, full3, bc, bc, full3, full3},
                   ps::itersAllParallel(3));

        Value out = ps::empty(rw, loc, outTy);
        out = tensor::InsertSliceOp::create(rw, loc, outEven, out, idxAttrs(rw, {0, 0, 0}), sizes, strides);
        out = tensor::InsertSliceOp::create(rw, loc, outOdd, out, idxAttrs(rw, {0, 0, 1}), sizes, strides);
        rw.replaceOp(op, out);
        return success();
    }
};

}  // namespace

void populateRopePatterns(RewritePatternSet & patterns) {
    patterns.add<RopeLowering>(patterns.getContext());
}

}  // namespace tsi::mlir_export
