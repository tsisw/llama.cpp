// ggml.permute, ggml.reshape, ggml.cont, ggml.concat, ggml.get_rows -> linalg/tensor.
#include "GgmlDialect.h"
#include "GgmlToLinalg.h"
#include "PatternSupport.h"

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

using namespace mlir;

namespace tsi::mlir_export {

namespace {

// ggml ne[i] for a tensor whose MLIR type has `rank` dims. ggml dims beyond the effective rank are
// 1, exactly as ggml itself reports them.
int64_t ne(RankedTensorType t, int i) {
    const int rank = t.getRank();
    return i < rank ? t.getShape()[rank - 1 - i] : 1;
}

// MLIR dims at an explicit rank R >= the type's own: ggml's trailing size-1 dims become MLIR
// leading size-1 dims.
SmallVector<int64_t> dimsAtRank(RankedTensorType t, int R) {
    SmallVector<int64_t> s;
    for (int i = R - 1; i >= 0; i--) {
        s.push_back(ne(t, i));
    }
    return s;
}

SmallVector<OpFoldResult> idxAttrs(OpBuilder & b, ArrayRef<int64_t> vals) {
    SmallVector<OpFoldResult> r;
    for (int64_t v : vals) {
        r.push_back(b.getIndexAttr(v));
    }
    return r;
}

// Groups dims so each group holds exactly one non-1 dim: size-1 dims attach to the preceding
// non-1's group, leading 1s to the first. This is the reassociation collapse/expand needs to go
// between a shape and its dense (all-1s-removed) form.
SmallVector<ReassociationIndices> size1Reassoc(ArrayRef<int64_t> s) {
    SmallVector<ReassociationIndices> groups;
    bool                              hasNonOne = false;
    for (int i = 0; i < (int) s.size(); i++) {
        if (s[i] != 1 && hasNonOne) {
            groups.push_back({});
            hasNonOne = false;
        }
        if (groups.empty()) {
            groups.push_back({});
        }
        groups.back().push_back(i);
        if (s[i] != 1) {
            hasNonOne = true;
        }
    }
    return groups;
}

// A shape change that moves no data, only adds or removes size-1 dims.
Value size1Reshape(ConversionPatternRewriter & rw, Location loc, Value src, ArrayRef<int64_t> srcDims,
                   ArrayRef<int64_t> dstDims, Type elem) {
    SmallVector<int64_t> dense;
    for (int64_t d : srcDims) {
        if (d != 1) {
            dense.push_back(d);
        }
    }

    Value cur = src;
    if (srcDims.size() != dense.size()) {
        cur = tensor::CollapseShapeOp::create(rw, loc, ps::typeOf(dense, elem), cur, size1Reassoc(srcDims));
    }
    if (dstDims.size() != dense.size()) {
        cur = tensor::ExpandShapeOp::create(rw, loc, ps::typeOf(dstDims, elem), cur, size1Reassoc(dstDims));
    }
    return cur;
}

struct PermuteLowering : public OpConversionPattern<ggml::PermuteOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ggml::PermuteOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter & rw) const override {
        Value    x     = adaptor.getInput();
        auto     xTy   = llvm::cast<RankedTensorType>(x.getType());
        auto     outTy = llvm::cast<RankedTensorType>(op.getResult().getType());
        Location loc   = op.getLoc();

        llvm::ArrayRef<int32_t> axes = op.getAxes();

        // If the permutation only reshuffles size-1 dims (the relative order of the non-1 dims is
        // preserved) it moves no data, so emit a reshape rather than a rank-preserving transpose.
        bool size1Only = true;
        for (int i = 0; i < 4 && size1Only; i++) {
            for (int j = i + 1; j < 4; j++) {
                if (ne(xTy, i) > 1 && ne(xTy, j) > 1 && axes[i] > axes[j]) {
                    size1Only = false;
                    break;
                }
            }
        }
        if (size1Only) {
            SmallVector<int64_t> srcDims(xTy.getShape().begin(), xTy.getShape().end());
            SmallVector<int64_t> dstDims(outTy.getShape().begin(), outTy.getShape().end());
            rw.replaceOp(op, size1Reshape(rw, loc, x, srcDims, dstDims, xTy.getElementType()));
            return success();
        }

        const int R = xTy.getRank();
        if (R < 2 || R > 3) {
            return rw.notifyMatchFailure(op, "only rank 2 or 3 is supported");
        }
        if (outTy.getRank() != R) {
            return rw.notifyMatchFailure(op, "only rank-preserving permutations are supported");
        }
        for (int g = 0; g < R; g++) {
            if (axes[g] < 0 || axes[g] >= R) {
                return rw.notifyMatchFailure(op, "only permutations confined to the effective rank are supported "
                                                 "(no promoting or demoting trivial dims)");
            }
        }

        // ggml sets result.ne[axes[g]] = src.ne[g]; translate that into an MLIR dim permutation.
        SmallVector<int64_t> perm(R);
        for (int g = 0; g < R; g++) {
            perm[R - 1 - axes[g]] = R - 1 - g;
        }

        Value init = ps::empty(rw, loc, outTy);
        Value res  = linalg::TransposeOp::create(rw, loc, x, init, perm).getResult()[0];
        rw.replaceOp(op, res);
        return success();
    }
};

// Shared by reshape and cont. Because the exporter materializes every value eagerly, a CONT's
// source is already what ggml's own cont would produce, so only the shape change needs emitting.
template <typename OpT>
struct ReshapeLikeLowering : public OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;
    using Adaptor = typename OpT::Adaptor;

    LogicalResult matchAndRewrite(OpT op, Adaptor adaptor, ConversionPatternRewriter & rw) const override {
        Value    x     = adaptor.getInput();
        auto     xTy   = llvm::cast<RankedTensorType>(x.getType());
        auto     outTy = llvm::cast<RankedTensorType>(op.getResult().getType());
        Location loc   = op.getLoc();

        if (xTy.getShape() == outTy.getShape()) {
            rw.replaceOp(op, x);   // plain ggml_cont() with no reshape: nothing to emit
            return success();
        }

        const int xr = xTy.getRank();
        const int or_ = outTy.getRank();

        // 2D -> 3D head split, the shape multi-head attention needs before per-head ops.
        if (xr == 2 && or_ == 3 && ne(outTy, 2) == ne(xTy, 1) && ne(outTy, 0) * ne(outTy, 1) == ne(xTy, 0)) {
            SmallVector<ReassociationIndices> re = {{0}, {1, 2}};
            rw.replaceOp(op, tensor::ExpandShapeOp::create(rw, loc, outTy, x, re));
            return success();
        }
        // 3D -> 2D head merge.
        if (xr == 3 && or_ == 2 && ne(xTy, 2) == ne(outTy, 1) && ne(xTy, 0) * ne(xTy, 1) == ne(outTy, 0)) {
            SmallVector<ReassociationIndices> re = {{0}, {1, 2}};
            rw.replaceOp(op, tensor::CollapseShapeOp::create(rw, loc, outTy, x, re));
            return success();
        }
        // n_tokens=1: 1D hidden -> 2D heads.
        if (xr == 1 && or_ == 2 && ne(outTy, 0) * ne(outTy, 1) == ne(xTy, 0)) {
            SmallVector<ReassociationIndices> re = {{0, 1}};
            rw.replaceOp(op, tensor::ExpandShapeOp::create(rw, loc, outTy, x, re));
            return success();
        }
        // n_tokens=1: 2D heads -> 1D hidden.
        if (xr == 2 && or_ == 1 && ne(xTy, 0) * ne(xTy, 1) == ne(outTy, 0)) {
            SmallVector<ReassociationIndices> re = {{0, 1}};
            rw.replaceOp(op, tensor::CollapseShapeOp::create(rw, loc, outTy, x, re));
            return success();
        }

        return rw.notifyMatchFailure(op, "only a same-shape passthrough, a 2D<->3D head split/merge, or a "
                                         "1D<->2D head split/merge are supported");
    }
};

// Built from empty + insert_slice rather than tensor.concat, which the TSI pipeline does not
// bufferize. `lhs` fills [0, lhs.dim) along the concat dim, `rhs` the remainder.
struct ConcatLowering : public OpConversionPattern<ggml::ConcatOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ggml::ConcatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter & rw) const override {
        auto     aTy   = llvm::cast<RankedTensorType>(adaptor.getLhs().getType());
        auto     bTy   = llvm::cast<RankedTensorType>(adaptor.getRhs().getType());
        auto     outTy = llvm::cast<RankedTensorType>(op.getResult().getType());
        Location loc   = op.getLoc();

        const int     R   = outTy.getRank();
        const int32_t dim = op.getDim();
        if (dim < 0 || dim >= R) {
            return rw.notifyMatchFailure(op, "concat dim is outside the result rank");
        }
        Type elem = outTy.getElementType();

        // An operand may report fewer ggml dims than the result (new K/V [.,.,1] vs a cache
        // [.,.,cur]); bring it up to rank R first.
        auto toRankR = [&](Value v, RankedTensorType ty) -> Value {
            if (ty.getRank() == R) {
                return v;
            }
            const int                         r = ty.getRank();
            SmallVector<ReassociationIndices> re;
            ReassociationIndices              lead;
            for (int i = 0; i <= R - r; i++) {
                lead.push_back(i);
            }
            re.push_back(lead);
            for (int i = R - r + 1; i < R; i++) {
                re.push_back({i});
            }
            return tensor::ExpandShapeOp::create(rw, loc, ps::typeOf(dimsAtRank(ty, R), elem), v, re);
        };

        Value av = toRankR(adaptor.getLhs(), aTy);
        Value bv = toRankR(adaptor.getRhs(), bTy);

        const int            md = R - 1 - dim;   // ggml dim -> MLIR dim
        SmallVector<int64_t> zeros(R, 0), ones(R, 1), boff(R, 0);
        boff[md] = ne(aTy, dim);

        SmallVector<OpFoldResult> strides = idxAttrs(rw, ones);
        Value                     init    = ps::empty(rw, loc, outTy);
        Value r0 = tensor::InsertSliceOp::create(rw, loc, av, init, idxAttrs(rw, zeros),
                                                 idxAttrs(rw, dimsAtRank(aTy, R)), strides);
        Value r1 = tensor::InsertSliceOp::create(rw, loc, bv, r0, idxAttrs(rw, boff),
                                                 idxAttrs(rw, dimsAtRank(bTy, R)), strides);
        rw.replaceOp(op, r1);
        return success();
    }
};

// Token embedding lookup. Unlike every other slice here, the indices are genuine runtime data, so
// each gathered row needs a DYNAMIC-offset extract_slice. The token count is still compile-time
// known, so this unrolls one extract/insert pair per token.
struct GetRowsLowering : public OpConversionPattern<ggml::GetRowsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ggml::GetRowsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter & rw) const override {
        Value    tbl   = adaptor.getTable();
        Value    ids   = adaptor.getIds();
        auto     tblTy = llvm::cast<RankedTensorType>(tbl.getType());
        auto     idsTy = llvm::cast<RankedTensorType>(ids.getType());
        auto     outTy = llvm::cast<RankedTensorType>(op.getResult().getType());
        Location loc   = op.getLoc();

        if (tblTy.getRank() != 2 || idsTy.getRank() != 1) {
            return rw.notifyMatchFailure(op, "only a 2D embedding table and a 1D token-id vector are supported");
        }

        const int64_t nEmbd   = tblTy.getShape()[1];
        const int64_t nTokens = idsTy.getShape()[0];
        Type          elem    = tblTy.getElementType();

        SmallVector<OpFoldResult> sizes   = idxAttrs(rw, {1, nEmbd});
        SmallVector<OpFoldResult> strides = idxAttrs(rw, {1, 1});

        auto tokenIndex = [&](int64_t t) -> Value {
            Value ct   = ps::constIndex(rw, loc, t);
            Value i32v = tensor::ExtractOp::create(rw, loc, ids, ValueRange{ct});
            return arith::IndexCastOp::create(rw, loc, rw.getIndexType(), i32v);
        };

        // n_tokens=1 (decode): the result collapses to rank 1, so a rank-reducing slice drops the
        // unit dim directly and no insert loop is needed.
        if (nTokens == 1 && outTy.getRank() == 1) {
            Value                     idx     = tokenIndex(0);
            SmallVector<OpFoldResult> offsets = {idx, rw.getIndexAttr(0)};
            rw.replaceOp(op, tensor::ExtractSliceOp::create(rw, loc, outTy, tbl, offsets, sizes, strides));
            return success();
        }

        auto  rowTy  = ps::typeOf({1, nEmbd}, elem);
        Value current = ps::empty(rw, loc, outTy);
        for (int64_t t = 0; t < nTokens; t++) {
            Value                     idx     = tokenIndex(t);
            SmallVector<OpFoldResult> offsets = {idx, rw.getIndexAttr(0)};
            Value row = tensor::ExtractSliceOp::create(rw, loc, rowTy, tbl, offsets, sizes, strides);
            current   = tensor::InsertSliceOp::create(rw, loc, row, current, idxAttrs(rw, {t, 0}), sizes, strides);
        }
        rw.replaceOp(op, current);
        return success();
    }
};

}  // namespace

void populateShapePatterns(RewritePatternSet & patterns) {
    patterns.add<PermuteLowering, ReshapeLikeLowering<ggml::ReshapeOp>, ReshapeLikeLowering<ggml::ContOp>,
                 ConcatLowering, GetRowsLowering>(patterns.getContext());
}

}  // namespace tsi::mlir_export
