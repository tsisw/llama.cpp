// ggml.mul_mat -> linalg.
//
// ggml_mul_mat(A, B) computes A * B^T and stores the raw result with ne[0] = A.ne[1],
// ne[1] = B.ne[1], i.e. read with the usual ne[0]=cols convention it is the transpose of the
// intuitive result. To keep every tensor in the exported graph on one convention, we transpose A
// (not B) and emit matmul(B, transpose(A)).
#include "GgmlDialect.h"
#include "GgmlToLinalg.h"
#include "PatternSupport.h"

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

using namespace mlir;

namespace tsi::mlir_export {

namespace {

// Static [offsets] [sizes] [strides] for a slice, as OpFoldResults.
SmallVector<OpFoldResult> idxAttrs(OpBuilder & b, ArrayRef<int64_t> vals) {
    SmallVector<OpFoldResult> r;
    for (int64_t v : vals) {
        r.push_back(b.getIndexAttr(v));
    }
    return r;
}

struct MulMatLowering : public OpConversionPattern<ggml::MulMatOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ggml::MulMatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter & rw) const override {
        Value a = adaptor.getLhs();
        Value b = adaptor.getRhs();
        auto  aTy = llvm::cast<RankedTensorType>(a.getType());
        auto  bTy = llvm::cast<RankedTensorType>(b.getType());
        auto  outTy = llvm::cast<RankedTensorType>(op.getResult().getType());

        const int ar = aTy.getRank();
        const int br = bTy.getRank();

        if (ar == 2 && br == 2) {
            rw.replaceOp(op, lower2d(rw, op.getLoc(), a, b, aTy, outTy));
            return success();
        }
        if (ar == 2 && br == 1) {
            // n_tokens=1 decode: b is ggml [k,1] collapsed to rank 1, and the result to rank 1.
            rw.replaceOp(op, lower2dVec(rw, op.getLoc(), a, b, aTy, bTy, outTy));
            return success();
        }
        if (ar == 3 && br == 3) {
            const int64_t aH = aTy.getShape()[0];
            const int64_t bH = bTy.getShape()[0];
            if (aH == bH) {
                rw.replaceOp(op, lowerBatchedCore(rw, op.getLoc(), a, aTy.getShape()[2], aTy.getShape()[1], aH, b,
                                                  bTy, outTy));
                return success();
            }
            if (aH != 0 && bH % aH == 0) {
                // Real attention's Q/KV head mismatch: repeat a's heads to match b's first, then run
                // the identical batched core. Matches ggml-cpu's own grouping (contiguous blocks of
                // bH/aH sharing one a-head).
                Value repeated = repeatHeads(rw, op.getLoc(), a, aTy, bH);
                rw.replaceOp(op, lowerBatchedCore(rw, op.getLoc(), repeated, aTy.getShape()[2], aTy.getShape()[1],
                                                  bH, b, bTy, outTy));
                return success();
            }
        }
        return rw.notifyMatchFailure(op, "only 2D, 2Dx1D, 3D with equal batch dims, or 3D GQA broadcast are "
                                         "supported");
    }

  private:
    static Value lower2d(ConversionPatternRewriter & rw, Location loc, Value a, Value b, RankedTensorType aTy,
                         RankedTensorType outTy) {
        Value zero   = ps::constF32(rw, loc, 0.0f);
        auto  atTy   = ps::lastTwoSwapped(aTy);
        Value atInit = ps::empty(rw, loc, atTy);
        Value at     = linalg::TransposeOp::create(rw, loc, a, atInit, ArrayRef<int64_t>{1, 0}).getResult()[0];
        Value filled = ps::fillWith(rw, loc, zero, outTy);
        return linalg::MatmulOp::create(rw, loc, TypeRange{outTy}, ValueRange{b, at}, ValueRange{filled})
            .getResult(0);
    }

    // Expand b to [1, k], reuse the transpose+matmul path, then collapse [1, n] back to [n].
    static Value lower2dVec(ConversionPatternRewriter & rw, Location loc, Value a, Value b, RankedTensorType aTy,
                            RankedTensorType bTy, RankedTensorType outTy) {
        const int64_t k = bTy.getShape()[0];
        const int64_t n = aTy.getShape()[0];
        Type          elem = aTy.getElementType();

        Value zero = ps::constF32(rw, loc, 0.0f);

        SmallVector<ReassociationIndices> flat = {ReassociationIndices{0, 1}};
        auto  b2Ty = ps::typeOf({1, k}, bTy.getElementType());
        Value b2   = tensor::ExpandShapeOp::create(rw, loc, b2Ty, b, flat);

        auto  atTy   = ps::lastTwoSwapped(aTy);
        Value atInit = ps::empty(rw, loc, atTy);
        Value at     = linalg::TransposeOp::create(rw, loc, a, atInit, ArrayRef<int64_t>{1, 0}).getResult()[0];

        auto  mmTy   = ps::typeOf({1, n}, elem);
        Value filled = ps::fillWith(rw, loc, zero, mmTy);
        Value mm =
            linalg::MatmulOp::create(rw, loc, TypeRange{mmTy}, ValueRange{b2, at}, ValueRange{filled}).getResult(0);

        return tensor::CollapseShapeOp::create(rw, loc, outTy, mm, flat);
    }

    // For each h: out[h] = mul_mat_2d(a[h], b[h]). Same transpose-A-then-matmul structure as the 2D
    // case with a leading batch dim carried through unchanged.
    static Value lowerBatchedCore(ConversionPatternRewriter & rw, Location loc, Value aVal, int64_t aNe0,
                                  int64_t aNe1, int64_t H, Value b, RankedTensorType bTy,
                                  RankedTensorType outTy) {
        Type  elem  = bTy.getElementType();
        Value zero  = ps::constF32(rw, loc, 0.0f);
        auto  atTy  = ps::typeOf({H, aNe0, aNe1}, elem);
        Value atInit = ps::empty(rw, loc, atTy);
        Value at = linalg::TransposeOp::create(rw, loc, aVal, atInit, ArrayRef<int64_t>{0, 2, 1}).getResult()[0];
        Value filled = ps::fillWith(rw, loc, zero, outTy);
        return linalg::BatchMatmulOp::create(rw, loc, TypeRange{outTy}, ValueRange{b, at}, ValueRange{filled})
            .getResult(0);
    }

    // Built from extract_slice/insert_slice rather than a linalg.generic with a floordiv indexing
    // map: the latter is valid MLIR but untested through this pipeline's tile/vectorize stages,
    // while slices are already proven by RoPE's deinterleave.
    static Value repeatHeads(ConversionPatternRewriter & rw, Location loc, Value a, RankedTensorType aTy,
                             int64_t targetH) {
        const int64_t srcH  = aTy.getShape()[0];
        const int64_t group = targetH / srcH;
        const int64_t ne1   = aTy.getShape()[1];
        const int64_t ne0   = aTy.getShape()[2];
        Type          elem  = aTy.getElementType();

        auto outTy   = ps::typeOf({targetH, ne1, ne0}, elem);
        auto sliceTy = ps::typeOf({1, ne1, ne0}, elem);

        SmallVector<OpFoldResult> sizes   = idxAttrs(rw, {1, ne1, ne0});
        SmallVector<OpFoldResult> strides = idxAttrs(rw, {1, 1, 1});

        Value current = ps::empty(rw, loc, outTy);
        for (int64_t hSrc = 0; hSrc < srcH; hSrc++) {
            Value slice = tensor::ExtractSliceOp::create(rw, loc, sliceTy, a, idxAttrs(rw, {hSrc, 0, 0}), sizes,
                                                         strides);
            for (int64_t g = 0; g < group; g++) {
                const int64_t hDst = hSrc * group + g;
                current = tensor::InsertSliceOp::create(rw, loc, slice, current, idxAttrs(rw, {hDst, 0, 0}), sizes,
                                                        strides);
            }
        }
        return current;
    }
};

}  // namespace

void populateMatmulPatterns(RewritePatternSet & patterns) {
    patterns.add<MulMatLowering>(patterns.getContext());
}

}  // namespace tsi::mlir_export
