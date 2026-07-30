#include "PatternSupport.h"

using namespace mlir;

namespace tsi::mlir_export::ps {

RankedTensorType typeOf(llvm::ArrayRef<int64_t> shape, Type elem) {
    return RankedTensorType::get(shape, elem);
}

RankedTensorType reducedType(RankedTensorType t) {
    llvm::SmallVector<int64_t> s(t.getShape().begin(), t.getShape().end());
    s.pop_back();
    return typeOf(s, t.getElementType());
}

RankedTensorType lastTwoSwapped(RankedTensorType t) {
    llvm::SmallVector<int64_t> s(t.getShape().begin(), t.getShape().end());
    std::swap(s[s.size() - 2], s[s.size() - 1]);
    return typeOf(s, t.getElementType());
}

Value constF32(OpBuilder & b, Location loc, float v) {
    return arith::ConstantOp::create(b, loc, b.getF32FloatAttr(v));
}

Value constIndex(OpBuilder & b, Location loc, int64_t v) {
    return arith::ConstantOp::create(b, loc, b.getIndexAttr(v));
}

Value empty(OpBuilder & b, Location loc, RankedTensorType ty) {
    return tensor::EmptyOp::create(b, loc, ty.getShape(), ty.getElementType());
}

Value fillWith(OpBuilder & b, Location loc, Value scalar, RankedTensorType ty) {
    Value init = empty(b, loc, ty);
    return linalg::FillOp::create(b, loc, ValueRange{scalar}, ValueRange{init}).getResult(0);
}

Value zeroFilled(OpBuilder & b, Location loc, RankedTensorType ty) {
    return fillWith(b, loc, constF32(b, loc, 0.0f), ty);
}

Value denseF32(OpBuilder & b, Location loc, llvm::ArrayRef<float> vals, RankedTensorType ty) {
    return arith::ConstantOp::create(b, loc, DenseElementsAttr::get(ty, vals));
}

Value castElements(OpBuilder & b, Location loc, Value v, Type toElem) {
    auto from = llvm::cast<RankedTensorType>(v.getType());
    if (from.getElementType() == toElem) {
        return v;
    }
    auto      to   = RankedTensorType::get(from.getShape(), toElem);
    const int rank = to.getRank();
    AffineMap id   = mapFull(b.getContext(), rank);
    return generic(b, loc, to, ValueRange{v}, empty(b, loc, to), {id, id}, itersAllParallel(rank),
                   [&](OpBuilder & nb, Location nloc, ValueRange args) -> Value {
                       const unsigned fw = from.getElementType().getIntOrFloatBitWidth();
                       if (toElem.getIntOrFloatBitWidth() > fw) {
                           return arith::ExtFOp::create(nb, nloc, toElem, args[0]);
                       }
                       return arith::TruncFOp::create(nb, nloc, toElem, args[0]);
                   });
}

AffineMap mapFull(MLIRContext * ctx, int rank) {
    return AffineMap::getMultiDimIdentityMap(rank, ctx);
}

AffineMap mapDropLast(MLIRContext * ctx, int rank) {
    llvm::SmallVector<AffineExpr> exprs;
    for (int i = 0; i < rank - 1; i++) {
        exprs.push_back(getAffineDimExpr(i, ctx));
    }
    return AffineMap::get(rank, 0, exprs, ctx);
}

AffineMap mapSelect(MLIRContext * ctx, int rank, llvm::ArrayRef<int> keep) {
    llvm::SmallVector<AffineExpr> exprs;
    for (int d : keep) {
        exprs.push_back(getAffineDimExpr(d, ctx));
    }
    return AffineMap::get(rank, 0, exprs, ctx);
}

llvm::SmallVector<utils::IteratorType> itersAllParallel(int rank) {
    return llvm::SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

llvm::SmallVector<utils::IteratorType> itersReduceLast(int rank) {
    llvm::SmallVector<utils::IteratorType> its(rank, utils::IteratorType::parallel);
    its.back() = utils::IteratorType::reduction;
    return its;
}

Value generic(OpBuilder & b, Location loc, RankedTensorType resultTy, ValueRange ins, Value out,
              llvm::ArrayRef<AffineMap> maps, llvm::ArrayRef<utils::IteratorType> iters, BodyFn body) {
    auto op = linalg::GenericOp::create(b, loc, TypeRange{resultTy}, ins, ValueRange{out}, maps, iters,
                                        [&](OpBuilder & nb, Location nloc, ValueRange args) {
                                            Value y = body(nb, nloc, args);
                                            linalg::YieldOp::create(nb, nloc, y);
                                        });
    return op.getResult(0);
}

}  // namespace tsi::mlir_export::ps
