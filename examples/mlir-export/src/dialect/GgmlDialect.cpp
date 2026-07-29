// ggml dialect registration and verifiers.
//
// Every verifier here checks an invariant of ggml ITSELF, something true of any valid ggml graph.
// Restrictions belonging to our lowering (rank <= 3, ROPE mode NORMAL, no YaRN scaling, which
// reshape shapes are handled) are deliberately absent: those are match failures in the conversion
// patterns, so this dialect stays a faithful representation of the source graph.
#include "GgmlDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

#include "GgmlOpsDialect.cpp.inc"
#define GET_OP_CLASSES
#include "GgmlOps.cpp.inc"

using namespace mlir;

void tsi::ggml::GgmlDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "GgmlOps.cpp.inc"
        >();
}

namespace tsi::ggml {

// Recall the shape convention: MLIR dim (rank-1) is ggml ne[0], MLIR dim (rank-1-d) is ggml ne[d].

LogicalResult SoftMaxOp::verify() {
    auto in = llvm::cast<RankedTensorType>(getInput().getType());
    if (getMask()) {
        auto m = llvm::cast<RankedTensorType>(getMask().getType());
        // ggml's own precondition: mask->ne[0] == x->ne[0], i.e. the innermost dims agree.
        if (m.getShape().back() != in.getShape().back()) {
            return emitOpError() << "mask innermost dim " << m.getShape().back()
                                 << " must equal input innermost dim " << in.getShape().back();
        }
    }
    return success();
}

LogicalResult MulMatOp::verify() {
    auto a   = llvm::cast<RankedTensorType>(getLhs().getType());
    auto b   = llvm::cast<RankedTensorType>(getRhs().getType());
    auto res = llvm::cast<RankedTensorType>(getResult().getType());

    // ggml_can_mul_mat: a->ne[0] == b->ne[0]. That is the innermost dim of both, the K dim.
    if (a.getShape().back() != b.getShape().back()) {
        return emitOpError() << "reduction dim mismatch: lhs innermost " << a.getShape().back()
                             << " vs rhs innermost " << b.getShape().back();
    }

    // ggml result ne[0] == a->ne[1], which is the innermost result dim vs lhs's second-innermost.
    if (a.getRank() >= 2 && res.getShape().back() != a.getShape()[a.getRank() - 2]) {
        return emitOpError() << "result innermost dim " << res.getShape().back()
                             << " must equal lhs second-innermost dim " << a.getShape()[a.getRank() - 2];
    }

    // GQA grouping rule: b->ne[2] % a->ne[2] == 0, the leading batch dim when both are rank 3.
    if (a.getRank() == 3 && b.getRank() == 3 && a.getShape()[0] != 0 && b.getShape()[0] % a.getShape()[0] != 0) {
        return emitOpError() << "rhs batch dim " << b.getShape()[0] << " must be a multiple of lhs batch dim "
                             << a.getShape()[0];
    }
    return success();
}

LogicalResult RopeOp::verify() {
    auto in  = llvm::cast<RankedTensorType>(getInput().getType());
    auto pos = llvm::cast<RankedTensorType>(getPositions().getType());

    if (pos.getRank() != 1) {
        return emitOpError() << "positions must be rank 1, got rank " << pos.getRank();
    }
    // ggml asserts pos->ne[0] == x->ne[2]. ggml ne[2] is MLIR dim (rank-3); absent dims are 1.
    const int64_t expected = in.getRank() >= 3 ? in.getShape()[in.getRank() - 3] : 1;
    if (pos.getShape()[0] != expected) {
        return emitOpError() << "positions has " << pos.getShape()[0] << " entries but input needs " << expected
                             << " (one per ggml ne[2])";
    }
    return success();
}

LogicalResult PermuteOp::verify() {
    llvm::ArrayRef<int32_t> axes = getAxes();
    if (axes.size() != 4) {
        return emitOpError() << "axes must have 4 entries (ggml op_params[0..3]), got " << axes.size();
    }
    bool seen[4] = {false, false, false, false};
    for (int32_t a : axes) {
        if (a < 0 || a > 3 || seen[a]) {
            return emitOpError() << "axes must be a permutation of 0..3";
        }
        seen[a] = true;
    }
    return success();
}

// Shared by reshape and cont: ggml asserts the element count is preserved.
static LogicalResult verifyReshapeLike(Operation * op, Value input, Value result) {
    auto in  = llvm::cast<RankedTensorType>(input.getType());
    auto out = llvm::cast<RankedTensorType>(result.getType());
    if (in.getNumElements() != out.getNumElements()) {
        return op->emitOpError() << "element count must be preserved: " << in.getNumElements() << " vs "
                                 << out.getNumElements();
    }
    return success();
}

LogicalResult ReshapeOp::verify() {
    return verifyReshapeLike(*this, getInput(), getResult());
}

LogicalResult ContOp::verify() {
    return verifyReshapeLike(*this, getInput(), getResult());
}

LogicalResult ConcatOp::verify() {
    auto a   = llvm::cast<RankedTensorType>(getLhs().getType());
    auto b   = llvm::cast<RankedTensorType>(getRhs().getType());
    auto res = llvm::cast<RankedTensorType>(getResult().getType());

    const int32_t dim = getDim();
    if (dim < 0 || dim > 3) {
        return emitOpError() << "dim must be in [0, 4), got " << dim;
    }

    // Operands may legitimately report a lower rank than the result (the decode graph concatenates
    // a rank-3 cache with a [.,.,1] new entry that collapses), so only check the equal-rank case.
    if (a.getRank() == res.getRank() && b.getRank() == res.getRank()) {
        const int64_t R  = res.getRank();
        const int64_t md = R - 1 - dim;   // ggml dim -> MLIR dim
        if (md < 0) {
            return emitOpError() << "dim " << dim << " is outside the result's rank " << R;
        }
        for (int64_t i = 0; i < R; i++) {
            if (i == md) {
                if (res.getShape()[i] != a.getShape()[i] + b.getShape()[i]) {
                    return emitOpError() << "concat dim: result " << res.getShape()[i] << " != " << a.getShape()[i]
                                         << " + " << b.getShape()[i];
                }
            } else if (a.getShape()[i] != b.getShape()[i] || a.getShape()[i] != res.getShape()[i]) {
                return emitOpError() << "non-concat dim " << i << " must match across operands and result";
            }
        }
    }
    return success();
}

LogicalResult GetRowsOp::verify() {
    auto tbl = llvm::cast<RankedTensorType>(getTable().getType());
    auto ids = llvm::cast<RankedTensorType>(getIds().getType());
    auto res = llvm::cast<RankedTensorType>(getResult().getType());

    if (ids.getRank() != 1) {
        return emitOpError() << "ids must be rank 1, got rank " << ids.getRank();
    }
    // The gathered row keeps the table's innermost dim (n_embd).
    if (res.getShape().back() != tbl.getShape().back()) {
        return emitOpError() << "result innermost dim " << res.getShape().back()
                             << " must equal table innermost dim " << tbl.getShape().back();
    }
    // One output row per id. When n_tokens == 1 the result collapses to rank 1, so only check when
    // the result kept its row dim.
    if (res.getRank() >= 2 && res.getShape()[res.getRank() - 2] != ids.getShape()[0]) {
        return emitOpError() << "result row count " << res.getShape()[res.getRank() - 2] << " must equal id count "
                             << ids.getShape()[0];
    }
    return success();
}

}  // namespace tsi::ggml
