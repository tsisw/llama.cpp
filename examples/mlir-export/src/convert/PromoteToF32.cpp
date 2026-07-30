// Rewrite every ggml-dialect op to compute in f32, extending half-precision inputs and truncating
// results back.
//
// Why a pass and not per-pattern handling: f32 accumulation is not optional. An f16 sum over 2048
// elements loses most of its significance, so every reduction in the model - matmul, rms_norm,
// soft_max - has to accumulate in f32 regardless of how the weights are stored. Promoting once, here,
// gets that for free and means no lowering pattern ever sees an f16 operand. The alternative, teaching
// five pattern files to widen their accumulators, is the same rule written five times.
//
// It also handles the mixed-precision case llama actually produces: an f16 weight matmul'd against an
// f32 activation. Extending each float operand independently makes that fall out with no special case.
//
// Runs after import, so the ggml dialect IR remains a faithful record of the graph and
// TSI_DUMP_GGML_IR still shows what ggml had. Set TSI_DUMP_GGML_IR=1 to see the promoted form too.
#include "Builder.h"
#include "GgmlDialect.h"
#include "GgmlToLinalg.h"
#include "PatternSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace tsi::mlir_export {

namespace {

bool isHalf(Type elem) {
    return elem.isF16() || elem.isBF16();
}

RankedTensorType asF32(RankedTensorType t) {
    return RankedTensorType::get(t.getShape(), Float32Type::get(t.getContext()));
}

}  // namespace

void promoteGgmlToF32(ModuleOp mod) {
    MLIRContext * ctx = mod.getContext();
    Type          f32 = Float32Type::get(ctx);

    // Collect first: the walk rewrites ops, and mutating while walking is a way to visit clones.
    SmallVector<Operation *> ops;
    mod.walk([&](Operation * op) {
        if (op->getDialect() != ctx->getLoadedDialect<ggml::GgmlDialect>()) {
            return;
        }
        const bool halfIn  = llvm::any_of(op->getOperandTypes(), [](Type t) {
            auto rt = llvm::dyn_cast<RankedTensorType>(t);
            return rt && isHalf(rt.getElementType());
        });
        const bool halfOut = llvm::any_of(op->getResultTypes(), [](Type t) {
            auto rt = llvm::dyn_cast<RankedTensorType>(t);
            return rt && isHalf(rt.getElementType());
        });
        if (halfIn || halfOut) {
            ops.push_back(op);
        }
    });

    for (Operation * op : ops) {
        OpBuilder b(op);
        Location  loc = op->getLoc();

        // Extend the float operands; leave i32 indices and positions alone.
        SmallVector<Value> operands;
        for (Value v : op->getOperands()) {
            auto rt = llvm::dyn_cast<RankedTensorType>(v.getType());
            operands.push_back(rt && isHalf(rt.getElementType()) ? ps::castElements(b, loc, v, f32) : v);
        }

        SmallVector<Type> resultTys;
        for (Type t : op->getResultTypes()) {
            auto rt = llvm::dyn_cast<RankedTensorType>(t);
            resultTys.push_back(rt && isHalf(rt.getElementType()) ? Type(asF32(rt)) : t);
        }

        // Same op, same attributes, f32 types. Cloning by OperationState keeps this generic: adding a
        // ggml op needs no change here.
        OperationState state(loc, op->getName(), operands, resultTys, op->getAttrs());
        Operation *    promoted = b.create(state);

        // Truncate back, so every consumer still sees the type the graph declared.
        SmallVector<Value> replacements;
        for (auto [old, neu] : llvm::zip(op->getResults(), promoted->getResults())) {
            auto rt = llvm::dyn_cast<RankedTensorType>(old.getType());
            replacements.push_back(rt && isHalf(rt.getElementType())
                                       ? ps::castElements(b, loc, neu, rt.getElementType())
                                       : neu);
        }
        op->replaceAllUsesWith(replacements);
        op->erase();
    }
}

}  // namespace tsi::mlir_export
