#include "GgmlToLinalg.h"

#include "Builder.h"
#include "GgmlDialect.h"
#include "PatternSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace tsi::mlir_export {

void convertGgmlToLinalg(ModuleOp mod) {
    MLIRContext * ctx = mod.getContext();

    ConversionTarget target(*ctx);
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect,
                           math::MathDialect, func::FuncDialect>();
    // applyFullConversion demands EVERY op be legal, the enclosing builtin.module included.
    target.addLegalOp<ModuleOp>();
    // Full conversion, so a ggml op left standing is a hard error with MLIR's own diagnostic naming
    // it, rather than something that silently survives into the printed output.
    target.addIllegalDialect<ggml::GgmlDialect>();

    RewritePatternSet patterns(ctx);
    populateElementwisePatterns(patterns);
    populateNormPatterns(patterns);
    populateMatmulPatterns(patterns);
    populateShapePatterns(patterns);
    populateRopePatterns(patterns);

    if (failed(applyFullConversion(mod, target, std::move(patterns)))) {
        unsupported("ggml-to-linalg lowering failed (diagnostics above name the op)");
    }
}

}  // namespace tsi::mlir_export
