#include "GgmlToLinalg.h"

#include "IRBuilder.h"
#include "GgmlDialect.h"
#include "PatternSupport.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace tsi::mlir_export {

void convertGgmlToLinalg(ModuleOp mod) {
    MLIRContext * ctx = mod.getContext();

    ConversionTarget target(*ctx);
    // memref and bufferization appear only around the KV cache: the importer emits them directly
    // because ggml cannot express a DRAM buffer or a strided view. They are already lowered, so they
    // pass through untouched.
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect,
                           math::MathDialect, func::FuncDialect, memref::MemRefDialect,
                           bufferization::BufferizationDialect>();
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
