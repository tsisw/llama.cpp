// The ggml -> linalg lowering.
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace tsi::mlir_export {

// Lowers every ggml-dialect op in `mod` to linalg/tensor/arith/math.
//
// Throws mlir_export_error if any op cannot be lowered. This is where OUR restrictions live (rank
// limits, ROPE mode, which reshape shapes are handled) as opposed to ggml's own invariants, which
// are dialect verifiers. A pattern that declines to match produces a diagnostic naming the op.
void convertGgmlToLinalg(mlir::ModuleOp mod);

// One populate function per op family, so each is independently testable.
void populateElementwisePatterns(mlir::RewritePatternSet & patterns);
void populateNormPatterns(mlir::RewritePatternSet & patterns);
void populateMatmulPatterns(mlir::RewritePatternSet & patterns);
void populateShapePatterns(mlir::RewritePatternSet & patterns);
void populateRopePatterns(mlir::RewritePatternSet & patterns);

}  // namespace tsi::mlir_export
