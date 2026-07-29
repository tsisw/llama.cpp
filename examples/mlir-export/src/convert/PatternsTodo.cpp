// Op families not yet lowered to linalg. Each populate() is intentionally empty: with a full
// conversion target, an unlowered ggml op fails the pass with a diagnostic naming it, which is the
// behavior we want while the port is in progress.
#include "GgmlToLinalg.h"

namespace tsi::mlir_export {

void populateNormPatterns(mlir::RewritePatternSet &) {}    // ggml.rms_norm, ggml.soft_max
void populateMatmulPatterns(mlir::RewritePatternSet &) {}  // ggml.mul_mat
void populateShapePatterns(mlir::RewritePatternSet &) {}    // permute, reshape, cont, concat, get_rows
void populateRopePatterns(mlir::RewritePatternSet &) {}     // ggml.rope

}  // namespace tsi::mlir_export
