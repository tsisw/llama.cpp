// Op families not yet lowered to linalg. Each populate() is intentionally empty: with a full
// conversion target, an unlowered ggml op fails the pass with a diagnostic naming it, which is the
// behavior we want while the port is in progress.
#include "GgmlToLinalg.h"

namespace tsi::mlir_export {

void populateRopePatterns(mlir::RewritePatternSet &) {}     // ggml.rope

}  // namespace tsi::mlir_export
