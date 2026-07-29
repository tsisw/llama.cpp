// ROPE emitters. Port in progress: not yet translated to the MLIR C++ API.
#include "Builder.h"

using namespace mlir;

namespace tsi::mlir_export {

Value GraphBuilder::emitRope(const ggml_tensor * node) {
    (void) node;
    unsupported("ROPE: not yet ported to the MLIR C++ API");
}

}  // namespace tsi::mlir_export
