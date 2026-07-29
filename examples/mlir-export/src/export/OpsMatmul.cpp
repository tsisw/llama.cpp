// MUL_MAT emitters. Port in progress: not yet translated to the MLIR C++ API.
#include "Builder.h"

using namespace mlir;

namespace tsi::mlir_export {

Value GraphBuilder::emitMulMat(const ggml_tensor * node) {
    (void) node;
    unsupported("MUL_MAT: not yet ported to the MLIR C++ API");
}

}  // namespace tsi::mlir_export
