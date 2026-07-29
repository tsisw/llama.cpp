// PERMUTE/GET_ROWS/CONCAT emitters. Port in progress: not yet translated to the MLIR C++ API.
#include "Builder.h"

using namespace mlir;

namespace tsi::mlir_export {

Value GraphBuilder::emitPermute(const ggml_tensor * node) {
    (void) node;
    unsupported("PERMUTE/GET_ROWS/CONCAT: not yet ported to the MLIR C++ API");
}

Value GraphBuilder::emitGetRows(const ggml_tensor * node) {
    (void) node;
    unsupported("PERMUTE/GET_ROWS/CONCAT: not yet ported to the MLIR C++ API");
}

Value GraphBuilder::emitConcat(const ggml_tensor * node) {
    (void) node;
    unsupported("PERMUTE/GET_ROWS/CONCAT: not yet ported to the MLIR C++ API");
}

Value GraphBuilder::emitReshapeLike(const ggml_tensor * node, const ggml_tensor * x) {
    (void) node;
    (void) x;
    unsupported("RESHAPE/CONT: not yet ported to the MLIR C++ API");
}

}  // namespace tsi::mlir_export
