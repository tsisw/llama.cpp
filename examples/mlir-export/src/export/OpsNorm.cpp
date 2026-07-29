// RMS_NORM/SOFT_MAX emitters. Port in progress: not yet translated to the MLIR C++ API.
#include "Builder.h"

using namespace mlir;

namespace tsi::mlir_export {

Value GraphBuilder::emitRmsNorm(const ggml_tensor * node) {
    (void) node;
    unsupported("RMS_NORM/SOFT_MAX: not yet ported to the MLIR C++ API");
}

Value GraphBuilder::emitSoftMax(const ggml_tensor * node) {
    (void) node;
    unsupported("RMS_NORM/SOFT_MAX: not yet ported to the MLIR C++ API");
}

}  // namespace tsi::mlir_export
