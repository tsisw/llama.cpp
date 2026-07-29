// ggml_cgraph -> ggml dialect. Internal to the exporter library.
#pragma once

#include "tsi/export/Exporter.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace tsi::mlir_export {

// Builds `module { func.func @<name>(args) -> results { <ggml dialect ops> return } }`.
//
// A faithful 1:1 translation: one dialect op per graph node, op_params carried as attributes, no
// lowering decisions. The only transformation is the shape convention (ggml ne reversed into MLIR
// dim order), applied once here. Anything ggml itself can express should import successfully; what
// our lowering cannot handle is rejected later, by the conversion pass.
//
// Throws mlir_export_error for an op with no dialect equivalent.
mlir::OwningOpRef<mlir::ModuleOp> importGraph(mlir::MLIRContext & ctx, ggml_cgraph * gf,
                                             const ExportOptions & opts,
                                             const std::vector<const ggml_tensor *> & outputs);

}  // namespace tsi::mlir_export
