// The ggml dialect. Internal to the exporter library: the public Exporter.h stays MLIR-free, so
// nothing outside src/ includes this.
#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "GgmlOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "GgmlOps.h.inc"
