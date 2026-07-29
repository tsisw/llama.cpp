// linalg-building helpers shared by the ggml-to-linalg conversion patterns.
//
// These are the MLIR-object equivalents of the string helpers the previous emitter carried
// (mlir_tensor_type, affine_map_full, iterator_types_reduce_last, ...). They take MLIR types rather
// than ggml_tensor*, because by the time a pattern runs the ggml graph is gone and the ggml-dialect
// op is the source of truth.
#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace tsi::mlir_export::ps {

// --- types ----------------------------------------------------------------------------------
mlir::RankedTensorType typeOf(llvm::ArrayRef<int64_t> shape, mlir::Type elem);

// Drops the innermost dim, the one every reduction here reduces over (ggml's ne[0]).
mlir::RankedTensorType reducedType(mlir::RankedTensorType t);

// Swaps the last two dims. For a rank-2 tensor that is a plain transpose; for rank 3 it transposes
// within each batch entry.
mlir::RankedTensorType lastTwoSwapped(mlir::RankedTensorType t);

// --- constants and inits --------------------------------------------------------------------
mlir::Value constF32(mlir::OpBuilder & b, mlir::Location loc, float v);
mlir::Value constIndex(mlir::OpBuilder & b, mlir::Location loc, int64_t v);
mlir::Value empty(mlir::OpBuilder & b, mlir::Location loc, mlir::RankedTensorType ty);
mlir::Value fillWith(mlir::OpBuilder & b, mlir::Location loc, mlir::Value scalar, mlir::RankedTensorType ty);
mlir::Value zeroFilled(mlir::OpBuilder & b, mlir::Location loc, mlir::RankedTensorType ty);
mlir::Value denseF32(mlir::OpBuilder & b, mlir::Location loc, llvm::ArrayRef<float> vals,
                     mlir::RankedTensorType ty);

// --- affine maps and iterator types ---------------------------------------------------------
mlir::AffineMap mapFull(mlir::MLIRContext * ctx, int rank);
mlir::AffineMap mapDropLast(mlir::MLIRContext * ctx, int rank);
mlir::AffineMap mapSelect(mlir::MLIRContext * ctx, int rank, llvm::ArrayRef<int> keep);
llvm::SmallVector<mlir::utils::IteratorType> itersAllParallel(int rank);
llvm::SmallVector<mlir::utils::IteratorType> itersReduceLast(int rank);

// --- linalg.generic with one result ---------------------------------------------------------
using BodyFn = llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)>;
mlir::Value generic(mlir::OpBuilder & b, mlir::Location loc, mlir::RankedTensorType resultTy,
                    mlir::ValueRange ins, mlir::Value out, llvm::ArrayRef<mlir::AffineMap> maps,
                    llvm::ArrayRef<mlir::utils::IteratorType> iters, BodyFn body);

}  // namespace tsi::mlir_export::ps
