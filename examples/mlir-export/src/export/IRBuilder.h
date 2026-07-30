// ggml -> MLIR type mapping and the graph value map, used by the importer.
//
// Named IRBuilder, not Builder: `build*` in .gitignore matches `Builder.*` on a case-insensitive
// filesystem, so a file called Builder.cpp is silently untracked on macOS.
//
// Replaces the previous emitter's string helpers (mlir_tensor_type, mlir_shape_dims,
// mlir_dense_literal, ...) with functions returning MLIR objects. The linalg-building helpers live
// separately in convert/PatternSupport.h, because by the time lowering runs the ggml graph is gone
// and the ggml-dialect op is the source of truth.
#pragma once

#include "tsi/export/Exporter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <map>

namespace tsi::mlir_export {

// Prints `msg` to stderr then throws, so callers that catch mlir_export_error to record a case as
// unsupported still see the reason. Mirrors the previous emitter's fprintf-then-throw pairs.
[[noreturn]] void unsupported(const char * fmt, ...);

class GraphBuilder {
  public:
    GraphBuilder(mlir::OpBuilder & b, mlir::Location loc) : b_(b), loc_(loc) {}

    // --- value bookkeeping ------------------------------------------------------------------
    void        setValue(const ggml_tensor * t, mlir::Value v) { values_[t] = v; }
    mlir::Value valueOf(const ggml_tensor * t) const;
    bool        hasValue(const ggml_tensor * t) const { return values_.count(t) != 0; }

    // --- type mapping -----------------------------------------------------------------------
    // MLIR shape is ggml's ne reversed over ggml_n_dims: ggml ne[0] is the innermost/fastest dim,
    // and MLIR's last dim is. This is the single place that reversal happens.
    mlir::Type                 elementType(const ggml_tensor * t) const;
    llvm::SmallVector<int64_t> dims(const ggml_tensor * t) const;
    llvm::SmallVector<int64_t> dimsRanked(const ggml_tensor * t, int rank) const;
    mlir::RankedTensorType     tensorType(const ggml_tensor * t) const;
    mlir::RankedTensorType     tensorTypeRanked(const ggml_tensor * t, int rank) const;
    mlir::RankedTensorType     typeOf(llvm::ArrayRef<int64_t> shape, mlir::Type elem) const;

    // Bakes a leaf's data in as arith.constant dense_resource<...>.
    //
    // dense_resource, not dense<...>: the data goes to a named blob outside the op, which the
    // bytecode writer stores as raw bytes. An inline dense<> attribute would be hex-printed at 2x
    // the size, and a full model's weights make that difference decisive.
    //
    // A contiguous, aligned tensor is referenced in place, with no copy - so t->data must stay
    // valid until exportGraph returns. Anything else (a strided view) is gathered through
    // t->nb[] into a temporary, because a packed read of a PERMUTE output would silently pick up
    // the wrong elements.
    mlir::Value bakedConstant(const ggml_tensor * t);

    // --- KV cache in DRAM -------------------------------------------------------------------------
    // The cache is one memref per CacheSpec, shaped [n_layers, ...slice dims..., cells] with the
    // slice's element type and memory space 1. Space 1 is mandatory: the compiler rejects any other
    // with "all memrefs should already be in DRAM memory space".
    mlir::MemRefType cacheType(const CacheSpec & spec) const;

    // Layer il's whole window, as a tensor the graph can consume.
    mlir::Value cacheRead(mlir::Value cache, const CacheSpec & spec, int64_t il);

    // Writes `src` into layer il at cell `slot`. The width comes from src, so one call serves both
    // prefill (N cells at once) and decode (a single cell).
    void cacheAppend(mlir::Value cache, const CacheSpec & spec, int64_t il, mlir::Value slot,
                     mlir::Value src);

  private:
    // Shared by cacheRead/cacheAppend: the subview of layer il starting at cell `first`, `width`
    // cells wide. `slot` is null for a static offset.
    mlir::Value cacheSlice(mlir::Value cache, const CacheSpec & spec, int64_t il, mlir::Value slot,
                           int64_t width);

    mlir::OpBuilder &                          b_;
    mlir::Location                             loc_;
    std::map<const ggml_tensor *, mlir::Value> values_;
};

}  // namespace tsi::mlir_export
