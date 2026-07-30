// Type mapping, constant/init helpers and affine-map helpers for the exporter.
//
// Each function here replaces a string-formatting helper from the previous emitter, returning the
// MLIR object instead of its textual spelling.
#include "Builder.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"   // AsmResourceBlob, for dense_resource constants

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace mlir;

namespace tsi::mlir_export {

void unsupported(const char * fmt, ...) {
    char    buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "mlir-export: %s\n", buf);
    throw mlir_export_error("unsupported graph construct (see message above)");
}

Value GraphBuilder::valueOf(const ggml_tensor * t) const {
    auto it = values_.find(t);
    if (it == values_.end()) {
        // A graph whose nodes are not in topological order, or a leaf that was neither declared a
        // runtime arg nor a baked constant. Either way the caller built the graph wrong.
        unsupported("no value for tensor '%s' (op %s) - it is used before it is defined",
                    t->name[0] ? t->name : "<unnamed>", ggml_op_name(t->op));
    }
    return it->second;
}

// --- type mapping ---------------------------------------------------------------------------

Type GraphBuilder::elementType(const ggml_tensor * t) const {
    if (t->type == GGML_TYPE_F32) {
        return b_.getF32Type();
    }
    if (t->type == GGML_TYPE_I32) {
        return b_.getI32Type();
    }
    unsupported("unsupported tensor type: %s", ggml_type_name(t->type));
}

llvm::SmallVector<int64_t> GraphBuilder::dims(const ggml_tensor * t) const {
    return dimsRanked(t, ggml_n_dims(t));
}

llvm::SmallVector<int64_t> GraphBuilder::dimsRanked(const ggml_tensor * t, int rank) const {
    llvm::SmallVector<int64_t> s;
    s.reserve(rank);
    for (int i = rank - 1; i >= 0; i--) {
        s.push_back(t->ne[i]);
    }
    return s;
}

RankedTensorType GraphBuilder::typeOf(llvm::ArrayRef<int64_t> shape, Type elem) const {
    return RankedTensorType::get(shape, elem);
}

RankedTensorType GraphBuilder::tensorType(const ggml_tensor * t) const {
    return typeOf(dims(t), elementType(t));
}

RankedTensorType GraphBuilder::tensorTypeRanked(const ggml_tensor * t, int rank) const {
    return typeOf(dimsRanked(t, rank), elementType(t));
}

// A blob name MLIR will accept as a resource key: keep the GGUF-ish characters, replace the rest.
static std::string blobName(const ggml_tensor * t) {
    std::string s = t->name[0] ? t->name : "const";
    for (char & c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '_' || c == '.';
        if (!ok) {
            c = '_';
        }
    }
    return s;
}

// Gather a strided tensor into a packed buffer in MLIR element order.
//
// MLIR row-major order walks ggml dim 0 fastest (MLIR's last dim IS ggml's ne[0]), so an odometer
// with dim 0 least-significant produces exactly the order the blob needs.
template <typename T>
static std::vector<T> gather(const ggml_tensor * t) {
    const int    rank  = ggml_n_dims(t);
    const size_t total = (size_t) ggml_nelements(t);

    std::vector<T> out;
    out.reserve(total);
    llvm::SmallVector<int64_t, 4> idx(rank, 0);
    for (size_t k = 0; k < total; k++) {
        size_t off = 0;
        for (int d = 0; d < rank; d++) {
            off += (size_t) idx[d] * t->nb[d];
        }
        out.push_back(*(const T *) ((const char *) t->data + off));
        for (int d = 0; d < rank; d++) {
            if (++idx[d] < t->ne[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
    return out;
}

Value GraphBuilder::bakedConstant(const ggml_tensor * t) {
    if (t->data == nullptr) {
        // Either the caller forgot to declare this leaf a runtime arg, or it declared a leaf whose
        // data was never bound. Both produce garbage silently if we bake whatever is at the pointer.
        unsupported("leaf '%s' is neither a declared runtime input nor holds data to bake",
                    t->name[0] ? t->name : "<unnamed>");
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) {
        unsupported("cannot bake a constant of type %s", ggml_type_name(t->type));
    }

    RankedTensorType  ty    = tensorType(t);
    const bool        isI32 = t->type == GGML_TYPE_I32;
    const std::string name  = blobName(t);
    const size_t      bytes = (size_t) ggml_nelements(t) * 4;   // f32 and i32 are both 4 bytes

    // Zero-copy when the layout already matches the blob's: contiguous and 4-byte aligned. This is
    // the case for every model weight, and it is the difference between one and two copies of the
    // model in memory during export.
    const bool inPlace = ggml_is_contiguous(t) && ((uintptr_t) t->data % 4) == 0;

    Attribute attr;
    if (inPlace) {
        auto blob = UnmanagedAsmResourceBlob::allocateWithAlign(
            llvm::ArrayRef<char>((const char *) t->data, bytes), 4);
        attr = isI32 ? Attribute(DenseI32ResourceElementsAttr::get(ty, name, std::move(blob)))
                     : Attribute(DenseF32ResourceElementsAttr::get(ty, name, std::move(blob)));
    } else if (isI32) {
        std::vector<int32_t> v = gather<int32_t>(t);
        attr = DenseI32ResourceElementsAttr::get(
            ty, name, HeapAsmResourceBlob::allocateAndCopyInferAlign(llvm::ArrayRef<int32_t>(v)));
    } else {
        std::vector<float> v = gather<float>(t);
        attr = DenseF32ResourceElementsAttr::get(
            ty, name, HeapAsmResourceBlob::allocateAndCopyInferAlign(llvm::ArrayRef<float>(v)));
    }
    return arith::ConstantOp::create(b_, loc_, cast<TypedAttr>(attr));
}

// --- KV cache in DRAM -------------------------------------------------------------------------

MemRefType GraphBuilder::cacheType(const CacheSpec & spec) const {
    if (spec.read.empty() || spec.read[0] == nullptr) {
        unsupported("cache '%s' has no per-layer slices", spec.name.c_str());
    }
    const ggml_tensor * slice = spec.read[0];

    // [n_layers, ...slice dims...]; the slice's last dim is already the cell count.
    SmallVector<int64_t> shape;
    shape.push_back(spec.n_layers);
    for (int64_t d : dims(slice)) {
        shape.push_back(d);
    }
    if (shape.back() != spec.cells) {
        unsupported("cache '%s': slice last dim %lld != cells %lld", spec.name.c_str(),
                    (long long) shape.back(), (long long) spec.cells);
    }
    // Memory space 1 is DRAM and is not optional: any other space is rejected downstream.
    return MemRefType::get(shape, elementType(slice), MemRefLayoutAttrInterface{},
                           b_.getI64IntegerAttr(1));
}

Value GraphBuilder::cacheSlice(Value cache, const CacheSpec & spec, int64_t il, Value slot,
                               int64_t width) {
    (void) spec;
    auto      ct   = cast<MemRefType>(cache.getType());
    const int rank = (int) ct.getRank();

    // Take one layer and `width` cells: offset [il, 0.., slot], size [1, full.., width].
    SmallVector<OpFoldResult> offsets, sizes, strides;
    SmallVector<int64_t>      resultShape;
    offsets.push_back(b_.getIndexAttr(il));
    sizes.push_back(b_.getIndexAttr(1));
    strides.push_back(b_.getIndexAttr(1));
    for (int d = 1; d < rank; d++) {
        const bool last = (d == rank - 1);
        if (last && slot) {
            offsets.push_back(slot);
        } else {
            offsets.push_back(b_.getIndexAttr(0));
        }
        const int64_t n = last ? width : ct.getDimSize(d);
        sizes.push_back(b_.getIndexAttr(n));
        strides.push_back(b_.getIndexAttr(1));
        resultShape.push_back(n);   // drops the leading layer dim
    }

    auto resTy = memref::SubViewOp::inferRankReducedResultType(resultShape, ct, offsets, sizes,
                                                               strides);
    return memref::SubViewOp::create(b_, loc_, cast<MemRefType>(resTy), cache, offsets, sizes,
                                     strides);
}

Value GraphBuilder::cacheRead(Value cache, const CacheSpec & spec, int64_t il) {
    Value sub = cacheSlice(cache, spec, il, Value(), spec.cells);
    auto  mt  = cast<MemRefType>(sub.getType());
    auto  tt  = RankedTensorType::get(mt.getShape(), mt.getElementType());
    // restrict: nothing else aliases this view. Not writable: reads only, the append writes.
    return bufferization::ToTensorOp::create(b_, loc_, tt, sub, /*restrict=*/true,
                                             /*writable=*/false);
}

void GraphBuilder::cacheAppend(Value cache, const CacheSpec & spec, int64_t il, Value slot,
                               Value src) {
    auto st = cast<RankedTensorType>(src.getType());
    auto ct = cast<MemRefType>(cache.getType());
    if (st.getElementType() != ct.getElementType()) {
        // Mixed precision needs an explicit truncf/extf here; not built yet.
        unsupported("cache '%s': append element type differs from the cache", spec.name.c_str());
    }
    Value sub = cacheSlice(cache, spec, il, slot, st.getShape().back());
    auto  op  = bufferization::MaterializeInDestinationOp::create(b_, loc_, TypeRange{}, src, sub);
    op.setWritable(true);   // the destination is a memref we own
}

}  // namespace tsi::mlir_export
