// Type mapping, constant/init helpers and affine-map helpers for the exporter.
//
// Each function here replaces a string-formatting helper from the previous emitter, returning the
// MLIR object instead of its textual spelling.
#include "IRBuilder.h"

#include "PatternSupport.h"   // ps::castElements, for the f32 -> cache-type narrowing

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

Type GraphBuilder::elementType(ggml_type t) const {
    switch (t) {
        case GGML_TYPE_F32:  return b_.getF32Type();
        case GGML_TYPE_F16:  return b_.getF16Type();
        case GGML_TYPE_BF16: return b_.getBF16Type();
        case GGML_TYPE_I32:  return b_.getI32Type();
        default:
            // Quantized types are the notable absence. They are not element types at all: a q8_0
            // block interleaves scales with quants, so a tensor of them has no MLIR equivalent and
            // would need dequantization in the graph.
            unsupported("unsupported tensor type: %s", ggml_type_name(t));
    }
}

Type GraphBuilder::elementType(const ggml_tensor * t) const {
    return elementType(t->type);
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
    RankedTensorType  ty   = tensorType(t);
    const std::string name = blobName(t);

    // Element width drives both the blob alignment and the gather type. f16/bf16 are copied as raw
    // 16-bit patterns: nothing here does arithmetic on them, and the host has no bf16 type anyway.
    size_t width = 0;
    switch (t->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_I32:  width = 4; break;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: width = 2; break;
        default:
            unsupported("cannot bake a constant of type %s", ggml_type_name(t->type));
    }

    // Zero-copy when the layout already matches the blob's: contiguous and naturally aligned. That is
    // the case for every model weight, and it is the difference between one and two copies of the
    // model in memory during export.
    const bool inPlace = ggml_is_contiguous(t) && ((uintptr_t) t->data % width) == 0;

    AsmResourceBlob blob;
    if (inPlace) {
        blob = UnmanagedAsmResourceBlob::allocateWithAlign(
            llvm::ArrayRef<char>((const char *) t->data, (size_t) ggml_nelements(t) * width), width);
    } else if (width == 4) {
        std::vector<int32_t> v = gather<int32_t>(t);   // f32 and i32 both move as 32-bit words
        blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(llvm::ArrayRef<int32_t>(v));
    } else {
        std::vector<uint16_t> v = gather<uint16_t>(t);
        blob = HeapAsmResourceBlob::allocateAndCopyInferAlign(llvm::ArrayRef<uint16_t>(v));
    }

    // The generic DenseResourceElementsAttr, not the typed DenseF32ResourceElementsAttr: the typed
    // ones exist only for f32/f64 and the integer widths, so f16 and bf16 have no variant to use.
    Attribute attr = DenseResourceElementsAttr::get(ty, name, std::move(blob));
    return arith::ConstantOp::create(b_, loc_, cast<TypedAttr>(attr));
}

// --- KV cache in DRAM -------------------------------------------------------------------------

MemRefType GraphBuilder::cacheType(const CacheSpec & spec) const {
    if (spec.read.empty() || spec.read[0] == nullptr) {
        unsupported("cache '%s' has no per-layer slices", spec.name.c_str());
    }
    const ggml_tensor * slice = spec.read[0];

    // [n_layers, cells, ...rest...]. Cells is the FIRST slice dim, not the last, because that is
    // llama's own layout: cell c of a layer starts at c * head_dim * n_head_kv, so one cell's values
    // are contiguous and cells are strided. A cells-last memref would be the transpose of llama's
    // cache, so it could never alias it and every read would need a shuffle.
    SmallVector<int64_t> shape;
    shape.push_back(spec.n_layers);
    for (int64_t d : dims(slice)) {
        shape.push_back(d);
    }
    if (shape.size() < 2 || shape[1] != spec.cells) {
        unsupported("cache '%s': slice's first dim %lld != cells %lld", spec.name.c_str(),
                    (long long) (shape.size() > 1 ? shape[1] : 0), (long long) spec.cells);
    }
    // Memory space 1 is DRAM and is not optional: any other space is rejected downstream.
    return MemRefType::get(shape, elementType(spec.elem_type), MemRefLayoutAttrInterface{},
                           b_.getI64IntegerAttr(1));
}

Value GraphBuilder::cacheSlice(Value cache, const CacheSpec & spec, int64_t il, Value slot,
                               int64_t width, int64_t resultRank) {
    (void) spec;
    auto      ct   = cast<MemRefType>(cache.getType());
    const int rank = (int) ct.getRank();

    // Take one layer and `width` cells: offset [il, slot, 0..], size [1, width, full..].
    SmallVector<OpFoldResult> offsets, sizes, strides;
    SmallVector<int64_t>      resultShape;
    offsets.push_back(b_.getIndexAttr(il));
    sizes.push_back(b_.getIndexAttr(1));
    strides.push_back(b_.getIndexAttr(1));
    for (int d = 1; d < rank; d++) {
        const bool cell = (d == 1);   // cells sit immediately after the layer dim
        if (cell && slot) {
            offsets.push_back(slot);
        } else {
            offsets.push_back(b_.getIndexAttr(0));
        }
        const int64_t n = cell ? width : ct.getDimSize(d);
        sizes.push_back(b_.getIndexAttr(n));
        strides.push_back(b_.getIndexAttr(1));
        resultShape.push_back(n);   // drops the leading layer dim
    }

    // ggml_n_dims ignores trailing 1s, so a tensor of ne [head_dim, n_head_kv, 1] - exactly what a
    // decode step appends - is rank 2 in MLIR, not 3. Its destination has to lose the unit cell dim
    // too, or materialize_in_destination fails verification with a rank mismatch. Only unit dims are
    // ever dropped, so this cannot silently discard real cells.
    while (resultRank > 0 && (int64_t) resultShape.size() > resultRank && resultShape.front() == 1) {
        resultShape.erase(resultShape.begin());
    }
    if (resultRank > 0 && (int64_t) resultShape.size() != resultRank) {
        unsupported("cache '%s': cannot reduce a %zu-dim slice to the appended value's rank %lld",
                    spec.name.c_str(), resultShape.size(), (long long) resultRank);
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
    Value read = bufferization::ToTensorOp::create(b_, loc_, tt, sub, /*restrict=*/true,
                                                   /*writable=*/false);

    // The cache is stored in whatever type llama uses (f16), while the graph computes in f32, so widen
    // here. cacheAppend narrows symmetrically. Keeping the conversion at the cache boundary is what
    // lets the storage type change without touching the reconstruction or its CPU reference.
    Type want = elementType(spec.read[il]);
    if (mt.getElementType() != want) {
        read = ps::castElements(b_, loc_, read, want);
    }
    return read;
}

void GraphBuilder::cacheAppend(Value cache, const CacheSpec & spec, int64_t il, Value slot,
                               Value src) {
    auto st = cast<RankedTensorType>(src.getType());
    auto ct = cast<MemRefType>(cache.getType());
    // The graph computes in f32 (see promoteGgmlToF32) but the cache is f16 to match llama bit for
    // bit, so the append is where the narrowing happens. Emitted as linalg.generic for the same
    // reason the promotion pass does: it is the form the rest of the lowering produces.
    if (st.getElementType() != ct.getElementType()) {
        src = ps::castElements(b_, loc_, src, ct.getElementType());
        st  = cast<RankedTensorType>(src.getType());
    }
    // How many cells this append covers. Normally the source's first MLIR dim, matching the
    // cells-first layout. But ggml_n_dims drops trailing 1s, so a 1-cell append arrives already
    // rank-reduced with its cell dim gone and its first dim meaning something else entirely - reading
    // the width off it there would append n_head_kv cells' worth at the wrong stride.
    const int64_t sliceRank = ct.getRank() - 1;   // the cache minus its layer dim
    int64_t       width;
    if (st.getRank() == sliceRank) {
        width = st.getShape().front();
    } else if (st.getRank() == sliceRank - 1) {
        width = 1;
    } else {
        unsupported("cache '%s': appended value has rank %lld, expected %lld or %lld",
                    spec.name.c_str(), (long long) st.getRank(), (long long) sliceRank,
                    (long long) (sliceRank - 1));
        return;
    }

    Value sub = cacheSlice(cache, spec, il, slot, width, st.getRank());
    auto  op  = bufferization::MaterializeInDestinationOp::create(b_, loc_, TypeRange{}, src, sub);
    op.setWritable(true);   // the destination is a memref we own
}

}  // namespace tsi::mlir_export
