// Device buffers and memref descriptors for one call of the compiled forward.
//
// The compiled code runs on the device and cannot read host pointers, so every argument is copied
// into a tsi_alloc buffer first. RAII because the earlier version leaked on every early return, and
// each of those returns was an error path where DRAM was already tight.
#pragma once

#include "TestModel.h"   // MemRefDescriptor<N>, tsi_alloc / tsi_dealloc via HostShimCAPI.h
#include "ggml.h"

#include <cstring>
#include <vector>

namespace tsi::driver {

// Heap MemRefDescriptor<N> over device pointer `p`, matching the exporter's type mapping: rank =
// ggml_n_dims, MLIR shape = ne reversed, row-major strides.
//
// malloc, not new, so one free() releases any rank without knowing which N it was.
// MemRefDescriptor is a trivially copyable aggregate, so there is nothing to construct.
template <int N>
void * makeDescN(const ggml_tensor * t, void * p) {
    auto * d  = (MemRefDescriptor<N> *) malloc(sizeof(MemRefDescriptor<N>));
    d->base   = p;
    d->data   = p;
    d->offset = 0;
    for (int i = 0; i < N; i++) {
        d->shape[i] = t->ne[N - 1 - i];
    }
    d->strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; i--) {
        d->strides[i] = d->strides[i + 1] * d->shape[i + 1];
    }
    return d;
}

inline void * makeDesc(const ggml_tensor * t, void * p) {
    switch (ggml_n_dims(t)) {
        case 1:  return makeDescN<1>(t, p);
        case 2:  return makeDescN<2>(t, p);
        case 3:  return makeDescN<3>(t, p);
        default: return makeDescN<4>(t, p);
    }
}

// Same, but for a shape given directly in MLIR order. The KV cache has no ggml_tensor to describe
// it: it is a memref the exporter synthesizes, one rank higher than any tensor in the graph.
template <int N>
void * makeDescShapeN(const int64_t * shape, void * p) {
    auto * d  = (MemRefDescriptor<N> *) malloc(sizeof(MemRefDescriptor<N>));
    d->base   = p;
    d->data   = p;
    d->offset = 0;
    for (int i = 0; i < N; i++) {
        d->shape[i] = shape[i];
    }
    d->strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; i--) {
        d->strides[i] = d->strides[i + 1] * d->shape[i + 1];
    }
    return d;
}

inline void * makeDescShape(const std::vector<int64_t> & shape, void * p) {
    switch (shape.size()) {
        case 1:  return makeDescShapeN<1>(shape.data(), p);
        case 2:  return makeDescShapeN<2>(shape.data(), p);
        case 3:  return makeDescShapeN<3>(shape.data(), p);
        case 4:  return makeDescShapeN<4>(shape.data(), p);
        default: return makeDescShapeN<5>(shape.data(), p);
    }
}

// argv for one forward call, in the ciface order the exporter documents:
// [runtime_args..., caches..., slot?, outputs...].
class DeviceArgs {
  public:
    ~DeviceArgs() {
        for (void * d : bufs_) {
            tsi_dealloc(d);
        }
        for (void * d : descs_) {
            free(d);   // malloc'd by makeDesc*, so free, not operator delete
        }
    }

    DeviceArgs(const DeviceArgs &)             = delete;
    DeviceArgs & operator=(const DeviceArgs &) = delete;
    DeviceArgs()                               = default;

    // Copies t->data to the device. Returns false when DRAM is exhausted, which is a report-and-skip
    // condition rather than a crash: raise USER_DRAM_SIZE and retry.
    bool addInput(const ggml_tensor * t) {
        if (!add(t, /*copy=*/true)) {
            return false;
        }
        ins_.push_back(bufs_.size() - 1);
        return true;
    }

    // Refresh the k'th input's bytes in place, for reusing one argv across many calls.
    //
    // A decode step only changes id, pos and mask, so rebuilding the whole argv per token would
    // re-allocate device memory every time for values that are a few kilobytes.
    void refreshInput(size_t k, const ggml_tensor * t) {
        memcpy(bufs_[ins_[k]], t->data, ggml_nbytes(t));
    }

    // Space for a result, uninitialized.
    bool addOutput(const ggml_tensor * t) {
        if (!add(t, /*copy=*/false)) {
            return false;
        }
        outs_.push_back(bufs_.size() - 1);
        return true;
    }

    // A KV cache: a descriptor over a buffer this object does NOT own.
    //
    // The buffer outlives the call by design - that is the whole point of a cache in DRAM - so it is
    // allocated once by the caller and passed in every step. Registering it in bufs_ would free it
    // in the destructor and the next step would read freed device memory.
    //
    // `shape` is in MLIR order, [n_layers, cells, ...rest...], matching CacheSpec's memref.
    bool addCache(void * dev, const std::vector<int64_t> & shape) {
        if (!dev || shape.empty()) {
            return false;
        }
        void * desc = makeDescShape(shape, dev);
        descs_.push_back(desc);
        argv_.push_back(desc);
        return true;
    }

    // An index-typed scalar, e.g. the cache slot.
    //
    // Passed BY VALUE in the argv slot, not by pointer. The generated shim declares every parameter
    // as void* and forwards a[i] straight into _mlir_ciface_forward(..., i64, ...), so an i64
    // argument reads the slot itself. Passing a pointer here makes the address the slot number,
    // which writes the new cell at a garbage offset instead of failing.
    // Returns the argv slot it landed in, so a later call can change it without rebuilding argv.
    size_t addScalar(int64_t v) {
        argv_.push_back((void *) (intptr_t) v);
        return argv_.size() - 1;
    }

    // Overwrite a scalar added earlier. Same by-value convention as addScalar.
    void setScalar(size_t argv_index, int64_t v) {
        argv_[argv_index] = (void *) (intptr_t) v;
    }

    void ** argv() { return argv_.data(); }

    // Device pointer of the k'th addOutput, for reading a result back. Indexed by output order
    // rather than argv position, because caches and scalars sit between the inputs and the outputs.
    void * output(size_t k) const { return bufs_[outs_[k]]; }

  private:
    bool add(const ggml_tensor * t, bool copy) {
        const size_t nb  = ggml_nbytes(t);
        void *       dev = tsi_alloc((int64_t) nb);
        if (!dev) {
            fprintf(stderr, "[tsi-mlir] tsi_alloc failed for %zu bytes. Raise USER_DRAM_SIZE (MiB).\n",
                    nb);
            return false;
        }
        bufs_.push_back(dev);
        if (copy) {
            memcpy(dev, t->data, nb);
        }
        void * desc = makeDesc(t, dev);
        descs_.push_back(desc);
        argv_.push_back(desc);
        return true;
    }

    std::vector<void *> bufs_;
    std::vector<void *> descs_;
    std::vector<void *> argv_;
    std::vector<size_t> ins_;    // indices into bufs_, in addInput order
    std::vector<size_t> outs_;   // indices into bufs_, in addOutput order
};

}  // namespace tsi::driver
