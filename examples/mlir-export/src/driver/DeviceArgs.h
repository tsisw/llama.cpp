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

// argv for one forward call, in the ciface order the exporter documents:
// [runtime_args..., caches..., slot?, outputs...].
class DeviceArgs {
  public:
    ~DeviceArgs() {
        for (void * d : bufs_) {
            tsi_dealloc(d);
        }
        for (void * d : descs_) {
            operator delete(d);
        }
    }

    DeviceArgs(const DeviceArgs &)             = delete;
    DeviceArgs & operator=(const DeviceArgs &) = delete;
    DeviceArgs()                               = default;

    // Copies t->data to the device. Returns false when DRAM is exhausted, which is a report-and-skip
    // condition rather than a crash: raise USER_DRAM_SIZE and retry.
    bool addInput(const ggml_tensor * t) { return add(t, /*copy=*/true); }

    // Space for a result, uninitialized.
    bool addOutput(const ggml_tensor * t) { return add(t, /*copy=*/false); }

    void ** argv() { return argv_.data(); }

    // Device pointer of the i'th added entry, for reading a result back.
    void * buffer(size_t i) const { return bufs_[i]; }

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
};

}  // namespace tsi::driver
