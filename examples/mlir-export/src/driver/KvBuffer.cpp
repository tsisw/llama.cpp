// A ggml buffer type that allocates from the TSI shared DRAM heap.
//
// This is what lets llama keep full ownership of the KV cache while the compiled graph reads it in
// place. llama allocates its cache_k_l*/cache_v_l* tensors from here, so their data already lives in
// device-visible DRAM; the exporter then takes a memref over the very same bytes. There is one cache,
// llama writes it, we read it, and every cache operation llama performs - slot allocation, context
// shift, defrag, seq_rm - keeps working because nothing else mutates it.
//
// The memory is reported as host memory (is_host = true) and that is the whole trick: tsi_alloc
// returns a host-addressable pointer into shared DRAM, so ggml's CPU backend runs SET_ROWS and the
// shift/defrag ops on it directly, with no copies and no backend-support questions. A buffer type that
// claimed to be device memory would force the scheduler to stage every access through a copy.
//
// Buffers are deliberately never freed. tsi_finalize() runs from an atexit handler (see Runtime.h),
// and llama frees its KV buffers from the llama_context destructor, which for a static context runs
// after that - releasing DRAM to a finalized runtime. Leaking a process-lifetime cache is the correct
// trade here: the alternative is a use-after-teardown on every exit.
#include "Config.h"
#include "Runtime.h"     // runtimeUp: tsi_initialize must precede the first tsi_alloc

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "TestModel.h"   // tsi_alloc via HostShimCAPI.h

#include <cstdio>

namespace {

const char * kvBufferTypeName(ggml_backend_buffer_type_t) {
    return "TSI_DRAM";
}

ggml_backend_buffer_t kvAllocBuffer(ggml_backend_buffer_type_t buft, size_t size) {
    // llama allocates its cache during context creation, long before the first compiled call, so the
    // runtime has to come up here rather than lazily in the decode path. Missing this segfaults inside
    // the allocator.
    tsi::driver::runtimeUp();

    void * p = tsi_alloc((int64_t) size);
    if (!p) {
        fprintf(stderr, "[tsi-mlir] tsi_alloc failed for a %zu-byte KV cache buffer. "
                        "Raise USER_DRAM_SIZE (MiB).\n", size);
        return nullptr;
    }

    // Reuse ggml's own from_ptr buffer: it wraps an existing host pointer and, being CPU memory, gets
    // the standard tensor-access implementations for free. Only the reported type is overridden, so
    // ggml_backend_buft_is_host() answers for this type rather than the generic CPU-mapped one.
    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(p, size);
    if (!buf) {
        return nullptr;
    }
    buf->buft = buft;
    fprintf(stderr, "[tsi-mlir] KV cache: %.2f MiB in TSI DRAM at %p\n",
            (double) size / (1024.0 * 1024.0), p);
    return buf;
}

size_t kvAlignment(ggml_backend_buffer_type_t) {
    // Matches ggml's CPU alignment. The cache's per-layer size is a large power-of-two multiple in
    // practice, so tensors land at a uniform stride, but nothing here relies on that: the exporter
    // takes one memref per layer from each tensor's own data pointer.
    return 32;
}

bool kvIsHost(ggml_backend_buffer_type_t) {
    return true;
}

}  // namespace

// Returns the DRAM buffer type, or nullptr when the export path is off, in which case llama keeps its
// normal CPU cache and nothing about its behavior changes.
extern "C" ggml_backend_buffer_type_t tsi_mlir_kv_buffer_type(void) {
    if (!tsi::driver::Config::get().enabled) {
        return nullptr;
    }

    static ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name       = */ kvBufferTypeName,
            /* .alloc_buffer   = */ kvAllocBuffer,
            /* .get_alignment  = */ kvAlignment,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ nullptr,
            /* .is_host        = */ kvIsHost,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };

    // ggml_backend_buft_* helpers reach through to buft->device for some queries, so borrow the CPU
    // device: this really is CPU-accessible memory, just carved from the DRAM heap.
    if (!buft.device) {
        if (ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type()) {
            buft.device = cpu->device;
        }
    }
    return &buft;
}
