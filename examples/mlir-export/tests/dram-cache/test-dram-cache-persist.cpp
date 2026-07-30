// Does a DRAM memref keep its contents between calls?
//
// The whole KV cache design assumes yes: allocate once, pass the same pointer every step, let the
// graph append in place. Compiling cache_persist.mlir proves the compiler accepts the types. It says
// nothing about the runtime semantics, which is what this checks.
//
// Two calls write different cells. The test passes only if the second call leaves the first cell
// untouched - that is the persistence claim.
//
// Values are raw uint16_t bit patterns. The MLIR is f16, but nothing here interprets them, so no
// half-precision conversion is needed and the comparison stays exact.
//
// Usage: test-dram-cache-persist <host.so>
#include "include/TestModel.h"   // MemRefDescriptor<N>, tsi_alloc, tsi_dealloc

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr int64_t N_CELLS = 8;
constexpr int64_t N_VALS  = 4;

// Descriptor over a device pointer. Same shape/stride convention as the backend's make_desc, so the
// compiled function sees the ABI it was built for.
template <int RANK> MemRefDescriptor<RANK> desc(void * p, const int64_t (&shape)[RANK]) {
    MemRefDescriptor<RANK> d;
    d.base = p;
    d.data = p;
    d.offset = 0;
    for (int i = 0; i < RANK; i++) {
        d.shape[i] = shape[i];
    }
    d.strides[RANK - 1] = 1;
    for (int i = RANK - 2; i >= 0; i--) {
        d.strides[i] = d.strides[i + 1] * d.shape[i + 1];
    }
    return d;
}

int failures = 0;

void expect_cell(const uint16_t * cache, int64_t cell, const uint16_t (&want)[N_VALS], const char * what) {
    const uint16_t * got = cache + cell * N_VALS;
    if (memcmp(got, want, sizeof(want)) == 0) {
        printf("  ok   %s\n", what);
        return;
    }
    printf("  FAIL %s\n       want:", what);
    for (int64_t i = 0; i < N_VALS; i++) printf(" %u", want[i]);
    printf("\n       got: ");
    for (int64_t i = 0; i < N_VALS; i++) printf(" %u", got[i]);
    printf("\n");
    failures++;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <host.so>\n", argv[0]);
        return 2;
    }

    void * lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", argv[1], dlerror());
        return 2;
    }
    auto fwd = (void (*)(void **)) dlsym(lib, "tsi_forward_argv");
    if (!fwd) {
        fprintf(stderr, "dlsym tsi_forward_argv failed: %s\n", dlerror());
        return 2;
    }

    tsi_initialize(1);

    // Allocate the "cache" once. Every call below reuses this pointer - that is the point.
    auto * cache = (uint16_t *) tsi_alloc(N_CELLS * N_VALS * (int64_t) sizeof(uint16_t));
    auto * src   = (uint16_t *) tsi_alloc(N_VALS * (int64_t) sizeof(uint16_t));
    if (!cache || !src) {
        fprintf(stderr, "tsi_alloc failed; raise USER_DRAM_SIZE\n");
        tsi_finalize();
        return 2;
    }
    memset(cache, 0, N_CELLS * N_VALS * sizeof(uint16_t));

    const int64_t cache_shape[2] = { N_CELLS, N_VALS };
    const int64_t src_shape[1]   = { N_VALS };

    auto call = [&](int64_t slot) {
        auto d_src   = desc<1>(src, src_shape);
        auto d_cache = desc<2>(cache, cache_shape);
        // argv order is [inputs..., outputs...]; this function has no results.
        //
        // Memrefs are passed as a POINTER to the descriptor, but a scalar (index) is passed BY VALUE
        // cast into the void* slot. The generated shim declares every parameter as void* and forwards
        // a[i] straight into _mlir_ciface_forward(ptr, i64, ptr), so an i64 argument reads the slot
        // itself, not what it points at. Passing &slot here makes the address the cell number.
        void * argv_[3] = { &d_src, (void *) (intptr_t) slot, &d_cache };
        fwd(argv_);
    };

    const uint16_t first[N_VALS]  = { 0x3C00, 0x4000, 0x4200, 0x4400 };
    const uint16_t second[N_VALS] = { 0x4500, 0x4600, 0x4700, 0x4800 };

    printf("call 1: append to cell 0\n");
    memcpy(src, first, sizeof(first));
    call(0);
    expect_cell(cache, 0, first, "cell 0 holds the first values");

    printf("call 2: append to cell 3\n");
    memcpy(src, second, sizeof(second));
    call(3);
    expect_cell(cache, 3, second, "cell 3 holds the second values");
    // The actual claim under test: call 2 must not disturb what call 1 wrote.
    expect_cell(cache, 0, first, "cell 0 SURVIVED the second call");

    // Untouched cells must still be zero, so a write cannot have spilled past its subview.
    const uint16_t zero[N_VALS] = { 0, 0, 0, 0 };
    expect_cell(cache, 1, zero, "cell 1 untouched");
    expect_cell(cache, 7, zero, "cell 7 untouched");

    tsi_dealloc(src);
    tsi_dealloc(cache);
    tsi_finalize();

    printf(failures ? "\nFAILED (%d)\n" : "\nPASSED\n", failures);
    return failures ? 1 : 0;
}
