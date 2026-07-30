// The TSI host runtime's lifecycle, which is mandatory and easy to get wrong.
//
// Both halves have bitten this code before:
//   - tsi_initialize() before the first tsi_alloc(). Missing it segfaults inside the allocator.
//     A TSI build's ggml-tsavorite backend does this during llama_backend_init, but a plain host/FFM
//     build has no such backend, so the driver must.
//   - tsi_finalize() before exit. Missing it hangs the process: the runtime keeps state alive past
//     main() and the wait never completes, so llama-cli looks like it is still computing long after
//     it printed its final timings.
//
// Finalize runs from atexit rather than after the first compiled call, because prefill and decode
// each need the runtime and whichever finished first would have torn it down for the other.
#pragma once

#include "TestModel.h"   // tsi_initialize / tsi_finalize via HostShimCAPI.h

#include <cstdio>
#include <cstdlib>

namespace tsi::driver {

// Brings the runtime up on first call and arranges exactly one teardown at exit.
inline void runtimeUp() {
    static bool up = [] {
        tsi_initialize(1);
        atexit([] { tsi_finalize(); });
        fprintf(stderr, "[tsi-mlir] TSI host runtime initialized (1 TXE)\n");
        return true;
    }();
    (void) up;
}

}  // namespace tsi::driver
