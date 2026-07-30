// Every environment variable the driver reads, parsed once, in one place.
//
// Kept separate so the rest of the driver never calls getenv: a knob that only exists here cannot
// drift out of sync with the documentation in ExportDriver.h.
#pragma once

#include <string>

namespace tsi::driver {

struct Config {
    bool enabled = false;   // TSI_MLIR_EXPORT
    bool verify  = false;   // TSI_MLIR_VERIFY
    bool cpu_ref = false;   // TSI_MLIR_CPU_REF
    bool dump    = false;   // TSI_MLIR_DUMP_GRAPH

    int         skip = 1;   // TSI_MLIR_SKIP: llama's first graph is a warmup, not the prompt
    std::string dir;        // TSI_MLIR_DIR
    std::string python;     // TSI_MLIR_PYTHON
    std::string script;     // TSI_MLIR_SCRIPT

    // Where libTsavRTShimCAPI lives. The compile script needs it to link host.o -> host.so, and
    // without it the link fails on undefined tsi_* symbols. Defaults to the path this binary was
    // itself linked against, so the single-command flow needs no extra environment.
    std::string rt_lib_dir;   // TSI_RT_LIB_DIR

    // Parsed on first use and cached. Env is read once per process, so a mid-run change is ignored
    // rather than half-applied.
    static const Config & get();
};

}  // namespace tsi::driver
