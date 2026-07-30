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

    // Pass weights as func arguments instead of baking them in as dense_resource constants.
    //
    // Baking is the default and the design: it lets the compiler fold and place the weights, and the
    // resulting binary needs no matching weight buffer at run time. But the whole constant pool has to
    // fit in one object file, and a big model's does not - TinyLlama-1.1B f32 bakes 4196 MiB and llc
    // fails with "cannot encode offset of relocations; object file too large", because Mach-O
    // relocation offsets are 32-bit. Passing them keeps the object small, at the cost of one argument
    // and one device buffer per weight.
    bool weight_args = false;   // TSI_MLIR_WEIGHT_ARGS

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
