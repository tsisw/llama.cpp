// Compile one exported module into a loadable host.so, and cache the result.
//
// A whole-model compile is expensive: SmolLM2-135M f32 measures ~140 s. The cache key is a hash of
// the module bytes, so identical IR reuses the previous binary and a rerun with the same model and
// prompt length skips the compile entirely. Different IR gets a different directory, never a stale
// hit.
#pragma once

#include <string>

namespace tsi::driver {

struct Config;

// The generated void** unpacking shim. Using it avoids a libffi dependency: the argument count is
// baked into the shim at compile time by compile_graph_fpga.py.
using forward_argv_fn = void (*)(void **);

// Writes `mlir` to <dir>/<phase>-<hash>/forward.mlirbc, compiles it if that directory has no
// host.so yet, then dlopens it.
//
// dlopen is RTLD_LOCAL: prefill and decode both define @forward at different arities, and a global
// load would let the first one satisfy the second's symbol lookup.
//
// Returns nullptr on any failure, after reporting the reason. The caller falls back to llama's own
// result rather than aborting the run.
forward_argv_fn buildForward(const std::string & mlir, const std::string & phase,
                             const Config & cfg);

}  // namespace tsi::driver
