// Public interface of the ggml-graph-to-linalg-MLIR exporter.
//
// Deliberately free of MLIR types. Consumers (WholeGraphHook.cpp, the tools, the test generator) see
// only ggml and std types, so they compile without MLIR's headers on their include path and merely
// link the library. That keeps their compile times unaffected and lets them stay at C++20 while the
// exporter itself is built at C++17 to match LLVM.
//
// Emits MLIR matching the entry-point conventions the TSI compiler expects: a func with the given
// name, `txe.name` attributes on every argument and result, and `llvm.emit_c_interface`.
#pragma once

#include "ggml.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace tsi::mlir_export {

// Thrown for any graph the exporter cannot express. A human-readable reason is printed to stderr
// before the throw, matching the previous emitter's behavior, so callers that catch this to mark a
// case unsupported still get the detail.
struct mlir_export_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A KV cache held in device DRAM as a memref argument and written in place.
//
// This cannot be expressed in the ggml graph: ggml has no memref type, and the exporter has no
// lowering for a strided view, so a per-layer slice of one big buffer is not representable. The cache
// is therefore described here and emitted around the graph rather than built into it.
//
// Emitted as one `memref<n_layers x d0 x d1 x cells, elem, 1>` argument (memory space 1 is DRAM).
// Wherever `read[il]` appears in the graph the exporter substitutes layer il's slice, and after the
// body it appends `append[il]` at cell `slot`. The append width comes from the appended tensor, so
// prefill writing N cells and decode writing 1 use the same path.
struct CacheSpec {
    std::string name;                            // becomes txe.name, e.g. "cache_k"
    int64_t     n_layers = 0;
    int64_t     cells    = 0;                    // capacity L

    // Per layer, both sized n_layers. `read` entries are graph leafs standing for that layer's
    // slice; a null `append` entry skips the write for that layer.
    std::vector<const ggml_tensor *> read;
    std::vector<const ggml_tensor *> append;
};

// How the module is encoded.
//
// Bytecode keeps constant blobs as raw binary; text prints them as hex, which is exactly twice the
// bytes and unusable for a real model's weights. Text stays the default because it is readable and
// every test reads it.
enum class Format {
    Text,
    Bytecode,
};

struct ExportOptions {
    // Function name. The TSI compiler and the host shim both expect "forward".
    std::string func_name = "forward";

    // Leafs that become func arguments, in %arg order -> `txe.name = "input_<i>"`.
    //
    // This list is the whole story: every OTHER leaf the exporter discovers is baked into the IR as
    // a constant. There is no flag and no name heuristic - a leaf is either a per-step input the
    // caller declares here, or it is a constant. Baked leafs must still hold live data when
    // exportGraph runs, since the values are read out there.
    std::vector<const ggml_tensor *> runtime_args;

    // Results, in order -> `txe.name = "res_<i>"`. Empty means "the graph's single output", i.e.
    // its last node. There is deliberately no separate multi-output entry point: one output is
    // just the N=1 case, and keeping two functions in sync had already let them drift apart.
    std::vector<const ggml_tensor *> outputs;

    // KV caches, appended to the argument list after runtime_args, followed by a scalar `index`
    // named "slot" when any cache is present. Caches are never results; they are written in place.
    std::vector<CacheSpec> caches;

    // Encoding of the returned module. Bytecode is required once real weights are baked in.
    Format format = Format::Text;
};

// Builds the graph as an MLIR module, verifies it, and returns the encoded IR.
//
// With Format::Bytecode the returned string holds raw bytes, NUL bytes included, so write it with
// an ofstream opened in binary mode and never treat it as text.
//
// Verification runs before encoding, so a structurally invalid graph fails here, naming the op,
// rather than surfacing later as an opaque parse error inside the Python compiler driver.
//
// Throws mlir_export_error on an unsupported construct or a failed verification.
std::string exportGraph(ggml_cgraph * gf, const ExportOptions & opts);

// Leaf/input tensors in first-seen order. Callers use this to decide which leafs to declare as
// runtime_args; the rest become constants.
std::vector<const ggml_tensor *> discoverLeafs(ggml_cgraph * gf);

}  // namespace tsi::mlir_export
