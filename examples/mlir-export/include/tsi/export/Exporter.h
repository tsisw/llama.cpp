// Public interface of the ggml-graph-to-linalg-MLIR exporter.
//
// Deliberately free of MLIR types. Consumers (tsi_wholegraph.cpp, decode_run.cpp, the tools) see
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

struct ExportOptions {
    // Function name. The TSI compiler and the host shim both expect "forward".
    std::string func_name = "forward";

    // Leafs that become func arguments, in %arg order -> `txe.name = "input_<i>"`.
    std::vector<const ggml_tensor *> runtime_args;

    // Leafs baked into the IR as arith.constant dense<...> instead of passed in. Weight data must
    // still be live when exportGraph runs, since the values are read out here.
    std::vector<const ggml_tensor *> const_leafs;

    // Results, in order -> `txe.name = "res_<i>"`. Empty means "the graph's single output", i.e.
    // its last node. There is deliberately no separate multi-output entry point: one output is
    // just the N=1 case, and keeping two functions in sync had already let them drift apart.
    std::vector<const ggml_tensor *> outputs;
};

// Builds the graph as an MLIR module, verifies it, and returns the printed IR.
//
// Verification runs before printing, so a structurally invalid graph fails here, naming the op,
// rather than surfacing later as an opaque parse error inside the Python compiler driver.
//
// Throws mlir_export_error on an unsupported construct or a failed verification.
std::string exportGraph(ggml_cgraph * gf, const ExportOptions & opts);

// Leaf/input tensors in first-seen order. Independent of which of them end up as arguments vs
// baked constants; that split is the caller's, expressed through ExportOptions.
std::vector<const ggml_tensor *> discoverLeafs(ggml_cgraph * gf);

}  // namespace tsi::mlir_export
