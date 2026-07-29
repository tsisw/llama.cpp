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

// True when a leaf is a model weight: a value fixed for the model's lifetime, as opposed to a
// per-step input (token ids, positions, attention mask, KV cache). Matches the GGUF naming
// convention - the core name, with any "BACKEND#" prefix and "#<copy>" suffix stripped, ends in
// ".weight" - and additionally requires live data of a bakeable element type (f32 or i32).
bool isModelWeight(const ggml_tensor * t);

// Split `leafs` into the per-step inputs that stay function arguments and the model weights to
// bake in as constants. Relative order within each output is preserved, so argument indices stay
// the graph's first-seen order with the baked entries removed.
//
// Baking trades IR size for compile-time visibility of the weight values: the compiler can fold,
// pre-tile and place them, and the resulting binary no longer depends on a matching weight buffer
// at run time. Constants are printed in full, so total weight bytes bound what is practical.
void partitionWeights(const std::vector<const ggml_tensor *> & leafs,
                      std::vector<const ggml_tensor *> & args,
                      std::vector<const ggml_tensor *> & consts);

}  // namespace tsi::mlir_export
