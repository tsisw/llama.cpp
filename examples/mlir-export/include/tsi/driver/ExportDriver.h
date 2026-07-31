// Automatic MLIR export, compile and run, hooked into llama_context::graph_compute().
//
// With TSI_MLIR_EXPORT=1 a plain `llama-cli -m model.gguf -p "..."` needs nothing else: the driver
// intercepts each forward graph, rebuilds it as one MLIR `func @forward`, compiles it through the TSI
// compiler, runs it, and hands the logits back to llama. There are no wrapper scripts and no manual
// steps, because every step is inside llama itself.
//
// Host-buildable: needs the MLIR exporter and the TSI runtime shim, NOT the tsavorite backend.
//
// | env var | meaning |
// |---|---|
// | TSI_MLIR_EXPORT   | 1 turns the driver on. Unset: every hook is a no-op. |
// | TSI_MLIR_VERIFY   | 1 also diffs the compiled logits against llama's own. |
// | TSI_MLIR_CPU_REF  | 1 also computes the reconstruction on CPU, for a 3-way split. |
// | TSI_MLIR_DIR      | where artifacts are written and cached (default "./tsi-mlir"). |
// | TSI_MLIR_SKIP     | graphs to skip before acting (default 1, llama's warmup graph). |
// | TSI_MLIR_PYTHON   | venv python that has the tsavorite package. |
// | TSI_MLIR_SCRIPT   | compile_graph_fpga.py, if not next to this source tree. |
// | TSI_MLIR_DUMP_GRAPH | 1 writes each intercepted graph's nodes to <dir>/graph-<phase>.txt. |
// | TSI_MLIR_WEIGHT_ARGS | 1 passes weights as arguments instead of baking them in as constants. |
// | TSI_MLIR_CACHE_SUM | 1 fingerprints the device KV cache and the logits after every decode step. |
//
// Comparison is layered because each level costs more than the last. TSI_MLIR_EXPORT alone runs the
// compiled forward and uses its result, comparing against nothing: the point is to run the model
// through the MLIR path, not to check it.
#pragma once

struct ggml_cgraph;
struct ggml_tensor;

// Before llama computes the graph. Classifies the phase and snapshots what the reconstruction will
// need while it is still valid. Always lets the caller continue.
void tsi_mlir_export_before_compute(struct ggml_cgraph * cgraph);

// Scheduler eval-callback (install with ggml_backend_sched_set_eval_callback before compute).
// Snapshots each weight's data while valid during compute; the reconstruction reads it back later.
extern "C" bool tsi_mlir_export_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

// After llama computed the graph, so the TSI runtime is up and the live output holds llama's own
// logits. Exports, compiles, runs, and overwrites the live logits with the compiled ones.
// Returns false; llama has already run and there is nothing for it to skip.
bool tsi_mlir_export_after_compute(struct ggml_cgraph * cgraph);
