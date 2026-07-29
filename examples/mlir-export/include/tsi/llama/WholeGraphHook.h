// Whole-graph interception, hooked directly into llama_context::graph_compute().
// Host-buildable: needs the MLIR exporter and the TSI runtime shim, NOT the tsavorite backend.
//
// Hooks the forward cgraph llama.cpp builds and, depending on $TSI_WHOLEGRAPH:
//   capture  export the whole forward as one MLIR `func @forward` to $TSI_WG_DIR/forward.mlir
//            (+ forward.manifest), then let the normal per-op path run.
//   dump     write $TSI_WG_DIR/graph.txt (every node + srcs), no export. Diagnostic.
//   verify   after the per-op path runs, reconstruct + run the compiled forward and diff its
//            next-token logits/argmax against the per-op reference. Report only.
//   run      like verify, but also overwrite the live logits so llama samples the compiled token.
//
// Env:
//   TSI_WHOLEGRAPH   capture | dump | verify | run   (unset: hooks are no-ops)
//   TSI_WG_DIR       output/manifest dir             (default ".")
//   TSI_WG_LIB       host.so for run/verify          (default "$TSI_WG_DIR/out_fpga/host/host.so")
//   TSI_WG_SKIP      graphs to skip before acting    (default 0; skips llama warmup)
#pragma once

struct ggml_cgraph;

// capture/dump. No-op otherwise. Always lets the caller continue with per-op execution.
void tsi_wholegraph_maybe_capture(struct ggml_cgraph * cgraph);

// Scheduler eval-callback (install with ggml_backend_sched_set_eval_callback before compute).
// Snapshots each weight's data while valid during compute; the reconstruction reads it back later.
extern "C" bool tsi_wholegraph_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

// run/verify. No-op otherwise. Call after the per-op path has run (TSI runtime up, live output holds
// the reference logits). Reconstructs the cache-free graph, runs the compiled forward, and diffs the
// next-token logits/argmax. run also overwrites the live output. Returns false (nothing to skip).
bool tsi_wholegraph_maybe_run(struct ggml_cgraph * cgraph);
